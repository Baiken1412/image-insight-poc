"""Advanced-mode analysis via Qwen3-VL, run on the local GPU by default.

特别需求补充.md 1.6 requires the whole system to run fully offline — no
photo may be uploaded to any external server. LocalQwenTransport loads
Qwen3-VL's weights once (a one-time download during setup, cached locally
afterward) and does inference on-device; nothing is called over the
network at analysis time. This remains the default backend.

RemoteQwenTransport is an opt-in alternative (see image_insight.config's
QWEN_BACKEND) for deployments where the app runs on hardware with no GPU
(e.g. a company desktop) and needs to call out to an OpenAI-compatible
chat-completions endpoint (vLLM/SGLang/Xinference/etc.) serving Qwen3-VL on
a GPU box elsewhere. Choosing that backend means photo bytes DO leave this
machine over the network — it is only appropriate when QWEN_REMOTE_BASE_URL
points at infrastructure the deploying organization itself controls (a
private server on a VPN/intranet), never a public third-party API. That
choice is a deployment decision made via environment variables, not
something this code can verify on its own.

The model call is isolated behind the ChatCompletionTransport protocol so
unit tests can inject a fake transport instead of loading real GPU weights,
and so both transports are interchangeable to QwenVisionAnalyzer/
FastVisionAnalyzer. Anti-hallucination is enforced twice: once via explicit
prompt instructions, and again structurally by image_insight.vlm.schema (a
field claiming "confirmed" without a value, or "unknown" while smuggling in
a value, fails pydantic validation and the whole item is discarded into
parse_error rather than shown to a user).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol

from pydantic import ValidationError

from image_insight.config import load_confidence_config
from image_insight.ocr.pp_ocr import DEFAULT_MIN_TEXT_CONFIDENCE, TextRecognizer, extract_high_confidence_text
from image_insight.taxonomy import (
    FALLBACK_CATEGORY,
    MAJOR_CATEGORIES,
    format_taxonomy_block,
    major_of_subcategory,
    normalize_category,
    normalize_subcategory,
)
from image_insight.vlm.confidence import needs_pending_review
from image_insight.vlm.json_repair import recover_array_objects
from image_insight.vlm.schema import AdvancedAnalysisResult, FieldStatus, ItemAnalysis, ObservedField

logger = logging.getLogger(__name__)
# Without this, logger.info (the [AdvancedVision] performance line below) is
# silently dropped: a logger with no level of its own inherits root's
# default (WARNING), and only logger.exception's ERROR-level OCR-failure
# calls happen to survive via Python's lastResort handler. Scoped to this
# module's own logger only (used exclusively by advanced-mode code — fast
# mode has no logger at all) rather than touching root/basicConfig, so this
# can't change what fast mode or anything else prints.
if not logger.handlers:
    _handler = logging.StreamHandler()
    _handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(message)s"))
    logger.addHandler(_handler)
    logger.setLevel(logging.INFO)

_CATEGORY_LIST = "\n".join(f"   - {c}" for c in MAJOR_CATEGORIES)
_TAXONOMY_BLOCK = format_taxonomy_block()

SYSTEM_PROMPT = f"""\
你是涉案财物拍照分析助手。仔细观察这张照片中的物品，输出结构化 JSON。

硬性规则（违反视为失败）：
1. 只描述你能从这张照片里直接看到的内容，禁止编造品牌、型号或其他你不确定的信息。
2. 下面的 JSON 格式示例中，除 category/subcategory 的可选值列表外，其余所有的值
   （品牌、颜色、文字、备注等）都只是占位符，用来说明格式，跟你现在看到的这张照片
   完全无关。绝对不能把示例里的具体文字（例如某个品牌名、某句备注）抄到你的回答
   里——你回答的每一个值都必须来自你对这张照片的实际观察，如果照片里没有摄像头、
   没有品牌标志，就不要提摄像头、不要写品牌。
3. 对每一个字段，你必须从以下三种状态中选择："confirmed"（清晰可见、确信）、
   "suspected"（有线索但不完全确定，例如部分遮挡的品牌标志）、"unknown"（无法判断）。
4. 如果 status 是 "unknown"，value 字段必须是 null，禁止同时给出 status=unknown 和一个值。
5. 如果 status 是 "confirmed" 或 "suspected"，value 字段必须给出具体内容，不能为空。
6. category、subcategory、brand、model、color、condition 这六个字段，每一个都必须是
   {{"value": ..., "status": ...}} 这样的对象，绝对不能直接写成字符串。
   错误示例（禁止）："category": "手机"
   正确示例："category": {{"value": "手机", "status": "confirmed"}}
7. subcategory 的 value 必须从以下体系中选一个最贴切的细类，一字不差地照抄细类
   名称；category 的 value 照抄该细类所在方括号里的大类名称（系统也会自动用细类
   反推大类，两者不一致时以细类为准）：
{_TAXONOMY_BLOCK}
   如果实在没有任何细类贴切，subcategory 的 value 设为 null、status 设为
   "unknown"；category 的 value 从上面的大类名中选最贴切的一个，都不贴切就写
   "其他/未分类"。
8. 给出一个 category_confidence 字段（0~1 的小数），表示你对 subcategory 这个
   判断的把握程度：非常确定给 0.8 以上，比较犹豫给 0.3~0.5，基本是随便猜给 0.2
   以下——不需要精确，只需要真实反映你的把握程度。
9. appearance_notes 只描述你能直接看到的表面特征本身（颜色、形状、位置、纹理、
   范围大小等），绝对禁止对痕迹、污渍或损伤的成因下结论——那是专业鉴定人员的
   工作，不是你的。控制在 40 字以内，只挑最关键的 1-2 个特征，不需要面面俱到。
   错误示例（禁止，属于诊断结论）："有血迹"、"烧灼痕迹"、"弹孔"、"火烧过的痕迹"
   正确示例（客观描述外观，不下结论）："边缘有大片深红褐色不规则渍迹"、
   "表面有多处黑褐色斑点，边界不规则"
   如果一处痕迹的性质无法仅凭外观确定，就只描述它长什么样，不要猜测它是什么
   造成的，也不要在 condition 或其他字段里换个说法重复下结论。
10. visible_text 最多列出 8 条、每条不超过 15 个字——只挑最有辨识度的关键文字
    （品牌名、型号/编号、年份等），不要逐字抄录说明书、防伪提示、大段介绍文字
    等长段落，超长的一段文字只保留其中最关键的品牌名/编号片段。同一段文字（包
    装、瓶身、标签、说明卡上出现的重复内容）只记一次，绝对不能把同一条文字重复
    列在数组里凑数——包装盒、瓶身、说明书上反复出现的同一句话（如产品名称、防
    伪说明）只算一条，不是看到几次就写几次。
11. 只输出最终的 JSON 结果本身，不要输出任何解释性文字、不要输出 markdown 代码块
    标记（如 ```json）、不要展示你的推理或检查过程。items 数组里每件实际看到的
    物品只输出一次——不要为了"确认"或"补充"而重复输出同一个物品、同一个字段或
    同一段描述。JSON 完整闭合（最后一个右花括号写完）后立即停止，不要在 JSON 后面
    继续写任何总结、说明或免责声明。

JSON 格式说明（items 是数组，图片中每件实际看到的物品各占一个元素；
以下字段名和结构必须照抄，但 value 的具体内容必须换成你对这张照片的真实观察）：
{{
  "items": [
    {{
      "category": {{"value": "电子设备及数字载体", "status": "confirmed"}},
      "subcategory": {{"value": "手机", "status": "confirmed"}},
      "category_confidence": 0.9,
      "brand": {{"value": "<照片上实际可见的品牌文字，看不到就整体省略或设为 unknown>", "status": "confirmed"}},
      "model": {{"value": null, "status": "unknown"}},
      "color": {{"value": "<物品的实际颜色>", "status": "confirmed"}},
      "count": 1,
      "visible_text": ["<最多8条最关键的文字，每条不超过15字，不要抄长段说明文字，没有就留空数组>"],
      "appearance_notes": "<只描述表面看到的样子（颜色/形状/位置/纹理），不要判断痕迹的成因>",
      "condition": {{"value": "<物品的实际成色/状态>", "status": "confirmed"}}
    }}
  ]
}}
"""

CODE_FENCE_PATTERN = re.compile(r"^```(?:json)?\s*|\s*```$", re.MULTILINE)

_OBSERVED_FIELD_KEYS = ("category", "subcategory", "brand", "model", "color", "condition")


def _normalize_observed_field(raw: object) -> object:
    """Coerce a raw JSON value into the {value, status} shape ObservedField expects.

    Qwen3-VL-2B doesn't always follow the nested-object instruction — observed
    in practice, it sometimes answers a field with a bare string or a bare
    null instead of {"value": ..., "status": ...}, even when it clearly does
    know the answer. Discarding an otherwise-good analysis over a pure
    formatting slip throws away real information, so this repairs the shape
    — conservatively: a bare non-empty value becomes "suspected", never
    "confirmed", because the model didn't explicitly commit to a confidence
    level through the required format, and that omission is itself a form of
    uncertainty worth flagging for manual review rather than trusting fully.
    A bare null/empty string stays "unknown".
    """
    if isinstance(raw, dict):
        return raw
    if raw is None or raw == "":
        return {"value": None, "status": "unknown"}
    return {"value": str(raw), "status": "suspected"}


def _normalize_item(item: dict) -> dict:
    normalized = dict(item)
    for key in _OBSERVED_FIELD_KEYS:
        if key in normalized:
            normalized[key] = _normalize_observed_field(normalized[key])

    category = normalized.get("category")
    subcategory = normalized.get("subcategory")
    resolved_sub = None
    if isinstance(subcategory, dict) and subcategory.get("value"):
        resolved_sub = normalize_subcategory(subcategory["value"])

    if resolved_sub is not None:
        subcategory["value"] = resolved_sub
        major = major_of_subcategory(resolved_sub)
        if major is not None and isinstance(category, dict):
            # 细类 is more specific evidence than whatever the model put in
            # category directly — let it win, but never invent a confidence
            # level stronger than what the model actually gave subcategory.
            category["value"] = major
            category["status"] = subcategory["status"]
    elif isinstance(category, dict) and category.get("value") is not None:
        category["value"] = normalize_category(category["value"])

    if isinstance(normalized.get("visible_text"), list):
        normalized["visible_text"] = _clamp_visible_text(normalized["visible_text"])

    return normalized


def _clamp_visible_text(raw_texts: list) -> list[str]:
    """Defense-in-depth backstop for SYSTEM_PROMPT rule 10 (visible_text
    capped at 8 short entries, no duplicates) — a prompt instruction alone
    is not reliable enough on its own (observed in testing: the model can
    still loosely comply, e.g. transcribing one long label verbatim as a
    single very long string, without necessarily looping the exact same
    string repeatedly). Applied unconditionally so this holds even for a
    well-formed, non-runaway response.
    """
    deduped: list[str] = []
    seen: set[str] = set()
    for text in raw_texts:
        if not isinstance(text, str):
            continue
        text = text.strip()
        if not text or text in seen:
            continue
        seen.add(text)
        if len(text) > ADVANCED_VISIBLE_TEXT_MAX_CHARS:
            text = text[:ADVANCED_VISIBLE_TEXT_MAX_CHARS] + "…"
        deduped.append(text)
        if len(deduped) >= ADVANCED_VISIBLE_TEXT_MAX_ITEMS:
            break
    return deduped


class ChatCompletionTransport(Protocol):
    def complete(
        self,
        *,
        image_base64: str,
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        """Return the raw text content of the model's reply.

        max_new_tokens/timeout_seconds/repetition_penalty are optional
        per-call overrides used by advanced mode (see QwenVisionAnalyzer) to
        run its own generation budget and anti-repetition-loop guard
        independent of whatever the transport's own default is — fast mode
        never passes them, so its behavior is unaffected.
        """
        ...


DEFAULT_MAX_IMAGE_PIXELS = 1024 * 1024  # ~1MP; caps vision-token count regardless of source photo size
DEFAULT_MIN_IMAGE_PIXELS = 256 * 16 * 16

# Advanced-mode-only image pre-downscale — a fixed rule, not (yet) an env
# var like AdvancedQwenConfig's fields. Distinct from DEFAULT_MAX_IMAGE_PIXELS
# above: that one bounds vision-tower tokens (already enforced inside the
# processor for both modes); this one avoids handing a huge source photo
# (e.g. 3072x4096) to PIL/base64/the processor at all when it's nowhere near
# needed at that resolution.
ADVANCED_RESIZE_PIXEL_THRESHOLD = 4_000_000
ADVANCED_RESIZE_MAX_SIDE = 2048

# Fallback only for QwenVisionAnalyzer instantiated without an explicit
# advanced config (e.g. direct construction in tests) — production wiring
# goes through image_insight.config.load_advanced_qwen_config() (see
# image_insight/api/main.py).
ADVANCED_DEFAULT_MAX_NEW_TOKENS = 4096
ADVANCED_DEFAULT_TIMEOUT_SECONDS = 300.0
ADVANCED_DEFAULT_REPETITION_PENALTY = 1.15

# Hard cap on visible_text: how many distinct strings and how long each may
# be, enforced defensively in _normalize_item AFTER the model responds (in
# addition to the prompt asking for this directly) — a text-dense photo
# (product packaging, labels) is exactly the case that pushed real requests
# to 16+ minutes by having the model transcribe/repeat long label text
# verbatim instead of extracting only the few most identifying strings.
ADVANCED_VISIBLE_TEXT_MAX_ITEMS = 8
ADVANCED_VISIBLE_TEXT_MAX_CHARS = 30


@dataclass
class ImageResizeInfo:
    """What _prepare_advanced_image actually did to one photo — logged
    verbatim by QwenVisionAnalyzer._log_performance, not otherwise used."""

    original_size: tuple[int, int]
    original_pixels: int
    processed_size: tuple[int, int]
    processed_pixels: int
    resized: bool


def _prepare_advanced_image(image_bytes: bytes) -> tuple[str, ImageResizeInfo]:
    """Advanced-mode-only pre-downscale for large source photos.

    A 3072x4096 (~12.6MP) photo doesn't need to survive PIL-decode +
    base64-encode + the processor's own internal resize at full resolution
    just to end up bounded to ~1MP by DEFAULT_MAX_IMAGE_PIXELS anyway — this
    does the size check up front and, above ADVANCED_RESIZE_PIXEL_THRESHOLD,
    downscales (preserving aspect ratio, never upscaling, never cropping) so
    the longer side is at most ADVANCED_RESIZE_MAX_SIDE before any of that.
    Never touches the file on disk — this operates on bytes already read
    into memory and returns a new in-memory copy; the original upload is
    untouched and OCR (image_insight.ocr) still reads the original path
    directly. Below the threshold, the original bytes are reused as-is (no
    re-encode), so small/medium photos pay zero extra cost.

    A photo PIL can't decode is not this function's problem to raise on —
    resizing is a performance optimization, not a validation gate, so an
    undecodable image is passed through byte-for-byte unchanged and the
    real decode failure surfaces exactly where it always did, inside
    LocalQwenTransport.complete's own Image.open call (caught by
    QwenVisionAnalyzer.analyze's try/except like any other transport error).
    """
    try:
        from PIL import Image

        with Image.open(io.BytesIO(image_bytes)) as img:
            img = img.convert("RGB")
            original_size = img.size
            original_pixels = original_size[0] * original_size[1]

            if original_pixels <= ADVANCED_RESIZE_PIXEL_THRESHOLD:
                processed_bytes = image_bytes
                processed_size = original_size
                resized = False
            else:
                scale = ADVANCED_RESIZE_MAX_SIDE / max(original_size)
                new_size = (
                    max(1, round(original_size[0] * scale)),
                    max(1, round(original_size[1] * scale)),
                )
                resized_img = img.resize(new_size, Image.LANCZOS)
                buffer = io.BytesIO()
                resized_img.save(buffer, format="PNG")
                processed_bytes = buffer.getvalue()
                processed_size = new_size
                resized = True
    except Exception:
        processed_bytes = image_bytes
        original_size = (0, 0)
        processed_size = (0, 0)
        resized = False

    original_pixels = original_size[0] * original_size[1]
    processed_pixels = processed_size[0] * processed_size[1]
    image_base64 = base64.b64encode(processed_bytes).decode("ascii")
    info = ImageResizeInfo(
        original_size=original_size,
        original_pixels=original_pixels,
        processed_size=processed_size,
        processed_pixels=processed_pixels,
        resized=resized,
    )
    return image_base64, info


@dataclass
class GenerationStats:
    """Per-call token/timing accounting, read back by QwenVisionAnalyzer for
    its performance log — see LocalQwenTransport.complete and
    LocalQwenTransport.last_usage. Fast mode never reads this."""

    prompt_tokens: int
    completion_tokens: int
    finish_reason: str  # "stop" | "length" | "timeout"
    inference_time: float
    model_start_time: float
    model_end_time: float


class _DeadlineStoppingCriteria:
    """Stops model.generate() once a wall-clock deadline passes.

    transformers checks every registered stopping criterion once per
    decoded token, so this actually halts generation (and frees the GPU)
    at the granularity of one more token past the deadline — unlike an
    HTTP-layer timeout, which would only abandon the response while
    inference kept running to completion in its worker thread. Duck-typed
    to the callable shape transformers.StoppingCriteriaList expects
    (input_ids, scores) -> bool; no need to import the base class.
    """

    def __init__(self, deadline: float):
        self._deadline = deadline
        self.timed_out = False

    def __call__(self, input_ids, scores, **kwargs) -> bool:  # noqa: ANN001 - transformers' own signature
        if time.monotonic() >= self._deadline:
            self.timed_out = True
            return True
        return False


class LocalQwenTransport:
    """Real transport: Qwen3-VL loaded on-device via transformers, no network calls.

    Model weights are downloaded once (cached under ~/.cache/huggingface)
    and every analyze() call after that runs purely on the local GPU/CPU.

    Qwen-VL's image processor tokenizes at native resolution by default
    (observed ceiling: 16.7 million pixels — effectively unbounded), so a
    single ~4000x3000 photo from this dataset can demand ~9-10GB just for
    the vision tower, which OOMs an 8GB card outright, or on a desktop GPU
    also shared with a browser/other apps, OOMs even sooner. max_pixels
    caps the resolution the model actually sees (downscaled, not cropped)
    so memory use is bounded no matter how large the source photo is.
    """

    def __init__(
        self,
        model_id: str,
        max_new_tokens: int = 1024,
        max_pixels: int = DEFAULT_MAX_IMAGE_PIXELS,
        min_pixels: int = DEFAULT_MIN_IMAGE_PIXELS,
    ):
        import torch
        from transformers import AutoProcessor, Qwen3VLForConditionalGeneration

        self._torch = torch
        self._model = Qwen3VLForConditionalGeneration.from_pretrained(
            model_id,
            dtype="auto" if torch.cuda.is_available() else torch.float32,
            device_map="auto",
            # Explicit rather than relying on transformers' default resolution —
            # sdpa (PyTorch's native fused attention) is broadly available and
            # meaningfully faster than the "eager" fallback some environments
            # silently pick. torch.compile was evaluated as a further speedup on
            # top of this but rejected: it hard-fails on this project's Windows
            # deployment target with "Cannot find a working triton installation"
            # (Triton has no reliable Windows wheel), so it's not a usable lever here.
            attn_implementation="sdpa",
        )
        self._processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self._max_new_tokens = max_new_tokens
        # Advanced mode reads this back immediately after complete() returns
        # (see QwenVisionAnalyzer._log_performance) via getattr(transport,
        # "last_usage", None) — fast mode never reads it, so populating it
        # unconditionally on every call is a no-op as far as fast mode is
        # concerned, just bookkeeping nobody looks at.
        self.last_usage: Optional[GenerationStats] = None

    def complete(
        self,
        *,
        image_base64: str,
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        from PIL import Image

        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        messages = [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": system_prompt},
                ],
            }
        ]
        inputs = self._processor.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=True,
            return_tensors="pt",
        ).to(self._model.device)

        effective_max_new_tokens = self._max_new_tokens if max_new_tokens is None else max_new_tokens
        effective_repetition_penalty = 1.15 if repetition_penalty is None else repetition_penalty

        # Only advanced-mode calls pass timeout_seconds — fast-mode calls
        # (which never pass it) get stopping_criteria=None, identical to
        # this method's behavior before this parameter existed.
        stopping_criteria = None
        deadline_criterion: Optional[_DeadlineStoppingCriteria] = None
        if timeout_seconds is not None:
            from transformers import StoppingCriteriaList

            deadline_criterion = _DeadlineStoppingCriteria(time.monotonic() + timeout_seconds)
            stopping_criteria = StoppingCriteriaList([deadline_criterion])

        try:
            prompt_tokens = int(inputs.input_ids.shape[-1])
            model_start_time = time.time()
            start = time.monotonic()
            # Greedy decoding + a repetition penalty: more reliable than
            # sampling for structured JSON extraction, where we want
            # consistency rather than creative variety. repetition_penalty
            # is a soft per-token logit adjustment (not a hard mask), so —
            # unlike no_repeat_ngram_size, which was tried and reverted here
            # (it forced the model into garbled/invalid output at any
            # threshold tested, a known risk of hard n-gram blocking under
            # greedy decoding) — raising it for advanced mode still lets the
            # model repeat a token when appropriate, just makes it costlier.
            # 1.15 (fast mode's value, used here when this isn't overridden)
            # reproduces this method's exact behavior before this override
            # param existed.
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=effective_max_new_tokens,
                do_sample=False,
                repetition_penalty=effective_repetition_penalty,
                stopping_criteria=stopping_criteria,
            )
            inference_time = time.monotonic() - start
            model_end_time = time.time()
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            completion_tokens = int(generated_ids_trimmed[0].shape[-1]) if generated_ids_trimmed else 0
            if deadline_criterion is not None and deadline_criterion.timed_out:
                finish_reason = "timeout"
            elif completion_tokens >= effective_max_new_tokens:
                finish_reason = "length"
            else:
                finish_reason = "stop"
            self.last_usage = GenerationStats(
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                finish_reason=finish_reason,
                inference_time=inference_time,
                model_start_time=model_start_time,
                model_end_time=model_end_time,
            )
            output_text = self._processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            return output_text[0]
        finally:
            # Release cached activations immediately rather than waiting for
            # Python GC — this is a long-lived server process handling many
            # differently-sized images back to back, so stale cached blocks
            # from a previous large image should not linger and fragment.
            del inputs
            if self._torch.cuda.is_available():
                self._torch.cuda.empty_cache()


class RemoteQwenTransport:
    """Alternative transport: calls a remote OpenAI-compatible chat-completions
    endpoint (vLLM/SGLang/Xinference/etc. serving Qwen3-VL) instead of loading
    weights on this machine — for deployments on GPU-less hardware (e.g. a
    company desktop with no local GPU) that need to reach a cloud/remote GPU
    box. See this module's docstring for the offline-requirement caveat: this
    is opt-in only (image_insight.config.QWEN_BACKEND), and the operator is
    responsible for pointing base_url at infrastructure they control, not a
    public third-party API.

    Implements the same ChatCompletionTransport shape as LocalQwenTransport
    (including the last_usage bookkeeping QwenVisionAnalyzer's performance
    log reads), so the two are drop-in interchangeable and QwenVisionAnalyzer/
    FastVisionAnalyzer need no changes to use either one.
    """

    def __init__(
        self,
        base_url: str,
        model_id: str,
        api_key: Optional[str] = None,
        max_new_tokens: int = 1024,
        request_timeout: float = 300.0,
    ):
        self._base_url = base_url.rstrip("/")
        self._model_id = model_id
        self._api_key = api_key
        self._max_new_tokens = max_new_tokens
        self._request_timeout = request_timeout
        # Same field name/shape as LocalQwenTransport.last_usage — read back
        # by QwenVisionAnalyzer._log_performance via getattr(..., None).
        self.last_usage: Optional[GenerationStats] = None

    def complete(
        self,
        *,
        image_base64: str,
        system_prompt: str,
        max_new_tokens: Optional[int] = None,
        timeout_seconds: Optional[float] = None,
        repetition_penalty: Optional[float] = None,
    ) -> str:
        import requests
        from PIL import Image

        # Re-encode as JPEG so the data: URI's declared mime type always
        # matches the actual bytes, regardless of what format the caller's
        # image_base64 happened to be in (PNG after _prepare_advanced_image's
        # resize path, or the original upload's own format otherwise) — a
        # server that trusts the declared mime over sniffing the bytes would
        # otherwise fail to decode a mismatched pairing.
        image = Image.open(io.BytesIO(base64.b64decode(image_base64))).convert("RGB")
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=90)
        data_url = "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode("ascii")

        effective_max_new_tokens = self._max_new_tokens if max_new_tokens is None else max_new_tokens
        effective_timeout = self._request_timeout if timeout_seconds is None else timeout_seconds

        payload = {
            "model": self._model_id,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "image_url", "image_url": {"url": data_url}},
                        {"type": "text", "text": system_prompt},
                    ],
                }
            ],
            "max_tokens": effective_max_new_tokens,
            # Greedy-equivalent, matching LocalQwenTransport's do_sample=False
            # — structured JSON extraction wants consistency, not variety.
            "temperature": 0,
        }
        if repetition_penalty is not None:
            # vLLM's OpenAI-compatible server accepts this as an extra
            # sampling param; a strictly spec-only server would reject an
            # unknown field with 422 rather than silently ignoring it, so
            # this is only safe to rely on against vLLM/SGLang-family
            # servers — which is the expected deployment target here.
            payload["repetition_penalty"] = repetition_penalty

        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"

        model_start_time = time.time()
        start = time.monotonic()
        try:
            response = requests.post(
                f"{self._base_url}/chat/completions",
                json=payload,
                headers=headers,
                timeout=effective_timeout,
            )
            response.raise_for_status()
        except requests.exceptions.Timeout as exc:
            raise TimeoutError(
                f"远程 Qwen 服务调用超时（超过 {effective_timeout:.0f} 秒）：{self._base_url}"
            ) from exc
        except requests.exceptions.RequestException as exc:
            raise RuntimeError(f"远程 Qwen 服务调用失败（{self._base_url}）：{exc}") from exc

        inference_time = time.monotonic() - start
        data = response.json()
        usage = data.get("usage") or {}
        choice = data["choices"][0]
        self.last_usage = GenerationStats(
            prompt_tokens=int(usage.get("prompt_tokens", 0)),
            completion_tokens=int(usage.get("completion_tokens", 0)),
            finish_reason=str(choice.get("finish_reason", "stop")),
            inference_time=inference_time,
            model_start_time=model_start_time,
            model_end_time=time.time(),
        )
        return choice["message"]["content"]


class QwenVisionAnalyzer:
    """Advanced-mode entry point: image path in, validated result out.

    Optionally cross-checks with a local OCR pass (特别需求补充.md's
    "VLM 负责看懂，OCR 负责看清字" principle — see image_insight/ocr/pp_ocr.py).
    OCR is supplementary: its failure must never void an otherwise-good VLM
    result, so it's isolated in its own try/except.
    """

    def __init__(
        self,
        transport: ChatCompletionTransport,
        model_version: str,
        ocr: Optional[TextRecognizer] = None,
        ocr_min_confidence: float = DEFAULT_MIN_TEXT_CONFIDENCE,
        max_new_tokens: int = ADVANCED_DEFAULT_MAX_NEW_TOKENS,
        timeout_seconds: float = ADVANCED_DEFAULT_TIMEOUT_SECONDS,
        repetition_penalty: float = ADVANCED_DEFAULT_REPETITION_PENALTY,
    ):
        self._transport = transport
        self._model_version = model_version
        self._ocr = ocr
        self._ocr_min_confidence = ocr_min_confidence
        # Advanced mode's own generation budget — independent of whatever
        # max_new_tokens the shared transport was constructed with (that one
        # still governs fast mode). See image_insight.config.AdvancedQwenConfig.
        self._max_new_tokens = max_new_tokens
        self._timeout_seconds = timeout_seconds
        self._repetition_penalty = repetition_penalty

    def analyze(self, image_path: Path) -> AdvancedAnalysisResult:
        start_total = time.monotonic()
        original_bytes = Path(image_path).read_bytes()
        image_base64, resize_info = _prepare_advanced_image(original_bytes)
        try:
            raw_text = self._transport.complete(
                image_base64=image_base64,
                system_prompt=SYSTEM_PROMPT,
                max_new_tokens=self._max_new_tokens,
                timeout_seconds=self._timeout_seconds,
                repetition_penalty=self._repetition_penalty,
            )
            result = self._parse(raw_text)
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to manual review, not crash
            result = AdvancedAnalysisResult(items=[], model_version=self._model_version, parse_error=str(exc))
            self._log_performance(image_path, resize_info, start_total, usage=None)
            return result

        usage = getattr(self._transport, "last_usage", None)
        if usage is not None and usage.finish_reason == "timeout" and not result.items:
            # Nothing usable was salvaged from the partial output before the
            # deadline hit — surface a specific, actionable message instead
            # of a bare JSONDecodeError.
            result = AdvancedAnalysisResult(
                items=[],
                model_version=self._model_version,
                parse_error=(
                    f"高级分析超时（超过 {self._timeout_seconds:.0f} 秒），"
                    "请重新分析或尝试降低图片复杂度。"
                ),
            )
        self._log_performance(image_path, resize_info, start_total, usage=usage)
        self._augment_with_ocr(result, image_path)
        return result

    def _log_performance(
        self,
        image_path: Path,
        resize_info: ImageResizeInfo,
        start_total: float,
        usage: Optional[GenerationStats],
    ) -> None:
        total_time = time.monotonic() - start_total
        parts = [
            f"file={Path(image_path).name}",
            f"original_size={resize_info.original_size[0]}x{resize_info.original_size[1]}",
            f"original_pixels={resize_info.original_pixels}",
            f"processed_size={resize_info.processed_size[0]}x{resize_info.processed_size[1]}",
            f"processed_pixels={resize_info.processed_pixels}",
            f"resized={'true' if resize_info.resized else 'false'}",
            f"max_new_tokens={self._max_new_tokens}",
        ]
        if usage is not None:
            parts += [
                f"prompt_tokens={usage.prompt_tokens}",
                f"completion_tokens={usage.completion_tokens}",
                f"total_tokens={usage.prompt_tokens + usage.completion_tokens}",
                f"model_start_time={usage.model_start_time:.3f}",
                f"model_end_time={usage.model_end_time:.3f}",
                f"inference_time={usage.inference_time:.1f}s",
                f"finish_reason={usage.finish_reason}",
                f"timeout={'true' if usage.finish_reason == 'timeout' else 'false'}",
            ]
        parts.append(f"total_time={total_time:.1f}s")
        logger.info("[AdvancedVision] " + " ".join(parts))

    def _parse(self, raw_text: str) -> AdvancedAnalysisResult:
        cleaned = CODE_FENCE_PATTERN.sub("", raw_text).strip()
        truncated = False
        try:
            payload = json.loads(cleaned)
            raw_items = payload["items"]
        except (json.JSONDecodeError, KeyError, TypeError) as exc:
            # Generation may have hit max_new_tokens before the JSON closed
            # (see image_insight.vlm.json_repair) — try to salvage whatever
            # items DID finish generating before giving up entirely.
            recovered = recover_array_objects(cleaned, "items")
            if not recovered:
                return AdvancedAnalysisResult(
                    items=[], model_version=self._model_version, parse_error=f"{type(exc).__name__}: {exc}"
                )
            raw_items = recovered
            truncated = True
        try:
            items = [ItemAnalysis(**_normalize_item(item)) for item in raw_items]
        except (TypeError, ValidationError) as exc:
            return AdvancedAnalysisResult(
                items=[], model_version=self._model_version, parse_error=f"{type(exc).__name__}: {exc}"
            )
        thresholds = load_confidence_config()
        for item in items:
            item.category_pending_review = needs_pending_review(
                item.category_confidence, low_confidence_threshold=thresholds.low_confidence_threshold
            )
        return AdvancedAnalysisResult(items=items, model_version=self._model_version, truncated=truncated)

    def _augment_with_ocr(self, result: AdvancedAnalysisResult, image_path: Path) -> None:
        if self._ocr is None or result.parse_error is not None:
            return
        try:
            boxes = self._ocr.read_text(image_path)
            texts = extract_high_confidence_text(boxes, min_confidence=self._ocr_min_confidence)
        except Exception:  # noqa: BLE001 - OCR is a supplementary cross-check, not a hard dependency
            logger.exception("OCR text extraction failed; continuing without OCR cross-check")
            return
        if not texts:
            return
        if len(result.items) == 1:
            # Only case where OCR text can be attributed to a specific item
            # with any confidence — no bounding boxes to localize otherwise.
            item = result.items[0]
            item.visible_text = list(dict.fromkeys(item.visible_text + texts))
        else:
            result.ocr_text = sorted(set(result.ocr_text) | set(texts))


_STATUS_PRIORITY = {FieldStatus.CONFIRMED: 2, FieldStatus.SUSPECTED: 1, FieldStatus.UNKNOWN: 0}


def _better_field(a: ObservedField, b: ObservedField) -> ObservedField:
    return a if _STATUS_PRIORITY[a.status] >= _STATUS_PRIORITY[b.status] else b


def _merge_item_bucket(bucket: list[ItemAnalysis]) -> ItemAnalysis:
    merged = bucket[0]
    for item in bucket[1:]:
        merged = ItemAnalysis(
            category=_better_field(merged.category, item.category),
            subcategory=_better_field(merged.subcategory, item.subcategory),
            brand=_better_field(merged.brand, item.brand),
            model=_better_field(merged.model_name, item.model_name),
            color=_better_field(merged.color, item.color),
            count=max(merged.count, item.count),
            visible_text=sorted(set(merged.visible_text) | set(item.visible_text)),
            appearance_notes="；".join(
                dict.fromkeys(text for text in (merged.appearance_notes, item.appearance_notes) if text)
            ),
            condition=_better_field(merged.condition, item.condition),
            category_confidence=max(merged.category_confidence, item.category_confidence),
            category_pending_review=merged.category_pending_review or item.category_pending_review,
        )
    return merged


def analyze_advanced_group(analyzer: QwenVisionAnalyzer, photo_paths: list[Path]) -> AdvancedAnalysisResult:
    """Merge advanced-mode results across multiple angle photos of the SAME
    physical item (特别需求补充.md §1.2), mirroring fast mode's
    analyze_goods_group.

    Unlike fast mode, ItemAnalysis has no free-text name to match items
    across photos on — the closest stable identifier available is the
    resolved category (大类). This is a known, honest simplification: if a
    single angle photo genuinely contains two DIFFERENT items of the same
    大类, they will be merged into one record here. That matches this
    system's actual use case (one goods_id = one physical item,
    photographed from several sides) but would misbehave for a truly
    multi-item scene photographed independently from multiple angles.
    """
    if not photo_paths:
        return AdvancedAnalysisResult(items=[], model_version="unknown")

    per_photo = [analyzer.analyze(p) for p in photo_paths]
    model_version = per_photo[0].model_version

    usable = [r for r in per_photo if r.parse_error is None]
    if not usable:
        errors = "; ".join(r.parse_error for r in per_photo if r.parse_error)
        return AdvancedAnalysisResult(items=[], model_version=model_version, parse_error=errors)

    buckets: dict[str, list[ItemAnalysis]] = {}
    for result in usable:
        for item in result.items:
            key = item.category.value or FALLBACK_CATEGORY
            buckets.setdefault(key, []).append(item)

    merged_items = [_merge_item_bucket(bucket) for bucket in buckets.values()]
    ocr_text = sorted({text for r in usable for text in r.ocr_text})
    truncated = any(r.truncated for r in usable)
    return AdvancedAnalysisResult(
        items=merged_items, model_version=model_version, ocr_text=ocr_text, truncated=truncated
    )

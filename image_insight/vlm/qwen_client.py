"""Advanced-mode analysis via Qwen3-VL, run entirely on the local GPU.

特别需求补充.md 1.6 requires the whole system to run fully offline — no
photo may be uploaded to any external server. LocalQwenTransport loads
Qwen3-VL's weights once (a one-time download during setup, cached locally
afterward) and does inference on-device; nothing is called over the
network at analysis time.

The model call is isolated behind the ChatCompletionTransport protocol so
unit tests can inject a fake transport instead of loading real GPU weights.
Anti-hallucination is enforced twice: once via explicit prompt instructions,
and again structurally by image_insight.vlm.schema (a field claiming
"confirmed" without a value, or "unknown" while smuggling in a value, fails
pydantic validation and the whole item is discarded into parse_error rather
than shown to a user).
"""
from __future__ import annotations

import base64
import io
import json
import logging
import re
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
   工作，不是你的。
   错误示例（禁止，属于诊断结论）："有血迹"、"烧灼痕迹"、"弹孔"、"火烧过的痕迹"
   正确示例（客观描述外观，不下结论）："边缘有大片深红褐色不规则渍迹"、
   "表面有多处黑褐色斑点，边界不规则"
   如果一处痕迹的性质无法仅凭外观确定，就只描述它长什么样，不要猜测它是什么
   造成的，也不要在 condition 或其他字段里换个说法重复下结论。
10. 只输出 JSON，不要输出任何解释性文字或 markdown 代码块标记。

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
      "visible_text": ["<照片上实际可见的文字，逐条列出，没有就留空数组>"],
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

    return normalized


class ChatCompletionTransport(Protocol):
    def complete(self, *, image_base64: str, system_prompt: str) -> str:
        """Return the raw text content of the model's reply."""
        ...


DEFAULT_MAX_IMAGE_PIXELS = 1024 * 1024  # ~1MP; caps vision-token count regardless of source photo size
DEFAULT_MIN_IMAGE_PIXELS = 256 * 16 * 16


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
            model_id, dtype="auto" if torch.cuda.is_available() else torch.float32, device_map="auto"
        )
        self._processor = AutoProcessor.from_pretrained(model_id, min_pixels=min_pixels, max_pixels=max_pixels)
        self._max_new_tokens = max_new_tokens

    def complete(self, *, image_base64: str, system_prompt: str) -> str:
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

        try:
            # Greedy decoding + a mild repetition penalty: more reliable
            # than sampling for structured JSON extraction, where we want
            # consistency rather than creative variety.
            generated_ids = self._model.generate(
                **inputs,
                max_new_tokens=self._max_new_tokens,
                do_sample=False,
                repetition_penalty=1.15,
            )
            generated_ids_trimmed = [
                out_ids[len(in_ids) :] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
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
    ):
        self._transport = transport
        self._model_version = model_version
        self._ocr = ocr
        self._ocr_min_confidence = ocr_min_confidence

    def analyze(self, image_path: Path) -> AdvancedAnalysisResult:
        image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        try:
            raw_text = self._transport.complete(image_base64=image_base64, system_prompt=SYSTEM_PROMPT)
            result = self._parse(raw_text)
        except Exception as exc:  # noqa: BLE001 - any failure must degrade to manual review, not crash
            return AdvancedAnalysisResult(items=[], model_version=self._model_version, parse_error=str(exc))
        self._augment_with_ocr(result, image_path)
        return result

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

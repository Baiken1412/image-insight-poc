"""Fast-mode analysis: open-vocabulary item naming and counting, in
natural language — not restricted to a fixed category list.

Originally fast mode used RF-DETR's pretrained COCO classes (see
image_insight/detection/phone_counter.py), which only covers 5 categories
(cell phone, laptop, backpack, handbag, suitcase). Per product feedback,
that's too narrow — fast mode should describe whatever it actually sees,
in the model's own words, same as a human would. This reuses the same
on-device Qwen3-VL model/transport as advanced mode (no extra VRAM cost),
just with a short, count-only prompt so it stays fast and simple, matching
快速模式's "快、准、结果简单" goal from the product spec — no brand/model/
text detail, only "what is it, how many."

Each item is also tagged with one of the official 84 细类 (see
image_insight/taxonomy.py), with its 大类 derived from that — per product
feedback that results must map onto the taxonomy already defined for this
system, not be displayed as an unstructured blob. A model reply that omits
或无法归入任何 细类 still gets a 大类-only classification rather than being
discarded.

特别需求补充.md §1.3/§1.4 also requires a confidence score and a
threshold-driven "待核实" fallback — see image_insight/vlm/confidence.py for
why that confidence is a single self-reported number rather than a ranked
Top-3 list (the ranked-list version was tried first and measurably slowed
fast mode down for little UI benefit).
"""
from __future__ import annotations

import base64
import json
from pathlib import Path
from typing import Optional

from pydantic import BaseModel, Field, ValidationError, computed_field, field_validator, model_validator

from image_insight.config import load_confidence_config
from image_insight.taxonomy import (
    format_taxonomy_block,
    major_of_subcategory,
    needs_forced_review,
    normalize_category,
    normalize_subcategory,
)
from image_insight.vlm.confidence import coerce_confidence, needs_pending_review
from image_insight.vlm.json_repair import recover_array_objects
from image_insight.vlm.qwen_client import CODE_FENCE_PATTERN, ChatCompletionTransport

_TAXONOMY_BLOCK = format_taxonomy_block()

FAST_MODE_PROMPT = f"""\
仔细观察这张照片，列出照片中出现的所有物品种类和对应数量。

规则：
1. 物品名称（name）用你自己的语言自然描述，不要局限于固定的类别列表，看到什么就
   写什么（例如"钱包"、"螺丝刀"、"运动鞋"、"文件夹"、"充电线"都可以）。
2. 细类（subcategory）必须从以下体系中选一个最贴切的，一字不差地照抄细类名称；
   category 字段照抄该细类所在方括号里的大类名称（系统也会自动用细类反推大类，
   两者不一致时以细类为准，不需要你反复检查）：
{_TAXONOMY_BLOCK}
   如果实在没有任何细类贴切，subcategory 留空字符串 ""，category 从上面的大类名
   中选最贴切的一个，都不贴切就写"其他/未分类"。
3. 只统计你能在照片里实际看清楚、数得清的物品，不要编造照片里没有的东西。
4. 同一种物品如果有多个，合并成一条并给出数量；不同种类分开列出。
5. 给出一个 confidence 字段（0~1 的小数），表示你对 subcategory 这个判断的把握
   程度：非常确定给 0.8 以上，比较犹豫给 0.3~0.5，基本是随便猜给 0.2 以下——不需要
   精确，只需要真实反映你的把握程度。
6. 不需要品牌、型号、文字等细节，只要名称、大类、细类、数量和置信度。
7. 只输出 JSON，不要输出任何解释性文字或 markdown 代码块标记。

输出格式：
{{"items": [{{"name": "手机", "category": "电子设备及数字载体", "subcategory": "手机",
  "count": 1, "confidence": 0.9}}]}}
"""


class FastModeItem(BaseModel):
    name: str
    category: str
    subcategory: Optional[str] = None
    count: int = Field(ge=1)
    confidence: float = 0.0

    @field_validator("confidence", mode="before")
    @classmethod
    def _coerce_confidence(cls, value: object) -> float:
        return coerce_confidence(value)

    # Set by FastVisionAnalyzer._parse after construction, per
    # 特别需求补充.md §1.4 — never computed at construction time, so plain
    # model instantiation (e.g. in tests) defaults to False.
    pending_review: bool = False

    @model_validator(mode="after")
    def _resolve_taxonomy(self) -> "FastModeItem":
        resolved_sub = normalize_subcategory(self.subcategory) if self.subcategory else None
        if resolved_sub is not None:
            self.subcategory = resolved_sub
            major = major_of_subcategory(resolved_sub)
            if major is not None:
                self.category = major
                return self
        else:
            self.subcategory = None
        self.category = normalize_category(self.category)
        return self

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_sensitive_category(self) -> bool:
        return needs_forced_review(self.category, self.subcategory)


class FastModeResult(BaseModel):
    items: list[FastModeItem]
    parse_error: Optional[str] = None
    # True when the model's raw response was cut off before the JSON closed
    # and _parse salvaged whatever complete items it could (see
    # image_insight.vlm.json_repair) — the photo may have had more items
    # than are listed here, not because the model finished but because
    # generation ran out of budget mid-way. See AdvancedAnalysisResult's
    # matching field for the fuller rationale.
    truncated: bool = False

    @property
    def counts(self) -> dict[str, int]:
        return {item.name: item.count for item in self.items}

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_manual_review(self) -> bool:
        return self.parse_error is not None or self.truncated or any(i.pending_review for i in self.items)


class FastVisionAnalyzer:
    """Fast-mode entry point: image path in, named+categorized+counted items out.

    Shares its transport (and therefore the loaded model) with advanced
    mode's QwenVisionAnalyzer — pass the same LocalQwenTransport instance
    to both to avoid loading Qwen3-VL twice.
    """

    def __init__(self, transport: ChatCompletionTransport):
        self._transport = transport

    def analyze(self, image_path: Path) -> FastModeResult:
        image_base64 = base64.b64encode(Path(image_path).read_bytes()).decode("ascii")
        try:
            raw_text = self._transport.complete(image_base64=image_base64, system_prompt=FAST_MODE_PROMPT)
            return self._parse(raw_text)
        except Exception as exc:  # noqa: BLE001 - any failure must degrade gracefully, not crash
            return FastModeResult(items=[], parse_error=str(exc))

    def _parse(self, raw_text: str) -> FastModeResult:
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
                return FastModeResult(items=[], parse_error=f"{type(exc).__name__}: {exc}")
            raw_items = recovered
            truncated = True
        try:
            items = [FastModeItem(**item) for item in raw_items]
        except (TypeError, ValidationError) as exc:
            return FastModeResult(items=[], parse_error=f"{type(exc).__name__}: {exc}")
        thresholds = load_confidence_config()
        for item in items:
            item.pending_review = needs_pending_review(
                item.confidence, low_confidence_threshold=thresholds.low_confidence_threshold
            )
        return FastModeResult(items=items, truncated=truncated)


def analyze_goods_group(analyzer: FastVisionAnalyzer, photo_paths: list[Path]) -> FastModeResult:
    """Merge items across multiple angle photos of the SAME physical item.

    Item names are free text now (not a fixed enum), so merging is by exact
    name match only — "手机" and "iPhone" from different angles of the same
    phone will NOT be merged. This is a known, honest limitation of moving
    to open-vocabulary naming; matches 特别需求补充.md 1.2's "one final
    result per item" intent as closely as exact-string matching allows.
    When the same name appears with a higher count on another angle, that
    higher-count occurrence's item (and its category) wins.

    Returns a FastModeResult (not a bare list) so a per-photo truncation or
    total parse failure survives the merge and reaches the caller — mirrors
    image_insight.vlm.qwen_client.analyze_advanced_group's shape for the
    same reason: silently dropping that signal during merging would hide
    exactly the kind of incomplete-result case this system must always
    surface, never paper over.
    """
    if not photo_paths:
        return FastModeResult(items=[])

    per_photo = [analyzer.analyze(path) for path in photo_paths]
    usable = [r for r in per_photo if r.parse_error is None]
    if not usable:
        errors = "; ".join(r.parse_error for r in per_photo if r.parse_error)
        return FastModeResult(items=[], parse_error=errors)

    merged: dict[str, FastModeItem] = {}
    for result in usable:
        for item in result.items:
            existing = merged.get(item.name)
            if existing is None or item.count > existing.count:
                merged[item.name] = item
    truncated = any(r.truncated for r in usable)
    return FastModeResult(items=list(merged.values()), truncated=truncated)

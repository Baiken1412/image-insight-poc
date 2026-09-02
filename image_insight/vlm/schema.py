"""Structured output schema for advanced-mode (Qwen3-VL) analysis.

Anti-hallucination is a hard requirement (see Image Insight product spec):
any field the model isn't sure about must be flagged, never guessed. This
module enforces that at the data level, not just via prompt wording — a
model_validator rejects any field whose status/value combination would
represent a silent guess dressed up as a confirmed fact.
"""
from __future__ import annotations

from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field, computed_field, field_validator, model_validator

from image_insight.taxonomy import needs_forced_review
from image_insight.vlm.confidence import coerce_confidence


class FieldStatus(str, Enum):
    CONFIRMED = "confirmed"  # 确认
    SUSPECTED = "suspected"  # 疑似
    UNKNOWN = "unknown"  # 无法确定


class ObservedField(BaseModel):
    """One described attribute (brand, model, color, condition, ...)."""

    value: Optional[str] = None
    status: FieldStatus

    @model_validator(mode="after")
    def _value_must_match_status(self) -> "ObservedField":
        if self.status is FieldStatus.UNKNOWN and self.value:
            raise ValueError(
                "status=unknown must not carry a value — that would hide "
                "a guess behind an 'unknown' label instead of surfacing it"
            )
        if self.status in (FieldStatus.CONFIRMED, FieldStatus.SUSPECTED) and not self.value:
            raise ValueError(
                f"status={self.status.value} requires a non-empty value"
            )
        return self

    @property
    def needs_manual_review(self) -> bool:
        return self.status is not FieldStatus.CONFIRMED

    def display(self) -> str:
        if self.status is FieldStatus.UNKNOWN:
            return "无法确定"
        if self.status is FieldStatus.SUSPECTED:
            return f"{self.value}（疑似，建议人工确认）"
        return self.value or ""


class ItemAnalysis(BaseModel):
    """Advanced-mode structured description of one detected item."""

    category: ObservedField
    # 特别需求补充.md §3's 细类 (subcategory) — defaults to unknown/unset so
    # existing callers that don't supply one (e.g. direct construction in
    # tests) keep working; deliberately excluded from needs_manual_review's
    # aggregate below so an absent subcategory alone doesn't force review.
    subcategory: ObservedField = Field(default_factory=lambda: ObservedField(value=None, status=FieldStatus.UNKNOWN))
    brand: ObservedField
    model_name: ObservedField = Field(alias="model")
    color: ObservedField
    count: int = Field(ge=1)
    visible_text: list[str] = Field(default_factory=list)
    appearance_notes: str = ""
    condition: ObservedField
    # 特别需求补充.md §1.3/§1.4 confidence + threshold-driven "待核实". A
    # single self-reported number, not a ranked candidate list — see
    # image_insight/vlm/confidence.py for why. category_pending_review is
    # set by QwenVisionAnalyzer._parse after construction, never by the
    # model itself — defaults to False so plain construction (tests) is
    # unaffected.
    category_confidence: float = 0.0
    category_pending_review: bool = False

    model_config = {"populate_by_name": True}

    @field_validator("category_confidence", mode="before")
    @classmethod
    def _coerce_category_confidence(cls, value: object) -> float:
        return coerce_confidence(value)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_manual_review(self) -> bool:
        return self.category_pending_review or any(
            f.needs_manual_review
            for f in (self.category, self.brand, self.model_name, self.color, self.condition)
        )

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_forced_manual_review(self) -> bool:
        """特别需求补充.md §2's forced-manual-review list, at 细类 precision
        when a subcategory was actually resolved, falling back to the
        coarser 大类-level list otherwise (see taxonomy.needs_forced_review).
        """
        sub = self.subcategory.value if self.subcategory.status is not FieldStatus.UNKNOWN else None
        return needs_forced_review(self.category.value or "", sub)


class AdvancedAnalysisResult(BaseModel):
    """Full advanced-mode result for one uploaded photo (possibly multi-item)."""

    items: list[ItemAnalysis]
    model_version: str
    parse_error: Optional[str] = None
    """Set when the raw model output could not be parsed/validated; when set,
    ``items`` is empty and the caller must fall back to manual review rather
    than presenting a partial/fabricated result."""
    # True when the model's raw response was cut off (hit max_new_tokens)
    # before the JSON closed, and QwenVisionAnalyzer._parse salvaged however
    # many complete items it could via image_insight.vlm.json_repair — see
    # that module's docstring. ``items`` here are genuine, individually
    # re-validated model output, but the photo may have contained MORE items
    # than what's listed: the tail got cut off mid-generation, not because
    # the model finished. Must always force manual review, never be
    # presented as a complete/normal result.
    truncated: bool = False
    # OCR text detected on the photo but not attributable to a single item
    # (multi-item photos, or zero items) — see QwenVisionAnalyzer._augment_with_ocr.
    ocr_text: list[str] = Field(default_factory=list)

    @computed_field  # type: ignore[prop-decorator]
    @property
    def needs_manual_review(self) -> bool:
        return self.parse_error is not None or self.truncated or any(i.needs_manual_review for i in self.items)

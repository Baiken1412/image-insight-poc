"""Runtime configuration, read entirely from environment variables.

No secrets are ever hardcoded here (see project security rules). Advanced
mode runs Qwen3-VL fully locally (特别需求补充.md 1.6: no photo may leave
the machine) — there is no API key to configure, only which local
checkpoint to load.
"""
from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass(frozen=True)
class QwenConfig:
    model_id: str
    max_new_tokens: int
    max_image_pixels: int
    min_image_pixels: int


def load_qwen_config() -> QwenConfig:
    return QwenConfig(
        model_id=os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct"),
        # Advanced mode's per-item schema (subcategory, category_confidence,
        # visible_text, appearance_notes, ...) can need more than 1024
        # tokens for a photo with several items before the JSON closes —
        # generation getting cut off mid-string was a real observed bug
        # (json.JSONDecodeError: "Unterminated string..."). 1536 gives more
        # headroom; image_insight.vlm.json_repair is the remaining safety
        # net for photos that still exceed this.
        max_new_tokens=int(os.environ.get("QWEN_MAX_NEW_TOKENS", "9999")),
        # Bounds vision-token count regardless of source photo resolution —
        # without this, a large real photo can demand ~9-10GB just for the
        # vision tower and OOM an 8GB card. See LocalQwenTransport docstring.
        max_image_pixels=int(os.environ.get("QWEN_MAX_IMAGE_PIXELS", str(1024 * 1024))),
        min_image_pixels=int(os.environ.get("QWEN_MIN_IMAGE_PIXELS", str(256 * 16 * 16))),
    )


@dataclass(frozen=True)
class AppConfig:
    rfdetr_variant: str
    min_phone_confidence: float
    photos_root: str


def load_app_config() -> AppConfig:
    return AppConfig(
        rfdetr_variant=os.environ.get("RFDETR_VARIANT", "medium"),
        min_phone_confidence=float(os.environ.get("MIN_PHONE_CONFIDENCE", "0.5")),
        photos_root=os.environ.get("PHOTOS_ROOT", "物品图片"),
    )


@dataclass(frozen=True)
class ClassificationConfidenceConfig:
    """特别需求补充.md §1.4's "待核实" thresholds — explicitly required to be
    configurable ("上述阈值应可配置"), to be tuned once real seized-item
    photos are available rather than fixed in code.
    """

    low_confidence_threshold: float
    close_margin_threshold: float


def load_confidence_config() -> ClassificationConfidenceConfig:
    return ClassificationConfidenceConfig(
        low_confidence_threshold=float(os.environ.get("CLASSIFICATION_LOW_CONFIDENCE_THRESHOLD", "0.35")),
        close_margin_threshold=float(os.environ.get("CLASSIFICATION_CLOSE_MARGIN_THRESHOLD", "0.05")),
    )


@dataclass(frozen=True)
class OcrConfig:
    enabled: bool
    min_text_confidence: float


def load_ocr_config() -> OcrConfig:
    return OcrConfig(
        enabled=os.environ.get("OCR_ENABLED", "true").strip().lower() not in ("0", "false", "no"),
        min_text_confidence=float(os.environ.get("OCR_MIN_TEXT_CONFIDENCE", "0.7")),
    )


@dataclass(frozen=True)
class DbConfig:
    path: str


def load_db_config() -> DbConfig:
    return DbConfig(path=os.environ.get("IMAGE_INSIGHT_DB_PATH", "image_insight.db"))

"""Runtime configuration, read entirely from environment variables.

No secrets are ever hardcoded here (see project security rules). Advanced
mode runs Qwen3-VL fully locally (特别需求补充.md 1.6: no photo may leave
the machine) — there is no API key to configure, only which local
checkpoint to load.

Module import time also does two bits of environment setup, before anything
else in the app gets a chance to read os.environ or import transformers:

1. Loads `.env` from the project root (python-dotenv), so `cp .env.example
   .env` + edit is enough — no more manually exporting env vars per shell.
   load_dotenv() never overrides a variable already set in the real process
   environment, so `setx`/shell exports still win over `.env`.
2. Defaults HF_HOME to <project root>/models/huggingface (only if HF_HOME
   isn't already set by the OS env or `.env`), so `pip install` + first run
   downloads Qwen3-VL into the app folder itself rather than the user's
   home-directory-wide ~/.cache/huggingface — the whole app (code + model)
   then stays one self-contained, movable folder, matching how the
   PyInstaller portable build's 启动服务.bat already points HF_HOME at its
   own models/ subfolder.
"""
from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv

_PROJECT_ROOT = Path(__file__).resolve().parent.parent

load_dotenv(_PROJECT_ROOT / ".env")
os.environ.setdefault("HF_HOME", str(_PROJECT_ROOT / "models" / "huggingface"))


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


def load_qwen_backend() -> str:
    """Which transport image_insight/api/main.py should wire up:
    "local" (default — image_insight.vlm.qwen_client.LocalQwenTransport,
    runs fully offline, see 特别需求补充.md 1.6) or "remote"
    (RemoteQwenTransport, calls out to load_remote_qwen_config()'s
    base_url).

    Defaults to "local" rather than inferring "remote" from the mere
    presence of QWEN_REMOTE_BASE_URL — this project's photos are
    case-property evidence with a documented no-external-upload
    requirement, so switching to a backend that sends them over the network
    must be an explicit, deliberate choice, never an accidental side effect
    of leaving an old .env value set.
    """
    backend = os.environ.get("QWEN_BACKEND", "local").strip().lower()
    if backend not in ("local", "remote"):
        raise ValueError(f"invalid QWEN_BACKEND={backend!r}; expected 'local' or 'remote'")
    return backend


@dataclass(frozen=True)
class RemoteQwenConfig:
    """Only used when load_qwen_backend() == "remote". base_url should point
    at an OpenAI-compatible chat-completions endpoint (vLLM/SGLang/
    Xinference/etc. serving Qwen3-VL) — see RemoteQwenTransport's docstring
    for why this must be infrastructure you control (private network/VPN),
    not a public third-party API.
    """

    base_url: str
    model_id: str
    api_key: Optional[str]
    request_timeout: float


def load_remote_qwen_config() -> Optional[RemoteQwenConfig]:
    """Returns None if QWEN_REMOTE_BASE_URL isn't set — callers (currently
    image_insight/api/main.py) treat that as a configuration error when
    QWEN_BACKEND=remote, since there is then nothing to connect to."""
    base_url = os.environ.get("QWEN_REMOTE_BASE_URL", "").strip()
    if not base_url:
        return None
    return RemoteQwenConfig(
        base_url=base_url,
        # Falls back to QWEN_MODEL_ID (the same variable local mode reads)
        # only for convenience when the remote server happens to be serving
        # a model under that same id/name; set QWEN_REMOTE_MODEL_ID
        # explicitly whenever the remote deployment's model name differs.
        model_id=os.environ.get("QWEN_REMOTE_MODEL_ID") or os.environ.get("QWEN_MODEL_ID", "Qwen/Qwen3-VL-2B-Instruct"),
        # Never hardcoded, never defaulted to a real value — a bare API key
        # in .env is fine (it's git-ignored, see .gitignore), but nothing in
        # this codebase supplies one implicitly.
        api_key=os.environ.get("QWEN_REMOTE_API_KEY") or None,
        # Network round-trip to another machine, so this needs its own
        # timeout distinct from QWEN_ADVANCED_TIMEOUT (which bounds local
        # model.generate() wall-clock time via a StoppingCriteria — a
        # mechanism that has no remote equivalent; a remote call can only be
        # abandoned at the HTTP layer, which is what this governs).
        request_timeout=float(os.environ.get("QWEN_REMOTE_REQUEST_TIMEOUT", "300")),
    )


@dataclass(frozen=True)
class AdvancedQwenConfig:
    """Advanced mode's own generation budget — deliberately separate from
    QwenConfig.max_new_tokens (shared with fast mode) so tuning one never
    moves the other. See image_insight/vlm/qwen_client.py's QwenVisionAnalyzer
    for how these are actually enforced (per-call override + a wall-clock
    StoppingCriteria, not just a bigger/smaller number)."""

    max_new_tokens: int
    timeout_seconds: float
    repetition_penalty: float


def load_advanced_qwen_config() -> AdvancedQwenConfig:
    return AdvancedQwenConfig(
        # A well-formed advanced-mode item list rarely needs more than a
        # couple thousand tokens; 4096 leaves headroom for a busy photo
        # without paying for the runaway/repetitive generation observed at
        # the old shared default (9999, formerly 99999) on complex photos.
        max_new_tokens=int(os.environ.get("QWEN_ADVANCED_MAX_NEW_TOKENS", "4096")),
        # Hard wall-clock ceiling on a single advanced-mode photo analysis,
        # enforced via a StoppingCriteria inside model.generate() itself (see
        # LocalQwenTransport.complete) — actually halts generation and frees
        # the GPU, rather than only abandoning the HTTP response while
        # inference keeps running in its worker thread.
        timeout_seconds=float(os.environ.get("QWEN_ADVANCED_TIMEOUT", "300")),
        # Root cause of real multi-minute advanced-mode hangs on text-dense
        # product-packaging photos: the model gets stuck looping the same
        # few label phrases inside visible_text verbatim (e.g. "MOUTAI" /
        # "防伪追溯器方法介绍" / "贵州茅台酒股份有限公司" cycling ~90 times),
        # never closing the JSON before hitting max_new_tokens.
        #
        # Two decoding-level fixes were tried and BOTH reverted after
        # breaking output on otherwise-clean photos (13.png/14.png, tested
        # repeatedly): transformers.no_repeat_ngram_size (hard n-gram
        # blocking) either let the loop through (low threshold) or forced
        # completely garbled/invalid Unicode output (high threshold, even on
        # the SAME Moutai photo it was meant to fix); raising
        # repetition_penalty to 1.3 did the same — garbled output on 13/14,
        # not even a full fix on the Moutai photo. This model + greedy
        # decoding is evidently too fragile for either lever at a strength
        # that actually suppresses the loop. repetition_penalty is kept at
        # fast mode's own proven-safe 1.15 (unchanged) rather than shipping
        # an override that trades a slow failure for a silently-corrupted
        # one. The actual fix for this hang is the trio of max_new_tokens
        # (4096, well below the old 9999/99999), the image pre-downscale,
        # and the visible_text prompt cap/post-parse truncation above/below
        # this field — worst case measured at ~166s (vs. 16+ minutes
        # before), safely inside QWEN_ADVANCED_TIMEOUT's 300s backstop.
        repetition_penalty=float(os.environ.get("QWEN_ADVANCED_REPETITION_PENALTY", "1.15")),
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

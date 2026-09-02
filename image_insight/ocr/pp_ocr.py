"""Text recognition via PaddleOCR — the "看清字" half of advanced mode.

Principle from the Image Insight spec: "VLM 负责看懂，OCR 负责看清字" — Qwen3-VL
handles overall understanding, PaddleOCR is the secondary pass for fields
that need precise character-level reading (screen text, labels, serials).
The real engine is isolated behind TextRecognizer so unit tests don't need
PaddleOCR's model weights loaded.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

DEFAULT_MIN_TEXT_CONFIDENCE = 0.7


@dataclass(frozen=True)
class OcrTextBox:
    text: str
    confidence: float


class TextRecognizer(Protocol):
    def read_text(self, image_path: Path) -> list[OcrTextBox]: ...


def extract_high_confidence_text(
    boxes: list[OcrTextBox],
    min_confidence: float = DEFAULT_MIN_TEXT_CONFIDENCE,
) -> list[str]:
    """Keep only text PaddleOCR is actually confident about.

    Low-confidence OCR reads are exactly the kind of "looks like it might
    say X" guess the product spec forbids surfacing as fact — drop them
    rather than showing a possibly-wrong serial/IMEI digit as if it were read.
    """
    return [b.text for b in boxes if b.confidence >= min_confidence]


class PaddleOcrTextRecognizer:
    """Real recognizer backed by PaddleOCR's PP-OCRv5 pipeline.

    Runs on CPU by default (device="cpu"), not GPU: a machine that also has
    torch/Qwen3-VL installed for GPU inference can end up with paddlepaddle-gpu
    and torch pulling in conflicting cuDNN DLL versions on Windows (observed:
    "OSError: The specified procedure could not be found" loading
    cudnn_cnn64_9.dll) — a real, hard-to-untangle native dependency conflict,
    not a paddleocr bug per se. OCR is a supplementary cross-check on a single
    still image, not the hot path, so CPU latency here is an acceptable
    trade-off against destabilizing the GPU setup the main VLM depends on.
    """

    def __init__(self, lang: str = "ch", device: str = "cpu"):
        from paddleocr import PaddleOCR

        self._ocr = PaddleOCR(
            lang=lang, device=device, use_doc_orientation_classify=False, use_doc_unwarping=False
        )

    def read_text(self, image_path: Path) -> list[OcrTextBox]:
        results = self._ocr.predict(str(image_path))
        boxes: list[OcrTextBox] = []
        for page in results:
            texts = page.get("rec_texts", [])
            scores = page.get("rec_scores", [])
            for text, score in zip(texts, scores):
                boxes.append(OcrTextBox(text=text, confidence=float(score)))
        return boxes

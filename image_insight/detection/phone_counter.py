"""Fast-mode item counting via a pretrained (zero-shot) RF-DETR detector.

No bbox-annotated training data exists yet for the 122 real item photos
(see Image-Insight-开发计划.md M0), so fast mode uses RF-DETR's official
COCO-pretrained checkpoint directly rather than a fine-tuned model. COCO
already includes a "cell phone" class (id 77), which is enough for the
first-priority milestone: phone counting. Bounding boxes are never exposed
outside this module — fast mode's contract is "name + count" only.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Protocol

COCO_CELL_PHONE_CLASS_ID = 77
DEFAULT_MIN_CONFIDENCE = 0.5

# COCO classes RF-DETR's pretrained checkpoint already covers reasonably well
# and that map onto items task-for-andy.docx names as high-frequency (手机,
# 笔记本电脑) or plausible bag/luggage proxies. There is no COCO class for
# "wallet" (钱包) specifically — labeling a handbag detection as a wallet
# would be a fabrication, so that one stays out of scope until fine-tuned.
FAST_MODE_CLASS_MAP: dict[int, str] = {
    COCO_CELL_PHONE_CLASS_ID: "手机",
    73: "笔记本电脑",
    27: "背包",
    31: "手提包",
    33: "行李箱",
}


@dataclass(frozen=True)
class DetectionBox:
    class_id: int
    class_name: str
    confidence: float


class ObjectDetector(Protocol):
    """The only capability counting logic needs — lets tests inject a fake
    detector instead of loading real GPU weights for every unit test."""

    def detect(self, image_path: Path) -> list[DetectionBox]: ...


def count_items_by_class(
    detections: list[DetectionBox],
    class_id: int,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> int:
    return sum(1 for d in detections if d.class_id == class_id and d.confidence >= min_confidence)


def count_phones_in_photo(
    detector: ObjectDetector,
    image_path: Path,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> int:
    return count_items_by_class(detector.detect(image_path), COCO_CELL_PHONE_CLASS_ID, min_confidence)


def count_phones_for_goods_group(
    detector: ObjectDetector,
    photo_paths: list[Path],
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> int:
    """Count phones for one physical item, given all its angle photos.

    Multiple angles of the SAME item must not multiply the apparent count
    (特别需求补充.md 1.2: 合并后输出一个最终结果) — so this takes the max
    count seen across angles, not the sum, on the assumption that the best
    angle for seeing every phone present is an upper bound on the true count
    and no single angle should ever show more phones than actually exist.
    """
    if not photo_paths:
        return 0
    return max(count_phones_in_photo(detector, p, min_confidence) for p in photo_paths)


def count_items_for_goods_group(
    detector: ObjectDetector,
    photo_paths: list[Path],
    class_map: dict[int, str] = FAST_MODE_CLASS_MAP,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> dict[str, int]:
    """Fast-mode "name + count" for every supported category at once.

    Same max-across-angles semantics as count_phones_for_goods_group, applied
    independently per class so one angle's strong laptop view and another
    angle's strong backpack view both count correctly.
    """
    if not photo_paths:
        return {name: 0 for name in class_map.values()}

    detections_per_photo = [detector.detect(p) for p in photo_paths]
    return {
        name: max(count_items_by_class(detections, class_id, min_confidence) for detections in detections_per_photo)
        for class_id, name in class_map.items()
    }


class RFDETRDetector:
    """Real detector backed by RF-DETR's pretrained COCO checkpoint.

    Not phone-specific — detect() returns every COCO class found; callers
    (count_phones_*, count_items_for_goods_group) decide which classes matter.
    """

    _VARIANTS = ("nano", "small", "medium", "base", "large")

    def __init__(self, model_variant: str = "medium"):
        if model_variant not in self._VARIANTS:
            raise ValueError(f"unknown model_variant {model_variant!r}, choose from {self._VARIANTS}")
        from rfdetr import RFDETRBase, RFDETRLarge, RFDETRMedium, RFDETRNano, RFDETRSmall

        model_classes = {
            "nano": RFDETRNano,
            "small": RFDETRSmall,
            "medium": RFDETRMedium,
            "base": RFDETRBase,
            "large": RFDETRLarge,
        }
        self._model = model_classes[model_variant]()

    def detect(self, image_path: Path) -> list[DetectionBox]:
        from rfdetr.util.coco_classes import COCO_CLASSES

        result = self._model.predict(str(image_path), threshold=0.3)
        return [
            DetectionBox(
                class_id=int(class_id),
                class_name=COCO_CLASSES.get(int(class_id), "unknown"),
                confidence=float(confidence),
            )
            for class_id, confidence in zip(result.class_id, result.confidence)
        ]

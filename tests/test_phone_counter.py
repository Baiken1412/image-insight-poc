from pathlib import Path

from image_insight.detection.phone_counter import (
    COCO_CELL_PHONE_CLASS_ID,
    DetectionBox,
    count_items_by_class,
    count_items_for_goods_group,
    count_phones_for_goods_group,
    count_phones_in_photo,
)


class FakeDetector:
    """Test double: returns pre-scripted detections per image path."""

    def __init__(self, responses: dict[str, list[DetectionBox]]):
        self._responses = responses

    def detect(self, image_path: Path) -> list[DetectionBox]:
        return self._responses[str(image_path)]


def phone(confidence: float) -> DetectionBox:
    return DetectionBox(class_id=COCO_CELL_PHONE_CLASS_ID, class_name="cell phone", confidence=confidence)


def other(confidence: float = 0.9) -> DetectionBox:
    return DetectionBox(class_id=67, class_name="dining table", confidence=confidence)


def test_count_items_by_class_ignores_other_classes():
    detections = [phone(0.9), other(0.9), other(0.95)]
    assert count_items_by_class(detections, COCO_CELL_PHONE_CLASS_ID) == 1


def test_count_items_by_class_filters_low_confidence_detections():
    detections = [phone(0.9), phone(0.2)]
    assert count_items_by_class(detections, COCO_CELL_PHONE_CLASS_ID, min_confidence=0.5) == 1


def test_count_phones_in_photo_counts_multiple_phones_in_one_frame():
    path = Path("multi_phone.jpg")
    detector = FakeDetector({str(path): [phone(0.9), phone(0.8), other(0.99)]})
    assert count_phones_in_photo(detector, path) == 2


def test_count_phones_for_goods_group_returns_zero_for_empty_group():
    detector = FakeDetector({})
    assert count_phones_for_goods_group(detector, []) == 0


def test_count_phones_for_goods_group_takes_max_across_angles_not_sum():
    # Same physical item, three angle photos. One angle happens to show
    # both phones clearly; the others are partially occluded. The true
    # count is 2, not 2+1+1=4 — angles must not multiply the count.
    front = Path("front.jpg")
    side = Path("side.jpg")
    back = Path("back.jpg")
    detector = FakeDetector(
        {
            str(front): [phone(0.9), phone(0.85)],
            str(side): [phone(0.9)],
            str(back): [phone(0.9)],
        }
    )
    assert count_phones_for_goods_group(detector, [front, side, back]) == 2


def test_count_phones_for_goods_group_single_photo_matches_single_count():
    path = Path("only.jpg")
    detector = FakeDetector({str(path): [phone(0.9)]})
    assert count_phones_for_goods_group(detector, [path]) == 1


def laptop(confidence: float = 0.9) -> DetectionBox:
    return DetectionBox(class_id=73, class_name="laptop", confidence=confidence)


def backpack(confidence: float = 0.9) -> DetectionBox:
    return DetectionBox(class_id=27, class_name="backpack", confidence=confidence)


def test_count_items_for_goods_group_reports_every_supported_category():
    path = Path("desk.jpg")
    detector = FakeDetector({str(path): [phone(0.9), laptop(0.9)]})
    counts = count_items_for_goods_group(detector, [path])
    assert counts == {"手机": 1, "笔记本电脑": 1, "背包": 0, "手提包": 0, "行李箱": 0}


def test_count_items_for_goods_group_takes_max_per_class_independently_across_angles():
    front = Path("front.jpg")
    back = Path("back.jpg")
    detector = FakeDetector(
        {
            str(front): [phone(0.9), phone(0.9)],  # 2 phones, no backpack visible
            str(back): [backpack(0.9)],  # backpack visible, phones occluded
        }
    )
    counts = count_items_for_goods_group(detector, [front, back])
    assert counts["手机"] == 2
    assert counts["背包"] == 1


def test_count_items_for_goods_group_returns_all_zero_for_empty_group():
    detector = FakeDetector({})
    counts = count_items_for_goods_group(detector, [])
    assert counts == {"手机": 0, "笔记本电脑": 0, "背包": 0, "手提包": 0, "行李箱": 0}

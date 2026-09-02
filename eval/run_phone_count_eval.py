"""Fast-mode (phone counting) evaluation against manually verified ground truth.

Usage: python eval/run_phone_count_eval.py

Prints per-goods_id predicted vs. expected counts and an accuracy summary.
Per Image-Insight-开发计划.md M1: the metric is count-accuracy, not mAP.
"""
from __future__ import annotations

import json
from pathlib import Path

from image_insight.detection.phone_counter import RFDETRDetector, count_phones_for_goods_group
from image_insight.goods import find_photos, group_photos_by_goods_id

GROUND_TRUTH_PATH = Path(__file__).parent / "phone_count_ground_truth.json"
PHOTOS_ROOT = Path(__file__).parent.parent / "物品图片"


def main() -> None:
    ground_truth = json.loads(GROUND_TRUTH_PATH.read_text(encoding="utf-8"))
    ground_truth.pop("_comment", None)

    groups_by_id = {g.goods_id: g for g in group_photos_by_goods_id(find_photos(PHOTOS_ROOT))}

    detector = RFDETRDetector(model_variant="medium")

    correct = 0
    rows = []
    for goods_id, expected in ground_truth.items():
        group = groups_by_id.get(goods_id)
        if group is None:
            rows.append((goods_id, expected, "MISSING", False))
            continue
        predicted = count_phones_for_goods_group(detector, list(group.photo_paths))
        is_correct = predicted == expected
        correct += is_correct
        rows.append((goods_id, expected, predicted, is_correct))

    print(f"{'goods_id':<28} {'expected':>8} {'predicted':>9}  ok")
    for goods_id, expected, predicted, is_correct in rows:
        print(f"{goods_id:<28} {expected:>8} {str(predicted):>9}  {'OK' if is_correct else 'MISS'}")

    total = len(rows)
    print(f"\naccuracy: {correct}/{total} = {correct / total:.1%}")


if __name__ == "__main__":
    main()

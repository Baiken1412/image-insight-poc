"""Group seized-item photos by goods_id and split them without leakage.

Filenames follow the pattern ``<goods_id>_<14-digit-timestamp><hex-hash>.jpg``.
Multiple photos of the same physical item share one goods_id (different
camera angles). A goods_id must never be split across train/valid/test —
see 特别需求补充.md and task for andy.docx for why (leakage inflates metrics).
"""
from __future__ import annotations

import random
import re
from dataclasses import dataclass
from pathlib import Path

_FILENAME_PATTERN = re.compile(
    r"^(?P<goods_id>.+)_(?P<suffix>\d{14}[0-9A-Fa-f]+)\.(?P<ext>jpe?g|png)$",
    re.IGNORECASE,
)


class UnrecognizedFilenameError(ValueError):
    """A filename didn't match the expected goods_id pattern."""


def parse_goods_id(filename: str) -> str:
    """Extract the goods_id prefix from a photo filename.

    Raises rather than guessing on a mismatch: a silently wrong goods_id
    would defeat the point of leakage-free grouping.
    """
    match = _FILENAME_PATTERN.match(filename)
    if not match:
        raise UnrecognizedFilenameError(
            f"filename does not match expected goods_id pattern: {filename!r}"
        )
    return match.group("goods_id")


@dataclass(frozen=True)
class GoodsGroup:
    goods_id: str
    photo_paths: tuple[Path, ...]


def find_photos(root: Path) -> list[Path]:
    """Recursively find jpg/jpeg/png photos under root, sorted for determinism."""
    exts = {".jpg", ".jpeg", ".png"}
    return sorted(p for p in root.rglob("*") if p.is_file() and p.suffix.lower() in exts)


def group_photos_by_goods_id(photo_paths: list[Path]) -> list[GoodsGroup]:
    """Group photo paths by goods_id, preserving first-seen order."""
    groups: dict[str, list[Path]] = {}
    for path in photo_paths:
        goods_id = parse_goods_id(path.name)
        groups.setdefault(goods_id, []).append(path)
    return [
        GoodsGroup(goods_id=goods_id, photo_paths=tuple(paths))
        for goods_id, paths in groups.items()
    ]


def group_uploaded_photos(uploads: list[tuple[Path, str]]) -> list[GoodsGroup]:
    """Group web-uploaded photos for analysis — tolerant version of
    group_photos_by_goods_id for the API's upload endpoints.

    ``uploads`` is a list of (actual_file_path, original_filename) pairs:
    uploaded files are saved under random temp names (see api/main.py), so
    the goods_id-bearing filename the browser sent has to be tracked
    separately from the path used to read file content.

    A filename matching the dataset's <goods_id>_<timestamp><hash> pattern
    groups with its siblings (multi-angle photos of one physical item, per
    特别需求补充.md §1.2). A filename that DOESN'T match — the common case
    for ad-hoc photos picked from a phone/camera roll, e.g. "IMG_1234.jpg"
    — becomes its own singleton group instead of raising
    UnrecognizedFilenameError, so a batch of unrelated photos can be
    uploaded and analyzed together in one request, each getting its own
    independent result. This is what lets one upload endpoint serve both
    "several angles of one item" and "a batch of unrelated photos" without
    the caller having to declare which case it is up front.
    """
    groups: dict[str, list[Path]] = {}
    order: list[str] = []
    for path, original_name in uploads:
        try:
            goods_id = parse_goods_id(original_name)
        except UnrecognizedFilenameError:
            goods_id = original_name  # unrecognized name -> its own singleton group
        if goods_id not in groups:
            order.append(goods_id)
        groups.setdefault(goods_id, []).append(path)
    return [GoodsGroup(goods_id=goods_id, photo_paths=tuple(groups[goods_id])) for goods_id in order]


def split_groups(
    groups: list[GoodsGroup],
    valid_ratio: float = 0.15,
    test_ratio: float = 0.15,
    seed: int = 42,
) -> tuple[list[GoodsGroup], list[GoodsGroup], list[GoodsGroup]]:
    """Split goods groups into train/valid/test by goods_id (never by photo).

    Splitting is done on the shuffled list of groups so a goods_id's photos
    always land together, avoiding the leakage task-for-andy.docx calls out
    as a hard requirement ("按物品实例划分...否则测试指标会虚高").
    """
    if not (0 <= valid_ratio < 1) or not (0 <= test_ratio < 1):
        raise ValueError("valid_ratio and test_ratio must each be in [0, 1)")
    if valid_ratio + test_ratio >= 1:
        raise ValueError("valid_ratio + test_ratio must be < 1 (train would be empty)")

    shuffled = list(groups)
    random.Random(seed).shuffle(shuffled)

    n = len(shuffled)
    n_valid = round(n * valid_ratio)
    n_test = round(n * test_ratio)

    valid = shuffled[:n_valid]
    test = shuffled[n_valid : n_valid + n_test]
    train = shuffled[n_valid + n_test :]
    return train, valid, test

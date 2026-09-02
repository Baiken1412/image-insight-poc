from pathlib import Path

import pytest

from image_insight.goods import (
    UnrecognizedFilenameError,
    group_photos_by_goods_id,
    group_uploaded_photos,
    parse_goods_id,
    split_groups,
)


def test_parse_goods_id_extracts_prefix_before_timestamp_hash():
    name = "W1102210800002026087026_20260819182721BEC656A28D6A207DF5.jpg"
    assert parse_goods_id(name) == "W1102210800002026087026"


def test_parse_goods_id_accepts_png_and_lowercase_hash():
    name = "abc123_20260101000000deadbeef.png"
    assert parse_goods_id(name) == "abc123"


def test_parse_goods_id_rejects_unrecognized_filename():
    with pytest.raises(UnrecognizedFilenameError):
        parse_goods_id("random_photo.jpg")


def test_group_photos_by_goods_id_merges_multi_angle_photos():
    paths = [
        Path("A001_20260101000000AAAA.jpg"),
        Path("A001_20260101000001BBBB.jpg"),
        Path("A002_20260101000002CCCC.jpg"),
    ]
    groups = group_photos_by_goods_id(paths)
    by_id = {g.goods_id: g.photo_paths for g in groups}
    assert set(by_id) == {"A001", "A002"}
    assert len(by_id["A001"]) == 2
    assert len(by_id["A002"]) == 1


def test_split_groups_never_splits_a_goods_id_across_sets():
    paths = [Path(f"G{i:03d}_20260101000000{i:04X}.jpg") for i in range(40)]
    groups = group_photos_by_goods_id(paths)

    train, valid, test = split_groups(groups, valid_ratio=0.2, test_ratio=0.2, seed=7)

    train_ids = {g.goods_id for g in train}
    valid_ids = {g.goods_id for g in valid}
    test_ids = {g.goods_id for g in test}

    assert train_ids.isdisjoint(valid_ids)
    assert train_ids.isdisjoint(test_ids)
    assert valid_ids.isdisjoint(test_ids)
    assert train_ids | valid_ids | test_ids == {g.goods_id for g in groups}


def test_split_groups_is_deterministic_given_same_seed():
    paths = [Path(f"G{i:03d}_20260101000000{i:04X}.jpg") for i in range(20)]
    groups = group_photos_by_goods_id(paths)

    split_a = split_groups(groups, seed=123)
    split_b = split_groups(groups, seed=123)

    ids_a = tuple(tuple(g.goods_id for g in part) for part in split_a)
    ids_b = tuple(tuple(g.goods_id for g in part) for part in split_b)
    assert ids_a == ids_b


def test_split_groups_rejects_ratios_that_leave_no_train_set():
    groups = group_photos_by_goods_id([Path("A_20260101000000AAAA.jpg")])
    with pytest.raises(ValueError):
        split_groups(groups, valid_ratio=0.6, test_ratio=0.5)


def test_group_uploaded_photos_merges_recognized_goods_id_names():
    uploads = [
        (Path("/tmp/a.jpg"), "A001_20260101000000AAAA.jpg"),
        (Path("/tmp/b.jpg"), "A001_20260101000001BBBB.jpg"),
    ]
    groups = group_uploaded_photos(uploads)
    assert len(groups) == 1
    assert groups[0].goods_id == "A001"
    assert set(groups[0].photo_paths) == {Path("/tmp/a.jpg"), Path("/tmp/b.jpg")}


def test_group_uploaded_photos_treats_unrecognized_names_as_singleton_groups():
    uploads = [
        (Path("/tmp/a.jpg"), "IMG_0001.jpg"),
        (Path("/tmp/b.jpg"), "IMG_0002.jpg"),
    ]
    groups = group_uploaded_photos(uploads)
    assert len(groups) == 2
    by_id = {g.goods_id: g.photo_paths for g in groups}
    assert by_id["IMG_0001.jpg"] == (Path("/tmp/a.jpg"),)
    assert by_id["IMG_0002.jpg"] == (Path("/tmp/b.jpg"),)


def test_group_uploaded_photos_handles_a_mix_of_recognized_and_unrecognized_names():
    uploads = [
        (Path("/tmp/a.jpg"), "A001_20260101000000AAAA.jpg"),
        (Path("/tmp/b.jpg"), "A001_20260101000001BBBB.jpg"),
        (Path("/tmp/c.jpg"), "random_photo.jpg"),
    ]
    groups = group_uploaded_photos(uploads)
    assert len(groups) == 2
    by_id = {g.goods_id: g.photo_paths for g in groups}
    assert set(by_id["A001"]) == {Path("/tmp/a.jpg"), Path("/tmp/b.jpg")}
    assert by_id["random_photo.jpg"] == (Path("/tmp/c.jpg"),)


def test_group_uploaded_photos_preserves_first_seen_order():
    uploads = [
        (Path("/tmp/z.jpg"), "IMG_0002.jpg"),
        (Path("/tmp/a.jpg"), "IMG_0001.jpg"),
    ]
    groups = group_uploaded_photos(uploads)
    assert [g.goods_id for g in groups] == ["IMG_0002.jpg", "IMG_0001.jpg"]


def test_group_uploaded_photos_empty_input():
    assert group_uploaded_photos([]) == []

from __future__ import annotations

import time
from pathlib import Path

from image_insight import db


def test_connect_creates_schema_on_an_empty_file(tmp_path: Path):
    conn = db.connect(tmp_path / "test.db")
    # schema exists and is queryable without error
    assert db.list_analysis_records(conn) == []


def test_upsert_then_get_round_trips_a_record():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(
        conn,
        goods_id="A001",
        mode="fast",
        source_files=["A001_front.jpg", "A001_back.jpg"],
        result={"items": [{"name": "手机", "count": 1}]},
    )

    record = db.get_analysis_record(conn, goods_id="A001", mode="fast")

    assert record is not None
    assert record.goods_id == "A001"
    assert record.mode == "fast"
    assert record.source_files == ["A001_front.jpg", "A001_back.jpg"]
    assert record.result == {"items": [{"name": "手机", "count": 1}]}


def test_get_returns_none_for_unknown_goods_id():
    conn = db.connect(":memory:")
    assert db.get_analysis_record(conn, goods_id="does-not-exist", mode="fast") is None


def test_reanalyzing_the_same_goods_id_overwrites_not_duplicates():
    # the core product requirement: analyzing the same item a second time
    # replaces the existing record rather than adding a new one.
    conn = db.connect(":memory:")
    db.upsert_analysis_record(
        conn, goods_id="A001", mode="fast", source_files=["v1.jpg"], result={"items": [{"name": "手机", "count": 1}]}
    )
    db.upsert_analysis_record(
        conn, goods_id="A001", mode="fast", source_files=["v2.jpg"], result={"items": [{"name": "手机", "count": 2}]}
    )

    all_records = db.list_analysis_records(conn)
    assert len(all_records) == 1  # not 2

    record = db.get_analysis_record(conn, goods_id="A001", mode="fast")
    assert record.source_files == ["v2.jpg"]  # latest write wins
    assert record.result == {"items": [{"name": "手机", "count": 2}]}


def test_different_goods_ids_are_separate_records():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=["a.jpg"], result={"n": 1})
    db.upsert_analysis_record(conn, goods_id="A002", mode="fast", source_files=["b.jpg"], result={"n": 2})

    assert len(db.list_analysis_records(conn)) == 2


def test_fast_and_advanced_are_separate_records_for_the_same_goods_id():
    # re-analyzing in a DIFFERENT mode must not clobber the other mode's
    # saved result for the same physical item.
    conn = db.connect(":memory:")
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=["a.jpg"], result={"mode": "fast"})
    db.upsert_analysis_record(
        conn, goods_id="A001", mode="advanced", source_files=["a.jpg"], result={"mode": "advanced"}
    )

    assert len(db.list_analysis_records(conn)) == 2
    fast = db.get_analysis_record(conn, goods_id="A001", mode="fast")
    advanced = db.get_analysis_record(conn, goods_id="A001", mode="advanced")
    assert fast.result == {"mode": "fast"}
    assert advanced.result == {"mode": "advanced"}


def test_list_analysis_records_filters_by_mode():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=["a.jpg"], result={})
    db.upsert_analysis_record(conn, goods_id="A002", mode="advanced", source_files=["b.jpg"], result={})

    fast_only = db.list_analysis_records(conn, mode="fast")
    assert [r.goods_id for r in fast_only] == ["A001"]


def test_list_analysis_records_orders_most_recently_updated_first():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=[], result={})
    time.sleep(0.01)  # ensure a distinct updated_at timestamp, not just insertion order
    db.upsert_analysis_record(conn, goods_id="A002", mode="fast", source_files=[], result={})
    time.sleep(0.01)
    # re-touch A001 so it becomes the most recently updated
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=[], result={"v": 2})

    records = db.list_analysis_records(conn)
    assert records[0].goods_id == "A001"


def test_upsert_flattens_advanced_mode_items_into_readable_columns():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(
        conn,
        goods_id="A001",
        mode="advanced",
        source_files=["a.jpg"],
        result={
            "items": [
                {
                    "category": {"value": "电子设备及数字载体", "status": "confirmed"},
                    "subcategory": {"value": "手机", "status": "confirmed"},
                    "brand": {"value": "HONOR", "status": "confirmed"},
                    "model": {"value": None, "status": "unknown"},
                    "color": {"value": "银灰色", "status": "confirmed"},
                    "count": 1,
                    "visible_text": ["HONOR", "SN123"],
                    "appearance_notes": "背面光滑金属质感",
                    "condition": {"value": "完好无损", "status": "confirmed"},
                    "category_confidence": 0.9,
                    "needs_manual_review": True,
                    "is_forced_manual_review": False,
                }
            ]
        },
    )

    items = db.list_analysis_items(conn, goods_id="A001", mode="advanced")

    assert len(items) == 1
    item = items[0]
    assert item["category"] == "电子设备及数字载体"
    assert item["subcategory"] == "手机"
    assert item["brand"] == "HONOR"
    assert item["model"] is None
    assert item["color"] == "银灰色"
    assert item["condition"] == "完好无损"
    assert item["appearance_notes"] == "背面光滑金属质感"
    assert item["visible_text"] == "HONOR; SN123"
    assert item["confidence"] == 0.9
    assert item["needs_manual_review"] == 1
    assert item["is_sensitive"] == 0


def test_upsert_flattens_fast_mode_items_into_readable_columns():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(
        conn,
        goods_id="A002",
        mode="fast",
        source_files=["b.jpg"],
        result={
            "items": [
                {
                    "name": "手机",
                    "category": "电子设备及数字载体",
                    "subcategory": "手机",
                    "count": 1,
                    "confidence": 0.9,
                    "pending_review": False,
                    "is_sensitive_category": False,
                }
            ]
        },
    )

    items = db.list_analysis_items(conn, goods_id="A002", mode="fast")

    assert len(items) == 1
    assert items[0]["name"] == "手机"
    assert items[0]["subcategory"] == "手机"
    assert items[0]["brand"] is None  # fast mode never has this field
    assert items[0]["confidence"] == 0.9


def test_reanalyzing_replaces_item_rows_not_accumulates_them():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(
        conn, goods_id="A001", mode="fast", source_files=["v1.jpg"], result={"items": [{"name": "手机"}]}
    )
    db.upsert_analysis_record(
        conn,
        goods_id="A001",
        mode="fast",
        source_files=["v2.jpg"],
        result={"items": [{"name": "手机"}, {"name": "充电器"}]},
    )

    items = db.list_analysis_items(conn, goods_id="A001", mode="fast")
    assert [i["name"] for i in items] == ["手机", "充电器"]


def test_list_analysis_items_with_no_items_key_returns_empty():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=[], result={})
    assert db.list_analysis_items(conn, goods_id="A001", mode="fast") == []


def test_delete_analysis_record_removes_record_and_its_items():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(
        conn, goods_id="A001", mode="fast", source_files=["a.jpg"], result={"items": [{"name": "手机"}]}
    )

    deleted = db.delete_analysis_record(conn, goods_id="A001", mode="fast")

    assert deleted is True
    assert db.get_analysis_record(conn, goods_id="A001", mode="fast") is None
    assert db.list_analysis_items(conn, goods_id="A001", mode="fast") == []


def test_delete_analysis_record_returns_false_when_nothing_to_delete():
    conn = db.connect(":memory:")
    assert db.delete_analysis_record(conn, goods_id="does-not-exist", mode="fast") is False


def test_delete_analysis_record_does_not_affect_the_other_mode():
    conn = db.connect(":memory:")
    db.upsert_analysis_record(conn, goods_id="A001", mode="fast", source_files=["a.jpg"], result={})
    db.upsert_analysis_record(conn, goods_id="A001", mode="advanced", source_files=["a.jpg"], result={})

    db.delete_analysis_record(conn, goods_id="A001", mode="fast")

    assert db.get_analysis_record(conn, goods_id="A001", mode="fast") is None
    assert db.get_analysis_record(conn, goods_id="A001", mode="advanced") is not None

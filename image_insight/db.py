"""Local SQLite persistence for analysis results, keyed by goods_id + mode.

特别需求补充.md 1.6 requires the whole system to run offline — SQLite (a
single local file, no server process, no network) is the natural fit here,
same reasoning as running Qwen3-VL/PaddleOCR fully on-device.

Storage is an UPSERT keyed on (goods_id, mode), not an append-only log: per
product requirement, re-analyzing the SAME physical item (same goods_id —
see image_insight/goods.py for how that's derived from filenames) a second
time in the same mode REPLACES its existing record rather than adding a new
one. fast mode and advanced mode are stored as separate records (they
capture different information), so running one never clobbers the other's
saved result for the same item.
"""
from __future__ import annotations

import json
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal, Optional

AnalysisMode = Literal["fast", "advanced"]

DEFAULT_DB_PATH = "image_insight.db"

_SCHEMA = """
CREATE TABLE IF NOT EXISTS analysis_records (
    goods_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    source_files TEXT NOT NULL,
    result TEXT NOT NULL,
    updated_at TEXT NOT NULL,
    PRIMARY KEY (goods_id, mode)
);

CREATE TABLE IF NOT EXISTS analysis_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    goods_id TEXT NOT NULL,
    mode TEXT NOT NULL,
    item_index INTEGER NOT NULL,
    name TEXT,
    category TEXT,
    subcategory TEXT,
    brand TEXT,
    model TEXT,
    color TEXT,
    condition TEXT,
    appearance_notes TEXT,
    visible_text TEXT,
    count INTEGER,
    confidence REAL,
    needs_manual_review INTEGER NOT NULL DEFAULT 0,
    is_sensitive INTEGER NOT NULL DEFAULT 0,
    updated_at TEXT NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_analysis_items_goods_mode ON analysis_items (goods_id, mode);
"""


def _field_value(raw: Any) -> Any:
    """Advanced mode wraps most fields as an ObservedField {value, status}
    dict (see image_insight/vlm/schema.py); fast mode stores plain values.
    Unwrap either shape down to the bare value for a flat, readable column.
    """
    if isinstance(raw, dict) and "value" in raw:
        return raw.get("value")
    return raw


def _item_to_columns(item: dict[str, Any], *, mode: AnalysisMode) -> dict[str, Any]:
    """Flatten one result item (fast or advanced mode shape — see
    FastModeItem / ItemAnalysis) into analysis_items columns. The two modes
    use different field names for the same concepts (e.g. fast mode's
    `confidence` vs advanced mode's `category_confidence`), so the mapping
    branches on mode rather than trying to unify the schemas upstream.
    """
    visible_text = item.get("visible_text") or []
    if mode == "fast":
        confidence = item.get("confidence")
        needs_review = bool(item.get("pending_review"))
        is_sensitive = bool(item.get("is_sensitive_category"))
    else:
        confidence = item.get("category_confidence")
        needs_review = bool(item.get("needs_manual_review"))
        is_sensitive = bool(item.get("is_forced_manual_review"))
    return {
        "name": item.get("name"),
        "category": _field_value(item.get("category")),
        "subcategory": _field_value(item.get("subcategory")),
        "brand": _field_value(item.get("brand")),
        "model": _field_value(item.get("model")),
        "color": _field_value(item.get("color")),
        "condition": _field_value(item.get("condition")),
        "appearance_notes": item.get("appearance_notes"),
        "visible_text": "; ".join(str(t) for t in visible_text) if visible_text else None,
        "count": item.get("count"),
        "confidence": confidence,
        "needs_manual_review": needs_review,
        "is_sensitive": is_sensitive,
    }


def _replace_items(
    conn: sqlite3.Connection, *, goods_id: str, mode: AnalysisMode, items: list[dict[str, Any]], updated_at: str
) -> None:
    """Re-analyzing a goods_id replaces its item rows too (same
    overwrite-not-duplicate rule as analysis_records — see module
    docstring), so old items are cleared before the current ones are
    inserted rather than accumulating stale rows from earlier item counts.
    """
    conn.execute("DELETE FROM analysis_items WHERE goods_id = ? AND mode = ?", (goods_id, mode))
    for index, item in enumerate(items):
        columns = _item_to_columns(item, mode=mode)
        conn.execute(
            """
            INSERT INTO analysis_items (
                goods_id, mode, item_index, name, category, subcategory, brand, model, color,
                condition, appearance_notes, visible_text, count, confidence,
                needs_manual_review, is_sensitive, updated_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                goods_id,
                mode,
                index,
                columns["name"],
                columns["category"],
                columns["subcategory"],
                columns["brand"],
                columns["model"],
                columns["color"],
                columns["condition"],
                columns["appearance_notes"],
                columns["visible_text"],
                columns["count"],
                columns["confidence"],
                int(columns["needs_manual_review"]),
                int(columns["is_sensitive"]),
                updated_at,
            ),
        )


def _backfill_items_from_existing_records(conn: sqlite3.Connection) -> None:
    """One-time migration for databases created before analysis_items
    existed: populate it from whatever's already in analysis_records so
    older data gets the same readable columns without re-running analysis.
    Runs only when analysis_items is empty, so it never overwrites items
    written by the current schema.
    """
    (count,) = conn.execute("SELECT COUNT(*) FROM analysis_items").fetchone()
    if count > 0:
        return
    rows = conn.execute("SELECT goods_id, mode, result, updated_at FROM analysis_records").fetchall()
    for goods_id, mode, result_json, updated_at in rows:
        result = json.loads(result_json)
        items = result.get("items") or []
        if items:
            _replace_items(conn, goods_id=goods_id, mode=mode, items=items, updated_at=updated_at)
    conn.commit()


def connect(db_path: str | Path = DEFAULT_DB_PATH) -> sqlite3.Connection:
    """Open (creating if needed) the local SQLite database and ensure the
    schema exists.

    check_same_thread=False: the connection is created once at app startup
    on the main thread, but FastAPI request handlers that write to it run
    as async endpoint code (also the main/event-loop thread here — DB
    writes are small and fast enough not to need run_in_threadpool the way
    the GPU calls do), so a single long-lived connection shared across
    requests is fine; SQLite serializes writes internally.
    """
    conn = sqlite3.connect(str(db_path), check_same_thread=False)
    # WAL mode lets a reader (e.g. a colleague with DB Browser for SQLite
    # open on this same file) read concurrently with the app's writes
    # instead of hitting "database is locked"; busy_timeout makes any
    # remaining brief contention retry instead of failing outright. Both
    # are file-level settings — set once, they stick across reconnects.
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA busy_timeout=5000")
    conn.executescript(_SCHEMA)
    conn.commit()
    _backfill_items_from_existing_records(conn)
    return conn


def upsert_analysis_record(
    conn: sqlite3.Connection,
    *,
    goods_id: str,
    mode: AnalysisMode,
    source_files: list[str],
    result: dict[str, Any],
) -> None:
    """Store (or overwrite) the analysis result for one goods_id + mode.

    Analyzing the same goods_id again in the same mode replaces the
    existing row (ON CONFLICT ... DO UPDATE) rather than inserting a
    second one — the table holds the latest known analysis per physical
    item, not a growing history of every run.
    """
    updated_at = datetime.now(timezone.utc).isoformat()
    conn.execute(
        """
        INSERT INTO analysis_records (goods_id, mode, source_files, result, updated_at)
        VALUES (?, ?, ?, ?, ?)
        ON CONFLICT(goods_id, mode) DO UPDATE SET
            source_files = excluded.source_files,
            result = excluded.result,
            updated_at = excluded.updated_at
        """,
        (
            goods_id,
            mode,
            json.dumps(source_files, ensure_ascii=False),
            json.dumps(result, ensure_ascii=False),
            updated_at,
        ),
    )
    _replace_items(conn, goods_id=goods_id, mode=mode, items=result.get("items") or [], updated_at=updated_at)
    conn.commit()


@dataclass(frozen=True)
class AnalysisRecord:
    goods_id: str
    mode: AnalysisMode
    source_files: list[str]
    result: dict[str, Any]
    updated_at: str


def _row_to_record(row: tuple) -> AnalysisRecord:
    goods_id, mode, source_files_json, result_json, updated_at = row
    return AnalysisRecord(
        goods_id=goods_id,
        mode=mode,
        source_files=json.loads(source_files_json),
        result=json.loads(result_json),
        updated_at=updated_at,
    )


def get_analysis_record(
    conn: sqlite3.Connection, *, goods_id: str, mode: AnalysisMode
) -> Optional[AnalysisRecord]:
    row = conn.execute(
        "SELECT goods_id, mode, source_files, result, updated_at "
        "FROM analysis_records WHERE goods_id = ? AND mode = ?",
        (goods_id, mode),
    ).fetchone()
    return _row_to_record(row) if row is not None else None


def delete_analysis_record(conn: sqlite3.Connection, *, goods_id: str, mode: AnalysisMode) -> bool:
    """Delete one goods_id + mode record and its flattened item rows —
    e.g. to remove a bad analysis (wrong photo uploaded, test data) without
    waiting for a re-analysis to overwrite it. Returns whether a record
    actually existed to delete.
    """
    cursor = conn.execute(
        "DELETE FROM analysis_records WHERE goods_id = ? AND mode = ?", (goods_id, mode)
    )
    conn.execute("DELETE FROM analysis_items WHERE goods_id = ? AND mode = ?", (goods_id, mode))
    conn.commit()
    return cursor.rowcount > 0


def list_analysis_records(
    conn: sqlite3.Connection, *, mode: Optional[AnalysisMode] = None
) -> list[AnalysisRecord]:
    """All stored records, most recently updated first."""
    base = "SELECT goods_id, mode, source_files, result, updated_at FROM analysis_records"
    if mode is None:
        rows = conn.execute(base + " ORDER BY updated_at DESC").fetchall()
    else:
        rows = conn.execute(base + " WHERE mode = ? ORDER BY updated_at DESC", (mode,)).fetchall()
    return [_row_to_record(row) for row in rows]


_ITEM_COLUMNS = (
    "goods_id",
    "mode",
    "item_index",
    "name",
    "category",
    "subcategory",
    "brand",
    "model",
    "color",
    "condition",
    "appearance_notes",
    "visible_text",
    "count",
    "confidence",
    "needs_manual_review",
    "is_sensitive",
    "updated_at",
)


def list_analysis_items(
    conn: sqlite3.Connection, *, goods_id: Optional[str] = None, mode: Optional[AnalysisMode] = None
) -> list[dict[str, Any]]:
    """Flat, human-readable rows from analysis_items — one per recognized
    item, with 品牌/颜色/特征/描述 etc. as real columns instead of a JSON
    blob. This is what a colleague opening the .db file in a SQLite viewer
    (e.g. DB Browser for SQLite) would actually browse.
    """
    query = f"SELECT {', '.join(_ITEM_COLUMNS)} FROM analysis_items"
    conditions = []
    params: list[str] = []
    if goods_id is not None:
        conditions.append("goods_id = ?")
        params.append(goods_id)
    if mode is not None:
        conditions.append("mode = ?")
        params.append(mode)
    if conditions:
        query += " WHERE " + " AND ".join(conditions)
    query += " ORDER BY updated_at DESC, goods_id, item_index"
    rows = conn.execute(query, params).fetchall()
    return [dict(zip(_ITEM_COLUMNS, row)) for row in rows]

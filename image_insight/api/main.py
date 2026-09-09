"""FastAPI app tying fast mode and advanced mode together — both backed by
the same on-device Qwen3-VL model, loaded once (see LocalQwenTransport).

Fast mode was originally RF-DETR-based (5 fixed COCO categories); replaced
with an open-vocabulary, natural-language Qwen3-VL prompt per product
feedback that a fixed category list was too narrow. See
image_insight/vlm/fast_analyzer.py for the rationale. RF-DETR's
zero-shot phone-counting code (image_insight/detection/phone_counter.py)
is kept as a standalone, tested, evaluated artifact (93.3% accuracy on a
manual ground-truth sample, see eval/) but is no longer wired into the app.

Both models run entirely on-device (no network calls at request time — see
特别需求补充.md 1.6) and are loaded once at startup via the lifespan handler,
injected through Depends — tests override get_fast_analyzer/get_analyzer
directly rather than loading real models or touching app.state.

model.generate() is a long, blocking, GPU-bound call. Running it directly
inside an `async def` endpoint would freeze FastAPI's single-threaded event
loop for the whole request — not just that one client, but every other
concurrent request, including trivial ones like loading "/". Both endpoints
below dispatch the actual analyze() call to a worker thread via
run_in_threadpool, and serialize GPU access through app.state.gpu_lock so
two requests never call generate() on the same model concurrently.
"""
from __future__ import annotations

import asyncio
import logging
import sqlite3
from contextlib import asynccontextmanager
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import List, Optional

from fastapi import Depends, FastAPI, File, HTTPException, UploadFile
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from starlette.concurrency import run_in_threadpool
from starlette.requests import Request

from image_insight import db
from image_insight.config import load_advanced_qwen_config, load_db_config, load_ocr_config, load_qwen_config
from image_insight.goods import group_uploaded_photos
from image_insight.ocr.pp_ocr import PaddleOcrTextRecognizer
from image_insight.vlm.fast_analyzer import FastVisionAnalyzer, analyze_goods_group
from image_insight.vlm.qwen_client import LocalQwenTransport, QwenVisionAnalyzer, analyze_advanced_group

logger = logging.getLogger(__name__)

BASE_DIR = Path(__file__).resolve().parent


def get_fast_analyzer(request: Request) -> FastVisionAnalyzer:
    analyzer = getattr(request.app.state, "fast_analyzer", None)
    if analyzer is None:
        raise HTTPException(
            status_code=503, detail="fast mode is unavailable — local Qwen3-VL model failed to load"
        )
    return analyzer


def get_analyzer(request: Request) -> Optional[QwenVisionAnalyzer]:
    return getattr(request.app.state, "analyzer", None)


def get_db(request: Request) -> Optional[sqlite3.Connection]:
    """Optional — persistence is a supplementary feature, mirroring how OCR
    is handled: if the database failed to open, analysis must still work
    and return results to the user, just without being saved. Callers check
    for None and skip the save rather than failing the whole request.
    """
    return getattr(request.app.state, "db", None)


def create_app(*, load_models: bool = True) -> FastAPI:
    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.fast_analyzer = None
        app.state.analyzer = None
        app.state.db = None
        app.state.gpu_lock = asyncio.Lock()
        if load_models:
            db_config = load_db_config()
            try:
                app.state.db = db.connect(db_config.path)
            except Exception:
                logger.exception(
                    "failed to open local database (%s) — analysis results will not be saved", db_config.path
                )
            qwen_config = load_qwen_config()
            ocr_config = load_ocr_config()
            logger.info("loading local Qwen3-VL (%s)...", qwen_config.model_id)
            try:
                transport = LocalQwenTransport(
                    qwen_config.model_id,
                    max_new_tokens=qwen_config.max_new_tokens,
                    max_pixels=qwen_config.max_image_pixels,
                    min_pixels=qwen_config.min_image_pixels,
                )
                # Fast and advanced mode share one loaded model instance —
                # only the prompt differs, so there's no reason to pay for
                # two copies of the weights.
                app.state.fast_analyzer = FastVisionAnalyzer(transport)

                ocr_recognizer = None
                if ocr_config.enabled:
                    try:
                        ocr_recognizer = PaddleOcrTextRecognizer()
                    except Exception:
                        # OCR is a supplementary "看清字" cross-check, not a
                        # hard dependency — advanced mode still works without
                        # it, just without the OCR cross-check on visible text.
                        logger.exception("failed to load PaddleOCR — advanced mode will run without OCR cross-check")

                advanced_qwen_config = load_advanced_qwen_config()
                app.state.analyzer = QwenVisionAnalyzer(
                    transport,
                    model_version=qwen_config.model_id,
                    ocr=ocr_recognizer,
                    ocr_min_confidence=ocr_config.min_text_confidence,
                    # Advanced mode's own max_new_tokens/timeout, independent
                    # of qwen_config.max_new_tokens above (which still governs
                    # fast_analyzer via the shared transport's default) — see
                    # image_insight.config.AdvancedQwenConfig.
                    max_new_tokens=advanced_qwen_config.max_new_tokens,
                    timeout_seconds=advanced_qwen_config.timeout_seconds,
                    repetition_penalty=advanced_qwen_config.repetition_penalty,
                )
            except Exception:
                logger.exception(
                    "failed to load local Qwen3-VL model (%s) — fast and advanced mode disabled",
                    qwen_config.model_id,
                )
        yield

    app = FastAPI(title="图像洞察", lifespan=lifespan)
    app.mount("/static", StaticFiles(directory=str(BASE_DIR / "static")), name="static")
    templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))

    @app.get("/", response_class=HTMLResponse)
    async def index(request: Request):
        # Reflects the REAL startup outcome (see lifespan above), not a
        # hardcoded "system healthy" chip — if Qwen3-VL failed to load,
        # the page must say so rather than claim readiness it doesn't have.
        model_ready = getattr(request.app.state, "fast_analyzer", None) is not None
        return templates.TemplateResponse(request, "index.html", {"model_ready": model_ready})

    def _labels_for(group, uploads: List[tuple]) -> List[str]:
        by_path = dict(uploads)
        return [by_path[p] for p in group.photo_paths]

    def _save_result(
        conn: Optional[sqlite3.Connection], *, goods_id: str, mode: db.AnalysisMode, source_files: List[str], result
    ) -> None:
        """Persist one group's result, keyed so a repeat analysis of the
        same goods_id overwrites rather than duplicates (see
        image_insight/db.py). Best-effort: a save failure must never break
        an otherwise-successful analysis response, so it's logged and
        swallowed here rather than propagated — same pattern as OCR.
        """
        if conn is None:
            return
        try:
            db.upsert_analysis_record(
                conn,
                goods_id=goods_id,
                mode=mode,
                source_files=source_files,
                result=result.model_dump(mode="json", by_alias=True),
            )
        except Exception:
            logger.exception("failed to save analysis result for goods_id=%s mode=%s", goods_id, mode)

    async def _save_uploads(files: List[UploadFile]) -> List[tuple]:
        """Save uploads to temp files, keeping (temp_path, original_filename)
        pairs — group_uploaded_photos needs the ORIGINAL filename to detect
        goods_id, which the random temp filename doesn't carry.
        """
        saved = []
        for i, upload in enumerate(files):
            data = await upload.read()
            original_name = upload.filename or f"upload_{i}.jpg"
            suffix = Path(original_name).suffix or ".jpg"
            tmp = NamedTemporaryFile(delete=False, suffix=suffix)
            tmp.write(data)
            tmp.close()
            saved.append((Path(tmp.name), original_name))
        return saved

    @app.post("/api/analyze/fast")
    async def analyze_fast(
        request: Request,
        files: List[UploadFile] = File(...),
        analyzer: FastVisionAnalyzer = Depends(get_fast_analyzer),
        db_conn: Optional[sqlite3.Connection] = Depends(get_db),
    ):
        """Fast mode: open-vocabulary name + count, no brand/model/text detail.

        Accepts one or more files, grouped by goods_id (see
        image_insight.goods.group_uploaded_photos): files whose names match
        the dataset's <goods_id>_<timestamp><hash> convention are treated as
        multiple angle photos of the SAME physical item and merged (特别需求
        补充.md 1.2, max count across angles, not summed); any other filename
        is its own independent item — this is what makes uploading a batch
        of unrelated photos in one request work, not just multi-angle sets.
        """
        uploads = await _save_uploads(files)
        try:
            groups = group_uploaded_photos(uploads)
            async with request.app.state.gpu_lock:
                results = []
                for group in groups:
                    result = await run_in_threadpool(analyze_goods_group, analyzer, list(group.photo_paths))
                    source_files = _labels_for(group, uploads)
                    _save_result(
                        db_conn, goods_id=group.goods_id, mode="fast", source_files=source_files, result=result
                    )
                    results.append({"label": group.goods_id, "source_files": source_files, "result": result})
        finally:
            for p, _ in uploads:
                p.unlink(missing_ok=True)
        return {"groups": results}

    @app.post("/api/analyze/advanced")
    async def analyze_advanced(
        request: Request,
        files: List[UploadFile] = File(...),
        analyzer: Optional[QwenVisionAnalyzer] = Depends(get_analyzer),
        db_conn: Optional[sqlite3.Connection] = Depends(get_db),
    ):
        """Advanced mode: structured brand/model/color/text/condition analysis.

        Same batch/multi-angle grouping as analyze_fast (see its docstring),
        merged per group via analyze_advanced_group.
        """
        if analyzer is None:
            raise HTTPException(
                status_code=503,
                detail="advanced mode is unavailable — local Qwen3-VL model failed to load, check server logs",
            )
        uploads = await _save_uploads(files)
        try:
            groups = group_uploaded_photos(uploads)
            async with request.app.state.gpu_lock:
                results = []
                for group in groups:
                    result = await run_in_threadpool(analyze_advanced_group, analyzer, list(group.photo_paths))
                    source_files = _labels_for(group, uploads)
                    _save_result(
                        db_conn,
                        goods_id=group.goods_id,
                        mode="advanced",
                        source_files=source_files,
                        result=result,
                    )
                    results.append({"label": group.goods_id, "source_files": source_files, "result": result})
        finally:
            for p, _ in uploads:
                p.unlink(missing_ok=True)
        return {"groups": results}

    @app.get("/api/records")
    async def list_records(mode: Optional[str] = None, db_conn: Optional[sqlite3.Connection] = Depends(get_db)):
        """List stored analysis records, most recently updated first — the
        durable "一物一档" view: what's currently on file per goods_id,
        after upserts, not a run-by-run history.
        """
        if mode not in (None, "fast", "advanced"):
            raise HTTPException(status_code=400, detail="mode must be 'fast' or 'advanced'")
        if db_conn is None:
            return {"records": []}
        records = db.list_analysis_records(db_conn, mode=mode)
        return {
            "records": [
                {
                    "goods_id": r.goods_id,
                    "mode": r.mode,
                    "source_files": r.source_files,
                    "result": r.result,
                    "updated_at": r.updated_at,
                }
                for r in records
            ]
        }

    @app.delete("/api/records")
    async def delete_record(
        goods_id: str, mode: str, db_conn: Optional[sqlite3.Connection] = Depends(get_db)
    ):
        """Remove one stored record (and its flattened items) — e.g. to
        clear a bad upload without waiting for a re-analysis to overwrite
        it (增删改查: the missing 删 alongside list/upsert/get).
        """
        if mode not in ("fast", "advanced"):
            raise HTTPException(status_code=400, detail="mode must be 'fast' or 'advanced'")
        if db_conn is None:
            raise HTTPException(status_code=503, detail="database is unavailable")
        deleted = db.delete_analysis_record(db_conn, goods_id=goods_id, mode=mode)
        if not deleted:
            raise HTTPException(status_code=404, detail="no record found for that goods_id/mode")
        return {"deleted": True}

    return app


app = create_app(load_models=True)

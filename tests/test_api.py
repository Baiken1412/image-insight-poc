from __future__ import annotations

import io
from pathlib import Path

from fastapi.testclient import TestClient

from image_insight import db
from image_insight.api.main import create_app, get_analyzer, get_db, get_fast_analyzer
from image_insight.vlm.fast_analyzer import FastModeItem, FastModeResult
from image_insight.vlm.schema import AdvancedAnalysisResult, FieldStatus, ItemAnalysis, ObservedField


class FakeFastAnalyzer:
    def __init__(self, result: FastModeResult):
        self._result = result

    def analyze(self, image_path: Path) -> FastModeResult:
        return self._result


class FakeAnalyzer:
    def __init__(self, result: AdvancedAnalysisResult):
        self._result = result

    def analyze(self, image_path: Path) -> AdvancedAnalysisResult:
        return self._result


def make_client(*, fast_analyzer=None, analyzer="__unset__", with_db=False) -> TestClient:
    app = create_app(load_models=False)
    if fast_analyzer is not None:
        app.dependency_overrides[get_fast_analyzer] = lambda: fast_analyzer
    if analyzer != "__unset__":
        app.dependency_overrides[get_analyzer] = lambda: analyzer
    if with_db:
        # in-memory, isolated per test — never touches a real file on disk.
        test_db = db.connect(":memory:")
        app.dependency_overrides[get_db] = lambda: test_db
    client = TestClient(app)
    client.__enter__()  # run lifespan startup so app.state.gpu_lock exists, as it does under a real server
    if with_db:
        client.db = test_db
    return client


def fake_image_bytes() -> bytes:
    return b"\xff\xd8\xff\xd9"  # minimal JPEG magic bytes; analyzer is faked, content is irrelevant


def test_index_page_serves_html():
    client = make_client()
    response = client.get("/")
    assert response.status_code == 200
    assert "图像洞察" in response.text


def test_fast_mode_returns_categorized_items_for_single_photo():
    fast = FakeFastAnalyzer(
        FastModeResult(
            items=[
                FastModeItem(name="手机", category="电子设备及数字载体", count=1),
                FastModeItem(name="钱包", category="贵重物品", count=1),
            ]
        )
    )
    client = make_client(fast_analyzer=fast)
    response = client.post(
        "/api/analyze/fast",
        files={"files": ("phone.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["groups"]) == 1
    group = body["groups"][0]
    assert group["label"] == "phone.jpg"  # unrecognized filename -> its own singleton group
    assert group["source_files"] == ["phone.jpg"]
    by_name = {item["name"]: item for item in group["result"]["items"]}
    assert by_name["手机"]["count"] == 1
    assert by_name["手机"]["category"] == "电子设备及数字载体"
    assert by_name["手机"]["is_sensitive_category"] is False
    assert by_name["钱包"]["count"] == 1


def test_fast_mode_merges_same_goods_id_angle_photos_by_taking_max_not_sum():
    fast = FakeFastAnalyzer(FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=2)]))
    client = make_client(fast_analyzer=fast)
    files = [
        ("files", ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
        ("files", ("A001_20260101000001BBBB.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
    ]
    response = client.post("/api/analyze/fast", files=files)
    assert response.status_code == 200
    body = response.json()
    # same goods_id prefix -> one merged group, not two
    assert len(body["groups"]) == 1
    group = body["groups"][0]
    assert group["label"] == "A001"
    assert set(group["source_files"]) == {"A001_20260101000000AAAA.jpg", "A001_20260101000001BBBB.jpg"}
    # fake analyzer reports 2 phones per photo regardless of angle count;
    # max-across-angles means the total stays 2, not 2 photos * 2 = 4.
    items = {item["name"]: item for item in group["result"]["items"]}
    assert items["手机"]["count"] == 2


def test_fast_mode_batch_uploads_of_unrelated_photos_get_independent_groups():
    fast = FakeFastAnalyzer(FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=1)]))
    client = make_client(fast_analyzer=fast)
    files = [
        ("files", ("IMG_0001.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
        ("files", ("IMG_0002.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
        ("files", ("IMG_0003.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
    ]
    response = client.post("/api/analyze/fast", files=files)
    assert response.status_code == 200
    body = response.json()
    # none of these filenames share a goods_id -> three independent groups,
    # NOT merged into one — this is the batch-processing behavior.
    assert len(body["groups"]) == 3
    labels = {g["label"] for g in body["groups"]}
    assert labels == {"IMG_0001.jpg", "IMG_0002.jpg", "IMG_0003.jpg"}


def test_fast_mode_returns_503_when_model_not_loaded():
    client = make_client()  # no fast_analyzer override -> app.state.fast_analyzer stays None
    response = client.post(
        "/api/analyze/fast",
        files={"files": ("phone.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )
    assert response.status_code == 503


def test_advanced_mode_returns_structured_result():
    result = AdvancedAnalysisResult(
        items=[
            ItemAnalysis(
                category=ObservedField(value="手机", status=FieldStatus.CONFIRMED),
                brand=ObservedField(value="华为", status=FieldStatus.CONFIRMED),
                model=ObservedField(value=None, status=FieldStatus.UNKNOWN),
                color=ObservedField(value="黑色", status=FieldStatus.CONFIRMED),
                count=1,
                condition=ObservedField(value="良好", status=FieldStatus.CONFIRMED),
            )
        ],
        model_version="qwen3-vl-test",
    )
    client = make_client(analyzer=FakeAnalyzer(result))
    response = client.post(
        "/api/analyze/advanced",
        files={"files": ("phone.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert len(body["groups"]) == 1
    group = body["groups"][0]
    assert group["label"] == "phone.jpg"
    assert group["result"]["items"][0]["brand"]["value"] == "华为"
    assert group["result"]["needs_manual_review"] is True  # model_name unknown


def test_advanced_mode_merges_same_goods_id_angle_photos_by_category():
    result = AdvancedAnalysisResult(
        items=[
            ItemAnalysis(
                category=ObservedField(value="手机", status=FieldStatus.CONFIRMED),
                brand=ObservedField(value="华为", status=FieldStatus.CONFIRMED),
                model=ObservedField(value=None, status=FieldStatus.UNKNOWN),
                color=ObservedField(value="黑色", status=FieldStatus.CONFIRMED),
                count=1,
                condition=ObservedField(value="良好", status=FieldStatus.CONFIRMED),
            )
        ],
        model_version="qwen3-vl-test",
    )
    client = make_client(analyzer=FakeAnalyzer(result))
    files = [
        ("files", ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
        ("files", ("A001_20260101000001BBBB.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
    ]
    response = client.post("/api/analyze/advanced", files=files)
    assert response.status_code == 200
    body = response.json()
    # the fake analyzer returns the same single-item result for every photo;
    # merging by category must not double the item into two entries, and the
    # shared goods_id must not split into two groups.
    assert len(body["groups"]) == 1
    result_items = body["groups"][0]["result"]["items"]
    assert len(result_items) == 1
    assert result_items[0]["brand"]["value"] == "华为"


def test_advanced_mode_batch_uploads_of_unrelated_photos_get_independent_groups():
    result = AdvancedAnalysisResult(
        items=[ItemAnalysis(
            category=ObservedField(value="手机", status=FieldStatus.CONFIRMED),
            brand=ObservedField(value=None, status=FieldStatus.UNKNOWN),
            model=ObservedField(value=None, status=FieldStatus.UNKNOWN),
            color=ObservedField(value=None, status=FieldStatus.UNKNOWN),
            count=1,
            condition=ObservedField(value=None, status=FieldStatus.UNKNOWN),
        )],
        model_version="qwen3-vl-test",
    )
    client = make_client(analyzer=FakeAnalyzer(result))
    files = [
        ("files", ("IMG_0001.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
        ("files", ("IMG_0002.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")),
    ]
    response = client.post("/api/analyze/advanced", files=files)
    assert response.status_code == 200
    body = response.json()
    assert len(body["groups"]) == 2
    labels = {g["label"] for g in body["groups"]}
    assert labels == {"IMG_0001.jpg", "IMG_0002.jpg"}


def test_advanced_mode_surfaces_truncated_flag_from_analyzer():
    result = AdvancedAnalysisResult(
        items=[
            ItemAnalysis(
                category=ObservedField(value="手机", status=FieldStatus.CONFIRMED),
                brand=ObservedField(value=None, status=FieldStatus.UNKNOWN),
                model=ObservedField(value=None, status=FieldStatus.UNKNOWN),
                color=ObservedField(value=None, status=FieldStatus.UNKNOWN),
                count=1,
                condition=ObservedField(value=None, status=FieldStatus.UNKNOWN),
            )
        ],
        model_version="qwen3-vl-test",
        truncated=True,
    )
    client = make_client(analyzer=FakeAnalyzer(result))
    response = client.post(
        "/api/analyze/advanced",
        files={"files": ("phone.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200
    body = response.json()
    assert body["groups"][0]["result"]["truncated"] is True
    assert body["groups"][0]["result"]["needs_manual_review"] is True


def test_advanced_mode_returns_503_when_not_configured():
    client = make_client(analyzer=None)
    response = client.post(
        "/api/analyze/advanced",
        files={"files": ("phone.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )
    assert response.status_code == 503
    assert "Qwen3-VL" in response.json()["detail"]


# ---------------------------------------------------------------------------
# Persistence: analysis results are saved, and re-analyzing the same item
# overwrites its record instead of adding a new one.
# ---------------------------------------------------------------------------


def test_fast_mode_analysis_is_saved_to_the_database():
    fast = FakeFastAnalyzer(FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=1)]))
    client = make_client(fast_analyzer=fast, with_db=True)
    client.post(
        "/api/analyze/fast",
        files={"files": ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )

    record = db.get_analysis_record(client.db, goods_id="A001", mode="fast")
    assert record is not None
    assert record.result["items"][0]["name"] == "手机"


def test_reanalyzing_the_same_goods_id_overwrites_the_saved_record_not_duplicates():
    client = make_client(
        fast_analyzer=FakeFastAnalyzer(
            FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=1)])
        ),
        with_db=True,
    )
    client.post(
        "/api/analyze/fast",
        files={"files": ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )

    # second analysis of the SAME goods_id, different result this time
    client.app.dependency_overrides[get_fast_analyzer] = lambda: FakeFastAnalyzer(
        FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=2)])
    )
    client.post(
        "/api/analyze/fast",
        files={"files": ("A001_20260101000001BBBB.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )

    all_records = db.list_analysis_records(client.db, mode="fast")
    assert len(all_records) == 1  # overwritten, not duplicated
    assert all_records[0].result["items"][0]["count"] == 2
    assert all_records[0].source_files == ["A001_20260101000001BBBB.jpg"]  # latest source, not accumulated


def test_advanced_mode_analysis_is_also_saved_and_keyed_separately_from_fast_mode():
    result = AdvancedAnalysisResult(
        items=[
            ItemAnalysis(
                category=ObservedField(value="手机", status=FieldStatus.CONFIRMED),
                brand=ObservedField(value="华为", status=FieldStatus.CONFIRMED),
                model=ObservedField(value=None, status=FieldStatus.UNKNOWN),
                color=ObservedField(value="黑色", status=FieldStatus.CONFIRMED),
                count=1,
                condition=ObservedField(value="良好", status=FieldStatus.CONFIRMED),
            )
        ],
        model_version="qwen3-vl-test",
    )
    client = make_client(analyzer=FakeAnalyzer(result), with_db=True)
    client.post(
        "/api/analyze/advanced",
        files={"files": ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )

    record = db.get_analysis_record(client.db, goods_id="A001", mode="advanced")
    assert record is not None
    assert record.result["items"][0]["brand"]["value"] == "华为"
    # fast mode's earlier save (if any) for this goods_id must be untouched
    assert db.get_analysis_record(client.db, goods_id="A001", mode="fast") is None


def test_analysis_still_succeeds_when_no_database_is_configured():
    # with_db defaults to False -> get_db returns None -> saving is skipped,
    # but the analysis response itself must be unaffected (best-effort save).
    fast = FakeFastAnalyzer(FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=1)]))
    client = make_client(fast_analyzer=fast)
    response = client.post(
        "/api/analyze/fast",
        files={"files": ("phone.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )
    assert response.status_code == 200


def test_list_records_endpoint_returns_saved_records():
    fast = FakeFastAnalyzer(FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=1)]))
    client = make_client(fast_analyzer=fast, with_db=True)
    client.post(
        "/api/analyze/fast",
        files={"files": ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )

    response = client.get("/api/records")
    assert response.status_code == 200
    records = response.json()["records"]
    assert len(records) == 1
    assert records[0]["goods_id"] == "A001"
    assert records[0]["mode"] == "fast"


def test_list_records_endpoint_returns_empty_when_no_database_is_configured():
    client = make_client()
    response = client.get("/api/records")
    assert response.status_code == 200
    assert response.json()["records"] == []


def test_list_records_endpoint_rejects_invalid_mode_query_param():
    client = make_client(with_db=True)
    response = client.get("/api/records", params={"mode": "bogus"})
    assert response.status_code == 400


def test_delete_record_endpoint_removes_a_saved_record():
    fast = FakeFastAnalyzer(FastModeResult(items=[FastModeItem(name="手机", category="电子设备及数字载体", count=1)]))
    client = make_client(fast_analyzer=fast, with_db=True)
    client.post(
        "/api/analyze/fast",
        files={"files": ("A001_20260101000000AAAA.jpg", io.BytesIO(fake_image_bytes()), "image/jpeg")},
    )

    response = client.delete("/api/records", params={"goods_id": "A001", "mode": "fast"})
    assert response.status_code == 200
    assert response.json() == {"deleted": True}
    assert client.get("/api/records").json()["records"] == []


def test_delete_record_endpoint_returns_404_for_unknown_record():
    client = make_client(with_db=True)
    response = client.delete("/api/records", params={"goods_id": "does-not-exist", "mode": "fast"})
    assert response.status_code == 404


def test_delete_record_endpoint_rejects_invalid_mode_query_param():
    client = make_client(with_db=True)
    response = client.delete("/api/records", params={"goods_id": "A001", "mode": "bogus"})
    assert response.status_code == 400


def test_delete_record_endpoint_returns_503_when_no_database_is_configured():
    client = make_client()
    response = client.delete("/api/records", params={"goods_id": "A001", "mode": "fast"})
    assert response.status_code == 503

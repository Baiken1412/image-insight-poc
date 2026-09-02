from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_insight.ocr.pp_ocr import OcrTextBox
from image_insight.vlm.qwen_client import QwenVisionAnalyzer, analyze_advanced_group
from image_insight.vlm.schema import FieldStatus


VALID_RESPONSE = {
    "items": [
        {
            "category": {"value": "手机", "status": "confirmed"},
            "brand": {"value": "华为", "status": "confirmed"},
            "model": {"value": None, "status": "unknown"},
            "color": {"value": "黑色", "status": "confirmed"},
            "count": 1,
            "visible_text": ["HUAWEI"],
            "appearance_notes": "背面摄像头模组明显磨损",
            "condition": {"value": "轻微磨损", "status": "suspected"},
        }
    ]
}


class FakeTransport:
    def __init__(self, response_text: str | None = None, error: Exception | None = None):
        self._response_text = response_text
        self._error = error
        self.last_call = None

    def complete(self, *, image_base64: str, system_prompt: str) -> str:
        self.last_call = {"image_base64": image_base64, "system_prompt": system_prompt}
        if self._error:
            raise self._error
        return self._response_text


@pytest.fixture
def tiny_jpeg(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")  # minimal valid-looking JPEG magic bytes
    return path


def test_analyze_parses_well_formed_json_response(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert len(result.items) == 1
    item = result.items[0]
    assert item.brand.value == "华为"
    assert item.model_name.status is FieldStatus.UNKNOWN
    assert result.needs_manual_review is True  # model/condition aren't confirmed


def test_analyze_strips_markdown_code_fences(tiny_jpeg):
    fenced = "```json\n" + json.dumps(VALID_RESPONSE) + "\n```"
    transport = FakeTransport(response_text=fenced)
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert len(result.items) == 1


def test_analyze_sends_base64_image_and_system_prompt(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    analyzer.analyze(tiny_jpeg)

    assert transport.last_call is not None
    assert transport.last_call["image_base64"]  # non-empty
    assert "禁止编造" in transport.last_call["system_prompt"]


def test_analyze_falls_back_to_manual_review_on_invalid_json(tiny_jpeg):
    transport = FakeTransport(response_text="this is not json at all")
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.items == []
    assert result.parse_error is not None
    assert result.needs_manual_review is True


def test_analyze_falls_back_to_manual_review_when_model_fabricates_a_confirmed_field_without_value(tiny_jpeg):
    # A field claiming "confirmed" but omitting a value would be a disguised
    # fabrication/omission; the schema must reject it rather than pass it through.
    broken = json.loads(json.dumps(VALID_RESPONSE))
    broken["items"][0]["brand"] = {"value": None, "status": "confirmed"}
    transport = FakeTransport(response_text=json.dumps(broken))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.items == []
    assert result.parse_error is not None


def test_analyze_falls_back_to_manual_review_when_transport_raises(tiny_jpeg):
    transport = FakeTransport(error=ConnectionError("network unreachable"))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.items == []
    assert "network unreachable" in result.parse_error
    assert result.needs_manual_review is True


def test_analyze_salvages_complete_items_from_truncated_generation(tiny_jpeg):
    # Real observed bug: advanced mode's per-item schema grew enough fields
    # that a photo with multiple items could hit max_new_tokens before the
    # JSON closed, raising json.JSONDecodeError("Unterminated string...")
    # and discarding the ENTIRE response — including the first item, which
    # finished generating perfectly cleanly before the cutoff.
    first_item = json.loads(json.dumps(VALID_RESPONSE))["items"][0]
    truncated_raw = (
        '{"items": [' + json.dumps(first_item) + ', {"category": {"value": "贵重物品", '
        '"status": "confirmed"}, "brand": {"value": "cut off mid'
    )
    transport = FakeTransport(response_text=truncated_raw)
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert result.truncated is True
    assert len(result.items) == 1
    assert result.items[0].brand.value == "华为"  # the complete first item survived intact
    assert result.needs_manual_review is True  # truncation alone must force review


def test_analyze_truncation_with_no_complete_items_falls_back_to_parse_error(tiny_jpeg):
    # If even the FIRST item is cut off, there's nothing genuine to salvage
    # — must behave exactly like today's plain-invalid-JSON case.
    transport = FakeTransport(response_text='{"items": [{"category": {"value": "cut off mid')
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.items == []
    assert result.truncated is False
    assert result.parse_error is not None


def test_analyze_never_raises_even_on_completely_malformed_missing_items_key(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps({"unexpected": "shape"}))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)  # must not raise

    assert result.parse_error is not None


def test_analyze_repairs_bare_string_fields_the_model_forgot_to_nest(tiny_jpeg):
    # Observed in real usage: Qwen3-VL-2B sometimes answers a field with a
    # bare string instead of {"value": ..., "status": ...} even when it
    # clearly knows the answer. This must not be discarded as parse_error —
    # but a bare value that skipped the confidence-labeling protocol is
    # downgraded to "suspected", never silently upgraded to "confirmed".
    flat = {
        "items": [
            {
                "category": "普通实物及作案工具",
                "brand": "XINGYU",
                "model": None,
                "color": "蓝色",
                "count": 1,
                "visible_text": [],
                "appearance_notes": "",
                "condition": "良好",
            }
        ]
    }
    transport = FakeTransport(response_text=json.dumps(flat))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    item = result.items[0]
    assert item.category.value == "普通实物及作案工具"
    assert item.category.status is FieldStatus.SUSPECTED  # not CONFIRMED — format was non-compliant
    assert item.brand.value == "XINGYU"
    assert item.brand.status is FieldStatus.SUSPECTED


def test_analyze_snaps_category_to_official_taxonomy_even_when_well_formed(tiny_jpeg):
    # Free-text category guesses (e.g. "手套") must not pass through raw —
    # they get snapped to the official 12-大类 taxonomy (image_insight/taxonomy.py)
    # so the UI always shows a meaningful, consistent classification.
    payload = json.loads(json.dumps(VALID_RESPONSE))
    payload["items"][0]["category"] = {"value": "手套", "status": "confirmed"}
    transport = FakeTransport(response_text=json.dumps(payload))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert result.items[0].category.value == "其他/未分类"
    assert result.items[0].category.status is FieldStatus.CONFIRMED  # shape was fine, only value snapped


def test_analyze_handles_mixed_nested_and_bare_fields_in_the_same_item(tiny_jpeg):
    mixed = {
        "items": [
            {
                "category": {"value": "手机", "status": "confirmed"},  # properly nested
                "brand": "华为",  # bare string
                "model": {"value": None, "status": "unknown"},
                "color": "",  # bare empty string -> unknown
                "count": 1,
                "condition": {"value": "良好", "status": "confirmed"},
            }
        ]
    }
    transport = FakeTransport(response_text=json.dumps(mixed))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    item = result.items[0]
    assert item.category.status is FieldStatus.CONFIRMED  # untouched, was already well-formed
    assert item.brand.status is FieldStatus.SUSPECTED  # repaired, downgraded
    assert item.color.value is None
    assert item.color.status is FieldStatus.UNKNOWN


def test_analyze_resolves_subcategory_and_overrides_category(tiny_jpeg):
    payload = json.loads(json.dumps(VALID_RESPONSE))
    payload["items"][0]["category"] = {"value": "普通实物及作案工具", "status": "confirmed"}
    payload["items"][0]["subcategory"] = {"value": "手机", "status": "confirmed"}
    transport = FakeTransport(response_text=json.dumps(payload))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    item = result.items[0]
    assert item.subcategory.value == "手机"
    assert item.category.value == "电子设备及数字载体"  # derived from subcategory, overriding the mismatch


def test_analyze_flags_category_pending_review_on_low_confidence(tiny_jpeg):
    payload = json.loads(json.dumps(VALID_RESPONSE))
    payload["items"][0]["category_confidence"] = 0.1
    transport = FakeTransport(response_text=json.dumps(payload))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].category_pending_review is True
    assert result.needs_manual_review is True


def test_analyze_tolerates_malformed_category_confidence_without_dropping_the_item(tiny_jpeg):
    # Same class of regression as fast mode: a bad confidence value must not
    # discard an otherwise-good structured result.
    payload = json.loads(json.dumps(VALID_RESPONSE))
    payload["items"][0]["category_confidence"] = "很高"
    transport = FakeTransport(response_text=json.dumps(payload))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert len(result.items) == 1


class FakeOcr:
    def __init__(self, boxes=None, error: Exception | None = None):
        self._boxes = boxes or []
        self._error = error
        self.calls = 0

    def read_text(self, image_path: Path):
        self.calls += 1
        if self._error:
            raise self._error
        return self._boxes


def test_analyze_merges_ocr_text_into_the_single_item(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    ocr = FakeOcr(boxes=[OcrTextBox(text="IMEI 123456", confidence=0.95)])
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test", ocr=ocr)

    result = analyzer.analyze(tiny_jpeg)

    assert "IMEI 123456" in result.items[0].visible_text
    assert "HUAWEI" in result.items[0].visible_text  # original VLM text preserved
    assert ocr.calls == 1


def test_analyze_low_confidence_ocr_text_is_dropped(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    ocr = FakeOcr(boxes=[OcrTextBox(text="???", confidence=0.1)])
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test", ocr=ocr)

    result = analyzer.analyze(tiny_jpeg)

    assert "???" not in result.items[0].visible_text


def test_analyze_ocr_failure_does_not_void_an_otherwise_good_result(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    ocr = FakeOcr(error=RuntimeError("paddle crashed"))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test", ocr=ocr)

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert result.items[0].brand.value == "华为"  # VLM result intact despite OCR failure


def test_analyze_puts_ocr_text_at_result_level_when_multiple_items(tiny_jpeg):
    two_items = json.loads(json.dumps(VALID_RESPONSE))
    two_items["items"].append(json.loads(json.dumps(VALID_RESPONSE))["items"][0])
    transport = FakeTransport(response_text=json.dumps(two_items))
    ocr = FakeOcr(boxes=[OcrTextBox(text="unattributed text", confidence=0.9)])
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test", ocr=ocr)

    result = analyzer.analyze(tiny_jpeg)

    assert "unattributed text" in result.ocr_text
    assert all("unattributed text" not in item.visible_text for item in result.items)


def test_analyze_no_ocr_configured_leaves_ocr_text_empty(tiny_jpeg):
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyzer.analyze(tiny_jpeg)

    assert result.ocr_text == []


# ---------------------------------------------------------------------------
# analyze_advanced_group — multi-angle merge (特别需求补充.md §1.2)
# ---------------------------------------------------------------------------


def test_analyze_advanced_group_returns_empty_result_for_no_photos():
    transport = FakeTransport(response_text=json.dumps(VALID_RESPONSE))
    analyzer = QwenVisionAnalyzer(transport, model_version="qwen3-vl-test")

    result = analyze_advanced_group(analyzer, [])

    assert result.items == []


def test_analyze_advanced_group_merges_same_category_across_angles(tmp_path: Path):
    front = tmp_path / "front.jpg"
    front.write_bytes(b"front")
    back = tmp_path / "back.jpg"
    back.write_bytes(b"back")

    front_response = json.loads(json.dumps(VALID_RESPONSE))
    front_response["items"][0]["model"] = {"value": None, "status": "unknown"}
    back_response = json.loads(json.dumps(VALID_RESPONSE))
    back_response["items"][0]["model"] = {"value": "Mate 60", "status": "confirmed"}
    back_response["items"][0]["visible_text"] = ["Mate 60 Pro"]

    import base64

    front_b64 = base64.b64encode(front.read_bytes()).decode("ascii")
    back_b64 = base64.b64encode(back.read_bytes()).decode("ascii")

    class ScriptedTransport:
        def complete(self, *, image_base64: str, system_prompt: str) -> str:
            if image_base64 == front_b64:
                return json.dumps(front_response)
            if image_base64 == back_b64:
                return json.dumps(back_response)
            raise KeyError(image_base64)

    analyzer = QwenVisionAnalyzer(ScriptedTransport(), model_version="qwen3-vl-test")

    result = analyze_advanced_group(analyzer, [front, back])

    assert len(result.items) == 1  # both angles' single item merged into one
    merged = result.items[0]
    assert merged.model_name.value == "Mate 60"  # best (confirmed) status wins over unknown
    assert set(merged.visible_text) == {"HUAWEI", "Mate 60 Pro"}  # union across angles


def test_analyze_advanced_group_reports_parse_error_when_every_angle_fails(tmp_path: Path):
    bad = tmp_path / "bad.jpg"
    bad.write_bytes(b"bad")

    class BrokenTransport:
        def complete(self, *, image_base64: str, system_prompt: str) -> str:
            return "not json"

    analyzer = QwenVisionAnalyzer(BrokenTransport(), model_version="qwen3-vl-test")

    result = analyze_advanced_group(analyzer, [bad])

    assert result.items == []
    assert result.parse_error is not None


def test_analyze_advanced_group_propagates_truncated_from_any_angle(tmp_path: Path):
    clean = tmp_path / "clean.jpg"
    clean.write_bytes(b"clean")
    cut = tmp_path / "cut.jpg"
    cut.write_bytes(b"cut")

    import base64

    clean_b64 = base64.b64encode(clean.read_bytes()).decode("ascii")
    cut_b64 = base64.b64encode(cut.read_bytes()).decode("ascii")

    first_item = json.loads(json.dumps(VALID_RESPONSE))["items"][0]
    truncated_raw = (
        '{"items": [' + json.dumps(first_item) + ', {"category": {"value": "贵重物品", '
        '"status": "confirmed"}, "brand": {"value": "cut off mid'
    )

    class ScriptedTransport:
        def complete(self, *, image_base64: str, system_prompt: str) -> str:
            if image_base64 == clean_b64:
                return json.dumps(VALID_RESPONSE)
            if image_base64 == cut_b64:
                return truncated_raw
            raise KeyError(image_base64)

    analyzer = QwenVisionAnalyzer(ScriptedTransport(), model_version="qwen3-vl-test")

    result = analyze_advanced_group(analyzer, [clean, cut])

    # one angle's response got truncated (but partially recovered) — that
    # must not silently disappear just because another angle was clean.
    assert result.truncated is True

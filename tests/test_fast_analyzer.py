from __future__ import annotations

import json
from pathlib import Path

import pytest

from image_insight.taxonomy import FALLBACK_CATEGORY
from image_insight.vlm.fast_analyzer import FastVisionAnalyzer, analyze_goods_group


class FakeTransport:
    def __init__(self, responses: dict[str, str] | None = None, default: str | None = None):
        self._responses = responses or {}
        self._default = default
        self.calls: list[str] = []

    def complete(self, *, image_base64: str, system_prompt: str) -> str:
        self.calls.append(system_prompt)
        key = image_base64
        if key in self._responses:
            return self._responses[key]
        if self._default is not None:
            return self._default
        raise KeyError(f"no scripted response for {key!r}")


@pytest.fixture
def tiny_jpeg(tmp_path: Path) -> Path:
    path = tmp_path / "tiny.jpg"
    path.write_bytes(b"\xff\xd8\xff\xd9")
    return path


def test_analyze_parses_open_vocabulary_items(tiny_jpeg):
    payload = json.dumps(
        {
            "items": [
                {"name": "螺丝刀", "category": "普通实物及作案工具", "count": 2},
                {"name": "钱包", "category": "贵重物品", "count": 1},
            ]
        }
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert result.counts == {"螺丝刀": 2, "钱包": 1}
    by_name = {item.name: item for item in result.items}
    assert by_name["螺丝刀"].category == "普通实物及作案工具"


def test_analyze_is_not_restricted_to_any_fixed_category_list(tiny_jpeg):
    # The whole point of the rewrite: arbitrary natural-language names must
    # work, not just a hardcoded enum like the old RF-DETR/COCO categories.
    payload = json.dumps(
        {
            "items": [
                {"name": "望远镜", "category": "特殊、大宗及新型财产", "count": 1},
                {"name": "登山杖", "category": "普通实物及作案工具", "count": 2},
            ]
        }
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.counts == {"望远镜": 1, "登山杖": 2}


def test_analyze_snaps_unrecognized_category_to_fallback(tiny_jpeg):
    payload = json.dumps({"items": [{"name": "神秘装置", "category": "外星科技", "count": 1}]})
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].category == FALLBACK_CATEGORY


def test_analyze_strips_markdown_code_fences(tiny_jpeg):
    payload = "```json\n" + json.dumps(
        {"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 1}]}
    ) + "\n```"
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert result.counts == {"手机": 1}


def test_analyze_falls_back_to_empty_on_invalid_json(tiny_jpeg):
    transport = FakeTransport(default="not json")
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items == []
    assert result.parse_error is not None


def test_analyze_falls_back_when_transport_raises(tiny_jpeg):
    class RaisingTransport:
        def complete(self, *, image_base64: str, system_prompt: str) -> str:
            raise RuntimeError("out of memory")

    analyzer = FastVisionAnalyzer(RaisingTransport())

    result = analyzer.analyze(tiny_jpeg)

    assert result.items == []
    assert "out of memory" in result.parse_error


def test_analyze_goods_group_merges_by_exact_name_taking_max_per_name(tmp_path: Path):
    front = tmp_path / "front.jpg"
    front.write_bytes(b"front")
    side = tmp_path / "side.jpg"
    side.write_bytes(b"side")

    import base64

    front_b64 = base64.b64encode(front.read_bytes()).decode("ascii")
    side_b64 = base64.b64encode(side.read_bytes()).decode("ascii")

    transport = FakeTransport(
        responses={
            front_b64: json.dumps(
                {"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 2}]}
            ),
            side_b64: json.dumps(
                {
                    "items": [
                        {"name": "手机", "category": "电子设备及数字载体", "count": 1},
                        {"name": "钱包", "category": "贵重物品", "count": 1},
                    ]
                }
            ),
        }
    )
    analyzer = FastVisionAnalyzer(transport)

    result = analyze_goods_group(analyzer, [front, side])

    assert result.parse_error is None
    by_name = {item.name: item for item in result.items}
    assert by_name["手机"].count == 2  # max(2,1)=2, not 2 photos summed
    assert by_name["钱包"].count == 1


def test_analyze_goods_group_returns_empty_result_for_no_photos():
    transport = FakeTransport()
    analyzer = FastVisionAnalyzer(transport)

    result = analyze_goods_group(analyzer, [])

    assert result.items == []
    assert result.parse_error is None


def test_analyze_goods_group_propagates_truncated_flag(tmp_path: Path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"a")
    # cut off mid-string, past a fully-formed first item
    truncated_payload = (
        '{"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 1, '
        '"confidence": 0.9}, {"name": "钱包", "category": "贵重物品", "appearance'
    )
    transport = FakeTransport(default=truncated_payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyze_goods_group(analyzer, [path])

    assert result.truncated is True
    assert [item.name for item in result.items] == ["手机"]


def test_analyze_goods_group_returns_parse_error_when_every_photo_fails(tmp_path: Path):
    path = tmp_path / "a.jpg"
    path.write_bytes(b"a")
    transport = FakeTransport(default="not json at all")
    analyzer = FastVisionAnalyzer(transport)

    result = analyze_goods_group(analyzer, [path])

    assert result.items == []
    assert result.parse_error is not None


def test_analyze_resolves_subcategory_and_derives_major_category(tiny_jpeg):
    payload = json.dumps(
        {"items": [{"name": "手机", "category": "电子设备及数字载体", "subcategory": "手机", "count": 1}]}
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].subcategory == "手机"
    assert result.items[0].category == "电子设备及数字载体"


def test_analyze_subcategory_wins_over_mismatched_category(tiny_jpeg):
    # subcategory is more specific evidence than category — if they
    # disagree, the major category derived from subcategory wins.
    payload = json.dumps(
        {"items": [{"name": "戒指", "category": "普通实物及作案工具", "subcategory": "戒指", "count": 1}]}
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].category == "贵重物品"


def test_analyze_missing_subcategory_still_normalizes_category(tiny_jpeg):
    payload = json.dumps({"items": [{"name": "螺丝刀", "category": "普通实物及作案工具", "count": 1}]})
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].subcategory is None
    assert result.items[0].category == "普通实物及作案工具"


def test_is_sensitive_category_flags_forced_review_subcategory_not_sibling(tiny_jpeg):
    payload = json.dumps(
        {
            "items": [
                {"name": "老照相机", "category": "贵重物品", "subcategory": "收藏品", "count": 1},
                {"name": "戒指", "category": "贵重物品", "subcategory": "戒指", "count": 1},
            ]
        }
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    by_name = {item.name: item for item in result.items}
    assert by_name["老照相机"].is_sensitive_category is True
    assert by_name["戒指"].is_sensitive_category is False


def test_analyze_flags_pending_review_when_confidence_is_low(tiny_jpeg):
    payload = json.dumps(
        {
            "items": [
                {
                    "name": "神秘物体",
                    "category": "特殊、大宗及新型财产",
                    "subcategory": "药品",
                    "count": 1,
                    "confidence": 0.2,
                }
            ]
        }
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].pending_review is True


def test_analyze_does_not_flag_pending_review_when_confident(tiny_jpeg):
    payload = json.dumps(
        {
            "items": [
                {
                    "name": "手机",
                    "category": "电子设备及数字载体",
                    "subcategory": "手机",
                    "count": 1,
                    "confidence": 0.92,
                }
            ]
        }
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].pending_review is False


def test_analyze_missing_confidence_defaults_to_pending_review(tiny_jpeg):
    # confidence defaults to 0.0 when the model omits the field, which is
    # below any reasonable threshold — an omission is itself a signal.
    payload = json.dumps({"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 1}]})
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.items[0].pending_review is True


def test_analyze_tolerates_malformed_confidence_without_dropping_the_whole_item(tiny_jpeg):
    # A real regression: a strict confidence field meant one bad value from
    # the model (non-numeric, or a 0-100 scale) discarded the ENTIRE
    # photo's results, not just that one field. Must degrade gracefully.
    payload = json.dumps(
        {"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 1, "confidence": "很高"}]}
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert len(result.items) == 1


def test_analyze_clamps_percentage_scale_confidence_instead_of_rejecting(tiny_jpeg):
    payload = json.dumps(
        {"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 1, "confidence": 90}]}
    )
    transport = FakeTransport(default=payload)
    analyzer = FastVisionAnalyzer(transport)

    result = analyzer.analyze(tiny_jpeg)

    assert result.parse_error is None
    assert result.items[0].confidence == 0.9
    assert result.items[0].pending_review is False

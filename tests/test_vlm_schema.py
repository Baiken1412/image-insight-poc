import pytest
from pydantic import ValidationError

from image_insight.vlm.schema import AdvancedAnalysisResult, FieldStatus, ItemAnalysis, ObservedField


def make_item(**overrides):
    defaults = dict(
        category=ObservedField(value="手机", status=FieldStatus.CONFIRMED),
        brand=ObservedField(value="华为", status=FieldStatus.CONFIRMED),
        model=ObservedField(value=None, status=FieldStatus.UNKNOWN),
        color=ObservedField(value="黑色", status=FieldStatus.CONFIRMED),
        count=1,
        condition=ObservedField(value="轻微磨损", status=FieldStatus.SUSPECTED),
    )
    defaults.update(overrides)
    return ItemAnalysis(**defaults)


def test_unknown_status_with_a_value_is_rejected():
    with pytest.raises(ValidationError):
        ObservedField(value="iPhone 15", status=FieldStatus.UNKNOWN)


def test_confirmed_status_without_a_value_is_rejected():
    with pytest.raises(ValidationError):
        ObservedField(value=None, status=FieldStatus.CONFIRMED)


def test_suspected_status_without_a_value_is_rejected():
    with pytest.raises(ValidationError):
        ObservedField(value="", status=FieldStatus.SUSPECTED)


def test_unknown_field_displays_as_cannot_confirm_not_a_fabricated_guess():
    field = ObservedField(value=None, status=FieldStatus.UNKNOWN)
    assert field.display() == "无法确定"


def test_suspected_field_display_flags_for_manual_review():
    field = ObservedField(value="小米", status=FieldStatus.SUSPECTED)
    assert "疑似" in field.display()
    assert field.needs_manual_review is True


def test_confirmed_field_does_not_need_manual_review():
    field = ObservedField(value="黑色", status=FieldStatus.CONFIRMED)
    assert field.needs_manual_review is False


def test_item_needs_manual_review_if_any_field_is_not_confirmed():
    item = make_item()  # model_name is UNKNOWN
    assert item.needs_manual_review is True


def test_item_does_not_need_manual_review_when_all_fields_confirmed():
    item = make_item(
        model=ObservedField(value="Mate 60", status=FieldStatus.CONFIRMED),
        condition=ObservedField(value="轻微磨损", status=FieldStatus.CONFIRMED),
    )
    assert item.needs_manual_review is False


def test_result_with_parse_error_forces_manual_review_even_with_no_items():
    result = AdvancedAnalysisResult(items=[], model_version="qwen3-vl-test", parse_error="invalid JSON")
    assert result.needs_manual_review is True


def test_result_needs_manual_review_propagates_from_items():
    result = AdvancedAnalysisResult(items=[make_item()], model_version="qwen3-vl-test")
    assert result.needs_manual_review is True

    all_confirmed = AdvancedAnalysisResult(
        items=[
            make_item(
                model=ObservedField(value="Mate 60", status=FieldStatus.CONFIRMED),
                condition=ObservedField(value="轻微磨损", status=FieldStatus.CONFIRMED),
            )
        ],
        model_version="qwen3-vl-test",
    )
    assert all_confirmed.needs_manual_review is False


def test_item_default_subcategory_does_not_force_manual_review_on_its_own():
    # subcategory defaults to unknown when a caller doesn't supply one
    # (e.g. direct construction) — that alone must not force review; only
    # category_pending_review (set by the analyzer) and the original five
    # fields participate in the aggregate.
    item = make_item(
        model=ObservedField(value="Mate 60", status=FieldStatus.CONFIRMED),
        condition=ObservedField(value="轻微磨损", status=FieldStatus.CONFIRMED),
    )
    assert item.needs_manual_review is False


def test_item_category_pending_review_forces_manual_review():
    item = make_item(
        model=ObservedField(value="Mate 60", status=FieldStatus.CONFIRMED),
        condition=ObservedField(value="轻微磨损", status=FieldStatus.CONFIRMED),
        category_pending_review=True,
    )
    assert item.needs_manual_review is True


def test_is_forced_manual_review_uses_subcategory_when_resolved():
    item = make_item(
        category=ObservedField(value="贵重物品", status=FieldStatus.CONFIRMED),
        subcategory=ObservedField(value="收藏品", status=FieldStatus.CONFIRMED),
    )
    assert item.is_forced_manual_review is True

    sibling = make_item(
        category=ObservedField(value="贵重物品", status=FieldStatus.CONFIRMED),
        subcategory=ObservedField(value="戒指", status=FieldStatus.CONFIRMED),
    )
    assert sibling.is_forced_manual_review is False


def test_is_forced_manual_review_falls_back_to_major_category():
    item = make_item(category=ObservedField(value="毒品及违禁品", status=FieldStatus.CONFIRMED))
    assert item.is_forced_manual_review is True


def test_truncated_result_forces_manual_review_even_with_confirmed_items():
    # truncated=True means the model's generation was cut off before the
    # JSON closed — the photo may have had more items than survived
    # salvage. This must force review regardless of how clean the items
    # that DID survive look.
    all_confirmed_item = make_item(
        model=ObservedField(value="Mate 60", status=FieldStatus.CONFIRMED),
        condition=ObservedField(value="轻微磨损", status=FieldStatus.CONFIRMED),
    )
    result = AdvancedAnalysisResult(
        items=[all_confirmed_item], model_version="qwen3-vl-test", truncated=True
    )
    assert result.needs_manual_review is True


def test_non_truncated_result_with_all_confirmed_items_does_not_need_review():
    all_confirmed_item = make_item(
        model=ObservedField(value="Mate 60", status=FieldStatus.CONFIRMED),
        condition=ObservedField(value="轻微磨损", status=FieldStatus.CONFIRMED),
    )
    result = AdvancedAnalysisResult(
        items=[all_confirmed_item], model_version="qwen3-vl-test", truncated=False
    )
    assert result.needs_manual_review is False

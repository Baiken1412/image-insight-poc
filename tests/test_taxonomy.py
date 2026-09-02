from image_insight.taxonomy import (
    FALLBACK_CATEGORY,
    MAJOR_CATEGORIES,
    SUBCATEGORIES,
    SUBCATEGORY_TO_MAJOR,
    is_forced_manual_review_subcategory,
    is_sensitive,
    major_of_subcategory,
    needs_forced_review,
    normalize_category,
    normalize_subcategory,
)


def test_normalize_category_passes_through_exact_official_match():
    assert normalize_category("电子设备及数字载体") == "电子设备及数字载体"


def test_normalize_category_maps_known_alias():
    assert normalize_category("电子产品") == "电子设备及数字载体"
    assert normalize_category("首饰") == "贵重物品"


def test_normalize_category_handles_substring_variants():
    assert normalize_category("这是电子设备及数字载体类") == "电子设备及数字载体"


def test_normalize_category_falls_back_for_unrecognized_input():
    assert normalize_category("外星科技") == FALLBACK_CATEGORY


def test_normalize_category_falls_back_for_none_or_empty():
    assert normalize_category(None) == FALLBACK_CATEGORY
    assert normalize_category("") == FALLBACK_CATEGORY
    assert normalize_category("   ") == FALLBACK_CATEGORY


def test_all_12_major_categories_present():
    assert len(MAJOR_CATEGORIES) == 12


def test_is_sensitive_flags_high_risk_categories():
    assert is_sensitive("毒品及违禁品") is True
    assert is_sensitive("枪支、弹药及管制器具") is True
    assert is_sensitive("危险及特殊保存物品") is True


def test_is_sensitive_false_for_ordinary_categories():
    assert is_sensitive("电子设备及数字载体") is False
    assert is_sensitive(FALLBACK_CATEGORY) is False


# ---------------------------------------------------------------------------
# 细类 (subcategory) — 特别需求补充.md §3/§2
# ---------------------------------------------------------------------------


def test_every_subcategory_maps_back_to_a_declared_major_category():
    for major, subs in SUBCATEGORIES.items():
        assert major in MAJOR_CATEGORIES
        for sub in subs:
            assert SUBCATEGORY_TO_MAJOR[sub] == major


def test_normalize_subcategory_passes_through_exact_match():
    assert normalize_subcategory("戒指") == "戒指"


def test_normalize_subcategory_handles_substring_variants():
    assert normalize_subcategory("一枚戒指") == "戒指"


def test_normalize_subcategory_returns_none_for_unrecognized_input():
    assert normalize_subcategory("外星科技") is None


def test_normalize_subcategory_returns_none_for_none_or_empty():
    assert normalize_subcategory(None) is None
    assert normalize_subcategory("") is None


def test_major_of_subcategory_resolves_correctly():
    assert major_of_subcategory("手机") == "电子设备及数字载体"
    assert major_of_subcategory("戒指") == "贵重物品"


def test_major_of_subcategory_returns_none_for_unknown_subcategory():
    assert major_of_subcategory("外星科技") is None


def test_forced_review_subcategories_are_flagged_but_siblings_are_not():
    # 特别需求补充.md §2: "收藏品" forces review, but "戒指" (same 大类
    # 贵重物品) does not — the forced list is defined at 细类 precision.
    assert is_forced_manual_review_subcategory("收藏品") is True
    assert is_forced_manual_review_subcategory("戒指") is False
    assert is_forced_manual_review_subcategory(None) is False


def test_needs_forced_review_uses_subcategory_precision_when_available():
    assert needs_forced_review("贵重物品", "收藏品") is True
    assert needs_forced_review("贵重物品", "戒指") is False


def test_needs_forced_review_falls_back_to_major_category_when_subcategory_unresolved():
    # No 细类 available -> fall back to the coarser high-risk 大类 check
    # rather than silently passing.
    assert needs_forced_review("毒品及违禁品", None) is True
    assert needs_forced_review("电子设备及数字载体", None) is False

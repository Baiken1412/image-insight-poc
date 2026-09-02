from __future__ import annotations

from image_insight.vlm.json_repair import recover_array_objects


def test_recover_array_objects_returns_all_items_from_valid_json():
    text = '{"items": [{"a": 1}, {"a": 2}]}'
    assert recover_array_objects(text, "items") == [{"a": 1}, {"a": 2}]


def test_recover_array_objects_salvages_complete_items_before_truncation():
    # exactly the real bug shape: generation cut off mid-string inside the
    # second item's field, no closing quote/brace/bracket at all.
    text = (
        '{"items": [{"name": "手机", "category": "电子设备及数字载体", "count": 1}, '
        '{"name": "钱包", "category": "贵重物品", "appearance_notes": "边缘有大片深红褐'
    )
    assert recover_array_objects(text, "items") == [{"name": "手机", "category": "电子设备及数字载体", "count": 1}]


def test_recover_array_objects_handles_truncation_right_after_a_comma():
    text = '{"items": [{"a": 1}, {"a": 2},'
    assert recover_array_objects(text, "items") == [{"a": 1}, {"a": 2}]


def test_recover_array_objects_handles_braces_inside_string_values():
    # a brace inside a quoted string must not confuse the depth counter.
    text = '{"items": [{"note": "contains a { brace } inside"}, {"note": "cut off'
    assert recover_array_objects(text, "items") == [{"note": "contains a { brace } inside"}]


def test_recover_array_objects_handles_escaped_quotes_inside_strings():
    text = r'{"items": [{"note": "she said \"hi\""}, {"note": "cut off'
    assert recover_array_objects(text, "items") == [{"note": 'she said "hi"'}]


def test_recover_array_objects_returns_empty_when_key_is_missing():
    assert recover_array_objects('{"other": []}', "items") == []


def test_recover_array_objects_returns_empty_when_first_item_is_itself_truncated():
    text = '{"items": [{"name": "手机", "appearance_notes": "cut off mid'
    assert recover_array_objects(text, "items") == []


def test_recover_array_objects_returns_empty_for_empty_array():
    assert recover_array_objects('{"items": []}', "items") == []


def test_recover_array_objects_stops_at_first_malformed_object_not_just_first_incomplete():
    # a syntactically complete-looking but invalid object (e.g. trailing
    # comma making it unparseable) should stop recovery there too, not be
    # silently skipped in favor of scanning past it.
    text = '{"items": [{"a": 1}, {"a": 2,}, {"a": 3}]}'
    assert recover_array_objects(text, "items") == [{"a": 1}]

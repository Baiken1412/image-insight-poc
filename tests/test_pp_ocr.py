from image_insight.ocr.pp_ocr import OcrTextBox, extract_high_confidence_text


def test_extract_high_confidence_text_keeps_confident_reads():
    boxes = [OcrTextBox(text="IMEI 353065109623245", confidence=0.98)]
    assert extract_high_confidence_text(boxes) == ["IMEI 353065109623245"]


def test_extract_high_confidence_text_drops_low_confidence_reads():
    boxes = [
        OcrTextBox(text="HUAWEI", confidence=0.95),
        OcrTextBox(text="???garbled???", confidence=0.2),
    ]
    assert extract_high_confidence_text(boxes) == ["HUAWEI"]


def test_extract_high_confidence_text_respects_custom_threshold():
    boxes = [OcrTextBox(text="maybe", confidence=0.6)]
    assert extract_high_confidence_text(boxes, min_confidence=0.7) == []
    assert extract_high_confidence_text(boxes, min_confidence=0.5) == ["maybe"]


def test_extract_high_confidence_text_returns_empty_for_no_boxes():
    assert extract_high_confidence_text([]) == []

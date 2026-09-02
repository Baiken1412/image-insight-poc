"""Recover whatever complete items survive a truncated model response.

Qwen3-VL's generation is capped by max_new_tokens (see LocalQwenTransport).
Advanced mode's per-item schema grew (subcategory, category_confidence,
visible_text, appearance_notes, ...), so a photo with several items — or
one item with a long appearance_notes — can hit that cap before the JSON
closes, producing output like `..."appearance_notes": "边缘有大片深红褐` with
no closing quote/brace. json.loads() on that raises JSONDecodeError
("Unterminated string...") and, without this module, the whole photo's
result — including any items that DID finish generating cleanly before the
cutoff — was discarded.

recover_array_objects extracts however many complete, individually
re-parseable objects exist in a `"<key>": [ ... ]` array, stopping at the
first object that doesn't parse (the truncated one, or anything after it).
This never fabricates content: every recovered object is exactly what the
model wrote, individually re-validated via json.loads. The caller is
responsible for flagging to the user that recovery happened (truncated=True
downstream) — dropping an incomplete tail item is not the same as knowing
the photo only had that many items.
"""
from __future__ import annotations

import json


def recover_array_objects(text: str, key: str) -> list[dict]:
    """Best-effort recovery of complete JSON objects from `"<key>": [...]`.

    Returns [] if the key/array can't even be located, or if the very first
    object is itself incomplete — there is nothing genuine to recover in
    that case, and the caller should fall back to its normal parse_error
    path rather than pretending an empty recovery means "zero items".
    """
    marker = f'"{key}"'
    key_idx = text.find(marker)
    if key_idx == -1:
        return []
    bracket_idx = text.find("[", key_idx)
    if bracket_idx == -1:
        return []

    objects: list[dict] = []
    i = bracket_idx + 1
    n = len(text)
    while i < n:
        while i < n and text[i] in " \t\r\n,":
            i += 1
        if i >= n or text[i] == "]":
            break
        if text[i] != "{":
            break  # malformed structure — stop rather than guess

        start = i
        depth = 0
        in_string = False
        escape_next = False
        closed = False
        while i < n:
            char = text[i]
            if in_string:
                if escape_next:
                    escape_next = False
                elif char == "\\":
                    escape_next = True
                elif char == '"':
                    in_string = False
            else:
                if char == '"':
                    in_string = True
                elif char == "{":
                    depth += 1
                elif char == "}":
                    depth -= 1
                    if depth == 0:
                        i += 1
                        closed = True
                        break
            i += 1

        if not closed:
            break  # ran off the end mid-object — this is the truncation point

        candidate = text[start:i]
        try:
            obj = json.loads(candidate)
        except json.JSONDecodeError:
            break
        objects.append(obj)

    return objects

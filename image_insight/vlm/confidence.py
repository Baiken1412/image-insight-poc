"""Self-reported classification confidence (特别需求补充.md §1.3/§1.4).

Qwen3-VL is a generative model, not a classifier — it has no native softmax
confidence over the taxonomy. Three ways to approximate one were considered:

1. Self-consistency voting: sample the model N times and use vote share as
   a confidence proxy. Statistically the most honest, but multiplies
   inference cost by N for every photo.
2. Self-reported Top-3 candidates + confidence, in the same JSON call the
   model already makes. First implementation — but asking for a ranked
   list of alternatives (not just a single number) meaningfully increases
   how much the model has to generate for every item, and generation is
   sequential/token-by-token, so this measurably slowed fast mode down
   (observed: tens of seconds for a single simple item). Also gave the UI
   a "候选细类" list nobody asked to see and that added little value over
   the top pick shown right above it.
3. Self-reported single confidence for the model's one chosen 细类. What
   this module implements now, per product feedback after (2). Cheaper to
   generate (one number, not a ranked list) and simpler to show in the UI
   (one confidence tag next to the category, not a chip list).

The tradeoff versus 特别需求补充.md §1.4's literal wording: that section's
second trigger ("前两名置信度差距过小") needs a second candidate to compare
against, which this single-confidence approach no longer collects. Only the
first trigger (confidence below a threshold) is checked now. Re-adding (2)
is a straightforward revert of this module + its two callers if the margin
check is needed again and the latency cost is acceptable.
"""
from __future__ import annotations


def coerce_confidence(value: object) -> float:
    """Tolerantly parse a self-reported confidence into [0, 1].

    Never raises: Qwen3-VL-2B sometimes writes a non-numeric value ("很高"),
    a 0-100 scale instead of 0-1, or omits the field entirely. A malformed
    confidence must never blow up the whole item over the least load-bearing
    field in it — it degrades to a low (not fabricated-high) confidence
    instead, which still pushes toward "待核实" rather than silently passing.
    """
    try:
        parsed = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return 0.0
    if parsed > 1.0:
        # Model wrote a 0-100 scale by mistake (e.g. 90 instead of 0.9).
        parsed /= 100.0
    return max(0.0, min(1.0, parsed))


def needs_pending_review(confidence: float, *, low_confidence_threshold: float) -> bool:
    """特别需求补充.md §1.4's "待核实" trigger (low-confidence half only —
    see module docstring for why the close-margin half was dropped).
    """
    return confidence < low_confidence_threshold


def confidence_label(confidence: float) -> str:
    """Qualitative tier for display — 特别需求补充.md §1.3 explicitly forbids
    showing the raw score as a percentage ("不对外直接展示为百分比概率"), so
    the UI shows this tier instead of the number itself.
    """
    if confidence >= 0.7:
        return "高"
    if confidence >= 0.35:
        return "中"
    return "低"

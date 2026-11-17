"""TDD Cycle 2 tests for Allegro hand: warm pad tone, both-hand fill, and roll receipts."""

from __future__ import annotations

import pytest

from src.shared.python.model_appearance.hand_backends.allegro_settings import (
    visual_settings,
)

pytestmark = pytest.mark.unit


def test_tip_pad_rgba_is_warm_skin_harmonized() -> None:
    """Fingertip pads should blend smoothly with warm skin tone rather than looking grey."""
    cfg = visual_settings()
    r, g, b, a = cfg.tip_pad_rgba
    assert a == pytest.approx(1.0)
    assert r >= g >= b
    assert (r - b) >= 0.14


def test_rendering_has_distinct_both_frame_fill_in_contract_band() -> None:
    """Both-hand views use configurable both_frame_fill inside [0.72, 0.78]."""
    cfg = visual_settings()
    assert cfg.render_frame_fill == pytest.approx(0.70)
    assert 0.72 <= cfg.both_frame_fill <= 0.78


def test_evidence_receipt_records_per_roll_pad_gaps() -> None:
    """Receipt generator must include per-roll pad_gaps_by_roll_m for -20, 0, +20 deg."""
    from docs.development.full_body_models.evidence.hand_backend_allegro import (
        render_allegro_hand_evidence as ev,
    )

    receipt = ev.build_evidence_receipt(renders=[], include_roll_audit=True)
    by_roll = receipt["pad_gaps_by_roll_m"]
    assert set(by_roll.keys()) == {"-20", "0", "+20"}
    for roll_key in ("-20", "0", "+20"):
        assert set(by_roll[roll_key].keys()) == {"lead", "trail"}
        for side in ("lead", "trail"):
            for digit in ("index", "middle", "ring", "thumb"):
                gap = by_roll[roll_key][side][digit]
                assert abs(gap - by_roll["0"][side][digit]) < 1e-6

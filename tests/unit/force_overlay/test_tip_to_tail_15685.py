"""Tip-to-tail ZTCF/ZVCF force arrows (#15685, K5).

The drift (ZTCF) arrow starts at the application point, the control (ZVCF)
arrow starts at the drift tip, and the total runs from the point to the
control tip, so the picture shows drift + control = total.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.tip_to_tail import (
    TipToTailStyle,
    tip_to_tail_arrows,
)

pytestmark = pytest.mark.unit

STYLE = ForceGlyphStyle(force_scale_m_per_n=0.001, max_length_m=5.0)


def test_the_control_arrow_starts_at_the_drift_tip_and_the_total_closes() -> None:
    point = (0.1, 0.2, 1.0)
    drift, control = (300.0, 0.0, 0.0), (0.0, 400.0, 0.0)
    arrows = tip_to_tail_arrows(point, drift, control, STYLE)
    by = {a.label: a for a in arrows}
    style = TipToTailStyle()
    d, c, t = by[style.drift_label], by[style.control_label], by[style.total_label]
    assert d.tail_m == point
    np.testing.assert_allclose(c.tail_m, d.tip_m)
    np.testing.assert_allclose(t.tail_m, point)
    np.testing.assert_allclose(t.tip_m, c.tip_m, atol=1e-12)
    np.testing.assert_allclose(np.subtract(d.tip_m, point), (0.3, 0.0, 0.0))
    assert t.magnitude == pytest.approx(500.0)
    assert t.shaft_radius_m < d.shaft_radius_m


def test_a_tiny_part_is_skipped_but_the_total_is_kept() -> None:
    arrows = tip_to_tail_arrows((0, 0, 0), (1e-6, 0, 0), (200.0, 0, 0), STYLE)
    labels = [a.label for a in arrows]
    style = TipToTailStyle()
    assert style.drift_label not in labels
    assert style.control_label in labels and style.total_label in labels


def test_one_shared_scale_so_lengths_add() -> None:
    clamped = ForceGlyphStyle(force_scale_m_per_n=0.001, max_length_m=0.2)
    with pytest.raises(ValueError, match="max_length_m"):
        tip_to_tail_arrows((0, 0, 0), (300.0, 0, 0), (0, 0, 0), clamped)


def test_inputs_are_checked() -> None:
    with pytest.raises(ValueError, match="finite"):
        tip_to_tail_arrows((0, 0, 0), (np.nan, 0, 0), (0, 0, 0), STYLE)
    with pytest.raises(ValueError, match="length 3"):
        tip_to_tail_arrows((0, 0), (1, 0, 0), (0, 0, 0), STYLE)
    with pytest.raises(ValueError, match="colour"):
        TipToTailStyle(drift_hex="blue")

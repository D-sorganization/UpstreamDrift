"""ZTCF-only, ZVCF-only and total arrows, force and torque (#15150 Track Z, Z3).

Each render shows one view of the drift/control split: tip to tail, the ZTCF
(drift: gravity, momentum, passive) alone, the ZVCF (control: the applied
torques alone) alone, or the total. A torque is drawn as an axial vector on
the torque scale, in N m. The colour key lists exactly what is drawn.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.force_overlay.contracts import WrenchKind
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle
from src.shared.python.force_overlay.palette import hex_to_rgba
from src.shared.python.force_overlay.tip_to_tail import (
    PART_VIEWS,
    QUANTITIES,
    TipToTailStyle,
    colour_key,
    part_arrows,
    tip_to_tail_arrows,
)

pytestmark = pytest.mark.unit

STYLE = ForceGlyphStyle(
    force_scale_m_per_n=0.001, torque_scale_m_per_nm=0.01, max_length_m=5.0
)
POINT, DRIFT, CONTROL = (0.1, 0.2, 1.0), (300.0, 0.0, 0.0), (0.0, 400.0, 0.0)


def test_tip_to_tail_view_is_the_existing_picture() -> None:
    assert part_arrows(POINT, DRIFT, CONTROL, STYLE, "tip_to_tail") == (
        tip_to_tail_arrows(POINT, DRIFT, CONTROL, STYLE)
    )


@pytest.mark.parametrize(
    ("view", "vector", "attr"),
    [
        ("ztcf", DRIFT, "drift"),
        ("zvcf", CONTROL, "control"),
        ("total", (300.0, 400.0, 0.0), "total"),
    ],
)
def test_a_single_part_view_draws_one_arrow_from_the_point(
    view: str, vector: tuple[float, float, float], attr: str
) -> None:
    look = TipToTailStyle()
    (arrow,) = part_arrows(POINT, DRIFT, CONTROL, STYLE, view)
    assert arrow.tail_m == POINT
    np.testing.assert_allclose(
        np.subtract(arrow.tip_m, POINT), 0.001 * np.array(vector)
    )
    assert arrow.label == getattr(look, f"{attr}_label")
    assert arrow.rgba == hex_to_rgba(getattr(look, f"{attr}_hex"))
    assert arrow.units == "N"


def test_torque_uses_the_torque_scale_and_units() -> None:
    (arrow,) = part_arrows(
        POINT, (0.0, 0.0, 20.0), (0.0, 0.0, 5.0), STYLE, "total", quantity="torque"
    )
    np.testing.assert_allclose(np.subtract(arrow.tip_m, POINT), (0.0, 0.0, 0.25))
    assert arrow.units == "N m"
    assert arrow.kind == WrenchKind.GRIP


def test_the_grip_kind_gain_lengthens_torque_arrows_too() -> None:
    gained = ForceGlyphStyle(
        force_scale_m_per_n=0.001,
        torque_scale_m_per_nm=0.01,
        max_length_m=5.0,
        kind_scale={WrenchKind.GRIP: 4.0},
    )
    (arrow,) = part_arrows(POINT, (1.0, 0, 0), (1.0, 0, 0), gained, "ztcf", "torque")
    np.testing.assert_allclose(np.subtract(arrow.tip_m, POINT), (0.04, 0.0, 0.0))


def test_a_label_prefix_names_the_hand() -> None:
    arrows = part_arrows(POINT, DRIFT, CONTROL, STYLE, "tip_to_tail", prefix="lead")
    assert all(a.label.startswith("lead ") for a in arrows)


def test_unknown_view_or_quantity_is_refused() -> None:
    with pytest.raises(ValueError, match="view"):
        part_arrows(POINT, DRIFT, CONTROL, STYLE, "both")
    with pytest.raises(ValueError, match="quantity"):
        part_arrows(POINT, DRIFT, CONTROL, STYLE, "ztcf", quantity="power")


def test_vocabularies_are_settings_not_literals() -> None:
    assert PART_VIEWS == ("tip_to_tail", "ztcf", "zvcf", "total")
    assert QUANTITIES == ("force", "torque")


@pytest.mark.parametrize("view", PART_VIEWS)
def test_colour_key_lists_what_the_view_draws(view: str) -> None:
    look = TipToTailStyle()
    drawn = {a.label: a.rgba for a in part_arrows(POINT, DRIFT, CONTROL, STYLE, view)}
    key = colour_key(view, look)
    assert len(key) == len(drawn)
    for (text, rgb), (label, rgba) in zip(key, drawn.items(), strict=True):
        assert text
        assert rgb == tuple(round(255 * c) for c in rgba[:3]), label


def test_colour_key_explains_gravity_lives_in_the_ztcf() -> None:
    ztcf, zvcf = (colour_key(v, TipToTailStyle())[0][0] for v in ("ztcf", "zvcf"))
    assert "gravity" in ztcf.lower()
    assert "gravity" not in zvcf.lower()

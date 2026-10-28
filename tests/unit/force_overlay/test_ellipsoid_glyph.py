"""EllipsoidGlyph contract, palette and serialization (#14003)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.force_overlay.glyphs import (
    EllipsoidGlyph,
    GlyphSet,
    LegendSpec,
)
from src.shared.python.force_overlay.palette import (
    HAND_ELLIPSOID_PALETTE,
    hand_ellipsoid_rgba,
)

RGBA = (0.2, 0.4, 0.6, 0.5)


def _glyph(**over):
    kw = {
        "centre": (0.1, 0.2, 0.3),
        "axes": np.eye(3),
        "radii": (0.03, 0.02, 0.01),
        "rgba": RGBA,
        "label": "mobility:lead_hand",
    }
    kw.update(over)
    return EllipsoidGlyph(**kw)


def test_valid_glyph_normalises_to_tuples() -> None:
    g = _glyph()
    assert g.centre == (0.1, 0.2, 0.3)
    assert g.axes[1] == (0.0, 1.0, 0.0)
    assert g.radii == (0.03, 0.02, 0.01)


def test_non_orthonormal_axes_raise() -> None:
    bad = np.eye(3)
    bad[:, 0] = (1.0, 0.1, 0.0)
    with pytest.raises(ValueError, match="orthonormal"):
        _glyph(axes=bad)


def test_left_handed_basis_is_flipped_not_rejected() -> None:
    left = np.diag([1.0, 1.0, -1.0])
    g = _glyph(axes=left)
    assert np.linalg.det(np.array(g.axes)) == pytest.approx(1.0)


@pytest.mark.parametrize("radii", [(0.0, 1, 1), (-1, 1, 1), (math.nan, 1, 1)])
def test_bad_radii_raise(radii) -> None:
    with pytest.raises(ValueError, match="radii"):
        _glyph(radii=radii)


def test_bad_rgba_or_centre_raise() -> None:
    with pytest.raises(ValueError, match="rgba"):
        _glyph(rgba=(0.0, 0.0, 0.0, 1.5))
    with pytest.raises(ValueError, match="centre"):
        _glyph(centre=(0.0, math.inf, 0.0))


def test_clamped_radii() -> None:
    g = _glyph(radii=(1e-5, 0.02, 5.0)).with_clamped_radii(0.005, 0.1)
    assert g.radii == (0.005, 0.02, 0.1)
    with pytest.raises(ValueError, match="min_radius"):
        g.with_clamped_radii(0.2, 0.1)


def test_hand_palette_distinct_and_semi_transparent() -> None:
    lead = hand_ellipsoid_rgba("lead")
    trail = hand_ellipsoid_rgba("trail")
    assert lead != trail
    assert 0.0 < lead[3] < 1.0 and 0.0 < trail[3] < 1.0
    assert set(HAND_ELLIPSOID_PALETTE) == {"lead", "trail"}
    with pytest.raises(ValueError, match="hand"):
        hand_ellipsoid_rgba("left")


def test_glyphset_roundtrip_and_default_unchanged() -> None:
    legend = LegendSpec()
    base = GlyphSet(time_s=0.0, arrows=(), torque_arcs=(), legend=legend)
    assert base.ellipsoids == ()
    assert "ellipsoids" not in base.to_dict()
    gs = GlyphSet(
        time_s=0.0, arrows=(), torque_arcs=(), legend=legend, ellipsoids=(_glyph(),)
    )
    back = GlyphSet.from_dict(gs.to_dict())
    assert back.ellipsoids == gs.ellipsoids

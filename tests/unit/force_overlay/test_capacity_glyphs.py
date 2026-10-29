"""Tests for capacity wireframe ellipsoid glyphs (#14045, CAP-5b)."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.force_overlay.capacity_glyphs import (
    CapacityWireGlyph,
    capacity_wire_segments,
)

pytestmark = pytest.mark.unit

_IDENTITY = (
    (1.0, 0.0, 0.0),
    (0.0, 1.0, 0.0),
    (0.0, 0.0, 1.0),
)


def _glyph(**kw) -> CapacityWireGlyph:
    base = {
        "centre": (0.0, 0.0, 0.0),
        "axes": _IDENTITY,
        "radii": (0.1, 0.05, 0.03),
        "rgba": (0.0, 0.45, 0.7, 0.95),
    }
    base.update(kw)
    return CapacityWireGlyph(**base)


def test_wire_segments_include_axes_rim_and_ticks():
    segments = capacity_wire_segments(_glyph(), rim_samples=16)
    assert len(segments) == 3 + 3 * 2 + 16  # axes + ticks + rim edges
    starts = np.array([s.start for s in segments])
    ends = np.array([s.end for s in segments])
    assert np.isfinite(starts).all() and np.isfinite(ends).all()


def test_principal_axis_endpoints_match_radii():
    segments = capacity_wire_segments(_glyph(), rim_samples=12, tick_length_m=0.01)
    axis_segs = segments[:3]
    radii = np.array([0.1, 0.05, 0.03])
    for i, seg in enumerate(axis_segs):
        start = np.asarray(seg.start)
        end = np.asarray(seg.end)
        assert np.allclose(start, -radii[i] * np.eye(3)[i], atol=1e-9)
        assert np.allclose(end, radii[i] * np.eye(3)[i], atol=1e-9)


def test_rim_is_closed_in_equatorial_plane():
    glyph = _glyph(radii=(0.08, 0.04, 0.02))
    segments = capacity_wire_segments(glyph, rim_samples=20)
    rim = segments[-20:]
    first = np.asarray(rim[0].start)
    last = np.asarray(rim[-1].end)
    assert np.allclose(first, last, atol=1e-9)
    for seg in rim:
        mid = 0.5 * (np.asarray(seg.start) + np.asarray(seg.end))
        assert abs(mid[2]) < 1e-9


def test_rejects_bad_axes_and_rim_samples():
    with pytest.raises(ValueError, match="orthonormal"):
        _glyph(axes=((1.0, 0.0, 0.0), (1.0, 0.0, 0.0), (0.0, 0.0, 1.0)))
    with pytest.raises(ValueError, match="rim_samples"):
        capacity_wire_segments(_glyph(), rim_samples=4)


def test_dashed_flag_propagates():
    dashed = capacity_wire_segments(_glyph(dashed=True), rim_samples=12)
    assert all(seg.dashed for seg in dashed)

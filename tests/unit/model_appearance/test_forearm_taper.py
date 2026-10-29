"""Forearm taper: a slimmer wrist end on the shared appearance (#13993).

The visual forearm was a uniform 72 mm capsule, as wide as the hand. The
shared library now tapers it toward the wrist (distal radius 0.8x the elbow
radius). Appearance only: no inertial property may change.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = pytest.mark.unit

SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
)
PRE_TAPER_FOREARM_RADIUS_M = 0.036


def _radial_extent(spec: dict, lo: float, hi: float) -> list[float]:
    """Max skin radius about the axis in the [lo, hi] length band, per forearm."""
    skeleton = derive_visual_skeleton(spec)
    meshes = body_meshes(spec, document_from_dict({"schema_version": "appearance-v1"}))
    out = []
    for cap in skeleton.capsules:
        if library.classify_body(cap.body) != "forearm":
            continue
        p0, p1 = np.asarray(cap.start_m), np.asarray(cap.end_m)
        axis = (p1 - p0) / np.linalg.norm(p1 - p0)
        length = float(np.linalg.norm(p1 - p0))
        for m in meshes:
            if m.body != cap.body or not m.label.startswith("skin"):
                continue
            rel = m.mesh_in_body_frame.vertices - p0
            s = rel @ axis / length
            radial = np.linalg.norm(rel - np.outer(rel @ axis, axis), axis=1)
            band = (s >= lo) & (s <= hi)
            assert band.any()
            out.append(float(radial[band].max()))
    assert out, "no forearm skin meshes"
    return out


def test_distal_forearm_is_at_most_80_percent_of_the_old_radius() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    for r in _radial_extent(spec, 0.85, 0.95):
        assert r <= 0.8 * PRE_TAPER_FOREARM_RADIUS_M + 1e-9, r


def test_distal_forearm_is_narrower_than_the_hand_and_the_proximal_end() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    distal = _radial_extent(spec, 0.85, 0.95)
    proximal = _radial_extent(spec, 0.05, 0.15)
    assert max(distal) < library.PART_RADIUS_M["hand"]
    for d, p in zip(distal, proximal, strict=True):
        assert p >= d


def test_taper_scale_is_validated_and_defaults_to_none() -> None:
    assert library.distal_scale("forearm") == pytest.approx(0.8)
    assert library.distal_scale("thigh") == 1.0
    for bad in (0.0, -0.5, 1.2, float("nan")):
        with pytest.raises(ValueError, match="taper"):
            library.validate_distal_scale(bad)

"""The forearm's wrist end shaped from the skinned hand's own wrist (#15264).

The forearm ended round and 29 mm in radius whatever hand was drawn; the
skinned hand's wrist is about 25 mm by 18 mm, so the forearm overlapped it as
a thicker stub. With the skinned hands drawn, the forearm ends in the hand's
own wrist section, a little larger, turned the way the hand is turned.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import (
    document_from_dict,
    geometry,
    library,
)
from src.shared.python.model_appearance import skinned_club_hands as sch
from src.shared.python.model_appearance import skinned_wrist_sections as sws
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.motion_matching.execution.skinned_hand_wrists import (
    apply_skinned_hand_wrists,
)
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = [pytest.mark.unit, pytest.mark.slow, pytest.mark.timeout(600)]
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)


@pytest.fixture(scope="module")
def base() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def skinned(base: dict[str, Any]) -> dict[str, Any]:
    return apply_skinned_hand_wrists(base)


def _wrist_rings(spec: dict[str, Any]) -> dict[str, tuple[np.ndarray, ...]]:
    """Per wrist-carrying forearm body: its skin ring at the wrist about the
    ring's own centre, the body's long axis, and that centre off the axis."""
    club = ca.assembly_from_spec(spec)
    doc = document_from_dict({"schema_version": "appearance-v1"})
    meshes = {m.body: m for m in body_meshes(spec, doc, club) if m.label == "skin0"}
    out = {}
    for cap in derive_visual_skeleton(spec).capsules:
        carries_wrist = any(
            j["parent"] == cap.body and library.classify_body(j["child"]) != "forearm"
            for j in spec["joints"]
        )
        if library.classify_body(cap.body) != "forearm" or not carries_wrist:
            continue
        line = np.subtract(cap.end_m, cap.start_m)
        axis = line / np.linalg.norm(line)
        v = meshes[cap.body].mesh_in_body_frame.vertices - cap.start_m
        ring = v[np.abs(v @ axis - cap.length_m()) < 1e-9]
        ring = ring - np.outer(ring @ axis, axis)
        centre = 0.5 * (ring.max(axis=0) + ring.min(axis=0))
        out[cap.body] = (ring - centre, axis, centre)
    assert len(out) == 2
    return out


def test_the_hand_wrist_is_wider_than_it_is_deep(base) -> None:
    club = ca.assembly_from_spec(base)
    assert club is not None
    for section in sws.hand_wrist_sections(club).values():
        assert 0.020 <= section.half_width_m <= 0.030
        assert 0.014 <= section.half_depth_m < section.half_width_m
        assert np.linalg.norm(section.width_axis) == pytest.approx(1.0)


def test_the_forearm_ends_in_the_hand_wrist_section(skinned) -> None:
    club = ca.assembly_from_spec(skinned)
    sections = sws.forearm_wrist_sections(skinned, club)
    rings = _wrist_rings(skinned)
    assert set(sections) == set(rings)
    hands = sorted(
        (s.half_width_m, s.half_depth_m) for s in sws.hand_wrist_sections(club).values()
    )
    margin = sws.wrist_margin()
    assert sorted(
        (s.half_width_m / margin, s.half_depth_m / margin) for s in sections.values()
    ) == pytest.approx(hands)
    for body, (ring, axis, centre) in rings.items():
        section = sections[body]
        # The hand's skin is not centred on its wrist joint; the forearm ends
        # on the skin's centre, not on the joint.
        off = section.centre_m - (section.centre_m @ axis) * axis
        assert 0.003 <= np.linalg.norm(off) <= 0.012
        assert centre == pytest.approx(off, abs=3e-4)
        radial = np.linalg.norm(ring, axis=1)
        # The capsule axis is the body's long axis; the width axis is made
        # perpendicular to it, so the widest reach is the half width.
        assert radial.max() == pytest.approx(section.half_width_m, abs=2e-4)
        assert radial[radial > 1e-6].min() == pytest.approx(
            section.half_depth_m, abs=6e-4
        )
        widest = ring[radial.argmax()] / radial.max()
        # The loft makes the width axis perpendicular to the forearm.
        across = section.width_axis - (section.width_axis @ axis) * axis
        assert abs(float(widest @ across / np.linalg.norm(across))) > 0.98


def test_other_hands_keep_the_round_library_wrist(base) -> None:
    assert sws.forearm_wrist_sections(base, ca.assembly_from_spec(base)) == {}
    plain = library.PART_RADIUS_M["forearm"] * library.distal_scale("forearm")
    for ring, _, centre in _wrist_rings(base).values():
        assert np.linalg.norm(centre) <= 1e-9
        radial = np.linalg.norm(ring, axis=1)
        assert radial.max() == pytest.approx(plain, abs=1e-6)
        assert np.ptp(radial[radial > 1e-6]) <= 1e-6


def test_a_shallow_end_cap_reaches_less_far_past_the_joint() -> None:
    start, end = np.zeros(3), np.array([0.0, 0.0, 0.3])
    shape = {"capsule_ends": True, "end_scale": 0.6}
    full = geometry.lofted_segment(start, end, 0.04, geometry.LoftOptions(**shape))
    flat = geometry.lofted_segment(
        start, end, 0.04, geometry.LoftOptions(end_cap=0.25, **shape)
    )
    assert full.vertices[:, 2].max() == pytest.approx(0.3 + 0.024)
    assert flat.vertices[:, 2].max() == pytest.approx(0.3 + 0.25 * 0.024)
    assert flat.vertices[:, 2].min() == pytest.approx(-0.04)  # near end unchanged
    for bad in (0.0, 1.5, float("nan")):
        with pytest.raises(ValueError, match="end_cap"):
            geometry.lofted_segment(
                start, end, 0.04, geometry.LoftOptions(end_cap=bad, **shape)
            )


def test_a_shifted_far_end_bends_the_centreline_smoothly() -> None:
    start, end = np.zeros(3), np.array([0.0, 0.0, 0.3])
    shift = np.array([0.008, 0.0, 0.05])  # the part along the limb is ignored
    opts = geometry.LoftOptions(
        capsule_ends=True, taper_end=0.8, flatten_from=0.5, end_shift=shift
    )
    v = geometry.lofted_segment(start, end, 0.04, opts).vertices

    def centre(lo: float, hi: float) -> np.ndarray:
        band = v[(v[:, 2] >= lo) & (v[:, 2] <= hi)]
        return 0.5 * (band.max(axis=0) + band.min(axis=0))[:2]

    assert centre(-0.05, 0.149) == pytest.approx([0.0, 0.0], abs=1e-9)
    assert centre(0.2401, 0.2999) == pytest.approx([0.008, 0.0], abs=1e-9)
    assert v[v[:, 2].argmax()][:2] == pytest.approx([0.008, 0.0], abs=1e-9)
    with pytest.raises(ValueError, match="end_shift"):
        geometry.lofted_segment(
            start,
            end,
            0.04,
            geometry.LoftOptions(end_shift=shift, taper_end=0.5, flatten_from=0.5),
        )


def test_the_forearm_stops_just_past_the_wrist_joint(skinned) -> None:
    # A ball of the half width overhung the back of the hand. The end is
    # round about the width axis instead: as deep along the arm as it is thick.
    club = ca.assembly_from_spec(skinned)
    sections = sws.forearm_wrist_sections(skinned, club)
    doc = document_from_dict({"schema_version": "appearance-v1"})
    meshes = {m.body: m for m in body_meshes(skinned, doc, club) if m.label == "skin0"}
    caps = {c.body: c for c in derive_visual_skeleton(skinned).capsules}
    for body, section in sections.items():
        cap = caps[body]
        line = np.subtract(cap.end_m, cap.start_m)
        reach = (meshes[body].mesh_in_body_frame.vertices - cap.end_m) @ (
            line / np.linalg.norm(line)
        )
        assert reach.max() == pytest.approx(
            sws.wrist_cap() * section.half_depth_m, abs=1e-6
        )


def test_the_hand_is_cut_on_one_flat_ring(skinned) -> None:
    # Dropping whole triangles left a ragged edge that showed past the forearm.
    club = ca.assembly_from_spec(skinned)
    meshes, reach = sch.skinned_hand_meshes(club), sch.forearm_reach_m(club)
    for side, (wrist, along) in reach.items():
        mesh = meshes[f"{side}_hand"]
        height = (mesh.vertices[np.unique(mesh.faces)] - wrist) @ along
        assert height.max() == pytest.approx(sch.forearm_cuff_m(), abs=1e-9)
        # The rim, plus the centre point that closes it.
        assert np.sum(np.abs(height - sch.forearm_cuff_m()) < 1e-9) >= 12


def test_bad_wrist_settings_are_refused(monkeypatch) -> None:
    for key, bad in (("forearm_wrist_margin", 0.5), ("forearm_wrist_cap", 0.0)):
        monkeypatch.setattr(
            sws, "load_hand_visual_quality", lambda k=key, b=bad: {k: b}
        )
        with pytest.raises(ValueError, match=key):
            (sws.wrist_margin if key.endswith("margin") else sws.wrist_cap)()

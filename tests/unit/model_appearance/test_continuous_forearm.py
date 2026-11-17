"""One continuous forearm across its rotation joint (#15262).

The forearm is two bodies joined by the pronation joint. Each was drawn as
its own tapered spindle, so the radius jumped at the joint and the golfer
looked as if it had two elbows. The two bodies now draw the two halves of one
forearm shape, cut flat and round at the joint, so the seam is the same circle
on both sides at every rotation angle. Appearance only.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.model_appearance import document_from_dict, library
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.geometry import LoftOptions, lofted_segment
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton

pytestmark = pytest.mark.unit
SPEC = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
)
#: The two sides of a seam must agree to this (m).
SEAM_M = 1e-6
Z = np.array([0.0, 0.0, 0.2])


def _radius_at(mesh: Any, station_m: float, axis: np.ndarray) -> float:
    """Largest distance from ``axis`` of the vertices at ``station_m`` along it."""
    along = mesh.vertices @ axis
    ring = mesh.vertices[np.abs(along - station_m) < 1e-9]
    assert len(ring), f"no ring at {station_m}"
    return float(np.linalg.norm(ring - np.outer(ring @ axis, axis), axis=1).max())


def _watertight(mesh: Any) -> bool:
    f = mesh.faces
    edges = np.sort(np.concatenate([f[:, [0, 1]], f[:, [1, 2]], f[:, [2, 0]]]), axis=1)
    return bool((np.unique(edges, axis=0, return_counts=True)[1] == 2).all())


@pytest.mark.parametrize("cut", [0.5, 0.35])
def test_two_windows_of_one_loft_meet_in_the_same_circle(cut: float) -> None:
    axis = Z / np.linalg.norm(Z)
    length = float(np.linalg.norm(Z))
    whole = LoftOptions(end_scale=0.8, belly=(0.12, 0.28, 0.45))
    near = lofted_segment(
        np.zeros(3), cut * Z, 0.036, LoftOptions(**{**vars(whole), "window": (0, cut)})
    )
    far = lofted_segment(
        cut * Z, Z, 0.036, LoftOptions(**{**vars(whole), "window": (cut, 1)})
    )
    seam = cut * length
    assert _radius_at(near, seam, axis) == pytest.approx(
        _radius_at(far, seam, axis), abs=SEAM_M
    )
    # Flat at the seam, rounded past the outer ends.
    assert float((near.vertices @ axis).max()) == pytest.approx(seam, abs=1e-12)
    assert float((far.vertices @ axis).min()) == pytest.approx(seam, abs=1e-12)
    assert float((near.vertices @ axis).min()) < 0.0
    assert float((far.vertices @ axis).max()) > length
    for mesh in (near, far):
        assert _watertight(mesh)
        assert mesh.volume() > 0
    one = lofted_segment(np.zeros(3), Z, 0.036, whole)
    assert near.volume() + far.volume() == pytest.approx(one.volume(), rel=0.02)


def test_the_belly_makes_the_loft_widest_near_its_position() -> None:
    axis = Z / np.linalg.norm(Z)
    mesh = lofted_segment(
        np.zeros(3), Z, 0.036, LoftOptions(end_scale=0.8, belly=(0.12, 0.28, 0.45))
    )
    radial = np.linalg.norm(
        mesh.vertices - np.outer(mesh.vertices @ axis, axis), axis=1
    )
    widest = float((mesh.vertices @ axis)[radial.argmax()] / np.linalg.norm(Z))
    assert 0.15 <= widest <= 0.45
    assert radial.max() > 0.036


@pytest.mark.parametrize(
    "options",
    [
        {"window": (0.5, 0.5)},
        {"window": (-0.1, 0.5)},
        {"window": (0.2, 1.1)},
        {"belly": (-0.1, 0.3, 0.2)},
        {"belly": (0.1, 0.3, 0.0)},
        {"belly": (float("nan"), 0.3, 0.2)},
    ],
)
def test_bad_window_or_belly_is_refused(options: dict[str, Any]) -> None:
    with pytest.raises(ValueError, match="window|belly"):
        lofted_segment(np.zeros(3), Z, 0.036, LoftOptions(**options))


def _forearm_skins(spec: dict[str, Any]) -> dict[str, tuple[Any, np.ndarray, float]]:
    """Skin mesh, unit axis and length of every forearm body."""
    caps = {
        cap.body: cap
        for cap in derive_visual_skeleton(spec).capsules
        if library.classify_body(cap.body) == "forearm"
    }
    doc = document_from_dict({"schema_version": "appearance-v1"})
    out = {}
    for mesh in body_meshes(spec, doc):
        if mesh.body in caps and mesh.label.startswith("skin"):
            cap = caps[mesh.body]
            line = np.subtract(cap.end_m, cap.start_m)
            out[mesh.body] = (
                mesh.mesh_in_body_frame,
                line / np.linalg.norm(line),
                cap.length_m(),
            )
    return out


def test_the_forearm_bodies_share_one_circle_at_the_rotation_joint() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    skins = _forearm_skins(spec)
    pairs = [
        (j["parent"], j["child"])
        for j in spec["joints"]
        if j["parent"] in skins and j["child"] in skins
    ]
    assert len(pairs) == 2  # lead and trail
    for upper, lower in pairs:
        near, near_axis, near_length = skins[upper]
        far, far_axis, far_length = skins[lower]
        # Each capsule starts at its body origin; the seam ends the upper one.
        seam = _radius_at(near, near_length, near_axis)
        assert seam == pytest.approx(_radius_at(far, 0.0, far_axis), abs=SEAM_M)
        assert float((near.vertices @ near_axis).max()) <= near_length + 1e-12
        assert float((far.vertices @ far_axis).min()) >= -1e-12
        # Round sections, so the seam holds at every rotation about the axis.
        ring = near.vertices[np.abs(near.vertices @ near_axis - near_length) < 1e-9]
        radii = np.linalg.norm(ring - np.outer(ring @ near_axis, near_axis), axis=1)
        assert np.ptp(radii[radii > 1e-6]) <= SEAM_M
        # One forearm shape: wider at the seam than at the wrist.
        along = far.vertices @ far_axis
        wrist = far.vertices[(along > 0.85 * far_length) & (along < far_length)]
        wrist_radius = np.linalg.norm(
            wrist - np.outer(wrist @ far_axis, far_axis), axis=1
        ).max()
        assert seam > float(wrist_radius)
        assert _watertight(near) and _watertight(far)


def _radii(mesh: Any, axis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    along = mesh.vertices @ axis
    return along, np.linalg.norm(mesh.vertices - np.outer(along, axis), axis=1)


def test_capsule_ends_keep_the_radius_to_the_joint_and_round_beyond_it() -> None:
    axis = Z / np.linalg.norm(Z)
    length = float(np.linalg.norm(Z))
    options = LoftOptions(end_scale=0.8, capsule_ends=True, taper_end=0.8)
    mesh = lofted_segment(np.zeros(3), Z, 0.036, options)
    along, radial = _radii(mesh, axis)
    # A spindle is down to two thirds of its radius at the joint; this is not.
    assert radial[np.abs(along) < 1e-9].max() == pytest.approx(0.036, abs=1e-9)
    end = radial[np.abs(along - length) < 1e-9].max()
    assert end == pytest.approx(0.8 * 0.036, abs=1e-9)
    # The taper is finished at ``taper_end`` of the length.
    late = (along > 0.8 * length) & (along <= length)
    assert radial[late].max() == pytest.approx(0.8 * 0.036, abs=1e-9)
    # Past each joint the skin is a ball about the joint: a neighbour turning
    # about that joint cannot open a gap.
    before, after = mesh.vertices[along < 0], mesh.vertices[along > length]
    assert len(before) and len(after)
    assert np.linalg.norm(before, axis=1).max() <= 0.036 + 1e-9
    assert np.linalg.norm(after - Z, axis=1).max() <= 0.8 * 0.036 + 1e-9
    assert along.min() == pytest.approx(-0.036, abs=1e-9)
    assert along.max() == pytest.approx(length + 0.8 * 0.036, abs=1e-9)
    assert _watertight(mesh) and mesh.volume() > 0
    with pytest.raises(ValueError, match="taper_end"):
        lofted_segment(np.zeros(3), Z, 0.036, LoftOptions(taper_end=0.0))


def test_the_forearm_is_full_width_at_the_elbow_and_the_wrist() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    skins = _forearm_skins(spec)
    radius = library.PART_RADIUS_M["forearm"]
    wrist = radius * library.distal_scale("forearm")
    seen = set()
    for joint in spec["joints"]:
        upper, lower = joint["parent"], joint["child"]
        if upper not in skins or lower not in skins:
            continue
        near, near_axis, _ = skins[upper]
        far, far_axis, far_length = skins[lower]
        along, radial = _radii(near, near_axis)
        # Elbow: no pinched neck, and a ball about the joint beyond it.
        assert radial[np.abs(along) < 1e-9].max() >= radius - 1e-9
        beyond = near.vertices[along < 0]
        assert np.linalg.norm(beyond, axis=1).max() <= radial.max() + 1e-9
        along, radial = _radii(far, far_axis)
        # Wrist: the tapered radius holds to the joint, then a ball.
        at_wrist = radial[np.abs(along - far_length) < 1e-9].max()
        assert at_wrist == pytest.approx(wrist, abs=1e-6)
        tip = far.vertices[along > far_length] - far_length * far_axis
        assert np.linalg.norm(tip, axis=1).max() <= wrist + 1e-6
        seen.add(upper)
    assert len(seen) == 2


def test_forearm_shape_settings_are_validated() -> None:
    assert "forearm" in library.CAPSULE_END_PARTS
    assert 0.0 < library.taper_end("forearm") < 1.0
    assert library.taper_end("thigh") == 1.0
    assert "forearm" in library.CONTINUOUS_PARTS
    gain, position, width = library.belly("forearm")
    assert gain > 0.0 and 0.0 < position < 0.5 and width > 0.0
    assert library.belly("thigh")[0] == 0.0

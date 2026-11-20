"""Upper arm, deltoid cap and neck girth against the CC0 body (#15299). Visual only."""

from __future__ import annotations

import math

import numpy as np
import pytest

from src.shared.python.model_appearance import library, neck
from src.shared.python.model_appearance.body_meshes import body_meshes
from src.shared.python.model_appearance.cross_section import body_sections
from src.shared.python.model_appearance.skinned_body import spec_body_poses
from src.shared.python.motion_matching.visual_skeleton import derive_visual_skeleton
from tests.unit.model_appearance.test_neck import DOC, HEAD_M, SPEC
from tests.unit.model_appearance.test_torso_shape import _solids, _thorax_rings

pytestmark = pytest.mark.unit

REFERENCE_M = library.LIMB_REFERENCE_STATURE_M
STATURE_M = float(SPEC["subject"]["stature_m"])
TALL = {**SPEC, "subject": {**SPEC["subject"], "stature_m": REFERENCE_M}}
# Girth over stature of the athletic male CC0 body (measured at 1.82 m): the
# upper arm at fractions of shoulder to elbow, and the neck under the jaw.
ARM_TARGET = {0.525: 355.0 / 1820.0, 0.95: 310.0 / 1820.0}
NECK_TARGET, NECK_TOLERANCE = 393.0 / 1820.0, 0.06
GIRTH_TOLERANCE = 0.08
# Deepest an upper arm (skin, sleeve) sat inside the shirt before the arm was
# reshaped, metres: the clearance to the chest that must not get worse.
ARM_IN_SHIRT_M = {"hang": 0.03844, "address": 0.09499}


def _arm(spec: dict, label: str, side: str = "R") -> tuple[np.ndarray, np.ndarray]:
    """Rings of one upper-arm mesh about the shoulder joint; its (ML, depth) axes."""
    cap = next(
        c
        for c in derive_visual_skeleton(spec).capsules
        if c.body[-9:] == f"{side}UpperArm"
    )
    mesh = next(
        m for m in body_meshes(spec, DOC) if (m.body, m.label) == (cap.body, label)
    )
    section = body_sections(spec)[cap.body]
    span = np.asarray(cap.end_m) - np.asarray(cap.start_m)
    local = mesh.mesh_in_body_frame.vertices - np.asarray(cap.start_m)
    frame = np.stack([section.ml_axis, section.depth_axis, span / (span @ span)])
    return local @ frame.T, frame


def _ring(spec: dict, frac: float, label: str = "skin0") -> np.ndarray:
    """Section at ``frac`` of shoulder to elbow: columns (width, depth), metres."""
    rings = _arm(spec, label)[0][:-2].reshape(-1, library.LOFT_SIDES, 3)
    at = rings[:, :, 2].mean(axis=1)
    i = int(np.searchsorted(at, frac))
    t = (frac - at[i - 1]) / (at[i] - at[i - 1])
    return ((1 - t) * rings[i - 1] + t * rings[i])[:, :2]


def _girth(ring: np.ndarray) -> float:
    return float(np.linalg.norm(ring - np.roll(ring, 1, axis=0), axis=1).sum())


@pytest.mark.parametrize("frac", sorted(ARM_TARGET))
@pytest.mark.parametrize("spec", [SPEC, TALL], ids=["default", "reference"])
def test_upper_arm_girth_matches_the_measured_body(spec: dict, frac: float) -> None:
    target = ARM_TARGET[frac] * spec["subject"]["stature_m"]
    assert _girth(_ring(spec, frac)) == pytest.approx(target, rel=GIRTH_TOLERANCE)


def test_upper_arm_is_deeper_than_wide_and_rounder_at_the_elbow() -> None:
    width, depth = np.ptp(_ring(SPEC, 0.525), axis=0)
    assert depth / width == pytest.approx(library.PART_ASPECT["upper_arm"], abs=0.02)
    assert 1.10 <= depth / width <= 1.25
    width, depth = np.ptp(_ring(SPEC, 0.95), axis=0)
    assert 1.0 <= depth / width <= 1.08


def test_upper_arm_radius_scales_with_stature() -> None:
    small, tall = (np.ptp(_ring(s, 0.525)[:, 0]) for s in (SPEC, TALL))
    assert small / tall == pytest.approx(STATURE_M / REFERENCE_M, rel=2e-3)
    assert tall == pytest.approx(2.0 * library.PART_RADIUS_M["upper_arm"], rel=0.02)


def test_sleeve_encloses_the_arm_skin() -> None:
    for frac in (0.2, 0.4, 0.6):
        sleeve, skin = (
            np.ptp(_ring(SPEC, frac, k), axis=0) for k in ("garment0", "skin0")
        )
        assert (sleeve > skin).all(), (frac, sleeve, skin)


def test_deltoid_cap_has_its_own_radius(monkeypatch: pytest.MonkeyPatch) -> None:
    size = (
        library.DELTOID_RADIUS_M * library.GARMENT_THICKNESS * STATURE_M / REFERENCE_M
    )
    across = library.DELTOID_HALF_SIZES[0] * size
    before = _arm(SPEC, "blend0")[0]
    assert np.ptp(before[:, 0]) / 2.0 == pytest.approx(across, rel=0.01)
    assert np.ptp(before[:, 0]) > np.ptp(_ring(SPEC, 0.3, "garment0")[:, 0])
    monkeypatch.setitem(library.PART_RADIUS_M, "upper_arm", 0.07)
    assert np.ptp(_ring(SPEC, 0.525)[:, 0]) > 0.12  # the arm did get thicker
    assert np.ptp(_arm(SPEC, "blend0")[0][:, 0]) == pytest.approx(np.ptp(before[:, 0]))
    monkeypatch.setattr(library, "DELTOID_RADIUS_M", 1.5 * library.DELTOID_RADIUS_M)
    assert np.ptp(_arm(SPEC, "blend0")[0][:, 0]) / 2.0 == pytest.approx(
        1.5 * across, rel=0.01
    )


@pytest.mark.parametrize("side", ["L", "R"])
def test_deltoid_cap_sits_outside_the_arm_axis(side: str) -> None:
    local, frame = _arm(SPEC, "blend0", side)
    centre = 0.5 * (local[:, 0].max() + local[:, 0].min())
    size = (
        library.DELTOID_RADIUS_M * library.GARMENT_THICKNESS * STATURE_M / REFERENCE_M
    )
    assert centre == pytest.approx(library.DELTOID_OUTWARD * size, rel=0.02)
    poses = spec_body_poses(SPEC)
    arm = next(pose for body, pose in poses.items() if body[-9:] == f"{side}UpperArm")
    outward = arm[:3, :3] @ frame[0]  # the ML axis, in the world at the zero pose
    assert outward[1] * arm[1, 3] > 0.0  # away from the mid-sagittal plane


def test_deltoid_settings_are_checked(monkeypatch: pytest.MonkeyPatch) -> None:
    assert library.deltoid_radius_m(0.5) == pytest.approx(
        0.5 * library.DELTOID_RADIUS_M
    )
    monkeypatch.setattr(library, "DELTOID_OUTWARD", 1.2)
    with pytest.raises(ValueError, match="deltoid"):
        library.deltoid_radius_m(1.0)


def _deepest_in_shirt(pose: str, label: str) -> float:
    """Deepest an upper-arm ``label`` vertex sits inside the shirt's rings (m)."""
    rings = _thorax_rings(pose, "garment")
    levels = rings[:, :, 2].mean(axis=1)
    deepest = -math.inf
    arms = [
        v
        for body, name, v, _ in _solids(pose)[0]
        if "UpperArm" in body and label in name
    ]
    for point in np.vstack(arms):
        if not levels[0] < point[2] < levels[-1]:
            continue
        ring = rings[int(np.searchsorted(levels, point[2]))][:, :2]
        centre = ring.mean(axis=0)
        spokes, spoke = ring - centre, point[:2] - centre
        angles = np.arctan2(spokes[:, 1], spokes[:, 0])
        order = np.argsort(angles)
        reach = np.interp(
            math.atan2(spoke[1], spoke[0]),
            angles[order],
            np.linalg.norm(spokes, axis=1)[order],
            period=2.0 * math.pi,
        )
        deepest = max(deepest, float(reach - np.linalg.norm(spoke)))
    return deepest


@pytest.mark.parametrize("pose", sorted(ARM_IN_SHIRT_M))
def test_upper_arm_sinks_no_deeper_into_the_chest(pose: str) -> None:
    sleeve = _deepest_in_shirt(pose, "garment")
    assert sleeve <= ARM_IN_SHIRT_M[pose] + 1e-4, sleeve
    assert _deepest_in_shirt(pose, "skin") <= sleeve
    assert _deepest_in_shirt(pose, "blend") <= sleeve  # the cap hides in the sleeve


def _neck_ring(height: float) -> np.ndarray:
    """Neck section whose centre is ``height`` head lengths above the neck point."""
    rings = neck.build_neck(HEAD_M).vertices[:-2].reshape(-1, 36, 3)
    at = rings[:, :, 2].mean(axis=1) / HEAD_M
    i = int(np.searchsorted(at, height))
    t = (height - at[i - 1]) / (at[i] - at[i - 1])
    return (1 - t) * rings[i - 1] + t * rings[i]


def test_neck_girth_matches_the_measured_body() -> None:
    ring = _neck_ring(neck.NECK_ACCEPTANCE.neck_mid_height)
    assert _girth(ring) == pytest.approx(NECK_TARGET * STATURE_M, rel=NECK_TOLERANCE)
    assert np.ptp(ring[:, 0]) >= 0.845 * np.ptp(ring[:, 1])  # deeper than it was


def test_neck_front_is_a_straight_line_from_the_collar_to_the_jaw() -> None:
    low, high = neck.NECK_ACCEPTANCE.shoulder_line_height, 0.0
    rings = [_neck_ring(h) for h in np.linspace(low, high, 9)]
    front = np.array([(r[:, 0].max(), r[:, 2].mean()) for r in rings])
    chord = front[-1] - front[0]
    off = (front - front[0]) @ np.array([chord[1], -chord[0]]) / np.linalg.norm(chord)
    assert np.abs(off).max() <= 0.02 * HEAD_M, off  # no bulge at the throat

"""Smooth segment meshes: closed, outward-wound and finite."""

import numpy as np
import pytest

from src.shared.python.model_appearance import ellipsoid_mesh, lofted_segment
from src.shared.python.model_appearance.geometry import LoftOptions

pytestmark = pytest.mark.unit
Z = np.array([0.0, 0.0, 0.4])


def test_spindle_is_closed_with_positive_volume() -> None:
    mesh = lofted_segment(np.zeros(3), Z, 0.05)
    assert mesh.volume() > 0
    assert np.isfinite(mesh.vertices).all()
    assert mesh.faces.max() == len(mesh.vertices) - 1
    edges = np.sort(
        np.concatenate(
            [mesh.faces[:, [0, 1]], mesh.faces[:, [1, 2]], mesh.faces[:, [2, 0]]]
        ),
        axis=1,
    )
    _, counts = np.unique(edges, axis=0, return_counts=True)
    assert (counts == 2).all()  # watertight


def test_garment_band_is_thicker_than_the_skin_it_covers() -> None:
    skin = lofted_segment(np.zeros(3), Z, 0.05, LoftOptions(coverage=(0.0, 0.6)))
    cloth = lofted_segment(
        np.zeros(3), Z, 0.05, LoftOptions(coverage=(0.0, 0.6), thickness=1.1)
    )
    assert cloth.volume() > skin.volume() > 0


def test_ellipsoid_volume_matches_the_analytic_value() -> None:
    half = np.array([0.05, 0.04, 0.03])
    mesh = ellipsoid_mesh(
        np.zeros(3), half, np.array([1.0, 1.0, 0.0]), rings=40, sides=60
    )
    assert mesh.volume() == pytest.approx(4 / 3 * np.pi * half.prod(), rel=0.02)


@pytest.mark.parametrize(
    "call",
    [
        lambda: lofted_segment(np.zeros(3), np.zeros(3), 0.05),
        lambda: lofted_segment(np.zeros(3), Z, -0.1),
        lambda: lofted_segment(np.zeros(3), Z, 0.05, LoftOptions(coverage=(0.6, 0.2))),
        lambda: lofted_segment(np.zeros(3), np.array([np.nan, 0, 1]), 0.05),
        lambda: ellipsoid_mesh(np.zeros(3), np.array([1, 1, 0.0]), Z),
        lambda: ellipsoid_mesh(np.zeros(3), np.ones(3), np.zeros(3)),
    ],
)
def test_preconditions_raise_value_error(call) -> None:
    with pytest.raises(ValueError):
        call()


# --- R1 (#12680): explicit width axis for non-round cross-sections -------

ANTHRO_SPECS = (
    "docs/development/full_body_models/full_body_spec_anthro_driver.json",
    "docs/development/full_body_models/full_body_spec_anthro_iron7.json",
)


def _rotated_about_y(angle_deg: float) -> np.ndarray:
    """Unit vector ``angle_deg`` away from +z toward +x."""
    a = np.deg2rad(angle_deg)
    return np.array([np.sin(a), 0.0, np.cos(a)])


@pytest.mark.parametrize(
    "hint",
    [None, Z, _rotated_about_y(5.0), _rotated_about_y(9.0)],
    ids=["missing", "parallel", "5deg", "9deg"],
)
def test_non_round_loft_requires_a_width_axis_off_the_segment(hint) -> None:
    with pytest.raises(ValueError):
        lofted_segment(np.zeros(3), Z, 0.05, LoftOptions(aspect=0.7, width_hint=hint))


def test_width_axis_eleven_degrees_off_the_segment_is_accepted() -> None:
    mesh = lofted_segment(
        np.zeros(3), Z, 0.05, LoftOptions(aspect=0.7, width_hint=_rotated_about_y(11))
    )
    assert mesh.volume() > 0


def test_round_loft_still_needs_no_width_axis() -> None:
    assert lofted_segment(np.zeros(3), Z, 0.05, LoftOptions(aspect=1.0)).volume() > 0


@pytest.mark.parametrize(
    "width", [np.array([0.0, 1.0, 0.0]), np.array([1.0, 1.0, 0.0])]
)
def test_cross_section_extents_follow_the_width_axis(width) -> None:
    radius, thickness, aspect = 0.1, 1.08, 0.72
    mesh = lofted_segment(
        np.zeros(3),
        Z,
        radius,
        LoftOptions(
            aspect=aspect, thickness=thickness, rings=15, sides=20, width_hint=width
        ),
    )
    w = width / np.linalg.norm(width)
    d = np.cross(Z / np.linalg.norm(Z), w)
    span = lambda axis: float(np.ptp(mesh.vertices @ axis))  # noqa: E731
    full = 2.0 * radius * thickness
    assert span(w) == pytest.approx(full, rel=1e-9)
    assert span(d) == pytest.approx(aspect * full, rel=1e-9)


def test_section_width_axis_is_perpendicular_to_segment_and_depth() -> None:
    from src.shared.python.model_appearance.geometry import section_width_axis

    width = section_width_axis(Z, np.array([1.0, 0.0, 0.0]))
    np.testing.assert_allclose(np.abs(width), [0.0, 1.0, 0.0], atol=1e-12)
    with pytest.raises(ValueError):
        section_width_axis(Z, np.array([0.05, 0.0, 1.0]))  # depth ~ segment


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    """Unsigned angle between two lines (sign of either vector ignored)."""
    c = abs(float(np.dot(a, b))) / float(np.linalg.norm(a) * np.linalg.norm(b))
    return float(np.degrees(np.arccos(min(1.0, c))))


def _landmark_ml(spec: dict, body: str) -> np.ndarray:
    """ML direction in ``body``'s frame, straight from the spec landmarks.

    Pelvis: the two hip joint origins. Trunk: the two shoulder joint origins
    (children of the shoulder hubs). Foot: ground normal x (calcn -> toes).
    """
    from src.shared.python.model_appearance.library import classify_body
    from src.shared.python.model_appearance.skinned_body import spec_body_poses

    poses = spec_body_poses(spec)
    rot = poses[body][:3, :3]

    def origin(joint: dict) -> np.ndarray:
        return (poses[joint["parent"]] @ np.asarray(joint["parent_to_base"]))[:3, 3]

    joints = spec["joints"]
    if classify_body(body) == "foot":
        side = body.rsplit("_", 1)[-1]
        mtp = next(j for j in joints if j["child"] == f"toes_{side}")
        toe = origin(mtp) - poses[mtp["parent"]][:3, 3]
        up = -np.asarray(spec["gravity_m_s2"], float)
        return rot.T @ np.cross(up, toe)
    hips = [j for j in joints if classify_body(j["child"]) == "thigh"]
    pair = (
        hips
        if all(j["parent"] == body for j in hips)
        else [j for j in joints if classify_body(j["child"]) == "upper_arm"]
    )
    assert len(pair) == 2
    return rot.T @ (origin(pair[0]) - origin(pair[1]))


@pytest.mark.parametrize("spec_path", ANTHRO_SPECS)
def test_width_axes_come_from_spec_landmarks(spec_path) -> None:
    import json
    from pathlib import Path

    from src.shared.python.model_appearance.cross_section import body_sections
    from src.shared.python.model_appearance.library import (
        PART_ASPECT,
        classify_body,
    )
    from src.shared.python.motion_matching.visual_skeleton import (
        derive_visual_skeleton,
    )

    root = Path(__file__).resolve().parents[3]
    spec = json.loads((root / spec_path).read_text(encoding="utf-8"))
    sections = body_sections(spec)
    non_round = {
        b["name"]
        for b in spec["bodies"]
        if b["name"] != "world" and PART_ASPECT.get(classify_body(b["name"]), 1.0) != 1
    }
    assert set(sections) == non_round and non_round
    for name, section in sections.items():
        ml = _landmark_ml(spec, name)
        assert _angle_deg(section.ml_axis, ml) < 5.0, name
        assert abs(float(np.dot(section.ml_axis, section.depth_axis))) < 1e-9
    for name in ("calcn_r", "calcn_l"):  # the OpenSim calcn lateral axis is z
        assert _angle_deg(sections[name].ml_axis, np.array([0, 0, 1.0])) < 5.0
    checked = 0
    for cap in derive_visual_skeleton(spec).capsules:
        if cap.body not in sections:
            continue
        axis = np.subtract(cap.end_m, cap.start_m)
        width = sections[cap.body].width_axis(cap.start_m, cap.end_m)
        assert abs(float(np.dot(width, axis))) < 1e-9 * np.linalg.norm(axis)
        if _angle_deg(axis, sections[cap.body].ml_axis) > 85.0:
            assert _angle_deg(width, sections[cap.body].ml_axis) < 5.0, cap.body
            checked += 1
    assert checked >= 4  # spine capsules and both shoes

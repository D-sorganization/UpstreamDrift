"""OSV-10 (#11759): face-orientation residual of the shared marker IK."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.motion_matching import club_face_target as cft

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
TRIAD = cft.HEAD_TRIAD_LABELS


def _spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _rot(axis: tuple[float, float, float], deg: float) -> np.ndarray:
    a = np.asarray(axis, float) / np.linalg.norm(axis)
    k = np.array([[0, -a[2], a[1]], [a[2], 0, -a[0]], [-a[1], a[0], 0]])
    t = math.radians(deg)
    return np.eye(3) + math.sin(t) * k + (1 - math.cos(t)) * k @ k


OFFSETS = {
    TRIAD[0]: np.array([-0.005, -0.047, -0.020]),
    TRIAD[1]: np.array([0.040, 0.034, -0.022]),
    TRIAD[2]: np.array([0.052, 0.015, 0.025]),
}
LABELS = (*TRIAD, "HeadTop")


def _capture(rotations: list[np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic capture: the triad moved rigidly by ``rotations``."""
    local = np.array(list(OFFSETS.values()))
    points = np.zeros((len(rotations), len(LABELS), 3))
    for k, rot in enumerate(rotations):
        points[k, :3] = local @ rot.T + np.array([0.3 * k, 1.0, 0.5])
        points[k, 3] = [0.0, 0.0, 1.7]
    return points, np.ones(points.shape[:2], dtype=bool)


# ------------------------------------------------------------------ weight


@pytest.mark.parametrize("weight", [-1.0, float("nan"), float("inf")])
def test_face_weight_rejects_negative_and_non_finite(weight: float) -> None:
    with pytest.raises(ValueError, match="face weight"):
        cft.validate_face_weight(weight)


@pytest.mark.parametrize("weight", [True, "1.0", None])
def test_face_weight_rejects_non_numbers(weight: object) -> None:
    with pytest.raises(TypeError, match="face weight"):
        cft.validate_face_weight(weight)  # type: ignore[arg-type]


def test_face_weight_default_is_explicit_and_positive() -> None:
    assert cft.validate_face_weight(cft.FACE_ORIENTATION_WEIGHT) > 0.0
    assert cft.validate_face_weight(0) == 0.0


# ----------------------------------------------------- capture observations


def test_triad_rotation_is_recovered_exactly() -> None:
    truth = [np.eye(3), _rot((0, 1, 0), 30.0), _rot((1, 2, 3), -75.0)]
    points, valid = _capture(truth)
    got = cft.observe_frame_rotations(points, valid, LABELS, OFFSETS)
    np.testing.assert_allclose(got, np.array(truth), atol=1e-12)


def test_unobserved_frames_are_nan_not_zero() -> None:
    points, valid = _capture([np.eye(3), np.eye(3), np.eye(3)])
    valid[1, 2] = False
    points[2, 0] = np.nan
    got = cft.observe_frame_rotations(points, valid, LABELS, OFFSETS)
    assert np.isfinite(got[0]).all()
    assert np.isnan(got[1]).all() and np.isnan(got[2]).all()


def test_observation_contracts() -> None:
    points, valid = _capture([np.eye(3)])
    with pytest.raises(ValueError, match="frames, markers, 3"):
        cft.observe_frame_rotations(points[0], valid, LABELS, OFFSETS)
    with pytest.raises(ValueError, match="labels"):
        cft.observe_frame_rotations(points, valid, LABELS[:3], OFFSETS)
    with pytest.raises(ValueError, match="not in the capture"):
        cft.observe_frame_rotations(
            points, valid, LABELS, {**OFFSETS, "Nope": np.zeros(3)}
        )
    two = dict(list(OFFSETS.items())[:2])
    with pytest.raises(ValueError, match="three"):
        cft.observe_frame_rotations(points, valid, LABELS, two)
    line = {k: np.array([0.01 * i, 0.0, 0.0]) for i, k in enumerate(OFFSETS)}
    with pytest.raises(ValueError, match="degenerate"):
        cft.observe_frame_rotations(points, valid, LABELS, line)
    bad = {**OFFSETS, TRIAD[0]: np.array([np.nan, 0.0, 0.0])}
    with pytest.raises(ValueError, match="finite 3-vector"):
        cft.observe_frame_rotations(points, valid, LABELS, bad)


# ----------------------------------------------------------- spec face axis


def test_face_axis_is_the_rendered_clubface_vector_in_the_marker_frame() -> None:
    spec = _spec()
    axis = cft.face_axis_in_frame(spec)
    assert np.linalg.norm(axis) == pytest.approx(1.0)
    placement = next(f for f in spec["frames"] if f["name"] == cft.FACE_FRAME)
    rot = np.asarray(placement["placement"], float)[:3, :3]
    expected = ca.clubface_vector(ca.assembly_from_spec(spec))
    np.testing.assert_allclose(rot @ axis, expected, atol=1e-12)


def test_face_axis_needs_a_club_frame() -> None:
    spec = _spec()
    with pytest.raises(ValueError, match="no frame named"):
        cft.face_axis_in_frame(spec, "NoSuchFrame")
    with pytest.raises(ValueError, match="not on the club body"):
        cft.face_axis_in_frame(spec, "Head")


# ------------------------------------------------------------- IK targets


def _attachments() -> dict[str, tuple[str, tuple[float, float, float]]]:
    return {k: (cft.FACE_FRAME, tuple(float(x) for x in v)) for k, v in OFFSETS.items()}


def test_targets_rotate_the_face_axis_with_the_capture_triad() -> None:
    truth = [np.eye(3), _rot((0, 1, 0), 40.0)]
    points, valid = _capture(truth)
    spec = _spec()
    targets = cft.face_axis_targets(
        points, valid, LABELS, _attachments(), spec, weight=2.5
    )
    axis = cft.face_axis_in_frame(spec)
    for rot, frame in zip(truth, targets, strict=True):
        assert frame is not None
        body_axis, world, weight = frame[cft.FACE_FRAME]
        np.testing.assert_allclose(body_axis, axis, atol=1e-12)
        np.testing.assert_allclose(world, rot @ axis, atol=1e-12)
        assert weight == 2.5


def test_zero_weight_or_missing_triad_is_the_marker_only_fit() -> None:
    points, valid = _capture([np.eye(3), np.eye(3)])
    spec = _spec()
    off = cft.face_axis_targets(points, valid, LABELS, _attachments(), spec, weight=0)
    assert off == [None, None]
    partial = dict(list(_attachments().items())[:2])
    assert cft.face_axis_targets(points, valid, LABELS, partial, spec) == [None] * 2
    with pytest.raises(ValueError, match="face weight"):
        cft.face_axis_targets(points, valid, LABELS, _attachments(), spec, weight=-1)


def test_merge_is_a_framewise_union() -> None:
    pit = [{"LS": ((1, 0, 0), (0, 0, 1), 0.01)}, None]
    face = [None, {"Clubhead": ((1, 0, 0), (0, 1, 0), 1.0)}]
    assert cft.merge_axis_targets(pit, None, face) == [pit[0], face[1]]
    assert cft.merge_axis_targets(None, None) is None
    assert cft.merge_axis_targets([None], [None]) == [None]
    with pytest.raises(ValueError, match="one entry per capture frame"):
        cft.merge_axis_targets(pit, [None])
    with pytest.raises(ValueError, match="two axis targets"):
        cft.merge_axis_targets(face, face)


def test_face_separation_is_the_angle_between_normals() -> None:
    m = np.array([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]])
    c = np.array([[0.0, 2.0, 0.0], [math.cos(0.1), math.sin(0.1), 0.0]])
    np.testing.assert_allclose(
        cft.face_separation_deg(m, c), [90.0, math.degrees(0.1)], atol=1e-9
    )
    with pytest.raises(ValueError, match="frames, 3"):
        cft.face_separation_deg(m, c[:1])


def test_face_target_turns_the_ik_club_face_toward_the_capture() -> None:
    """The shared LM pose solve honours the face residual (MuJoCo provider)."""
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.contact_law import GroundPlane
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    spec = _spec()
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
        and label
        in ("BackTop", "BackLeft", "BackRight", "LWristTop", "RWristTop", *TRIAD)
    }
    kin = get_plant("mujoco", spec).create_ik(attachments)
    q0 = np.zeros(kin.nq)
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=-5.0)
    axis = cft.face_axis_in_frame(spec)
    rot0, _ = kin.body_poses(q0, [cft.FACE_FRAME])[cft.FACE_FRAME]
    target = _rot((0.3, 1.0, 0.2), 35.0) @ rot0 @ axis
    targets = kin.marker_positions(q0)
    valid = np.ones(len(kin.labels), dtype=bool)
    valid[[kin.labels.index(t) for t in TRIAD]] = False  # face residual only
    fit = kin.solve_pose(
        targets,
        valid,
        q0,
        ground=ground,
        ground_weight=0.0,
        closure_weight=0.0,
        prior_weight=1e-4,
        axis_targets={cft.FACE_FRAME: (tuple(axis), tuple(target), 10.0)},
    )
    rot, _ = kin.body_poses(fit.q, [cft.FACE_FRAME])[cft.FACE_FRAME]
    before = cft.face_separation_deg((rot0 @ axis)[None], target[None])[0]
    after = cft.face_separation_deg((rot @ axis)[None], target[None])[0]
    assert before > 10.0
    assert after < 1.0


def test_capture_face_follows_the_rigid_triad() -> None:
    truth = [np.eye(3), _rot((1, 0, 0), 25.0)]
    points, valid = _capture(truth)
    spec = _spec()
    normals, centres = cft.observe_capture_face(
        points, valid, LABELS, _attachments(), spec
    )
    axis = cft.face_axis_in_frame(spec)
    centre = cft.face_centre_in_frame(spec)
    for k, rot in enumerate(truth):
        shift = np.array([0.3 * k, 1.0, 0.5])
        np.testing.assert_allclose(normals[k], rot @ axis, atol=1e-12)
        np.testing.assert_allclose(centres[k], rot @ centre + shift, atol=1e-12)
    valid[1, 0] = False
    normals, centres = cft.observe_capture_face(
        points, valid, LABELS, _attachments(), spec
    )
    assert np.isnan(normals[1]).all() and np.isnan(centres[1]).all()
    with pytest.raises(ValueError, match="not attached"):
        cft.observe_capture_face(points, valid, LABELS, {}, spec)


def test_face_centre_is_the_rendered_head_face_centre() -> None:
    spec = _spec()
    placement = next(f for f in spec["frames"] if f["name"] == cft.FACE_FRAME)
    pose = np.asarray(placement["placement"], float)
    club_point = pose[:3, :3] @ cft.face_centre_in_frame(spec) + pose[:3, 3]
    assembly = ca.assembly_from_spec(spec)
    head = ca.assembly_meshes(assembly)["head"].vertices
    # On the rendered head surface, on the side the face normal points to.
    assert np.linalg.norm(head - club_point, axis=1).min() < 0.005
    normal = ca.clubface_vector(assembly)
    assert (club_point - head.mean(axis=0)) @ normal > 0.0

"""ROM-aware whole-body orientation IK tests (issue #11689).

The real-model tests need the verified MyoFullBody cache and skip without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.myofullbody import assets, mapping, share_ik

pytestmark = pytest.mark.unit
SPEC = Path("docs/development/full_body_models/full_body_spec_v2.json")


def test_left_jacobian_inverse_matches_finite_differences() -> None:
    rng = np.random.default_rng(3)
    for _ in range(5):
        m = Rotation.from_rotvec(rng.normal(0, 0.5, 3)).as_matrix()
        r = Rotation.from_matrix(m).as_rotvec()
        a = rng.normal(0, 1, 3)
        h = 1e-6
        plus = Rotation.from_rotvec(h * a).as_matrix() @ m
        fd = (Rotation.from_matrix(plus).as_rotvec() - r) / h
        np.testing.assert_allclose(share_ik.left_jacobian_inv(r) @ a, fd, atol=1e-5)


def test_left_jacobian_inverse_is_identity_at_zero() -> None:
    np.testing.assert_allclose(share_ik.left_jacobian_inv(np.zeros(3)), np.eye(3))


def test_tolerances_are_validated() -> None:
    with pytest.raises(ValueError):
        share_ik.ShareConfig(tolerance_deg={"pelvis": 0.0})
    with pytest.raises(ValueError):
        share_ik.ShareConfig(tolerance_deg={"not_a_segment": 5.0})


@pytest.fixture(scope="module")
def mapper():
    pytest.importorskip("mujoco")
    tree = assets.cached_tree()
    if tree is None or not SPEC.exists():
        pytest.skip("MyoFullBody cache or spec document absent")
    model, _ = assets.load_myofullbody(tree)
    spec_bytes = SPEC.read_bytes()
    order = json.loads(spec_bytes)["coordinate_order"]
    return mapping.MyoMapper(spec_bytes, model, np.zeros(len(order)), "clamp"), order


def _pose(order, torso: float) -> np.ndarray:
    """An in-range posture of the spec skeleton with a chosen torso twist (rad).

    The spec zero pose sits outside MyoFullBody's hip and lumbar ranges, so the
    hips are flexed and the elbows set to bring every joint inside its limits at
    ``torso = 1`` (found by search on the extended-ROM mapping).
    """
    q = np.zeros(len(order))
    q[order.index("hip_flexion_l")] = q[order.index("hip_flexion_r")] = 0.8
    q[order.index("LEInput")], q[order.index("REInput")] = 0.1, -0.1
    q[order.index("TorsoInput")] = torso
    return q


def test_in_range_pose_is_fitted_with_small_error(mapper) -> None:
    mp, order = mapper
    pose = share_ik.map_pose(mp, _pose(order, 1.0))
    for name in ("pelvis", "thorax", "humerus_l", "humerus_r"):
        assert pose.error_deg[name] < 0.5, name
    # the legs trade up to ~1 degree against the structural shank error
    assert pose.error_deg["femur_l"] < 2.0 and pose.error_deg["femur_r"] < 2.0


def test_every_joint_stays_inside_its_limits(mapper) -> None:
    mp, order = mapper
    pose = share_ik.map_pose(mp, _pose(order, 2.6))
    for names in mapping.SEGMENT_JOINTS.values():
        for name in names:
            lo, hi = mp.bounds((name,), "clamp")
            value = pose.qpos[mp.model.joint(name).qposadr[0]]
            assert lo[0] - 1e-9 <= value <= hi[0] + 1e-9, name


def test_excess_twist_is_shared_between_pelvis_and_thorax(mapper) -> None:
    mp, order = mapper
    q = _pose(order, 2.6)
    clamp = mp.map_pose(q)
    shared = share_ik.map_pose(mp, q)
    assert clamp.error_deg["thorax"] > 10.0 and clamp.error_deg["pelvis"] == 0.0
    assert shared.error_deg["thorax"] < clamp.error_deg["thorax"]
    assert 0.0 < shared.error_deg["pelvis"] <= 15.0
    assert shared.error_deg["femur_l"] < 2.0


def test_global_velocity_map_matches_finite_differences(mapper) -> None:
    mp, order = mapper
    q = _pose(order, 1.0)
    v = np.zeros(len(order))
    v[order.index("TorsoInput")] = 1.0
    v[order.index("hip_flexion_r")] = 0.5
    pose = share_ik.map_pose(mp, q)
    phi = share_ik.velocity_map(mp, q, pose)
    h = 1e-5
    ahead = share_ik.map_pose(mp, q + h * v, guess=pose.qpos)
    behind = share_ik.map_pose(mp, q - h * v, guess=pose.qpos)
    for name in ("axial_rotation", "hip_flexion_r", "flex_extension"):
        j = mp.model.joint(name)
        fd = (ahead.qpos[j.qposadr[0]] - behind.qpos[j.qposadr[0]]) / (2 * h)
        assert phi[j.dofadr[0]] @ v == pytest.approx(fd, abs=0.05), name


def test_sequence_warm_start_is_continuous(mapper) -> None:
    mp, order = mapper
    q = np.stack([_pose(order, t) for t in (2.5, 2.55, 2.6)])
    poses = share_ik.map_sequence(mp, q, np.arange(3))
    jump = np.abs(poses[1].qpos - poses[0].qpos).max()
    assert jump < 0.2
    assert len(poses) == 3


def test_requires_the_clamp_policy() -> None:
    class Fake:
        rom_policy = "extend"

    with pytest.raises(ValueError):
        share_ik.map_pose(Fake(), np.zeros(3))

"""Hip-calibration-aware MyoSuite leg retarget (#12052, OSV-6 #11737).

The ground-support pipeline fits address coordinates in a hip-calibrated spec
whose ``hip_{l,r}`` ``parent_to_base`` is rewritten, so the same
``hip_*`` values describe a different femur orientation than in stock
MyoSuite. These tests pin the orientation-based retarget against an
independent spec FK (``body_poses_from_state``) and, where MyoSuite is
installed, against the ``myolegs`` MJCF itself.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.engines.physics_engines.myosuite.python.hip_retarget import (
    MYOSUITE_HIP_AXES,
    MYOSUITE_PELVIS_IN_SPEC_PELVIS,
    calibrated_hip_targets,
    myosuite_hip_angles,
    spec_femur_in_pelvis,
)
from src.engines.physics_engines.myosuite.python.retarget import (
    default_retarget_map,
    retarget_frame,
    retarget_trajectory,
    source_coordinate_index,
)
from src.tools.tour_matching_viewer.core import body_poses_from_state

pytestmark = pytest.mark.unit

REPO = Path(__file__).resolve().parents[4]
MODELS = REPO / "docs/development/full_body_models"
STOCK_SPEC = MODELS / "full_body_spec_anthro_driver.json"
EVIDENCE = MODELS / "evidence/ground_support"
CALIBRATED_SPECS = (
    EVIDENCE / "anthro_driver/full_body_spec_hipcal_scaled.json",
    EVIDENCE / "anthro_iron/full_body_spec_hipcal_scaled.json",
)
PELVIS = "solid_reference:GolfSwing3D_Kinetic/Hips and Torso Inputs/LowerTorso"
SIDES = ("r", "l")
HIP = ("flexion", "adduction", "rotation")
#: Acceptance (#12052): segment orientations relative to the pelvis within 1 deg.
ORIENTATION_TOLERANCE_DEG = 1.0


def _load(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _angle_deg(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.degrees(Rotation.from_matrix(a.T @ b).magnitude()))


def _random_leg_pose(order: list[str], seed: int) -> dict[str, float]:
    """A plausible address-like leg pose (radians) with every other value zero."""
    rng = np.random.default_rng(seed)
    coords = dict.fromkeys(order, 0.0)
    for side in SIDES:
        coords[f"hip_flexion_{side}"] = rng.uniform(-0.3, 0.6)
        coords[f"hip_adduction_{side}"] = rng.uniform(-0.5, 0.3)
        coords[f"hip_rotation_{side}"] = rng.uniform(-0.7, 0.7)
        coords[f"knee_angle_{side}"] = rng.uniform(-0.6, 0.0)
        coords[f"ankle_angle_{side}"] = rng.uniform(-0.3, 0.3)
        coords[f"subtalar_angle_{side}"] = rng.uniform(-0.3, 0.3)
    return coords


def _myosuite_femur(side: str, angles: Mapping[str, float]) -> np.ndarray:
    """MyoSuite femur in its pelvis: hinge rotations in body order (MJCF)."""
    rot = np.eye(3)
    for name, axis in zip(HIP, MYOSUITE_HIP_AXES[side], strict=True):
        rot = (
            rot
            @ Rotation.from_rotvec(
                np.asarray(axis) * angles[f"hip_{name}_{side}"]
            ).as_matrix()
        )
    return rot


def _spec_relative(spec: Mapping, coords: Mapping[str, float], body: str) -> np.ndarray:
    poses = body_poses_from_state(spec, coords)
    return poses[PELVIS][:3, :3].T @ poses[body][:3, :3]


def _body_frame(spec: Mapping, body: str) -> np.ndarray:
    """Rotation of the child body frame in its joint follower frame."""
    joint = next(j for j in spec["joints"] if j["child"] == body)
    return np.asarray(joint["child_to_follower"], dtype=float)[:3, :3].T


def test_stock_spec_hip_retarget_is_the_identity() -> None:
    """On the uncalibrated spec the orientation map reproduces the coordinates."""
    spec = _load(STOCK_SPEC)
    for seed in range(5):
        coords = _random_leg_pose(list(spec["coordinate_order"]), seed)
        targets = calibrated_hip_targets(spec, coords)
        for side in SIDES:
            for name in HIP:
                key = f"hip_{name}_{side}"
                assert targets[key] == pytest.approx(coords[key], abs=1e-9), key


@pytest.mark.parametrize("spec_path", CALIBRATED_SPECS, ids=("driver", "iron7"))
def test_spec_femur_in_pelvis_matches_independent_fk(spec_path: Path) -> None:
    """The hip-only rotation equals the full spec FK's femur body frame."""
    spec = _load(spec_path)
    coords = _random_leg_pose(list(spec["coordinate_order"]), 7)
    for side in SIDES:
        expected = _spec_relative(spec, coords, f"femur_{side}") @ _body_frame(
            spec, f"femur_{side}"
        )
        got = spec_femur_in_pelvis(spec, coords, side)
        assert _angle_deg(got, expected) < 1e-9


@pytest.mark.parametrize("spec_path", CALIBRATED_SPECS, ids=("driver", "iron7"))
def test_calibrated_femur_orientation_matches_spec_fk(spec_path: Path) -> None:
    """MyoSuite femur relative to the pelvis matches the calibrated spec within 1 deg."""
    spec = _load(spec_path)
    for seed in range(5):
        coords = _random_leg_pose(list(spec["coordinate_order"]), seed)
        targets = calibrated_hip_targets(spec, coords)
        for side in SIDES:
            spec_femur = spec_femur_in_pelvis(spec, coords, side)
            myo = MYOSUITE_PELVIS_IN_SPEC_PELVIS @ _myosuite_femur(side, targets)
            assert _angle_deg(myo, spec_femur) < ORIENTATION_TOLERANCE_DEG


def test_calibration_changes_the_hip_coordinates() -> None:
    """The identity map is wrong on a calibrated spec: values must differ."""
    spec = _load(CALIBRATED_SPECS[0])
    coords = _random_leg_pose(list(spec["coordinate_order"]), 3)
    targets = calibrated_hip_targets(spec, coords)
    diffs = [abs(targets[k] - coords[k]) for k in targets]
    assert max(diffs) > np.radians(5.0)


def test_myosuite_hip_angles_round_trip() -> None:
    rng = np.random.default_rng(11)
    for side in SIDES:
        angles = {
            f"hip_{name}_{side}": float(v)
            for name, v in zip(HIP, rng.uniform(-0.6, 0.6, 3), strict=True)
        }
        got = myosuite_hip_angles(_myosuite_femur(side, angles), side)
        for key, value in angles.items():
            assert got[key] == pytest.approx(value, abs=1e-12)


def test_myosuite_hip_angles_rejects_gimbal_lock() -> None:
    lock = Rotation.from_rotvec([np.pi / 2, 0.0, 0.0]).as_matrix()
    with pytest.raises(ValueError, match="gimbal"):
        myosuite_hip_angles(lock, "r")


def test_myosuite_hip_angles_rejects_unknown_side() -> None:
    with pytest.raises(ValueError, match="side"):
        myosuite_hip_angles(np.eye(3), "x")


def test_calibrated_hip_targets_needs_hip_coordinates() -> None:
    spec = _load(CALIBRATED_SPECS[0])
    with pytest.raises(ValueError, match="hip_flexion_r"):
        calibrated_hip_targets(spec, {})


def test_retarget_frame_with_hip_spec_overwrites_only_hips() -> None:
    spec = _load(CALIBRATED_SPECS[0])
    rmap = default_retarget_map()
    order = list(spec["coordinate_order"])
    coords = _random_leg_pose(order, 5)
    q = np.array([coords[n] for n in order])[source_coordinate_index(order, rmap)]
    plain = retarget_frame(q, rmap)
    calibrated = retarget_frame(q, rmap, hip_spec=spec)
    targets = calibrated_hip_targets(spec, coords)
    for idx, name in enumerate(rmap.target_names):
        expected = targets.get(name, plain[idx])
        assert calibrated[idx] == pytest.approx(expected, abs=1e-12), name
    traj = retarget_trajectory(np.stack([q, q]), rmap, hip_spec=spec)
    np.testing.assert_allclose(traj[1], calibrated, atol=1e-12)


#: ``knee_angle_{side}`` hinge axes in the ``myolegs`` tibia frames (MJCF).
MYOLEGS_KNEE_AXES = {"r": (0.0, -0.0707, -0.9975), "l": (0.0, 0.0707, -0.9975)}


@pytest.mark.parametrize("spec_path", (STOCK_SPEC, *CALIBRATED_SPECS))
def test_map_knee_signs_follow_the_spec_knee_axes(spec_path: Path) -> None:
    """Map sign = sign of (spec knee rotation axis . MyoSuite knee axis).

    The spec's left knee flexes negative about the OpenSim axis while
    MyoSuite flexes both knees positive, so ``knee_angle_l`` maps with -1.
    """
    spec = _load(spec_path)
    rmap = default_retarget_map()
    for side in SIDES:
        joint = next(j for j in spec["joints"] if j["child"] == f"tibia_{side}")
        assert [p["primitive"] for p in joint["primitives"]] == ["Rz"]
        # Rz turns the follower about its z; in the tibia body frame that axis
        # is the z column of child_to_follower.
        axis = np.asarray(joint["child_to_follower"], dtype=float)[:3, 2]
        dot = float(np.dot(axis, MYOLEGS_KNEE_AXES[side]))
        assert abs(dot) > 0.99, (side, dot)
        _, sign = rmap.source_to_target[f"knee_angle_{side}"]
        assert np.sign(sign) == np.sign(dot), side


def _myolegs_model():
    myosuite = pytest.importorskip("myosuite")
    mujoco = pytest.importorskip("mujoco")
    xml = Path(myosuite.__file__).resolve().parent / ("simhive/myo_sim/leg/myolegs.xml")
    if not xml.is_file():
        pytest.skip(f"myolegs.xml not found at {xml}")
    return mujoco, mujoco.MjModel.from_xml_path(str(xml))


def test_myosuite_hip_axes_match_myolegs_mjcf() -> None:
    mujoco, model = _myolegs_model()
    for side in SIDES:
        femur = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, f"femur_{side}")
        np.testing.assert_allclose(model.body_quat[femur], [1, 0, 0, 0], atol=1e-12)
        for name, axis in zip(HIP, MYOSUITE_HIP_AXES[side], strict=True):
            joint = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_JOINT, f"hip_{name}_{side}"
            )
            assert model.jnt_bodyid[joint] == femur
            np.testing.assert_allclose(model.jnt_axis[joint], axis, atol=1e-12)
    pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
    base = np.zeros(9)
    mujoco.mju_quat2Mat(base, model.body_quat[pelvis])
    np.testing.assert_allclose(
        base.reshape(3, 3), MYOSUITE_PELVIS_IN_SPEC_PELVIS, atol=1e-6
    )


@pytest.mark.parametrize("spec_path", CALIBRATED_SPECS, ids=("driver", "iron7"))
def test_myolegs_segments_match_calibrated_spec_within_one_degree(
    spec_path: Path,
) -> None:
    """Acceptance: femur, tibia and calcn relative to the pelvis within 1 deg.

    The MyoSuite pose is the retargeted vector applied to the ``myolegs``
    MJCF (knee helper joints at 0, as in ``scripts/address_foot_progression_engines``).
    """
    mujoco, model = _myolegs_model()
    spec = _load(spec_path)
    rmap = default_retarget_map()
    order = list(spec["coordinate_order"])
    for seed in range(3):
        coords = _random_leg_pose(order, seed)
        q = np.array([coords[n] for n in order])[source_coordinate_index(order, rmap)]
        q_target = retarget_frame(q, rmap, hip_spec=spec)
        data = mujoco.MjData(model)
        for name, value in zip(rmap.target_names, q_target, strict=True):
            joint = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            if joint >= 0:
                data.qpos[model.jnt_qposadr[joint]] = value
        mujoco.mj_kinematics(model, data)
        pelvis = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "pelvis")
        r_pelvis = data.xmat[pelvis].reshape(3, 3)
        for side in SIDES:
            for body in (f"femur_{side}", f"tibia_{side}", f"calcn_{side}"):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
                myo = MYOSUITE_PELVIS_IN_SPEC_PELVIS @ (
                    r_pelvis.T @ data.xmat[bid].reshape(3, 3)
                )
                spec_rel = _spec_relative(spec, coords, body) @ _body_frame(spec, body)
                err = _angle_deg(myo, spec_rel)
                assert err < ORIENTATION_TOLERANCE_DEG, (body, seed, err)

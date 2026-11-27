"""Whole-body COM and H reference on multi-tree golf MJCF models (B3, #15409 / #15435)."""

from __future__ import annotations

import warnings

import mujoco
import numpy as np
import pytest

from src.engines.physics_engines.mujoco import golf_swing_models_xml
from src.shared.python.body_part_viz.mujoco_angular_momentum_anchors import (
    angular_momentum_anchors,
)
from src.shared.python.body_part_viz.mujoco_segment_angular_momentum import (
    MujocoSegmentAngularMomentumProvider,
    whole_body_com_kinematics,
)
from src.shared.python.body_part_viz.mujoco_segment_group_catalog import (
    infer_segment_group_catalog,
)
from tests.unit.body_part_viz.independent_angmom_oracle import (
    oracle_ball_mass_weight,
    oracle_club_angular_momentum,
    oracle_mass_weighted_com_and_velocity,
    oracle_system_angular_momentum,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_GOLF_MULTITREE: dict[str, str] = {
    "full_body_golf_swing": golf_swing_models_xml.FULL_BODY_GOLF_SWING_XML,
    "advanced_biomechanical_golf_swing": (
        golf_swing_models_xml.ADVANCED_BIOMECHANICAL_GOLF_SWING_XML
    ),
}


def _catalog(model: mujoco.MjModel):
    names = [model.body(i).name for i in range(1, model.nbody) if model.body(i).name]
    return infer_segment_group_catalog(names)


def _random_state(model: mujoco.MjModel, seed: int) -> mujoco.MjData:
    data = mujoco.MjData(model)
    rng = np.random.default_rng(seed)
    data.qpos[:] = rng.uniform(-0.15, 0.15, model.nq)
    data.qvel[:] = rng.uniform(-0.8, 0.8, model.nv)
    mujoco.mj_forward(model, data)
    return data


def _ball_free_joint_slices(
    model: mujoco.MjModel, ball_body_id: int
) -> tuple[slice, slice]:
    for joint_id in range(model.njnt):
        if int(model.jnt_bodyid[joint_id]) != ball_body_id:
            continue
        qpos_adr = int(model.jnt_qposadr[joint_id])
        qvel_adr = int(model.jnt_dofadr[joint_id])
        if int(model.jnt_type[joint_id]) == int(mujoco.mjtJoint.mjJNT_FREE):
            return slice(qpos_adr, qpos_adr + 7), slice(qvel_adr, qvel_adr + 6)
        raise AssertionError("expected ball to use a free joint")
    raise AssertionError("no joint found for ball body")


def default_first_mass_body_com(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    mujoco.mj_subtreeVel(model, data)
    for body_id in range(1, model.nbody):
        if model.body_mass[body_id] > 0.0:
            return np.asarray(data.subtree_com[body_id], dtype=float)
    raise AssertionError("no massive body")


@pytest.mark.parametrize(
    "model_name", tuple(_GOLF_MULTITREE), ids=tuple(_GOLF_MULTITREE)
)
def test_whole_body_anchor_matches_independent_com(model_name: str) -> None:
    model = mujoco.MjModel.from_xml_string(_GOLF_MULTITREE[model_name])
    data = _random_state(model, 15398)
    catalog = _catalog(model)
    qvel = np.asarray(data.qvel, dtype=float)
    ref_independent, _ = oracle_mass_weighted_com_and_velocity(model, data, qvel)
    ref_production, _ = whole_body_com_kinematics(model, data, qvel)
    anchors = angular_momentum_anchors(model, data, catalog)
    np.testing.assert_allclose(ref_production, ref_independent, rtol=0.0, atol=1e-12)
    np.testing.assert_allclose(
        anchors["whole_body"], ref_independent, rtol=0.0, atol=1e-12
    )
    first_mass = default_first_mass_body_com(model, data)
    assert float(np.linalg.norm(ref_independent - first_mass)) > 0.05


@pytest.mark.parametrize(
    "model_name", tuple(_GOLF_MULTITREE), ids=tuple(_GOLF_MULTITREE)
)
def test_whole_body_and_club_h_match_independent_oracle(model_name: str) -> None:
    model = mujoco.MjModel.from_xml_string(_GOLF_MULTITREE[model_name])
    data = _random_state(model, 15399)
    catalog = _catalog(model)
    qvel = np.asarray(data.qvel, dtype=float)
    frame = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=0.0
    ).get_segment_angular_momentum()
    assert frame is not None and frame.whole_body_nm_s is not None
    reference_h = oracle_system_angular_momentum(model, data, qvel)
    np.testing.assert_allclose(
        frame.whole_body_nm_s,
        reference_h,
        rtol=0.0,
        atol=1e-9,
    )
    club = frame.groups_nm_s.get("club")
    assert club is not None
    club_reference = oracle_club_angular_momentum(model, data, qvel, catalog.club)
    np.testing.assert_allclose(club, club_reference, rtol=0.0, atol=1e-9)


@pytest.mark.parametrize(
    "model_name", tuple(_GOLF_MULTITREE), ids=tuple(_GOLF_MULTITREE)
)
def test_ball_motion_does_not_shift_whole_body_com_or_h(model_name: str) -> None:
    """Ball is excluded from COM and whole-body H (N2 regression)."""
    model = mujoco.MjModel.from_xml_string(_GOLF_MULTITREE[model_name])
    data = _random_state(model, 15400)
    catalog = _catalog(model)
    ball_id, ball_mass = oracle_ball_mass_weight(model)
    assert ball_mass > 0.0
    qvel = np.asarray(data.qvel, dtype=float).copy()
    baseline_com, _ = oracle_mass_weighted_com_and_velocity(model, data, qvel)
    baseline_h = oracle_system_angular_momentum(model, data, qvel)
    frame_before = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=0.0
    ).get_segment_angular_momentum()
    assert frame_before is not None and frame_before.whole_body_nm_s is not None

    qpos_sl, qvel_sl = _ball_free_joint_slices(model, ball_id)
    data.qvel[:] = qvel
    pos = np.asarray(data.qpos[qpos_sl], dtype=float)
    pos[0:3] += np.array([0.5, -0.3, 0.2])
    data.qpos[qpos_sl] = pos
    ball_qvel = np.zeros(6, dtype=float)
    ball_qvel[0:3] = (8.0, -6.0, 5.0)
    ball_qvel[3:6] = (4.0, -3.0, 2.0)
    data.qvel[qvel_sl] = ball_qvel
    mujoco.mj_forward(model, data)

    perturbed_com, _ = oracle_mass_weighted_com_and_velocity(model, data, qvel)
    perturbed_h = oracle_system_angular_momentum(model, data, qvel)
    frame_after = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=0.0
    ).get_segment_angular_momentum()
    assert frame_after is not None and frame_after.whole_body_nm_s is not None

    np.testing.assert_allclose(baseline_com, perturbed_com, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(baseline_h, perturbed_h, rtol=0.0, atol=1e-11)
    np.testing.assert_allclose(
        frame_before.whole_body_nm_s,
        frame_after.whole_body_nm_s,
        rtol=0.0,
        atol=1e-11,
    )
    assert float(np.linalg.norm(np.asarray(data.xipos[ball_id]) - baseline_com)) > 0.1


def test_root_body_id_emits_deprecation_warning() -> None:
    model = mujoco.MjModel.from_xml_string(_GOLF_MULTITREE["full_body_golf_swing"])
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    catalog = _catalog(model)
    with pytest.warns(DeprecationWarning, match="root_body_id is deprecated"):
        MujocoSegmentAngularMomentumProvider(
            model, data, catalog=catalog, root_body_id=2
        )
    with pytest.warns(DeprecationWarning, match="root_body_id is deprecated"):
        whole_body_com_kinematics(model, data, data.qvel, root_body_id=2)
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", DeprecationWarning)
        angular_momentum_anchors(model, data, catalog, root_body_id=2)
    assert any("root_body_id is deprecated" in str(w.message) for w in caught)

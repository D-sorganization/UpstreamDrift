"""Whole-body COM and H reference on multi-tree golf MJCF models (B3, #15409)."""

from __future__ import annotations

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
    _body_spin_and_motion,
)
from src.shared.python.body_part_viz.mujoco_segment_group_catalog import (
    infer_segment_group_catalog,
)
from src.shared.python.body_part_viz.segment_angular_momentum import (
    body_angular_momentum_about_point,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

_MULTITREE = (
    golf_swing_models_xml.FULL_BODY_GOLF_SWING_XML,
    golf_swing_models_xml.ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
)


def _independent_system_angular_momentum(
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> np.ndarray:
    qvel = np.asarray(data.qvel, dtype=float)
    ref_com, ref_vel = whole_body_com_kinematics(model, data, qvel)
    total = np.zeros(3, dtype=float)
    for body_id in range(1, model.nbody):
        name = model.body(body_id).name or ""
        (
            com_pos,
            com_vel,
            ang_vel,
            inertia,
            mass,
            rot,
        ) = _body_spin_and_motion(model, data, body_id, qvel)
        if mass <= 0.0 or not np.isfinite(inertia).all():
            continue
        contrib, _spin = body_angular_momentum_about_point(
            com_world=com_pos,
            com_vel_world=com_vel,
            ang_vel_world=ang_vel,
            inertia_principal=inertia,
            rot_body_to_world=rot,
            mass=mass,
            reference_com_world=ref_com,
            reference_com_vel_world=ref_vel,
        )
        total += contrib
    return total


def _catalog(model: mujoco.MjModel):
    names = [model.body(i).name for i in range(1, model.nbody) if model.body(i).name]
    return infer_segment_group_catalog(names)


@pytest.mark.parametrize("xml", _MULTITREE)
def test_whole_body_anchor_equals_system_com(xml: str) -> None:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(15398)
    data.qpos[:] = rng.uniform(-0.15, 0.15, model.nq)
    data.qvel[:] = rng.uniform(-0.8, 0.8, model.nv)
    mujoco.mj_forward(model, data)
    catalog = _catalog(model)
    ref_com, _ = whole_body_com_kinematics(
        model, data, np.asarray(data.qvel, dtype=float)
    )
    anchors = angular_momentum_anchors(model, data, catalog)
    np.testing.assert_allclose(
        anchors["whole_body"],
        ref_com,
        rtol=0.0,
        atol=1e-12,
    )
    # Must not coincide with the first-massive-body subtree COM (left-foot bug).
    first_mass = default_first_mass_body_com(model, data)
    assert float(np.linalg.norm(np.asarray(anchors["whole_body"]) - first_mass)) > 0.05


def default_first_mass_body_com(
    model: mujoco.MjModel, data: mujoco.MjData
) -> np.ndarray:
    mujoco.mj_subtreeVel(model, data)
    for body_id in range(1, model.nbody):
        if model.body_mass[body_id] > 0.0:
            return np.asarray(data.subtree_com[body_id], dtype=float)
    raise AssertionError("no massive body")


@pytest.mark.parametrize("xml", _MULTITREE)
def test_whole_body_and_club_h_match_independent_sum(xml: str) -> None:
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    rng = np.random.default_rng(15398)
    data.qpos[:] = rng.uniform(-0.15, 0.15, model.nq)
    data.qvel[:] = rng.uniform(-0.8, 0.8, model.nv)
    mujoco.mj_forward(model, data)
    catalog = _catalog(model)
    frame = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=0.0
    ).get_segment_angular_momentum()
    assert frame is not None and frame.whole_body_nm_s is not None
    reference = _independent_system_angular_momentum(model, data)
    np.testing.assert_allclose(
        frame.whole_body_nm_s,
        reference,
        rtol=0.0,
        atol=1e-9,
    )
    club = frame.groups_nm_s.get("club")
    assert club is not None
    club_members = catalog.club
    club_ref = np.zeros(3, dtype=float)
    for name in club_members:
        vec = frame.contribution_nm_s.get(name)
        if vec is not None:
            club_ref += np.asarray(vec, dtype=float)
    np.testing.assert_allclose(club, club_ref, rtol=0.0, atol=1e-9)

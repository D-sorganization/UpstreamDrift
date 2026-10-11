"""Mapping and receipt-helper tests (issue #11644).

The real-model tests need the verified MyoFullBody cache and skip without it.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.myofullbody import assets, mapping, mapping_report
from src.shared.python.myofullbody.swing_pipeline import frame_steps

pytestmark = pytest.mark.unit
SPEC = Path("docs/development/full_body_models/full_body_spec_v2.json")


def test_frame_steps_include_key_frames_sorted_unique() -> None:
    keys = mapping_report.KeyFrames(0, 33, 61, 98)
    steps = frame_steps(100, 10, keys)
    assert list(steps) == [0, 10, 20, 30, 33, 40, 50, 60, 61, 70, 80, 90, 98]
    with pytest.raises(ValueError):
        frame_steps(50, 10, keys)


def test_tolerance_is_tight_for_three_dof_chains_only() -> None:
    assert mapping_report.tolerance_for("thorax") == 1.0
    assert mapping_report.tolerance_for("femur_l") == 1.0
    assert mapping_report.tolerance_for("tibia_r") == 15.0
    assert mapping_report.tolerance_for("forearm_l") == 15.0


@pytest.fixture(scope="module")
def mapper():
    pytest.importorskip("mujoco")
    tree = assets.cached_tree()
    if tree is None or not SPEC.exists():
        pytest.skip("MyoFullBody cache or spec document absent")
    model, _ = assets.load_myofullbody(tree)
    spec_bytes = SPEC.read_bytes()
    order = json.loads(spec_bytes)["coordinate_order"]
    q0 = np.zeros(len(order))
    return mapping.MyoMapper(spec_bytes, model, q0, rom_policy="extend"), order


def test_spec_zero_pose_maps_to_a_exact_segment_orientations(mapper) -> None:
    mp, order = mapper
    pose = mp.map_pose(np.zeros(len(order)))
    for name in ("pelvis", "thorax", "humerus_l", "humerus_r"):
        assert pose.error_deg[name] < 1e-3
    assert np.isfinite(pose.qpos).all()


def test_dependent_joints_satisfy_their_equalities(mapper) -> None:
    import mujoco

    mp, order = mapper
    q = np.zeros(len(order))
    q[order.index("LSInputX")] = 0.3
    pose = mp.map_pose(q)
    data = mujoco.MjData(mp.model)
    data.qpos[:] = pose.qpos
    mujoco.mj_forward(mp.model, data)
    eq = np.asarray(data.efc_pos)[
        np.asarray(data.efc_type) == mujoco.mjtConstraint.mjCNSTR_EQUALITY
    ]
    assert eq.size == 0 or np.abs(eq).max() < 1e-6


def test_velocity_map_matches_finite_differences(mapper) -> None:
    mp, order = mapper
    q = np.zeros(len(order))
    q[order.index("TorsoInput")] = 0.2
    q[order.index("hip_flexion_r")] = 0.3
    v = np.zeros(len(order))
    v[order.index("TorsoInput")] = 1.0
    v[order.index("hip_flexion_r")] = 0.5
    pose = mp.map_pose(q)
    phi = mp.velocity_map(q, pose)
    h = 1e-5
    ahead = mp.map_pose(q + h * v, guess=pose.qpos)
    behind = mp.map_pose(q - h * v, guess=pose.qpos)
    adr = mp.model.jnt_qposadr
    dof = mp.model.jnt_dofadr
    for j in range(mp.model.njnt):
        if mp.model.jnt_type[j] != 3:  # hinge only
            continue
        fd = (ahead.qpos[adr[j]] - behind.qpos[adr[j]]) / (2 * h)
        assert phi[dof[j]] @ v == pytest.approx(fd, abs=2e-3), mp.model.joint(j).name

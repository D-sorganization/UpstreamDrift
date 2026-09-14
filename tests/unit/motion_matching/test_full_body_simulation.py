"""Tests for ground-supported forward dynamics of the MuJoCo full-body model."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python import full_body_simulation as module
from src.engines.physics_engines.mujoco.python.full_body_markers import (
    FullBodyMarkerKinematics,
)
from src.engines.physics_engines.mujoco.python.full_body_model import (
    NativeMujocoFullBodyModel,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_v1.json"
CANDIDATE = (
    ROOT
    / "docs/development/full_body_models/evidence/native_candidates"
    / "returned81_candidate.json"
)


@pytest.fixture(scope="module")
def simulator() -> module.FullBodySimulator:
    return module.FullBodySimulator(NativeMujocoFullBodyModel(SPEC.read_bytes()))


def standing_pose(simulator: module.FullBodySimulator) -> np.ndarray:
    """Upper body at the returned81 q0, legs solved for flat feet and CoM over the feet."""
    candidate = json.loads(CANDIDATE.read_text())
    spec = json.loads(SPEC.read_text())
    q = np.zeros(simulator.nv)
    for name, value in zip(candidate["coordinate_names"], candidate["q0"], strict=True):
        q[simulator.names.index(name)] = value
    attachments = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    kinematics = FullBodyMarkerKinematics(simulator.adapter, attachments)
    locked = {
        name: q[simulator.names.index(name)]
        for name in candidate["coordinate_names"][6:]
    }
    fit = kinematics.solve_pose(
        np.zeros((len(attachments), 3)),
        np.zeros(len(attachments), bool),
        q,
        ground=simulator.adapter.ground_plane,
        closure_weight=0.0,
        flat_feet=True,
        ground_weight=1e4,
        balance_weight=1e4,
        prior_weight=1e-1,
        locked=locked,
    )
    assert kinematics.support_offset(fit.q, simulator.adapter.ground_plane) < 1e-3
    return fit.q


def test_root_is_unactuated_and_free_fall_is_gravity(
    simulator: module.FullBodySimulator,
) -> None:
    q = np.zeros(simulator.nv)
    q[2] = 2.0  # feet well above the ground: no contact
    v = np.zeros(simulator.nv)
    tau = np.zeros(simulator.nv)
    tau[:6] = 100.0  # root efforts must be ignored
    a = simulator.acceleration(q, v, tau)
    axes = simulator.root_translation_axes(q)
    np.testing.assert_allclose(axes @ a[:3], simulator.gravity, atol=1e-9)
    np.testing.assert_allclose(a[3:], 0.0, atol=1e-9)
    report, lowest = simulator.support(q, v)
    assert report.total_normal_force_n == 0.0 and lowest > 1.0
    with pytest.raises(ValueError):
        simulator.acceleration(q, v, tau[:-1])


def test_balanced_standing_pose_is_carried_by_the_feet(
    simulator: module.FullBodySimulator,
) -> None:
    q0 = module.preload_feet(simulator, standing_pose(simulator))
    report0, lowest = simulator.support(q0, np.zeros(simulator.nv))
    assert lowest == pytest.approx(-simulator.static_penetration_m(), abs=1e-9)
    assert report0.weight_fraction == pytest.approx(1.0, abs=0.05)
    controller = module.hold_pose_controller(
        simulator, q0, omega_rad_s=30.0, zeta=1.0, balance=(60.0, 15.0)
    )
    record = simulator.run(
        q0,
        np.zeros(simulator.nv),
        controller,
        duration_s=1.0,
        dt_s=1e-3,
        record_every=50,
    )
    assert record.time_s[0] == 0.0 and record.q.shape[1] == simulator.nv
    assert record.weight_fraction[-1] == pytest.approx(1.0, abs=0.03)
    assert bool(record.inside_support_polygon[-1])
    assert np.linalg.norm(record.v[-1]) < 0.05
    assert np.abs(record.q[-1, 6:] - q0[6:]).max() < 0.01
    com0 = simulator.centre_of_mass(q0)[0]
    com1 = simulator.centre_of_mass(record.q[-1])[0]
    assert np.linalg.norm(com1[:2] - com0[:2]) < 0.02
    assert (
        record.lowest_sphere_height_m[-1] < 0.0
    )  # compliant contact carries the weight
    np.testing.assert_allclose(record.tau[:, :6], 0.0)
    with pytest.raises(ValueError):
        simulator.run(q0, np.zeros(simulator.nv), controller, duration_s=0.0, dt_s=1e-3)


def test_tracking_controller_validates_reference(
    simulator: module.FullBodySimulator,
) -> None:
    with pytest.raises(ValueError):
        module.tracking_controller(
            simulator, [0.0, 0.0], np.zeros((2, simulator.nv)), omega_rad_s=10.0
        )
    with pytest.raises(ValueError):
        module.hold_pose_controller(simulator, np.zeros(simulator.nv), omega_rad_s=-1.0)
    q = np.zeros(simulator.nv)
    q[2] = 2.0
    omega = module.joint_natural_frequencies(simulator, upper_body=30.0, lower_limb=8.0)
    assert omega[simulator.lower_limb[0]] == 8.0 and omega[6] == 30.0
    with pytest.raises(ValueError):
        module.joint_natural_frequencies(simulator, upper_body=0.0, lower_limb=8.0)
    controller = module.tracking_controller(
        simulator, [0.0, 1.0], np.stack([q, q]), omega_rad_s=omega
    )
    tau = controller(0.5, q, np.zeros(simulator.nv))
    assert tau.shape == (simulator.nv,) and np.all(tau[:6] == 0.0)
    # Computed torque reproduces any achievable joint acceleration exactly.
    q0 = module.preload_feet(simulator, standing_pose(simulator))
    v0 = np.zeros(simulator.nv)
    affine, offset = simulator.affine_dynamics(q0, v0)
    probe = np.zeros(simulator.nv)
    probe[6:] = np.linspace(-2.0, 2.0, simulator.actuated.size)
    target = (affine @ probe[6:] + offset)[simulator.actuated]
    tau = simulator.inverse_dynamics(q0, v0, target)
    achieved = simulator.acceleration(q0, v0, tau)[simulator.actuated]
    assert np.abs(achieved - target).max() < 1e-6
    with pytest.raises(ValueError):
        simulator.inverse_dynamics(q0, v0, target[:-1])

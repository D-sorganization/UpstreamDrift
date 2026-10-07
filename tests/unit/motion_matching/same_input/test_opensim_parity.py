"""OpenSim same-input parity against Pinocchio and MuJoCo (#11611, epic #11605).

Gates L0 (mass and mass matrix), L1 (pointwise constrained accelerations on the
closure manifold) and L2 (30 ms open-loop replay) for the OpenSim/Simbody
adapter, using the committed ground-support fixture exactly as the Drake and
Pinocchio parity tests do.  The adapter uses the shared contact law and the
exact weld KKT; OpenSim supplies kinematics, mass matrix and bias only.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.same_input import (
    VectorPlant,
    generate_reference_bundle,
    open_loop,
    project_to_closure,
)

pytestmark = [
    pytest.mark.unit,
    pytest.mark.integration,
    pytest.mark.requires_opensim,
    pytest.mark.requires_pinocchio,
    pytest.mark.requires_mujoco,
]

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
FRAMES = (0, 400, 420)
ACCELERATION_ATOL = 1e-6
ACCELERATION_RTOL = 1e-8
MASS_RTOL = 1e-9


@pytest.fixture(scope="module")
def spec_bytes() -> bytes:
    for module in ("opensim", "pinocchio", "mujoco", "pydrake"):
        pytest.importorskip(module)
    return (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()


@pytest.fixture(scope="module")
def opensim(spec_bytes) -> VectorPlant:
    return VectorPlant("opensim", spec_bytes)


@pytest.fixture(scope="module")
def pinocchio(spec_bytes) -> VectorPlant:
    return VectorPlant("pinocchio", spec_bytes)


@pytest.fixture(scope="module")
def record() -> dict[str, np.ndarray]:
    with np.load(EVIDENCE / "dynamics_record.npz") as data:
        return {key: data[key] for key in ("q", "v", "tau", "time_s")}


def _mass_matrix_pinocchio(plant: VectorPlant, q: np.ndarray) -> np.ndarray:
    model = plant._adapter
    raw = model.mass_matrix(plant._named(q))
    idx = [model._velocity_indices[name] for name in plant.coordinate_order]
    raw = raw[np.ix_(idx, idx)]
    return np.triu(raw) + np.triu(raw, 1).T  # crba fills the upper triangle


def test_coordinate_order_is_the_spec_order(opensim, pinocchio) -> None:
    assert opensim.coordinate_order == pinocchio.coordinate_order


def test_l0_total_mass_agrees(opensim, pinocchio) -> None:
    pin_mass = sum(
        pinocchio._adapter.model.inertias[i].mass
        for i in range(1, pinocchio._adapter.model.njoints)
    )
    assert opensim._adapter.mass_kg == pytest.approx(pin_mass, rel=MASS_RTOL)


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_l0_mass_matrix_and_kinematics_agree(opensim, pinocchio, seed) -> None:
    q = np.random.default_rng(seed).uniform(-0.6, 0.6, opensim.nv)
    mass_o = opensim._adapter.mass_matrix(opensim._named(q))
    mass_p = _mass_matrix_pinocchio(pinocchio, q)
    assert np.abs(mass_o - mass_p).max() <= MASS_RTOL * np.abs(mass_p).max()
    frames_o = opensim.kinematic_frames(q)
    frames_p = pinocchio.kinematic_frames(q)
    assert frames_o.keys() == frames_p.keys()
    for name, pose in frames_o.items():
        np.testing.assert_allclose(pose, frames_p[name], rtol=0, atol=1e-12)


@pytest.mark.parametrize("frame", FRAMES)
def test_l1_projection_matches_pinocchio(opensim, pinocchio, record, frame) -> None:
    q, v = record["q"][frame], record["v"][frame]
    result_o = project_to_closure(opensim, q, v)
    result_p = project_to_closure(pinocchio, q, v)
    assert result_o.pose_residual_after <= 1e-13
    assert result_o.rate_residual_after <= 1e-12
    np.testing.assert_allclose(result_o.q, result_p.q, rtol=0, atol=1e-12)
    np.testing.assert_allclose(result_o.v, result_p.v, rtol=0, atol=1e-12)


@pytest.mark.parametrize("frame", FRAMES)
def test_l1_accelerations_match_pinocchio(opensim, pinocchio, record, frame) -> None:
    state = project_to_closure(pinocchio, record["q"][frame], record["v"][frame])
    tau = record["tau"][frame]
    reference = pinocchio.acceleration(state.q, state.v, tau)
    bound = ACCELERATION_ATOL + ACCELERATION_RTOL * np.abs(reference).max()
    np.testing.assert_allclose(
        opensim.acceleration(state.q, state.v, tau), reference, rtol=0, atol=bound
    )


def test_closure_rate_residual_is_linear_in_the_rates(opensim, record) -> None:
    q = record["q"][400]
    v = record["v"][400]
    _, rate_full = opensim.closure_residuals(q, v)
    _, rate_half = opensim.closure_residuals(q, 0.5 * v)
    np.testing.assert_allclose(rate_half, 0.5 * rate_full, rtol=0, atol=1e-12)


def test_l2_open_loop_replay_matches_mujoco_reference(spec_bytes, record) -> None:
    bundle = generate_reference_bundle(
        spec_bytes, record["time_s"], record["q"], duration_s=0.03
    )
    rollout = open_loop(
        VectorPlant("opensim", bundle.spec_bytes),
        bundle.q0,
        bundle.v0,
        bundle.efforts,
        dt_s=bundle.dt_s,
    )
    np.testing.assert_allclose(rollout.q, bundle.reference_q, rtol=0, atol=1e-10)
    np.testing.assert_allclose(rollout.v, bundle.reference_v, rtol=0, atol=1e-8)


def test_adapter_rejects_an_unsupported_schema(spec_bytes) -> None:
    spec = json.loads(spec_bytes)
    spec["schema_version"] = "not-full-body"
    with pytest.raises(ValueError, match="schema"):
        VectorPlant("opensim", json.dumps(spec).encode())

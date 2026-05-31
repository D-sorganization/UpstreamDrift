"""JaxSim/Pinocchio free-floating dynamics parity gate.

The live test is intentionally tiny: one floating rigid body with unit mass
and unit body inertia. That makes the expected mass matrix insensitive to
joint-topology details while still exercising the JaxSim backend adapter,
velocity-representation normalization, Pinocchio free-flyer dynamics, and the
per-quantity tolerance policy from ``CROSS_ENGINE_PARITY_SPEC.md``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.jaxsim import JaxSimBackend

MASS_MATRIX_REL_TOL = 0.20
BIAS_FORCE_ABS_TOL = 1.0e-8
GRAVITY_REL_TOL = 0.20
CORIOLIS_ABS_TOL = 1.0e-8


@dataclass(frozen=True)
class DynamicsSnapshot:
    """Free-floating dynamics terms normalized to suite vector ordering."""

    mass_matrix: np.ndarray
    bias_forces: np.ndarray
    gravity_forces: np.ndarray
    coriolis_matrix: np.ndarray


def _relative_error(candidate: np.ndarray, reference: np.ndarray) -> float:
    numerator = float(np.linalg.norm(candidate - reference))
    denominator = float(np.linalg.norm(reference))
    if denominator == 0.0:
        return numerator
    return numerator / denominator


def assert_snapshot_within_parity(
    candidate: DynamicsSnapshot,
    reference: DynamicsSnapshot,
) -> None:
    """Assert dynamics terms meet the documented cross-engine tolerance policy."""

    assert candidate.mass_matrix.shape == reference.mass_matrix.shape
    assert candidate.bias_forces.shape == reference.bias_forces.shape
    assert candidate.gravity_forces.shape == reference.gravity_forces.shape
    assert candidate.coriolis_matrix.shape == reference.coriolis_matrix.shape

    mass_rel = _relative_error(candidate.mass_matrix, reference.mass_matrix)
    gravity_rel = _relative_error(candidate.gravity_forces, reference.gravity_forces)

    assert mass_rel <= MASS_MATRIX_REL_TOL
    np.testing.assert_allclose(
        candidate.bias_forces,
        reference.bias_forces,
        atol=BIAS_FORCE_ABS_TOL,
        rtol=0.0,
    )
    assert gravity_rel <= GRAVITY_REL_TOL
    np.testing.assert_allclose(
        candidate.coriolis_matrix,
        reference.coriolis_matrix,
        atol=CORIOLIS_ABS_TOL,
        rtol=0.0,
    )


def _jaxsim_snapshot() -> DynamicsSnapshot:
    backend = JaxSimBackend()
    backend.load_from_path(str(Path("tests/fixtures/jaxsim/single_link.sdf")))
    q, v = backend.get_state()
    backend.set_state(q, np.zeros_like(v))
    mass = backend.compute_mass_matrix()
    bias = backend.compute_bias_forces()
    gravity = backend.compute_gravity_forces()
    coriolis = backend.compute_coriolis_matrix()
    _assert_dynamics_invariants(mass, bias, gravity, coriolis)
    return DynamicsSnapshot(
        mass_matrix=mass,
        bias_forces=bias,
        gravity_forces=gravity,
        coriolis_matrix=coriolis,
    )


def _pinocchio_snapshot() -> DynamicsSnapshot:
    pinocchio = pytest.importorskip("pinocchio")

    model = pinocchio.Model()
    joint_id = model.addJoint(
        0,
        pinocchio.JointModelFreeFlyer(),
        pinocchio.SE3.Identity(),
        "base_joint",
    )
    model.appendBodyToJoint(
        joint_id,
        pinocchio.Inertia(1.0, np.zeros(3), np.eye(3)),
        pinocchio.SE3.Identity(),
    )
    data = model.createData()
    q = pinocchio.neutral(model)
    v = np.zeros(model.nv)

    mass = np.asarray(pinocchio.crba(model, data, q), dtype=np.float64)
    nonlinear = np.asarray(
        pinocchio.nonLinearEffects(model, data, q, v), dtype=np.float64
    )
    gravity = np.asarray(
        pinocchio.computeGeneralizedGravity(model, data, q), dtype=np.float64
    )
    bias = nonlinear - gravity
    coriolis = np.asarray(
        pinocchio.computeCoriolisMatrix(model, data, q, v), dtype=np.float64
    )
    _assert_dynamics_invariants(mass, bias, gravity, coriolis)
    return DynamicsSnapshot(
        mass_matrix=mass,
        bias_forces=bias,
        gravity_forces=gravity,
        coriolis_matrix=coriolis,
    )


def _assert_dynamics_invariants(
    mass: np.ndarray,
    bias: np.ndarray,
    gravity: np.ndarray,
    coriolis: np.ndarray,
) -> None:
    assert mass.shape == (6, 6)
    np.testing.assert_allclose(mass, mass.T, atol=1.0e-10)
    assert np.all(np.linalg.eigvalsh(mass) > 0.0)
    assert bias.shape == (6,)
    assert gravity.shape == (6,)
    assert coriolis.shape == (6, 6)
    assert np.all(np.isfinite(mass))
    assert np.all(np.isfinite(bias))
    assert np.all(np.isfinite(gravity))
    assert np.all(np.isfinite(coriolis))


def test_jaxsim_pinocchio_tolerance_policy_accepts_documented_envelope() -> None:
    reference = DynamicsSnapshot(
        mass_matrix=np.eye(6),
        bias_forces=np.zeros(6),
        gravity_forces=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.81]),
        coriolis_matrix=np.zeros((6, 6)),
    )
    candidate = DynamicsSnapshot(
        mass_matrix=np.eye(6) * 1.10,
        bias_forces=np.zeros(6),
        gravity_forces=np.array([0.0, 0.0, 0.0, 0.0, 0.0, -9.70]),
        coriolis_matrix=np.zeros((6, 6)),
    )

    assert_snapshot_within_parity(candidate, reference)


@pytest.mark.requires_jaxsim
@pytest.mark.requires_pinocchio
@pytest.mark.parity
def test_jaxsim_pinocchio_single_body_dynamics_equivalence() -> None:
    """Compare normalized JaxSim and Pinocchio terms for one free rigid body."""

    pytest.importorskip("jaxsim")
    pytest.importorskip("jaxsim.api")
    pytest.importorskip("pinocchio")

    jaxsim_snapshot = _jaxsim_snapshot()
    pinocchio_snapshot = _pinocchio_snapshot()

    assert_snapshot_within_parity(jaxsim_snapshot, pinocchio_snapshot)

"""Unit tests for CF-5: Forward and branched ZTCF qualification under epic #10286.

Tests:
1. CutState creation, immutability, and validation.
2. Explicit rejection of ZVCF forward rollouts.
3. Cut-instant consistency at t = t_cut (matching coordinates, rates, ZTCF acceleration, reaction wrench).
4. Evolving dynamics and branch immutability.
5. Mechanical energy conservation on unforced conservative systems.
6. Holonomic constraint closure maintenance along forward rollouts.
7. Spatial power, work, and impulse accounting along branches.
8. Conversion to engine-neutral CounterfactualTrajectory and InteractionEvidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any
import numpy as np
import pytest

from src.shared.python.motion_matching.counterfactual import (
    CutState,
    ForwardZTCFBranch,
    NativeConstrainedCounterfactualProvider,
    SpatialWrench,
    simulate_forward_ztcf,
    simulate_forward_zvcf,
)

pytestmark = pytest.mark.unit


class _HarmonicConstrainedModel:
    """Analytical 2-DOF constrained model: M qdd + K q = u + J^T lambda, with constraint q1 - q2 = 0.

    Has well-defined analytical kinetic, potential, and total mechanical energy.
    """

    def __init__(self, k_spring: float = 10.0, mass: float = 2.0) -> None:
        self.mass = mass
        self.k_spring = k_spring
        self.coordinates = ["theta1", "theta2"]

    def evaluate_dynamics(
        self,
        q: Mapping[str, float],
        v: Mapping[str, float],
        u: Mapping[str, float],
    ) -> tuple[dict[str, float], SpatialWrench]:
        q1, q2 = q["theta1"], q["theta2"]
        v1, v2 = v["theta1"], v["theta2"]
        u1, u2 = u.get("theta1", 0.0), u.get("theta2", 0.0)

        # Mass matrix M = diag(m, m)
        # Spring forces: F_spring = [-k*q1, -k*q2]
        # Constraint: q1 - q2 = 0 -> J = [1, -1]
        # KKT system:
        # [ m  0 -1 ] [ a1     ]   [ u1 - k*q1 ]
        # [ 0  m  1 ] [ a2     ] = [ u2 - k*q2 ]
        # [ 1 -1  0 ] [ lambda ]   [ 0         ]
        # From constraint: a1 = a2.
        # Adding row 1 and row 2: 2*m*a1 = (u1 + u2) - k*(q1 + q2)
        # a1 = a2 = ((u1 + u2) - k*(q1 + q2)) / (2*m)
        # lambda = m*a1 - (u1 - k*q1)
        a_val = ((u1 + u2) - self.k_spring * (q1 + q2)) / (2.0 * self.mass)
        lam = self.mass * a_val - (u1 - self.k_spring * q1)

        accels = {"theta1": float(a_val), "theta2": float(a_val)}
        wrench = SpatialWrench(
            force_N=np.array([float(lam), 0.0, 0.0]),
            torque_Nm=np.array([0.0, 0.0, float(lam * 0.5)]),
            point_of_application_m=np.array([0.0, 0.5, 0.0]),
            frame="world",
            action_direction="proximal_on_distal",
        )
        return accels, wrench

    def compute_energy(
        self, q: Mapping[str, float], v: Mapping[str, float]
    ) -> tuple[float, float, float]:
        """Return (kinetic_energy, potential_energy, mechanical_energy)."""
        q1, q2 = q["theta1"], q["theta2"]
        v1, v2 = v["theta1"], v["theta2"]
        ke = 0.5 * self.mass * (v1**2 + v2**2)
        pe = 0.5 * self.k_spring * (q1**2 + q2**2)
        return ke, pe, ke + pe

    def constraint_violation(
        self, q: Mapping[str, float], v: Mapping[str, float]
    ) -> float:
        """Return position constraint violation |q1 - q2|."""
        return abs(q["theta1"] - q["theta2"])


# ============================================================================
# 1. CutState Creation, Immutability & Validation
# ============================================================================


def test_cut_state_creation_and_immutability() -> None:
    q = {"theta1": 0.15, "theta2": 0.15}
    v = {"theta1": 1.5, "theta2": 1.5}
    cut = CutState(
        cut_time_s=0.35,
        coordinates=q,
        rates=v,
        parent_run_id="baseline_run_01",
        parent_sample_index=126,
    )

    assert cut.cut_time_s == 0.35
    assert cut.coordinates["theta1"] == 0.15
    assert cut.rates["theta2"] == 1.5
    assert cut.parent_run_id == "baseline_run_01"
    assert cut.parent_sample_index == 126

    # Immutable mapping views: modifications must raise
    with pytest.raises(TypeError):
        cut.coordinates["theta1"] = 99.0  # type: ignore[index]

    with pytest.raises(TypeError):
        cut.rates["theta2"] = 99.0  # type: ignore[index]


def test_cut_state_rejects_invalid_inputs() -> None:
    # Nonfinite cut time
    with pytest.raises(ValueError, match="finite"):
        CutState(cut_time_s=float("nan"), coordinates={"q": 0.0}, rates={"q": 0.0})

    # Mismatched coordinates and rates
    with pytest.raises(ValueError, match="match"):
        CutState(
            cut_time_s=0.1,
            coordinates={"q1": 0.0, "q2": 0.0},
            rates={"q1": 0.0},
        )

    # Nonfinite values
    with pytest.raises(ValueError, match="finite"):
        CutState(
            cut_time_s=0.1,
            coordinates={"q1": float("inf")},
            rates={"q1": 0.0},
        )


# ============================================================================
# 2. Rejection of ZVCF Forward Rollout
# ============================================================================


def test_reject_zvcf_forward_rollout() -> None:
    """ZVCF is strictly an instantaneous diagnostic and must not be treated as a rollout."""
    cut = CutState(cut_time_s=0.2, coordinates={"q": 0.0}, rates={"q": 0.0})
    model = _HarmonicConstrainedModel()

    with pytest.raises(ValueError, match="ZVCF.*instantaneous.*cannot be integrated"):
        simulate_forward_zvcf(model, cut, duration_s=0.1)

    with pytest.raises(ValueError, match="ZVCF.*instantaneous.*cannot be integrated"):
        simulate_forward_ztcf(model, cut, duration_s=0.1, intervention="zvcf")  # type: ignore[call-arg]


# ============================================================================
# 3. Cut-Instant Consistency at t = t_cut
# ============================================================================


def test_forward_ztcf_cut_instant_consistency() -> None:
    model = _HarmonicConstrainedModel(k_spring=20.0, mass=2.5)
    provider = NativeConstrainedCounterfactualProvider(model)

    q_cut = {"theta1": 0.2, "theta2": 0.2}
    v_cut = {"theta1": 0.8, "theta2": 0.8}
    cut = CutState(cut_time_s=0.4, coordinates=q_cut, rates=v_cut)

    # Evaluate expected pointwise ZTCF at the cut state
    pt_sample = provider.evaluate_pointwise(
        q_cut, v_cut, applied_efforts={"theta1": 10.0, "theta2": 5.0}, time_s=0.4
    )
    expected_ztcf_a = pt_sample.ztcf_acceleration
    expected_ztcf_w = pt_sample.ztcf_reaction_wrench

    # Simulate forward ZTCF branch
    branch: ForwardZTCFBranch = simulate_forward_ztcf(
        model, cut, duration_s=0.2, dt_s=0.01
    )

    assert branch.is_cut_consistent
    assert branch.time_s[0] == 0.4
    assert np.isclose(branch.coordinates[0, 0], q_cut["theta1"])
    assert np.isclose(branch.coordinates[0, 1], q_cut["theta2"])
    assert np.isclose(branch.rates[0, 0], v_cut["theta1"])
    assert np.isclose(branch.rates[0, 1], v_cut["theta2"])

    # Acceleration at t_cut must match pointwise ZTCF acceleration
    assert np.isclose(branch.accelerations[0, 0], expected_ztcf_a["theta1"])
    assert np.isclose(branch.accelerations[0, 1], expected_ztcf_a["theta2"])

    # Spatial reaction wrench at t_cut must match pointwise ZTCF wrench
    assert np.allclose(branch.reaction_wrenches[0], expected_ztcf_w.vector)


# ============================================================================
# 4. Evolving Dynamics & Branch Immutability
# ============================================================================


def test_forward_ztcf_evolving_dynamics_and_immutability() -> None:
    model = _HarmonicConstrainedModel(k_spring=15.0, mass=1.5)
    cut = CutState(
        cut_time_s=0.1,
        coordinates={"theta1": 0.3, "theta2": 0.3},
        rates={"theta1": 1.0, "theta2": 1.0},
    )

    branch = simulate_forward_ztcf(model, cut, duration_s=0.3, dt_s=0.005)

    n_samples = branch.time_s.size
    assert n_samples == 61  # (0.3 / 0.005) + 1
    assert branch.coordinates.shape == (n_samples, 2)
    assert branch.rates.shape == (n_samples, 2)
    assert branch.accelerations.shape == (n_samples, 2)
    assert branch.reaction_wrenches.shape == (n_samples, 6)

    # Dynamics must evolve over time (state changes from initial cut state)
    terminal_q = branch.coordinates[-1]
    initial_q = branch.coordinates[0]
    assert not np.allclose(terminal_q, initial_q)

    # Immutable arrays: writing to any array must raise ValueError
    with pytest.raises(ValueError):
        branch.coordinates[0, 0] = 99.0

    with pytest.raises(ValueError):
        branch.rates[0, 0] = 99.0

    with pytest.raises(ValueError):
        branch.accelerations[0, 0] = 99.0

    with pytest.raises(ValueError):
        branch.reaction_wrenches[0, 0] = 99.0


# ============================================================================
# 5. Mechanical Energy Conservation
# ============================================================================


def test_forward_ztcf_energy_conservation() -> None:
    """For an unforced conservative system, mechanical energy E = T + V must be conserved."""
    model = _HarmonicConstrainedModel(k_spring=25.0, mass=2.0)
    cut = CutState(
        cut_time_s=0.0,
        coordinates={"theta1": 0.25, "theta2": 0.25},
        rates={"theta1": 0.5, "theta2": 0.5},
    )

    branch = simulate_forward_ztcf(model, cut, duration_s=0.5, dt_s=0.002)

    assert branch.mechanical_energy_J is not None
    assert branch.kinetic_energy_J is not None
    assert branch.potential_energy_J is not None

    e0 = branch.mechanical_energy_J[0]
    energy_drift = np.abs(branch.mechanical_energy_J - e0)

    # RK4 energy drift on harmonic oscillator over 0.5s with dt=2ms should be < 1e-4 J
    assert np.max(energy_drift) < 1e-4

    # Delta E method
    delta_e = branch.energy_change()
    assert delta_e is not None
    assert np.isclose(delta_e[0], 0.0)
    assert np.max(np.abs(delta_e)) < 1e-4


# ============================================================================
# 6. Holonomic Constraint Closure Maintenance
# ============================================================================


def test_forward_ztcf_constraint_closure() -> None:
    """Along the rollout, loop constraint q1 - q2 = 0 must be maintained."""
    model = _HarmonicConstrainedModel(k_spring=20.0, mass=3.0)
    cut = CutState(
        cut_time_s=0.5,
        coordinates={"theta1": 0.1, "theta2": 0.1},
        rates={"theta1": -0.4, "theta2": -0.4},
    )

    branch = simulate_forward_ztcf(model, cut, duration_s=0.4, dt_s=0.005)

    # Constraint error |q1(t) - q2(t)|
    q_diff = np.abs(branch.coordinates[:, 0] - branch.coordinates[:, 1])
    assert np.max(q_diff) < 1e-12
    assert branch.max_constraint_violation < 1e-12
    assert branch.solver_converged


# ============================================================================
# 7. Spatial Power, Work, and Impulse Accounting Along Branch
# ============================================================================


def test_forward_ztcf_power_work_and_impulse() -> None:
    model = _HarmonicConstrainedModel(k_spring=10.0, mass=2.0)
    cut = CutState(
        cut_time_s=0.0,
        coordinates={"theta1": 0.2, "theta2": 0.2},
        rates={"theta1": 0.6, "theta2": 0.6},
    )

    branch = simulate_forward_ztcf(model, cut, duration_s=0.2, dt_s=0.01)

    power = branch.power()
    assert power.shape == branch.time_s.shape
    assert np.all(np.isfinite(power))

    work = branch.work()
    assert work.shape == branch.time_s.shape
    assert work[0] == 0.0
    assert np.all(np.isfinite(work))

    lin_imp = branch.linear_impulse()
    assert lin_imp.shape == (branch.time_s.size, 3)
    assert np.allclose(lin_imp[0], 0.0)

    ang_imp = branch.angular_impulse()
    assert ang_imp.shape == (branch.time_s.size, 3)
    assert np.allclose(ang_imp[0], 0.0)


# ============================================================================
# 8. Conversion to Standalone CounterfactualTrajectory
# ============================================================================


def test_forward_ztcf_to_counterfactual_trajectory() -> None:
    model = _HarmonicConstrainedModel(k_spring=10.0, mass=2.0)
    cut = CutState(
        cut_time_s=0.2,
        coordinates={"theta1": 0.1, "theta2": 0.1},
        rates={"theta1": 0.3, "theta2": 0.3},
        parent_run_id="run_harmonic",
    )

    branch = simulate_forward_ztcf(model, cut, duration_s=0.1, dt_s=0.01)
    traj = branch.to_counterfactual_trajectory()

    assert traj.parent_run_id == "run_harmonic"
    assert traj.time_s.size == branch.time_s.size
    assert np.allclose(traj.actual_accelerations, branch.accelerations)
    assert np.allclose(traj.ztcf_accelerations, branch.accelerations)
    assert np.allclose(traj.actual_wrenches, branch.reaction_wrenches)
    assert np.allclose(traj.ztcf_wrenches, branch.reaction_wrenches)

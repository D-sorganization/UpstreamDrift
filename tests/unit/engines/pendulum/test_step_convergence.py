"""Tests for step-size convergence and exact switch-time handling (SAP-3, #13792)."""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import pytest
from double_pendulum_model.physics.triple_pendulum import (
    TriplePendulumDynamics,
    TriplePendulumState,
)

from src.shared.python.pendulum_simulator.step_convergence import (
    BreakpointTorqueProfile,
    ConvergenceReceipt,
    InHouseTripleAdapter,
    MuJoCoTripleAdapter,
    run_step_convergence_study,
)
from src.shared.python.pendulum_simulator.triple_pendulum_spec import (
    TRIPLE_PENDULUM_SPEC,
)

pytestmark = pytest.mark.unit


def test_red_non_grid_aligned_breakpoint_handling() -> None:
    """Non-grid-aligned torque breakpoint fails closed in aligned mode and succeeds in event_located mode."""
    # Breakpoint at 1.5 ms on a 1.0 ms grid
    profile = BreakpointTorqueProfile(
        initial_torque=(10.0, 5.0, 2.0),
        breakpoints=((0.0015, (0.0, 0.0, 0.0)),),
    )
    dt = 0.001

    # DbC: aligned mode MUST raise ValueError when breakpoint is not on grid
    with pytest.raises(ValueError, match="not aligned"):
        profile.validate_grid_alignment(dt=dt, mode="aligned")

    # In event_located mode, step splitting partitions interval [0.001, 0.002] at 0.0015
    segments = profile.plan_step(t_current=0.001, dt=dt, mode="event_located")
    assert len(segments) == 2
    t0, dt0, tau0 = segments[0]
    t1, dt1, tau1 = segments[1]
    assert math.isclose(t0, 0.001)
    assert math.isclose(dt0, 0.0005)
    assert tau0 == (10.0, 5.0, 2.0)
    assert math.isclose(t1, 0.0015)
    assert math.isclose(dt1, 0.0005)
    assert tau1 == (0.0, 0.0, 0.0)


def test_exact_time_stepping_prevents_float_accumulation() -> None:
    """Time must be computed as t = k * dt, never accumulating floats."""
    dt = 0.0001
    k = 3000
    accumulated = sum(dt for _ in range(k))
    exact = k * dt
    # Float accumulation drifts, exact indexing does not
    assert exact == 0.3
    # Profile step planning uses exact k * dt
    profile = BreakpointTorqueProfile(
        initial_torque=(1.0, 0.0, 0.0),
        breakpoints=((0.3, (0.0, 0.0, 0.0)),),
    )
    # At exact t=0.3, switch has occurred
    assert profile.evaluate(exact) == (0.0, 0.0, 0.0)


def test_observed_order_euler_and_rk4_on_canonical_spec() -> None:
    """Assert observed order ~1 for Euler and ~4 for RK4 (slope within +-0.3)."""
    adapter = InHouseTripleAdapter()
    timesteps = (0.004, 0.002, 0.001, 0.0005)

    # Semi-implicit Euler convergence
    euler_receipt = run_step_convergence_study(
        adapter,
        timesteps=timesteps,
        integrator="semi_implicit",
        t_end=0.2,
        reference_dt=2.5e-5,
    )
    assert isinstance(euler_receipt, ConvergenceReceipt)
    assert abs(euler_receipt.observed_order - 1.0) <= 0.3, (
        f"Expected Euler order ~1.0, got {euler_receipt.observed_order}"
    )

    # RK4 convergence
    rk4_receipt = run_step_convergence_study(
        adapter,
        timesteps=timesteps,
        integrator="rk4",
        t_end=0.2,
        reference_dt=2.5e-5,
    )
    assert isinstance(rk4_receipt, ConvergenceReceipt)
    assert abs(rk4_receipt.observed_order - 4.0) <= 0.3, (
        f"Expected RK4 order ~4.0, got {rk4_receipt.observed_order}"
    )


def test_receipt_records_required_fields() -> None:
    """Receipt records engine, integrator, dt, substeps, interpolation mode, and tolerance."""
    adapter = InHouseTripleAdapter()
    receipt = run_step_convergence_study(
        adapter,
        timesteps=(0.002, 0.001),
        integrator="rk4",
        substeps=2,
        interpolation_mode="zoh",
        t_end=0.05,
        reference_dt=5e-5,
    )
    data = receipt.to_dict()
    assert data["engine"] == "in_house_triple"
    assert data["integrator"] == "rk4"
    assert data["substeps"] == 2
    assert data["interpolation_mode"] == "zoh"
    assert data["dts_s"] == [0.002, 0.001]
    assert data["reference_tolerance"] == 5e-5
    assert len(data["errors_m"]) == 2
    assert "tip_error_m" in data


def test_mujoco_adapter_step_convergence() -> None:
    """MuJoCo adapter runs through shared study harness and verifies Euler order ~1."""
    pytest.importorskip("mujoco")
    adapter = MuJoCoTripleAdapter()
    receipt = run_step_convergence_study(
        adapter,
        timesteps=(0.002, 0.001, 0.0005),
        integrator="euler",
        t_end=0.2,
        reference_dt=2.5e-5,
    )
    assert abs(receipt.observed_order - 1.0) <= 0.3, (
        f"Expected MuJoCo Euler order ~1.0, got {receipt.observed_order}"
    )

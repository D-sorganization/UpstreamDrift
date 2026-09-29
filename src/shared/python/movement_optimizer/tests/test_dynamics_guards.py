# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Guards on the batch/scalar dynamics: finite inputs and slow-movement Coriolis.

Covers:
    * #499 — batch inverse dynamics rejects non-finite inputs with a clear error
      instead of silently propagating NaN torques.
    * #491 — a velocity-threshold warning fires when the slow-movement Coriolis
      assumption (cross-velocity terms omitted) is violated.
"""

from __future__ import annotations

import logging

import numpy as np
import pytest

from movement_optimizer.constants import CORIOLIS_SLOW_LIMIT_RAD_S
from movement_optimizer.models.body_model import BodyModel
from movement_optimizer.models.lagrangian_dynamics import LagrangianDynamics


@pytest.fixture
def squat_dyn() -> LagrangianDynamics:
    body = BodyModel(height=1.8, body_mass=80.0)
    return LagrangianDynamics(body, body.m_squat.copy(), body.I_squat.copy(), 0.0)


@pytest.mark.parametrize("which", ["q", "qd", "qdd"])
def test_batch_rejects_nan_inputs(squat_dyn: LagrangianDynamics, which: str) -> None:
    n = 5
    q = np.zeros((n, 3))
    qd = np.zeros((n, 3))
    qdd = np.zeros((n, 3))
    arrays = {"q": q, "qd": qd, "qdd": qdd}
    arrays[which][2, 1] = np.nan

    with pytest.raises(ValueError, match=rf"non-finite value\(s\) in '{which}'"):
        squat_dyn.inverse_dynamics_batch(q, qd, qdd)


def test_batch_rejects_inf_inputs(squat_dyn: LagrangianDynamics) -> None:
    n = 4
    q = np.zeros((n, 3))
    qd = np.zeros((n, 3))
    qdd = np.zeros((n, 3))
    qd[0, 0] = np.inf

    with pytest.raises(ValueError, match="non-finite"):
        squat_dyn.inverse_dynamics_batch(q, qd, qdd)


def test_batch_accepts_finite_inputs(squat_dyn: LagrangianDynamics) -> None:
    rng = np.random.default_rng(0)
    q = rng.standard_normal((6, 3))
    qd = rng.standard_normal((6, 3)) * 0.1
    qdd = rng.standard_normal((6, 3))
    tau = squat_dyn.inverse_dynamics_batch(q, qd, qdd)
    assert tau.shape == (6, 3)
    assert np.isfinite(tau).all()


def test_slow_movement_does_not_warn(
    squat_dyn: LagrangianDynamics, caplog: pytest.LogCaptureFixture
) -> None:
    q = np.array([0.2, 0.1, 0.0])
    qd = np.full(3, CORIOLIS_SLOW_LIMIT_RAD_S * 0.5)
    qdd = np.zeros(3)
    with caplog.at_level(logging.WARNING):
        squat_dyn.inverse_dynamics(q, qd, qdd)
    assert "slow-movement" not in caplog.text


def test_fast_movement_warns_once(
    squat_dyn: LagrangianDynamics, caplog: pytest.LogCaptureFixture
) -> None:
    q = np.array([0.2, 0.1, 0.0])
    fast = CORIOLIS_SLOW_LIMIT_RAD_S + 1.0
    qd = np.array([fast, 0.0, 0.0])
    qdd = np.zeros(3)
    with caplog.at_level(logging.WARNING):
        squat_dyn.inverse_dynamics(q, qd, qdd)
        squat_dyn.inverse_dynamics(q, qd, qdd)
    warnings = [r for r in caplog.records if "slow-movement" in r.message]
    assert len(warnings) == 1
    assert "may be underestimated" in warnings[0].message


def test_fast_movement_warns_in_batch(
    squat_dyn: LagrangianDynamics, caplog: pytest.LogCaptureFixture
) -> None:
    n = 3
    q = np.zeros((n, 3))
    qd = np.zeros((n, 3))
    qd[1, 2] = CORIOLIS_SLOW_LIMIT_RAD_S + 0.5
    qdd = np.zeros((n, 3))
    with caplog.at_level(logging.WARNING):
        squat_dyn.inverse_dynamics_batch(q, qd, qdd)
    assert any("slow-movement" in r.message for r in caplog.records)

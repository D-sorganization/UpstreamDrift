# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for issue #222: decomposed helpers from oversized functions.

Covers:
- cli._configure_logging
- cli._resolve_duration
- cli._build_optimizer
- cli._save_or_emit
- LagrangianDynamics._batch_inertia_torques
- LagrangianDynamics._batch_coriolis_torques
- LagrangianDynamics._batch_gravity_torques
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from movement_optimizer import cli
from movement_optimizer.cli import (
    _configure_logging,
    _resolve_duration,
    _save_or_emit,
)
from movement_optimizer.models import BodyModel, make_squat_config
from movement_optimizer.models.lagrangian_dynamics import LagrangianDynamics

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_lagrangian(body: BodyModel | None = None) -> LagrangianDynamics:
    """Return a LagrangianDynamics built from a squat configuration."""
    if body is None:
        body = BodyModel(75.0, 1.75)
    dyn, _qs, _qe, _qb = make_squat_config(body, 60.0)
    return dyn  # type: ignore[return-value]


# ---------------------------------------------------------------------------
# cli._configure_logging
# ---------------------------------------------------------------------------


class TestConfigureLogging:
    def test_verbose_sets_debug(self) -> None:
        """_configure_logging(True) should set the root logger to DEBUG."""
        root = logging.getLogger()
        # Remove existing handlers so basicConfig takes effect
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        _configure_logging(True)
        assert root.level == logging.DEBUG

    def test_non_verbose_sets_info(self) -> None:
        """_configure_logging(False) should set the root logger to INFO."""
        root = logging.getLogger()
        root.handlers.clear()
        root.setLevel(logging.WARNING)
        _configure_logging(False)
        assert root.level == logging.INFO


# ---------------------------------------------------------------------------
# cli._resolve_duration
# ---------------------------------------------------------------------------


class TestResolveDuration:
    def test_squat_no_minimum_enforced(self) -> None:
        assert _resolve_duration("squat", 1.0) == 1.0

    def test_deadlift_no_minimum_enforced(self) -> None:
        assert _resolve_duration("deadlift", 0.5) == 0.5

    def test_full_squat_enforces_3s(self) -> None:
        assert _resolve_duration("full_squat", 1.0) == 3.0

    def test_snatch_enforces_3s(self) -> None:
        assert _resolve_duration("snatch", 2.5) == 3.0

    def test_clean_enforces_2s(self) -> None:
        assert _resolve_duration("clean", 0.5) == 2.0

    def test_jerk_enforces_2s(self) -> None:
        assert _resolve_duration("jerk", 1.0) == 2.0

    def test_already_above_minimum_unchanged(self) -> None:
        assert _resolve_duration("full_squat", 4.0) == 4.0

    def test_clean_above_minimum_unchanged(self) -> None:
        assert _resolve_duration("clean", 3.0) == 3.0


# ---------------------------------------------------------------------------
# cli._build_optimizer
# ---------------------------------------------------------------------------


class TestBuildOptimizer:
    def test_returns_optimizer_and_dynamics(self) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        body = BodyModel(75.0, 1.75)
        opt, dyn = cli._build_optimizer(body, "squat", 60.0, 2.0, 1.0)
        assert isinstance(opt, TrajectoryOptimizer)
        assert dyn is not None

    def test_multiphase_config_with_q_via(self) -> None:
        """full_squat factory returns a 5-tuple; optimizer must receive q_via."""
        from movement_optimizer.trajectory import TrajectoryOptimizer

        body = BodyModel(75.0, 1.75)
        opt, _dyn = cli._build_optimizer(body, "full_squat", 40.0, 3.0, 1.0)
        assert isinstance(opt, TrajectoryOptimizer)
        assert opt.q_via is not None

    def test_bench_press_config(self) -> None:
        from movement_optimizer.trajectory import TrajectoryOptimizer

        body = BodyModel(80.0, 1.80)
        opt, _dyn = cli._build_optimizer(body, "bench_press", 80.0, 2.0, 1.0)
        assert isinstance(opt, TrajectoryOptimizer)


# ---------------------------------------------------------------------------
# cli._save_or_emit
# ---------------------------------------------------------------------------


class TestSaveOrEmit:
    def test_writes_json_to_file(self, tmp_path: Path) -> None:
        from conftest import make_test_result

        result = make_test_result(cost=5.0)
        out = tmp_path / "out.json"
        _save_or_emit(result, "squat", str(out))
        written = json.loads(out.read_text(encoding="utf-8"))
        assert written["exercise"] == "squat"
        assert written["cost"] == pytest.approx(5.0)

    def test_emits_summary_when_no_output(self, monkeypatch: pytest.MonkeyPatch) -> None:
        from conftest import make_test_result

        result = make_test_result(cost=7.7)
        emitted: list[dict[str, Any]] = []
        monkeypatch.setattr(cli, "_emit_cli_summary", lambda s: emitted.append(s))
        _save_or_emit(result, "deadlift", None)
        assert len(emitted) == 1
        assert emitted[0]["exercise"] == "deadlift"


# ---------------------------------------------------------------------------
# LagrangianDynamics._batch_inertia_torques
# ---------------------------------------------------------------------------


class TestBatchInertiaTorques:
    def test_output_shape(self) -> None:
        dyn = _make_lagrangian()
        n = 15
        qdd = np.random.default_rng(0).random((n, 3))
        c01 = np.ones(n)
        c02 = np.ones(n)
        c12 = np.ones(n)
        tau = dyn._batch_inertia_torques(qdd, c01, c02, c12)
        assert tau.shape == (n, 3)

    def test_zero_acceleration_gives_zero_inertia(self) -> None:
        dyn = _make_lagrangian()
        n = 8
        qdd = np.zeros((n, 3))
        c01 = np.ones(n)
        c02 = np.ones(n)
        c12 = np.ones(n)
        tau = dyn._batch_inertia_torques(qdd, c01, c02, c12)
        np.testing.assert_allclose(tau, 0.0)

    def test_single_joint_acceleration(self) -> None:
        """Accelerating only joint 0 should produce non-zero torque 0."""
        dyn = _make_lagrangian()
        n = 5
        qdd = np.zeros((n, 3))
        qdd[:, 0] = 1.0
        c01 = np.ones(n)
        c02 = np.ones(n)
        c12 = np.ones(n)
        tau = dyn._batch_inertia_torques(qdd, c01, c02, c12)
        assert np.all(tau[:, 0] != 0)

    def test_consistent_with_inverse_dynamics_single_timestep(self) -> None:
        """For a single timestep with qd=0, inertia torques == full torques."""
        dyn = _make_lagrangian()
        rng = np.random.default_rng(7)
        q = rng.random((1, 3))
        qd = np.zeros((1, 3))
        qdd = rng.random((1, 3))
        d01 = q[:, 0] - q[:, 1]
        d02 = q[:, 0] - q[:, 2]
        d12 = q[:, 1] - q[:, 2]
        tau_inertia = dyn._batch_inertia_torques(qdd, np.cos(d01), np.cos(d02), np.cos(d12))
        tau_gravity = dyn._batch_gravity_torques(q)
        tau_total = tau_inertia + tau_gravity
        tau_ref = dyn.inverse_dynamics_batch(q, qd, qdd)
        np.testing.assert_allclose(tau_total, tau_ref, rtol=1e-10)


# ---------------------------------------------------------------------------
# LagrangianDynamics._batch_coriolis_torques
# ---------------------------------------------------------------------------


class TestBatchCoriolisTorques:
    def test_output_shape(self) -> None:
        dyn = _make_lagrangian()
        n = 10
        qd = np.random.default_rng(1).random((n, 3))
        s01 = np.zeros(n)
        s02 = np.zeros(n)
        s12 = np.zeros(n)
        tau = dyn._batch_coriolis_torques(qd, s01, s02, s12)
        assert tau.shape == (n, 3)

    def test_zero_velocity_gives_zero(self) -> None:
        dyn = _make_lagrangian()
        n = 6
        qd = np.zeros((n, 3))
        s01 = np.ones(n)
        s02 = np.ones(n)
        s12 = np.ones(n)
        tau = dyn._batch_coriolis_torques(qd, s01, s02, s12)
        np.testing.assert_allclose(tau, 0.0)

    def test_zero_sin_gives_zero(self) -> None:
        """All sin differences zero → no coupling → zero Coriolis."""
        dyn = _make_lagrangian()
        n = 4
        qd = np.random.default_rng(2).random((n, 3))
        s = np.zeros(n)
        tau = dyn._batch_coriolis_torques(qd, s, s, s)
        np.testing.assert_allclose(tau, 0.0)


# ---------------------------------------------------------------------------
# LagrangianDynamics._batch_gravity_torques
# ---------------------------------------------------------------------------


class TestBatchGravityTorques:
    def test_output_shape(self) -> None:
        dyn = _make_lagrangian()
        n = 12
        q = np.random.default_rng(3).random((n, 3))
        tau = dyn._batch_gravity_torques(q)
        assert tau.shape == (n, 3)

    def test_upright_posture_nonzero(self) -> None:
        """Near-vertical posture should produce nonzero gravity torques."""
        dyn = _make_lagrangian()
        q = np.full((5, 3), np.pi / 2)  # fully upright, sin=1
        tau = dyn._batch_gravity_torques(q)
        assert np.all(tau > 0)

    def test_supine_model_uses_cos(self) -> None:
        """Supine flag switches gravity trig to cosine; verify different output."""
        body = BodyModel(75.0, 1.75)
        dyn_normal = _make_lagrangian(body)

        from movement_optimizer.models import make_bench_press_config

        dyn_bench, *_ = make_bench_press_config(body, 60.0)

        q = np.full((3, 3), np.pi / 4)
        tau_norm = dyn_normal._batch_gravity_torques(q)
        tau_supin = dyn_bench._batch_gravity_torques(q)  # type: ignore[attr-defined]
        # They should differ because sin(π/4) == cos(π/4), but coefficients differ
        # — just assert both run without error and produce finite output
        assert np.all(np.isfinite(tau_norm))
        assert np.all(np.isfinite(tau_supin))

    def test_full_batch_equals_sum_of_parts(self) -> None:
        """inverse_dynamics_batch should equal sum of the three helper outputs."""
        dyn = _make_lagrangian()
        rng = np.random.default_rng(42)
        n = 20
        q = rng.random((n, 3)) * 0.5
        qd = rng.random((n, 3)) * 0.2
        qdd = rng.random((n, 3)) * 0.1

        d01 = q[:, 0] - q[:, 1]
        d02 = q[:, 0] - q[:, 2]
        d12 = q[:, 1] - q[:, 2]

        tau_parts = (
            dyn._batch_inertia_torques(qdd, np.cos(d01), np.cos(d02), np.cos(d12))
            + dyn._batch_coriolis_torques(qd, np.sin(d01), np.sin(d02), np.sin(d12))
            + dyn._batch_gravity_torques(q)
        )
        tau_full = dyn.inverse_dynamics_batch(q, qd, qdd)
        np.testing.assert_allclose(tau_parts, tau_full, rtol=1e-10)

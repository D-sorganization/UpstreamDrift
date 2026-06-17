# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Shared test fixtures."""

from __future__ import annotations

# ---------------------------------------------------------------------------
# Fleet Testing Standards §5: environment setup BEFORE heavy imports.
# See: D-sorganization/Repository_Management docs/FLEET_TESTING_STANDARDS.md
# ---------------------------------------------------------------------------
import os

# C-extension thread safety. Movement-Optimizer uses numpy/scipy (MKL/OpenBLAS)
# heavily; pin to single-threaded to avoid xdist worker crashes from MKL/
# OpenBLAS forking. Production code can re-thread itself if needed.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

# matplotlib headless backend, set before any matplotlib import.
os.environ.setdefault("MPLBACKEND", "Agg")

# Qt headless backend, since this repo's GUI is PyQt6.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np
import pytest

from movement_optimizer.models import (
    BodyModel,
    JointTorqueSet,
    make_bench_press_config,
    make_bench_press_torque_set,
    make_deadlift_config,
    make_default_torque_set,
    make_full_squat_config,
    make_squat_config,
)
from movement_optimizer.trajectory import OptimizationResult, TrajectoryOptimizer


@pytest.fixture(scope="session")
def qapp():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance()
    if app is None:
        app = QApplication([])
    return app


def make_test_result(seed: int = 42, cost: float = 42.5) -> OptimizationResult:
    """Create a minimal OptimizationResult for testing.

    Shared helper to avoid duplicating this factory across test modules.
    """
    n = 10
    rng = np.random.default_rng(seed)
    t = np.linspace(0, 2, n)
    q = rng.random((n, 3))
    qd = rng.random((n, 3))
    qdd = rng.random((n, 3))
    torques = rng.random((n, 3))
    power = torques * qd
    com = rng.random((n, 2))
    bar = rng.random((n, 2))
    return OptimizationResult(
        t=t,
        q=q,
        qd=qd,
        qdd=qdd,
        torques=torques,
        power=power,
        com=com,
        bar=bar,
        success=True,
        cost=cost,
        com_horizontal_range_cm=3.2,
        elapsed_s=1.5,
        n_evals=100,
    )


@pytest.fixture()
def default_body() -> BodyModel:
    return BodyModel(75.0, 1.75)


@pytest.fixture()
def custom_body() -> BodyModel:
    return BodyModel(
        80.0,
        1.80,
        seg_multipliers={"lower_leg": 1.1, "upper_leg": 0.9, "torso": 1.05},
    )


@pytest.fixture()
def squat_dynamics(default_body: BodyModel):
    return make_squat_config(default_body, 60.0)


@pytest.fixture()
def deadlift_dynamics(default_body: BodyModel):
    return make_deadlift_config(default_body, 60.0)


@pytest.fixture()
def full_squat_config(default_body: BodyModel):
    return make_full_squat_config(default_body, 60.0)


@pytest.fixture()
def bench_press_config(default_body: BodyModel):
    return make_bench_press_config(default_body, 60.0)


@pytest.fixture()
def default_torque_set() -> JointTorqueSet:
    return make_default_torque_set()


@pytest.fixture()
def bench_torque_set() -> JointTorqueSet:
    return make_bench_press_torque_set()


@pytest.fixture()
def squat_optimizer():
    body = BodyModel(75.0, 1.75)
    dyn, qs, qe, qb = make_squat_config(body, 60.0)
    opt = TrajectoryOptimizer(
        body,
        dyn,
        "squat",
        60.0,
        qs,
        qe,
        qb,
        duration=2.0,
        n_waypoints=6,
        n_eval=20,
        n_starts=1,
    )
    return opt, body, dyn, qs, qe


@pytest.fixture()
def full_squat_optimizer():
    body = BodyModel(75.0, 1.75)
    dyn, qs, qe, qb, q_via = make_full_squat_config(body, 60.0)
    opt = TrajectoryOptimizer(
        body,
        dyn,
        "full_squat",
        60.0,
        qs,
        qe,
        qb,
        q_via=q_via,
        duration=4.0,
        n_waypoints=8,
        n_eval=20,
        n_starts=1,
    )
    return opt, body, dyn

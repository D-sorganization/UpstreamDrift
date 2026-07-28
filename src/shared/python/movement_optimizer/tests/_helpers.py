# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Importable test helpers for the movement_optimizer suite.

This lives outside ``conftest.py`` on purpose: the bare module name
``conftest`` resolves to the *repository root* conftest at import time
(the rootdir is first on ``pythonpath``), so ``from conftest import
make_test_result`` silently binds to the wrong module and fails.
Import from this module instead.
"""

from __future__ import annotations

import numpy as np

from movement_optimizer.trajectory import OptimizationResult


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

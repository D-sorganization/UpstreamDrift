# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Regression guards for issue #498: behavioural constants must live in the
constants/tuning modules and be referenced (not re-hardcoded) at the use sites.

These tests assert the *wiring* — that the use sites read the named constants —
so a future edit that re-inlines a literal is caught.
"""

from __future__ import annotations

import inspect

import numpy as np

from movement_optimizer import constants as c
from movement_optimizer.gui import _sidebar_state
from movement_optimizer.trajectory import optimizer, optimizer_cost, tuning


def test_endpoint_constants_exist_with_expected_values() -> None:
    assert tuning.ENDPOINT_ACCEL_WEIGHT_RATIO == 0.1
    assert tuning.ENDPOINT_DAMP_SAMPLE_FRACTION == 0.125
    assert tuning.ENDPOINT_DAMP_MIN_SAMPLES == 2


def test_progress_and_layout_constants_exist() -> None:
    assert c.PROGRESS_MAX_PCT == 95
    assert c.PROGRESS_EVAL_SCALE == 500.0
    assert c.PROGRESS_PHASE_BOUNDARY_EVALS == 200
    assert c.STALL_HINT_ELAPSED_S == 120.0
    assert c.PLOT_GRID_ROWS == 3
    assert c.PLOT_GRID_COLS == 4
    assert c.PLOT_GRID_HEIGHT_RATIOS == (3, 1, 1)
    assert set(c.PLOT_GRID_MARGINS) == {"left", "right", "top", "bottom"}


def test_n_damp_fraction_matches_legacy_divisor() -> None:
    # int(n * 0.125) must equal the historical n // 8 for all positive ints.
    f = tuning.ENDPOINT_DAMP_SAMPLE_FRACTION
    m = tuning.ENDPOINT_DAMP_MIN_SAMPLES
    for n in (7, 8, 16, 20, 40, 60, 100):
        assert max(m, int(n * f)) == max(2, n // 8)


def test_use_sites_reference_constants_not_literals() -> None:
    # optimizer_cost uses the endpoint accel ratio symbol, not a bare 0.1.
    src = inspect.getsource(optimizer_cost.compute_endpoint_damping_cost)
    assert "ENDPOINT_ACCEL_WEIGHT_RATIO" in src

    # optimizer wires n_damp from the tuning constants.
    opt_src = inspect.getsource(optimizer.TrajectoryOptimizer.__init__)
    assert "ENDPOINT_DAMP_SAMPLE_FRACTION" in opt_src
    assert "// 8" not in opt_src

    # sidebar progress text reads the progress constants.
    sb_src = inspect.getsource(_sidebar_state.update_progress)
    assert "PROGRESS_MAX_PCT" in sb_src
    assert "PROGRESS_PHASE_BOUNDARY_EVALS" in sb_src


def test_plot_grid_height_ratios_sum_is_stable() -> None:
    # Guards the animation-row-dominant layout (tall top row).
    ratios = np.asarray(c.PLOT_GRID_HEIGHT_RATIOS, dtype=float)
    assert ratios[0] > ratios[1:].max()

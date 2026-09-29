"""Coverage tests for ``motion_matching.fit_result.CanonicalFitResult``.

Pin every legacy alias property so removing one would fail loudly.
"""

from __future__ import annotations

import warnings

import numpy as np
import pytest
from src.shared.python.motion_matching.cost import CostBreakdown
from src.shared.python.motion_matching.fit_result import CanonicalFitResult


def _make() -> CanonicalFitResult:
    return CanonicalFitResult(
        theta_optimal=np.zeros(7, dtype=np.float64),
        final_cost=1.5,
        final_rmse_m=0.01,
        solver_status="success",
        iterations=42,
        n_evaluations=99,
        wall_clock_s=3.14,
        message="ok",
        history=(2.0, 1.5),
        method="lbfgs",
        git_commit="deadbeef",
        engine_version="0.1",
        target_hash="abc",
        timestamp_utc="2024-01-01T00:00:00Z",
        cost_breakdown=CostBreakdown(
            position=0.5,
            orientation=0.5,
            impact_anchor=0.5,
            body_marker=0.0,
            regularizer=0.0,
            total=1.5,
        ),
    )


def _check_deprecation(fn, *args):
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        out = fn(*args)
    assert any(issubclass(x.category, DeprecationWarning) for x in w)
    return out


@pytest.mark.parametrize(
    "prop, expected",
    [
        ("coefficients", np.zeros(7, dtype=np.float64)),
        ("theta", np.zeros(7, dtype=np.float64)),
        ("mujoco_version", "0.1"),
        ("cost", 1.5),
        ("n_iter", 42),
        ("n_eval", 99),
        ("n_evals", 99),
        ("success", True),
        ("duration_s", 3.14),
        ("elapsed_s", 3.14),
        ("solver", "lbfgs"),
    ],
)
def test_deprecated_aliases_emit_warning(prop: str, expected) -> None:
    """Pin: every legacy alias emits ``DeprecationWarning`` and forwards."""
    res = _make()
    out = _check_deprecation(lambda: getattr(res, prop))
    if isinstance(expected, np.ndarray):
        assert np.array_equal(out, expected)
    else:
        assert out == expected


def test_success_false_when_status_not_success() -> None:
    """Pin: ``success`` is False unless solver_status == 'success'."""
    res = CanonicalFitResult(
        theta_optimal=np.zeros(7, dtype=np.float64),
        final_cost=0.0,
        final_rmse_m=0.0,
        solver_status="failed",
        iterations=0,
        n_evaluations=0,
        wall_clock_s=0.0,
        message="",
        history=(),
        method="m",
        git_commit="g",
        engine_version="v",
        target_hash="h",
        timestamp_utc="2024-01-01T00:00:00Z",
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        assert res.success is False

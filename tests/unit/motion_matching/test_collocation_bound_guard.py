"""Returned optimizer states must satisfy independently checked chart bounds."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("scale", ["0", "-1", "nan", "inf"])
def test_invalid_closure_scale_fails_before_loading_model(
    monkeypatch: pytest.MonkeyPatch, scale: str
) -> None:
    runner = Path(__file__).resolve().parents[3] / (
        "docs/development/simscape_tour_matching/probe_retracted_native_collocation.py"
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            str(runner),
            "--model",
            "missing.json",
            "--path",
            "missing.json",
            "--output",
            "unused.json",
            "--closure-scale",
            scale,
        ],
    )
    with pytest.raises(ValueError, match="Closure scale"):
        runpy.run_path(str(runner))["main"]()


def test_bound_guard_rejects_historical_infeasible_candidate() -> None:
    runner = Path(__file__).resolve().parents[3] / (
        "docs/development/simscape_tour_matching/probe_retracted_native_collocation.py"
    )
    guard = runpy.run_path(str(runner))["validate_chart_bounds"]
    for values in ([0.07306001724], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="chart bounds"):
            guard(np.asarray(values), 0.01)
    assert guard(np.array([-0.01, 0.0, 0.01]), 0.01) == 0.01


def test_clamped_path_preserves_initial_rate_and_node_sensitivity() -> None:
    runner = Path(__file__).resolve().parents[3] / (
        "docs/development/simscape_tour_matching/probe_retracted_native_collocation.py"
    )
    spline = runpy.run_path(str(runner))["make_path_spline"]
    times = np.array([0.0, 0.05, 0.1, 0.15])
    q = np.array([[1.0], [1.2], [1.4], [1.6]])
    rate = np.array([0.7])
    actual = spline(times, q, rate)
    np.testing.assert_allclose(actual(times[0], 1), rate, atol=1e-12)
    maps = spline(times, np.eye(4), np.zeros(4))
    perturbed = q.copy()
    perturbed[2] += 1e-5
    finite_difference = (
        spline(times, perturbed, rate)(times, 1) - actual(times, 1)
    ) / 1e-5
    np.testing.assert_allclose(finite_difference[:, 0], maps(times, 1)[:, 2], atol=1e-8)

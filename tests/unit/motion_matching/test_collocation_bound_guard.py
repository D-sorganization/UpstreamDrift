"""Returned optimizer states must satisfy independently checked chart bounds."""

import runpy
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_bound_guard_rejects_historical_infeasible_candidate() -> None:
    runner = Path(__file__).resolve().parents[3] / (
        "docs/development/simscape_tour_matching/probe_retracted_native_collocation.py"
    )
    guard = runpy.run_path(str(runner))["validate_chart_bounds"]
    for values in ([0.07306001724], [np.nan], [np.inf]):
        with pytest.raises(ValueError, match="chart bounds"):
            guard(np.asarray(values), 0.01)
    assert guard(np.array([-0.01, 0.0, 0.01]), 0.01) == 0.01

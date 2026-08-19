"""Research tests for parameter uncertainty and sensitivity analysis on articulated tier (#8752).

Verifies Latin Hypercube Sampling parameter coverage, PRCC sensitivity bounds,
and failure map generation across uncertain physiological, grip, and shaft parameters.
"""

from __future__ import annotations

import numpy as np
import pytest

pytestmark = [pytest.mark.scientific]

from scripts.research.proximal_distal_energy.articulated_uncertainty_study import (
    ArticulatedUncertaintyConfig,
    OUTPUT_METRICS,
    UNCERTAINTY_PARAMETERS,
    run_articulated_uncertainty_study,
)


def test_articulated_uncertainty_sweep_and_prcc_sensitivity() -> None:
    """Uncertainty sweep must execute, maintain energy closure, and compute valid PRCC."""
    config = ArticulatedUncertaintyConfig(
        sample_count=20,
        seed=1234,
        duration_s=0.01,
        time_step_s=0.001,
    )
    record, arrays = run_articulated_uncertainty_study(config)

    assert record["results"]["sample_count"] == 20
    assert record["results"]["all_simulations_energy_closed"] is True

    # Verify PRCC matrix shape and bounds in [-1, 1]
    prcc = arrays["prcc_sensitivity_matrix"]
    assert prcc.shape == (len(OUTPUT_METRICS), len(UNCERTAINTY_PARAMETERS))
    assert np.all(prcc >= -1.0 - 1e-12)
    assert np.all(prcc <= 1.0 + 1e-12)

    # Verify failure classification map
    failures = record["results"]["failure_distribution"]
    assert sum(failures.values()) == 20
    assert all(count > 0 for count in failures.values())

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
    resolve_uncertain_closed_state,
    run_articulated_uncertainty_study,
)


def test_uncertain_anthropometry_regenerates_closed_contact_state() -> None:
    """Geometry perturbations must re-solve closure and retain domain evidence."""

    resolved = resolve_uncertain_closed_state(
        profile_index=2,
        grip_span_m=0.18,
        sample_time_s=0.12,
        height_scale=0.97,
        mass_scale=1.04,
        joint_limit_scale=0.9,
    )
    assert resolved.feasible is True
    assert resolved.failure_class == "feasible"
    assert resolved.q.shape == resolved.qd.shape == (resolved.model.nq,)
    assert resolved.maximum_closure_error_m <= 5.0e-4
    assert resolved.minimum_joint_limit_margin_rad >= 0.0
    assert resolved.minimum_collision_clearance_m >= 0.0


def test_uncertain_closed_state_contract_fails_closed() -> None:
    with pytest.raises(ValueError, match="profile_index"):
        resolve_uncertain_closed_state(
            profile_index=99,
            grip_span_m=0.18,
            sample_time_s=0.12,
            height_scale=1.0,
            mass_scale=1.0,
            joint_limit_scale=1.0,
        )
    with pytest.raises(ValueError, match="joint_limit_scale"):
        resolve_uncertain_closed_state(
            profile_index=0,
            grip_span_m=0.18,
            sample_time_s=0.12,
            height_scale=1.0,
            mass_scale=1.0,
            joint_limit_scale=0.1,
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

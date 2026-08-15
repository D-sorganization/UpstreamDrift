"""Tests for the paired scapulothoracic closed-contact geometry screen."""

from __future__ import annotations

import numpy as np
import pytest
import json
from pathlib import Path

from scripts.research.proximal_distal_energy.scapulothoracic_contact_screen import (
    ScapulothoracicConfig,
    ellipsoid_surface_point,
    run_scapulothoracic_contact_screen,
)
from scripts.research.proximal_distal_energy.subject_scaled_spatial_geometry import (
    default_synthetic_profiles,
)

pytestmark = [pytest.mark.unit, pytest.mark.scientific]
ROOT = Path(__file__).resolve().parents[2]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"


def test_neutral_ellipsoid_points_reproduce_fixed_shoulder_offsets() -> None:
    config = ScapulothoracicConfig()

    lead = ellipsoid_surface_point(config, "lead", 0.0, 0.0, linear_scale=1.0)
    trail = ellipsoid_surface_point(config, "trail", 0.0, 0.0, linear_scale=1.0)

    np.testing.assert_allclose(lead, [0.0, 0.20, 0.18], atol=1.0e-12)
    np.testing.assert_allclose(trail, [0.0, -0.20, 0.18], atol=1.0e-12)
    for point in (lead, trail):
        normalized = np.sum((point / np.asarray(config.ellipsoid_radii_m)) ** 2)
        assert normalized == pytest.approx(1.0, abs=1.0e-12)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"closure_tolerance_m": 0.0},
        {"finite_difference_step_rad": 0.0},
        {"protraction_limit_rad": -0.1},
    ],
)
def test_config_rejects_nonphysical_numerical_contracts(
    kwargs: dict[str, float],
) -> None:
    with pytest.raises(ValueError):
        ScapulothoracicConfig(**kwargs)


def test_paired_screen_is_nested_deterministic_and_retains_adverse_control() -> None:
    profile = default_synthetic_profiles()[1]
    options = {
        "profiles": (profile,),
        "grip_spans_m": np.array([0.18]),
        "time_s": np.array([0.0, 0.12]),
        "adverse_grip_span_m": 2.0,
    }

    first_record, first_arrays = run_scapulothoracic_contact_screen(**options)
    second_record, second_arrays = run_scapulothoracic_contact_screen(**options)

    assert first_record == second_record
    for key in first_arrays:
        np.testing.assert_array_equal(first_arrays[key], second_arrays[key])
    assert first_record["design"]["paired_state_count"] == 2
    assert first_record["model"]["scapular_coordinates_per_side"] == 4
    assert first_record["boundaries"]["human_strategy"] == "not_identified"
    assert first_record["adverse_control"]["scapular_contact_closed"] is False
    assert np.all(
        first_arrays["scapular_max_contact_error_m"]
        <= first_arrays["fixed_max_contact_error_m"] + 1.0e-9
    )
    assert np.all(first_arrays["fixed_contact_jacobian_rank"] == 6)
    assert np.all(first_arrays["scapular_contact_jacobian_rank"] == 6)
    assert np.all(first_arrays["scapular_contact_jacobian_nullity"] >= 10)
    assert np.all(np.isfinite(first_arrays["scapular_shoulder_excursion_m"]))
    assert np.all(first_arrays["fixed_solver_termination_success"])
    assert np.count_nonzero(first_arrays["scapular_solver_termination_success"]) >= 1
    assert np.all(np.isfinite(first_arrays["scapular_minimum_bound_margin_rad"]))


def test_committed_screen_preserves_discrepancy_and_inference_boundaries() -> None:
    record = json.loads(
        (ARTICLE / "data/scapulothoracic_contact_screen.json").read_text()
    )
    arrays = np.load(ARTICLE / "data/scapulothoracic_contact_screen.npz")

    assert record["design"]["paired_state_count"] == 54
    assert record["results"]["fixed_contact_closed_count"] == 0
    assert 0 < record["results"]["scapular_contact_closed_count"] < 54
    assert record["results"]["scapular_never_worse_than_nested_fixed"] is True
    assert record["results"]["fixed_solver_termination_success_count"] == 54
    assert 0 < record["results"]["scapular_qualified_contact_count"] < 54
    assert record["results"]["scapular_bound_active_count"] > 0
    assert record["adverse_control"]["scapular_contact_closed"] is False
    assert np.all(arrays["fixed_contact_jacobian_rank"] == 6)
    assert np.all(arrays["scapular_contact_jacobian_rank"] == 6)
    assert record["boundaries"]["forward_force_power_or_transfer"] == ("not_evaluated")
    assert (ARTICLE / "figures/fig_scapulothoracic_contact_screen.pdf").is_file()
    assert (ARTICLE / "figures/fig_scapulothoracic_contact_screen.svg").is_file()

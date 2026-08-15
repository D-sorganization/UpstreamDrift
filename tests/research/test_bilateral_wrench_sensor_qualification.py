from __future__ import annotations

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.bilateral_wrench_sensor_qualification import (
    SensorQualificationConfig,
    run_sensor_qualification,
)

pytestmark = pytest.mark.scientific


def test_augmented_point_force_map_recovers_noiseless_trajectory() -> None:
    result = run_sensor_qualification(
        SensorQualificationConfig(
            sample_count=121,
            trial_count=2,
            normalized_noise_std=0.0,
            normalized_cross_talk=0.0,
            contact_migration_m=0.0,
        )
    )

    assert result.augmented.allocation_rmse_n < 1e-10
    assert result.augmented.normalized_net_wrench_rmse < 1e-10
    assert result.augmented.axial_mode_rmse_n < 1e-10
    assert result.net_wrench_only.axial_mode_rmse_n > 5.0


def test_calibrated_cross_talk_correction_improves_allocation_recovery() -> None:
    common = {
        "sample_count": 151,
        "trial_count": 12,
        "normalized_noise_std": 0.001,
        "normalized_cross_talk": 0.01,
        "contact_migration_m": 0.0,
        "seed": 731,
    }
    uncorrected = run_sensor_qualification(
        SensorQualificationConfig(**common, apply_cross_talk_correction=False)
    )
    corrected = run_sensor_qualification(
        SensorQualificationConfig(
            **common,
            apply_cross_talk_correction=True,
            cross_talk_calibration_error_fraction=0.0,
        )
    )

    assert (
        corrected.augmented.allocation_rmse_n < uncorrected.augmented.allocation_rmse_n
    )
    assert (
        corrected.augmented.normalized_net_wrench_rmse
        < uncorrected.augmented.normalized_net_wrench_rmse
    )


def test_contact_tracking_removes_contact_migration_model_bias() -> None:
    common = {
        "sample_count": 151,
        "trial_count": 4,
        "normalized_noise_std": 0.0,
        "normalized_cross_talk": 0.0,
        "contact_migration_m": 0.008,
        "seed": 919,
    }
    fixed_contacts = run_sensor_qualification(
        SensorQualificationConfig(**common, track_contact_centers=False)
    )
    tracked_contacts = run_sensor_qualification(
        SensorQualificationConfig(**common, track_contact_centers=True)
    )

    assert tracked_contacts.augmented.allocation_rmse_n < 1e-10
    assert fixed_contacts.augmented.allocation_rmse_n > 0.1
    assert (
        fixed_contacts.augmented.normalized_net_wrench_rmse
        > tracked_contacts.augmented.normalized_net_wrench_rmse
    )


def test_sensor_qualification_is_seed_reproducible() -> None:
    config = SensorQualificationConfig(sample_count=101, trial_count=6, seed=17)

    first = run_sensor_qualification(config)
    second = run_sensor_qualification(config)

    assert first == second


def test_sensor_qualification_contracts_reject_invalid_configuration() -> None:
    with pytest.raises(ValueError, match="sample_count"):
        SensorQualificationConfig(sample_count=2)
    with pytest.raises(ValueError, match="normalized_noise_std"):
        SensorQualificationConfig(normalized_noise_std=-0.1)
    with pytest.raises(ValueError, match="channel_scales"):
        SensorQualificationConfig(channel_scales=(1.0,) * 6)
    with pytest.raises(ValueError, match="positive"):
        SensorQualificationConfig(channel_scales=(1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 1.0))


def test_sensor_qualification_retains_declared_scope_boundary() -> None:
    result = run_sensor_qualification(
        SensorQualificationConfig(sample_count=101, trial_count=3)
    )

    assert result.scope == "synthetic_point_force_sensor_qualification"
    assert result.human_validation == "untested"
    assert result.anatomical_strategy == "not_identified"
    assert np.isfinite(result.augmented.allocation_p95_n)

"""Contracts for the bounded two-hand preactivation hypothesis study."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.two_hand_preactivation_hypothesis import (
    FIGURE_STEM,
    build_study,
    write_study,
)

pytestmark = pytest.mark.scientific


@pytest.fixture(scope="module")
def study() -> tuple[dict[str, object], dict[str, np.ndarray]]:
    return build_study()


def test_source_couple_closes_and_negative_action_is_mostly_pointwise_drift(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, arrays = study
    np.testing.assert_allclose(
        arrays["base_couple_nm"],
        arrays["pointwise_ztcf_couple_nm"] + arrays["control_residual_nm"],
        rtol=0.0,
        atol=2e-12,
    )
    reversal = record["source_reversal"]
    assert reversal["pointwise_ztcf_minimum_nm"] < -10.0
    assert 0.75 < reversal["drift_fraction_at_ztcf_minimum"] < 1.0
    assert reversal["control_residual_peak_abs_nm"] < (
        reversal["base_peak_abs_nm"] / 5.0
    )


def test_preview_reduces_delayed_residual_tracking_error_without_claiming_speed(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, _ = study
    reference = record["reference_actuator"]
    assert reference["time_constant_s"] == pytest.approx(0.03)
    assert 0.0 < reference["best_preview_s"] < 0.06
    assert reference["preview_rmse_nm"] < reference["reactive_rmse_nm"]
    assert reference["naive_net_target_rmse_nm"] > 10.0 * reference["reactive_rmse_nm"]
    assert record["claim_boundary"]["clubhead_speed_outcome"] == "not_evaluated"
    assert record["claim_boundary"]["human_preactivation"] == "not_established"


def test_time_constant_sensitivity_reports_bounded_optima(
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    record, _ = study
    rows = record["time_constant_sensitivity"]
    assert [row["time_constant_s"] for row in rows] == [0.01, 0.02, 0.03, 0.04, 0.05]
    for row in rows:
        assert 0.0 <= row["best_preview_s"] <= 0.08
        assert row["best_preview_rmse_nm"] < row["reactive_rmse_nm"]
        assert row["improvement_percent"] > 0.0


def test_study_is_deterministic_and_writes_traceable_artifacts(
    tmp_path: Path,
    study: tuple[dict[str, object], dict[str, np.ndarray]],
) -> None:
    first_record, first_arrays = study
    second_record, second_arrays = build_study()
    assert json.dumps(first_record, sort_keys=True) == json.dumps(
        second_record, sort_keys=True
    )
    for name in first_arrays:
        np.testing.assert_array_equal(first_arrays[name], second_arrays[name])

    paths = write_study(tmp_path)
    assert paths["json"].stat().st_size > 1_000
    assert paths["npz"].stat().st_size > 1_000
    assert (tmp_path / "figures" / f"{FIGURE_STEM}.svg").stat().st_size > 1_000
    assert (tmp_path / "figures" / f"{FIGURE_STEM}.pdf").stat().st_size > 1_000

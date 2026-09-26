"""Unit tests for full-body receipt gate evaluation (#10960 P0-9).

Validates:
- evaluate_gates helper behavior on pass, breach, NaN, missing metric.
- DbC input contracts: unknown gate names or negative thresholds raise ValueError.
- Diagnostic fallback when no thresholds exist.
- Per-script status-building functions for FB-3, FB-4, FB-5, FB-6:
  - RMSE above threshold -> not PASSED
  - Closure error of pi rad -> not PASSED
  - All metrics within threshold -> PASSED
"""

from __future__ import annotations

import math
from typing import Any

import pytest

from docs.development.full_body_models.evidence._gates import (
    FB3_DRAKE_THRESHOLDS,
    FB4_CALIBRATION_THRESHOLDS,
    FB5_MATCHING_THRESHOLDS,
    FB6_PARITY_THRESHOLDS,
    evaluate_gates,
)
from docs.development.full_body_models.evidence.fb3_drake.verify_drake_full_body import (
    build_status_from_metrics as fb3_build_status,
)
from docs.development.full_body_models.evidence.fb4_calibration.verify_full_body_calibration import (
    build_status_from_metrics as fb4_build_status,
)
from docs.development.full_body_models.evidence.fb5_matching.verify_full_body_matching import (
    build_status_from_metrics as fb5_build_status,
)
from docs.development.full_body_models.evidence.fb6_parity.verify_cross_engine_parity import (
    build_status_from_metrics as fb6_build_status,
)

pytestmark = pytest.mark.unit


class TestEvaluateGatesHelper:
    """Test suite for the shared evaluate_gates helper."""

    def test_all_within_threshold_passes(self) -> None:
        metrics = {"whole_marker_rmse_m": 0.020, "max_closure_residual_m": 0.003}
        thresholds = {"whole_marker_rmse_m": 0.060, "max_closure_residual_m": 0.005}
        result = evaluate_gates(metrics, thresholds)

        assert result["status"] == "PASSED"
        assert result["passed"] is True
        assert result["gates"]["whole_marker_rmse_m"]["passed"] is True
        assert result["gates"]["whole_marker_rmse_m"]["value"] == 0.020
        assert result["gates"]["max_closure_residual_m"]["passed"] is True

    def test_metric_above_threshold_fails(self) -> None:
        metrics = {"whole_marker_rmse_m": 2.84, "max_closure_residual_m": 0.003}
        thresholds = {"whole_marker_rmse_m": 0.060, "max_closure_residual_m": 0.005}
        result = evaluate_gates(metrics, thresholds)

        assert result["status"] != "PASSED"
        assert result["status"] == "FAILED"
        assert result["passed"] is False
        assert result["gates"]["whole_marker_rmse_m"]["passed"] is False
        assert "2.84" in result["gates"]["whole_marker_rmse_m"]["reason"]

    def test_missing_metric_fails_with_reason(self) -> None:
        metrics = {"max_closure_residual_m": 0.003}
        thresholds = {"whole_marker_rmse_m": 0.060, "max_closure_residual_m": 0.005}
        result = evaluate_gates(metrics, thresholds)

        assert result["status"] == "FAILED"
        assert result["gates"]["whole_marker_rmse_m"]["passed"] is False
        assert "missing" in result["gates"]["whole_marker_rmse_m"]["reason"].lower()

    def test_nan_or_inf_fails_with_reason(self) -> None:
        metrics = {"whole_marker_rmse_m": float("nan"), "max_closure_residual_m": 0.003}
        thresholds = {"whole_marker_rmse_m": 0.060, "max_closure_residual_m": 0.005}
        result = evaluate_gates(metrics, thresholds)

        assert result["status"] == "FAILED"
        assert result["gates"]["whole_marker_rmse_m"]["passed"] is False
        assert "non-finite" in result["gates"]["whole_marker_rmse_m"]["reason"].lower()

    def test_dbc_unknown_gate_name_in_thresholds_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown gate name in thresholds"):
            evaluate_gates({}, {"invented_gate_name": 1.0})

    def test_dbc_unknown_gate_name_in_metrics_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown gate name in metrics"):
            evaluate_gates({"invented_gate_name": 1.0}, {"whole_marker_rmse_m": 0.06})

    def test_dbc_negative_threshold_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Threshold .* cannot be negative"):
            evaluate_gates(
                {"whole_marker_rmse_m": 0.01}, {"whole_marker_rmse_m": -0.05}
            )

    def test_no_thresholds_returns_diagnostic_status(self) -> None:
        result = evaluate_gates({"whole_marker_rmse_m": 0.02}, {})
        assert result["status"] == "DIAGNOSTIC"
        assert "note" in result
        assert result["passed"] is False


class TestFb3DrakeGateStatus:
    """FB-3 Drake verification status building."""

    def test_closure_error_pi_rad_not_passed(self) -> None:
        metrics = {
            "gate_a_fk_diff_m": 0.0,
            "gate_a_mass_matrix_diff": 0.0,
            "gate_b_fk_diff_m": 0.0,
            "gate_c_normal_force_diff_n": 0.0,
            "gate_c_friction_force_diff_n": 0.0,
            "gate_c_penetration_diff_m": 0.0,
            "gate_d_position_residual_diff": 0.0,
            "gate_d_velocity_residual_diff": 0.0,
            "max_closure_pos_error": 3.14158898,
            "max_closure_vel_error": 0.0,
            "accelerations_all_finite": True,
        }
        res = fb3_build_status(metrics)
        assert res["status"] != "PASSED"
        assert res["status"] == "FAILED"
        assert res["gates"]["max_closure_pos_error"]["passed"] is False

    def test_all_within_threshold_passes(self) -> None:
        metrics = {
            "gate_a_fk_diff_m": 1e-13,
            "gate_a_mass_matrix_diff": 1e-13,
            "gate_b_fk_diff_m": 1e-13,
            "gate_c_normal_force_diff_n": 1e-13,
            "gate_c_friction_force_diff_n": 1e-13,
            "gate_c_penetration_diff_m": 1e-13,
            "gate_d_position_residual_diff": 1e-13,
            "gate_d_velocity_residual_diff": 1e-13,
            "max_closure_pos_error": 0.001,
            "max_closure_vel_error": 0.001,
            "accelerations_all_finite": True,
        }
        res = fb3_build_status(metrics)
        assert res["status"] == "PASSED"
        assert res["passed"] is True

    @pytest.mark.parametrize("flag", [False, 0.0, 1.0, None])
    def test_non_true_finite_flag_fails(self, flag: object) -> None:
        metrics = dict.fromkeys(FB3_DRAKE_THRESHOLDS, 0.0)
        metrics["accelerations_all_finite"] = flag
        res = fb3_build_status(metrics)
        assert res["status"] == "FAILED"
        assert res["gates"]["accelerations_all_finite"]["passed"] is False


class TestFb4CalibrationGateStatus:
    """FB-4 Calibration verification status building."""

    def test_rmse_above_threshold_not_passed(self) -> None:
        metrics = {
            "total_rms_m": 0.138,
            "closure_max_error_m": 0.016,
        }
        res = fb4_build_status(metrics)
        assert res["status"] != "PASSED"
        assert res["status"] == "FAILED"
        assert res["gates"]["total_rms_m"]["passed"] is False

    def test_all_within_threshold_passes(self) -> None:
        metrics = {
            "total_rms_m": 0.045,
            "closure_max_error_m": 0.002,
        }
        res = fb4_build_status(metrics)
        assert res["status"] == "PASSED"
        assert res["passed"] is True


class TestFb5MatchingGateStatus:
    """FB-5 Forward-dynamics matching verification status building."""

    def test_rmse_and_force_above_threshold_not_passed(self) -> None:
        metrics = {
            "whole_marker_rmse_m": 2.843,
            "max_normal_force_n": 110251.0,
            "max_penetration_m": 0.110,
            "max_closure_residual_m": 2.67,
            "max_defect_norm": 0.0,
        }
        res = fb5_build_status(metrics)
        assert res["status"] != "PASSED"
        assert res["status"] == "FAILED"
        assert res["gates"]["whole_marker_rmse_m"]["passed"] is False
        assert res["gates"]["max_normal_force_n"]["passed"] is False

    def test_all_within_threshold_passes(self) -> None:
        metrics = {
            "whole_marker_rmse_m": 0.055,
            "max_normal_force_n": 1800.0,
            "max_penetration_m": 0.008,
            "max_closure_residual_m": 0.003,
            "max_defect_norm": 0.001,
        }
        res = fb5_build_status(metrics)
        assert res["status"] == "PASSED"
        assert res["passed"] is True


class TestFb6ParityGateStatus:
    """FB-6 Cross-engine parity verification status building."""

    def test_rmse_and_force_above_threshold_not_passed(self) -> None:
        metrics = {
            "whole_marker_rmse_m": 2.392,
            "max_normal_force_n": 35397.0,
            "max_penetration_m": 0.094,
            "max_closure_residual_m": 1.908,
            "max_q_difference": 0.010,
        }
        res = fb6_build_status(metrics)
        assert res["status"] != "PASSED"
        assert res["status"] == "FAILED"
        assert res["gates"]["whole_marker_rmse_m"]["passed"] is False

    def test_all_within_threshold_passes(self) -> None:
        metrics = {
            "whole_marker_rmse_m": 0.050,
            "max_normal_force_n": 1500.0,
            "max_penetration_m": 0.005,
            "max_closure_residual_m": 0.002,
            "max_q_difference": 0.005,
        }
        res = fb6_build_status(metrics)
        assert res["status"] == "PASSED"
        assert res["passed"] is True

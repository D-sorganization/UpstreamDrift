"""Tests for forgiveness metrics (issue #8614).

Sensitivity of ball launch to entry point, attack angle, face-open angle and speed.
"""

from __future__ import annotations

from collections.abc import Callable

import pytest

from bunkershot3d.metrics.forgiveness import (
    compute_forgiveness_metrics,
    SensitivityGradient,
)


class TestSensitivityGradient:
    """Gradient of outputs w.r.t. error inputs."""

    def test_sensitivity_to_entry_point(self) -> None:
        """d(ball_speed)/d(entry_depth) [m/s per m]."""

        # A typical bunker shot: deeper entry = shorter carry
        # Sensitivity ~ -100 m/s per m of depth (i.e., 10 m/s per cm)
        def ball_speed_fn(entry_depth_m: float) -> float:
            # Simple linear model: speed drops with depth
            return 20.0 - 100.0 * entry_depth_m

        gradient = SensitivityGradient.from_finite_diff(
            ball_speed_fn,
            nominal_value=0.02,  # 2cm entry
            perturbation=0.001,  # 1mm
        )

        assert gradient.value == pytest.approx(-100.0, rel=0.1)
        assert gradient.units == "m/s per m"

    def test_sensitivity_to_attack_angle(self) -> None:
        """d(launch_angle)/d(attack_angle) [deg per deg]."""

        def launch_angle_fn(attack_angle_deg: float) -> float:
            # Steeper attack = higher launch (up to a point)
            return 30.0 + 0.5 * attack_angle_deg

        gradient = SensitivityGradient.from_finite_diff(
            launch_angle_fn,
            nominal_value=45.0,
            perturbation=1.0,
            input_units="deg",
            output_units="deg",
        )

        assert gradient.value == pytest.approx(0.5, rel=0.1)
        assert gradient.units == "deg per deg"


class TestForgivenessComputation:
    """Full forgiveness analysis across multiple input dimensions."""

    def test_forgiveness_score_computed(self) -> None:
        """Forgiveness is inverse of gradient magnitude."""

        # Simulate: a forgiving club has low sensitivity to errors
        def outcome_model(
            entry_depth_m: float,
            attack_angle_deg: float,
            face_open_deg: float,
            speed_m_s: float,
        ) -> dict:
            # Simple linear response model
            ball_speed = speed_m_s * 0.8 - entry_depth_m * 100
            launch_angle = 30 + attack_angle_deg * 0.3
            side_angle = face_open_deg * 0.5
            return {
                "ball_speed_m_s": ball_speed,
                "launch_angle_deg": launch_angle,
                "side_angle_deg": side_angle,
            }

        nominal = {
            "entry_depth_m": 0.02,
            "attack_angle_deg": 45.0,
            "face_open_deg": 0.0,
            "speed_m_s": 25.0,
        }
        perturbations = {
            "entry_depth_m": 0.005,  # 5mm
            "attack_angle_deg": 2.0,  # 2 deg
            "face_open_deg": 3.0,  # 3 deg
            "speed_m_s": 2.0,  # 2 m/s
        }

        metrics = compute_forgiveness_metrics(outcome_model, nominal, perturbations)

        # Should have gradients for each input-output pair
        assert "entry_depth_m" in metrics.gradients
        assert "ball_speed_m_s" in metrics.gradients["entry_depth_m"]

    def test_forgiveness_index_normalized(self) -> None:
        """Forgiveness index is 0-1 normalized, higher = more forgiving."""

        def stiff_model(entry_depth_m: float, **kw) -> dict:
            # Very sensitive to depth
            return {"ball_speed_m_s": 20 - 500 * entry_depth_m}

        def forgiving_model(entry_depth_m: float, **kw) -> dict:
            # Less sensitive to depth
            return {"ball_speed_m_s": 20 - 50 * entry_depth_m}

        nominal = {"entry_depth_m": 0.02, "attack_angle_deg": 45.0}
        perturbations = {"entry_depth_m": 0.005, "attack_angle_deg": 2.0}

        stiff_metrics = compute_forgiveness_metrics(stiff_model, nominal, perturbations)
        forgiving_metrics = compute_forgiveness_metrics(
            forgiving_model, nominal, perturbations
        )

        assert 0 <= stiff_metrics.forgiveness_index <= 1
        assert 0 <= forgiving_metrics.forgiveness_index <= 1
        # More forgiving model should have higher index
        assert forgiving_metrics.forgiveness_index > stiff_metrics.forgiveness_index


class TestPlayabilityAcrossConditions:
    """How does forgiveness vary with sand conditions?"""

    def test_playability_across_firmness_levels(self) -> None:
        """Compute forgiveness for firm, normal, soft sand."""

        def make_model(firmness: str) -> Callable:
            # Firmness affects sensitivity
            depth_coeff = {
                "firm": -150,  # Firm sand is less forgiving
                "normal": -100,
                "soft": -80,  # Soft sand is more forgiving
            }[firmness]

            def model(entry_depth_m: float, **kw) -> dict:
                return {"ball_speed_m_s": 20 + depth_coeff * entry_depth_m}

            return model

        nominal = {"entry_depth_m": 0.02}
        perturbations = {"entry_depth_m": 0.005}

        firm = compute_forgiveness_metrics(make_model("firm"), nominal, perturbations)
        normal = compute_forgiveness_metrics(
            make_model("normal"), nominal, perturbations
        )
        soft = compute_forgiveness_metrics(make_model("soft"), nominal, perturbations)

        # Soft should be most forgiving
        assert soft.forgiveness_index > normal.forgiveness_index
        assert normal.forgiveness_index > firm.forgiveness_index

    def test_playability_report_generated(self) -> None:
        """Generate a playability report across conditions."""
        conditions = ["firm", "fluffy", "wet", "plugged"]

        def make_model(condition: str) -> Callable:
            coeffs = {
                "firm": -150,
                "fluffy": -60,
                "wet": -120,
                "plugged": -200,
            }

            def model(entry_depth_m: float, **kw) -> dict:
                return {"ball_speed_m_s": 20 + coeffs[condition] * entry_depth_m}

            return model

        nominal = {"entry_depth_m": 0.02}
        perturbations = {"entry_depth_m": 0.005}

        report = {}
        for cond in conditions:
            metrics = compute_forgiveness_metrics(
                make_model(cond), nominal, perturbations
            )
            report[cond] = metrics.forgiveness_index

        # Report should have all conditions
        assert set(report.keys()) == set(conditions)
        # Fluffy should be most forgiving
        assert report["fluffy"] > report["firm"]
        assert report["fluffy"] > report["plugged"]


class TestErrorBudget:
    """How much error is tolerable for a given outcome variance?"""

    def test_error_budget_computation(self) -> None:
        """Given acceptable variance, compute allowable input error."""

        def model(entry_depth_m: float, **kw) -> dict:
            # d(ball_speed)/d(entry) = -100 m/s per m
            return {"ball_speed_m_s": 20 - 100 * entry_depth_m}

        nominal = {"entry_depth_m": 0.02}
        perturbations = {"entry_depth_m": 0.005}

        metrics = compute_forgiveness_metrics(model, nominal, perturbations)

        # If we want ball speed variance <= 2 m/s,
        # and sensitivity is 100 m/s per m,
        # allowable entry error is 2/100 = 0.02 m = 2 cm
        error_budget = metrics.compute_error_budget(
            output_name="ball_speed_m_s",
            max_variance=2.0,
        )

        assert error_budget["entry_depth_m"] == pytest.approx(0.02, rel=0.1)

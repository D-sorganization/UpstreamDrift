"""Population accounting and raw-target contracts with analytic test physics."""

from types import SimpleNamespace
import pytest
from src.tools.shot_pattern_analysis.matched_config import matched_preset
from src.tools.shot_pattern_analysis.matched_physics import UnsupportedShot
from src.tools.shot_pattern_analysis.matched_runner import (
    ExperimentConfig,
    run_matched_analysis,
)
from src.tools.shot_pattern_analysis.noise import NoiseSpec
from src.tools.shot_pattern_analysis.physics import ShotOutcome

pytestmark = pytest.mark.unit


class AnalyticPhysics:
    def simulate(self, *, face_deg, path_deg, config, **kwargs):
        return ShotOutcome(
            100 * config.club_speed_mps / 36, face_deg - 0.5 * path_deg, 100, 0, 1000, 0
        )

    def simulate_delivery(
        self,
        *,
        nominal_face_deg,
        path_deg,
        config,
        shaft_rotation_deg=0,
        contact_horizontal_m=0,
        contact_vertical_m=0,
        **kwargs,
    ):
        if abs(contact_horizontal_m) > 0.005:
            raise UnsupportedShot("off_face")
        face = nominal_face_deg - shaft_rotation_deg
        return SimpleNamespace(
            outcome=self.simulate(face_deg=face, path_deg=path_deg, config=config),
            delivered_face_deg=face,
            contact_face_deg=face,
            contact_loft_deg=config.loft_deg - config.shaft_lean_deg,
            delivered_loft_deg=config.loft_deg - config.shaft_lean_deg,
        )


def test_zero_noise_hits_raw_common_target_without_aim_rotation():
    request = ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((0,) * 7), n_shots=16
    )
    result = run_matched_analysis(
        request, physics=AnalyticPhysics(), include_scoring=False
    )
    assert len(result.shots) == 48
    for shot in result.shots:
        assert shot.carry_x_m == pytest.approx(100, abs=0.05)
        assert shot.carry_y_m == pytest.approx(0, abs=0.05)
    assert result.target_x_m == pytest.approx(100)
    assert set(result.nominals) == {"Straight", "Draw", "Fade"}


def test_unsupported_draws_are_retained_and_pairing_is_explicit():
    request = ExperimentConfig(
        matched_preset("seven_iron"),
        NoiseSpec((1, 0.5, 0.44704, 0.00635, 0.00635, 0, 0)),
        n_shots=64,
    )
    result = run_matched_analysis(
        request, physics=AnalyticPhysics(), include_scoring=False
    )
    assert len(result.shots) == 192
    by = {p: [s for s in result.shots if s.pattern == p] for p in result.nominals}
    for a, b in zip(by["Straight"], by["Draw"], strict=True):
        assert a.shot_index == b.shot_index
        assert a.shaft_rotation_deg == b.shaft_rotation_deg
        assert a.contact_horizontal_m == b.contact_horizontal_m
    assert 0 < sum(s.status != "supported" for s in by["Straight"]) < 64
    assert all(s.carry_x_m is None for s in result.shots if s.status != "supported")
    assert result.summary["Straight"]["total_n"] == 64
    assert result.summary["Straight"]["supported_n"] < 64
    assert (
        result.summary["Straight"]["statistics_conditioning"]
        == "supported_airborne_only"
    )


@pytest.mark.parametrize(
    "kwargs", [{"n_shots": 1}, {"seed": -1}, {"curve_scale": 0}, {"target_x_m": 0}]
)
def test_invalid_experiment_contracts(kwargs):
    with pytest.raises(ValueError):
        ExperimentConfig(
            matched_preset("seven_iron"), NoiseSpec((1, 0, 0, 0, 0, 0, 0)), **kwargs
        )


def test_numeric_physics_failure_is_not_hidden_as_unsupported():
    class Broken(AnalyticPhysics):
        def simulate_delivery(self, **kwargs):
            raise RuntimeError("contact solver failed")

    request = ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((0,) * 7), n_shots=4
    )
    with pytest.raises(RuntimeError, match="contact solver failed"):
        run_matched_analysis(request, physics=Broken(), include_scoring=False)


def test_custom_approach_radius_reaches_course_scoring(monkeypatch):
    import src.tools.shot_pattern_analysis.matched_runner as module

    received = {}

    def score(endpoints, **kwargs):
        received.update(kwargs)
        return {}

    monkeypatch.setattr(module, "score_course_endpoints", score)
    request = ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((0,) * 7), n_shots=4, target_radius_m=9
    )
    run_matched_analysis(request, physics=AnalyticPhysics())
    assert received["green_radius_m"] == 9


def test_joint_covariance_has_units_and_null_constant_correlations():
    request = ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((1, 0.5, 0, 0, 0, 0, 0)), n_shots=32
    )
    result = run_matched_analysis(
        request, physics=AnalyticPhysics(), include_scoring=False
    )
    joint = result.summary["Straight"]["joint_delivery_outcome_diagnostics"]
    assert joint["n"] == 32
    assert joint["units"]["spin_axis_tilt_deg"] == "degree; positive draw"
    index = joint["fields"].index("speed_mps")
    assert all(v is None for v in joint["correlation"][index])
    assert result.shots[0].launch_azimuth_deg == 0


def test_paired_accuracy_uncertainty_preserves_units_and_reference():
    request = ExperimentConfig(
        matched_preset("seven_iron"), NoiseSpec((1, 0.5, 0, 0, 0, 0, 0)), n_shots=32
    )
    result = run_matched_analysis(
        request, physics=AnalyticPhysics(), include_scoring=False
    )
    for key in (
        "mean_squared_target_error_difference_m2",
        "containment_fraction_difference",
    ):
        for name in ("Draw", "Fade"):
            interval = result.paired_comparisons[key][name]
            assert interval["estimate"] == pytest.approx(0, abs=1e-5)
            assert interval["lower_95"] <= interval["estimate"] <= interval["upper_95"]

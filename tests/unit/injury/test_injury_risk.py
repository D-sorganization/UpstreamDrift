"""Tests for src.shared.python.injury.injury_risk module."""

from __future__ import annotations

from src.shared.python.injury.injury_risk import (
    InjuryRiskReport,
    InjuryRiskScorer,
    InjuryType,
    RiskFactor,
    RiskLevel,
)


class TestRiskLevel:
    """Tests for RiskLevel enum."""

    def test_all_levels(self) -> None:
        assert RiskLevel.LOW.value == "low"
        assert RiskLevel.MODERATE.value == "moderate"
        assert RiskLevel.HIGH.value == "high"
        assert RiskLevel.VERY_HIGH.value == "very_high"


class TestInjuryType:
    """Tests for InjuryType enum."""

    def test_all_types(self) -> None:
        assert InjuryType.LOW_BACK.value == "low_back"
        assert InjuryType.HIP.value == "hip"
        assert InjuryType.SHOULDER.value == "shoulder"
        assert InjuryType.ELBOW.value == "elbow"
        assert InjuryType.WRIST.value == "wrist"
        assert InjuryType.KNEE.value == "knee"


class TestRiskFactor:
    """Tests for RiskFactor dataclass."""

    def test_construction(self) -> None:
        rf = RiskFactor(
            name="test",
            value=50.0,
            threshold_safe=30.0,
            threshold_high=80.0,
        )
        assert rf.name == "test"
        assert rf.value == 50.0
        assert rf.weight == 1.0

    def test_modifiable_default(self) -> None:
        rf = RiskFactor(name="x", value=0, threshold_safe=0, threshold_high=0)
        assert rf.modifiable is True


class TestInjuryRiskReport:
    """Tests for InjuryRiskReport dataclass."""

    def test_default_construction(self) -> None:
        report = InjuryRiskReport()
        assert report.overall_risk_score == 0.0
        assert report.overall_risk_level == RiskLevel.LOW
        assert report.risk_factors == []
        assert report.recommendations == []
        assert report.confidence == 1.0


class TestInjuryRiskScorer:
    """Tests for InjuryRiskScorer class."""

    def test_construction(self) -> None:
        scorer = InjuryRiskScorer()
        assert scorer is not None

    def test_score_no_inputs(self) -> None:
        """Scoring with no data should return a valid low-risk report."""
        scorer = InjuryRiskScorer()
        report = scorer.score()
        assert isinstance(report, InjuryRiskReport)
        assert report.overall_risk_score >= 0.0

    def test_score_with_spinal_result(self) -> None:
        """Test scoring with a mock spinal result."""

        class MockSpinal:
            peak_compression_bw = 5.5
            peak_lateral_shear_bw = 0.8

        scorer = InjuryRiskScorer()
        report = scorer.score(spinal_result=MockSpinal())
        assert isinstance(report, InjuryRiskReport)
        assert report.overall_risk_score >= 0.0

    def test_score_with_joint_results(self) -> None:
        """Test scoring with mock joint results."""

        class MockJoint:
            def __init__(self, risk_score: float, impingement_risk: bool = False):
                self.risk_score = risk_score
                self.impingement_risk = impingement_risk

        joints = {
            "hip_lead": MockJoint(45, True),
            "shoulder_trail": MockJoint(55),
        }
        scorer = InjuryRiskScorer()
        report = scorer.score(joint_results=joints)
        assert len(report.risk_factors) > 0

    def test_score_with_swing_metrics(self) -> None:
        """Test scoring with swing metrics."""
        metrics = {
            "club_head_speed": 55.0,  # m/s
            "x_factor_stretch": 52.0,  # degrees
        }
        scorer = InjuryRiskScorer()
        report = scorer.score(swing_metrics=metrics)
        assert isinstance(report, InjuryRiskReport)

    def test_score_with_training_load(self) -> None:
        """Test scoring with training load data."""
        training = {
            "sessions_per_week": 6,
            "balls_per_session": 200,
            "weeks_trained": 10,
        }
        scorer = InjuryRiskScorer()
        report = scorer.score(training_load=training)
        assert isinstance(report, InjuryRiskReport)

    def test_high_risk_produces_recommendations(self) -> None:
        """High spinal loads should produce recommendations."""

        class HighSpinal:
            peak_compression_bw = 10.0
            peak_lateral_shear_bw = 2.5

        scorer = InjuryRiskScorer()
        report = scorer.score(spinal_result=HighSpinal())
        assert report.overall_risk_score > 0

    def test_value_to_score_below_safe(self) -> None:
        scorer = InjuryRiskScorer()
        score = scorer._value_to_score(10.0, safe=50.0, high=100.0)
        assert score == 0.0

    def test_value_to_score_above_high(self) -> None:
        scorer = InjuryRiskScorer()
        score = scorer._value_to_score(150.0, safe=50.0, high=100.0)
        assert score == 100.0

    def test_value_to_score_midrange(self) -> None:
        scorer = InjuryRiskScorer()
        score = scorer._value_to_score(75.0, safe=50.0, high=100.0)
        assert 0 < score < 100

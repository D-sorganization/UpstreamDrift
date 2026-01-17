
import pytest
from unittest.mock import Mock, MagicMock
from shared.python.injury.injury_risk import (
    InjuryRiskScorer,
    RiskLevel,
    InjuryType,
    InjuryRiskReport,
    RiskFactor
)

class TestInjuryRiskScorer:
    @pytest.fixture
    def scorer(self):
        return InjuryRiskScorer()

    @pytest.fixture
    def mock_spinal_result(self):
        result = Mock()
        result.peak_compression_bw = 5.0  # Caution range (4-6)
        result.peak_lateral_shear_bw = 0.8  # Caution range (0.5-1.0)
        
        x_factor = Mock()
        x_factor.x_factor_stretch = 48.0  # Caution range (45-55)
        result.x_factor = x_factor
        
        return result

    @pytest.fixture
    def mock_joint_results(self):
        def create_result(score, impingement=False):
            result = Mock()
            result.risk_score = score
            result.impingement_risk = impingement
            return result

        return {
            "hip_lead": create_result(45, True),
            "hip_trail": create_result(30),
            "shoulder_lead": create_result(35),
            "shoulder_trail": create_result(55),
            "elbow_lead": create_result(60),
            "elbow_trail": create_result(40),
            "wrist_lead": create_result(50),
            "wrist_trail": create_result(35),
        }

    @pytest.fixture
    def mock_swing_metrics(self):
        return {
            "sequence_timing_error": 0.10,  # Middle of range
            "tempo_ratio": 3.5,            # Error of 0.5
            "early_extension": 10.0,       # Middle of range
        }

    @pytest.fixture
    def mock_training_load(self):
        return {
            "acwr": 1.4,          # High (>1.3)
            "weekly_swings": 800, # High (500-1000)
        }

    def test_score_spinal_risks(self, scorer, mock_spinal_result):
        report = InjuryRiskReport()
        scorer._score_spinal_risks(mock_spinal_result, report)
        
        # Verify specific risk factors were added
        factor_names = [f.name for f in scorer.risk_factors]
        assert "spinal_compression" in factor_names
        assert "spinal_shear" in factor_names
        assert "x_factor_stretch" in factor_names
        
        # Verify region score calculation
        assert InjuryType.LOW_BACK in report.region_scores
        score = report.region_scores[InjuryType.LOW_BACK]
        assert score > 0
        assert score < 100

    def test_score_joint_risks(self, scorer, mock_joint_results):
        report = InjuryRiskReport()
        scorer._score_joint_risks(mock_joint_results, report)
        
        # Verify region scores are populated
        assert InjuryType.HIP in report.region_scores
        assert InjuryType.SHOULDER in report.region_scores
        assert InjuryType.ELBOW in report.region_scores
        assert InjuryType.WRIST in report.region_scores
        
        # Verify scores match averages
        # Hip: (45 + 30) / 2 = 37.5
        assert report.region_scores[InjuryType.HIP] == 37.5

    def test_score_technique_risks(self, scorer, mock_swing_metrics):
        report = InjuryRiskReport()
        scorer._score_technique_risks(mock_swing_metrics, report)
        
        assert report.technique_risk_score > 0
        
        factor_names = [f.name for f in scorer.risk_factors]
        assert "kinematic_sequence" in factor_names
        assert "swing_tempo" in factor_names
        assert "early_extension" in factor_names

    def test_score_training_load_risks(self, scorer, mock_training_load):
        report = InjuryRiskReport()
        scorer._score_training_load_risks(mock_training_load, report)
        
        assert report.chronic_risk_score > 0
        
        factor_names = [f.name for f in scorer.risk_factors]
        assert "acwr" in factor_names
        assert "weekly_swings" in factor_names

    def test_compute_overall_scores(self, scorer):
        report = InjuryRiskReport()
        report.acute_risk_score = 60.0
        report.chronic_risk_score = 40.0
        report.technique_risk_score = 50.0
        
        # Add a dummy risk factor to sort
        scorer.risk_factors = [
            RiskFactor("high_risk", 10.0, 5.0, 8.0),
            RiskFactor("low_risk", 2.0, 5.0, 8.0)
        ]
        
        scorer._compute_overall_scores(report)
        
        # Expected: 0.5*60 + 0.3*40 + 0.2*50 = 30 + 12 + 10 = 52
        assert report.overall_risk_score == pytest.approx(52.0)
        assert report.overall_risk_level == RiskLevel.HIGH  # 50-75 range
        assert "high_risk" in report.top_risks

    def test_generate_recommendations(self, scorer):
        report = InjuryRiskReport()
        
        # Add high risk factors
        scorer.risk_factors = [
            RiskFactor("spinal_compression", 8.0, 4.0, 6.0, modifiable=True),  # High score
            RiskFactor("elbow_stress", 60.0, 25.0, 50.0, modifiable=True)      # High score
        ]
        report.overall_risk_level = RiskLevel.HIGH
        
        scorer._generate_recommendations(report)
        
        print(f"Recommendations: {report.recommendations}")
        
        assert len(report.recommendations) > 0
        # Should have specific recommendations + general advice
        assert any("spinal compression" in r.lower() for r in report.recommendations)
        assert any("elbow" in r.lower() for r in report.recommendations)
        assert any("consult" in r.lower() and "professional" in r.lower() for r in report.recommendations)

    def test_full_scoring_pipeline(self, scorer, mock_spinal_result, mock_joint_results, mock_swing_metrics, mock_training_load):
        report = scorer.score(
            spinal_result=mock_spinal_result,
            joint_results=mock_joint_results,
            swing_metrics=mock_swing_metrics,
            training_load=mock_training_load
        )
        
        assert isinstance(report, InjuryRiskReport)
        assert report.overall_risk_score > 0
        assert len(report.recommendations) > 0
        assert len(report.top_risks) > 0
        assert len(report.region_scores) > 0

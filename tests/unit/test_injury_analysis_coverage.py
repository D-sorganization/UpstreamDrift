import pytest
from shared.python.injury.injury_risk import InjuryRiskScorer, InjuryType, RiskLevel


class MockSpinalResult:
    def __init__(self, compression=3.0, shear=0.4, twist=40.0):
        self.peak_compression_bw = compression
        self.peak_lateral_shear_bw = shear
        self.x_factor = type("obj", (object,), {"x_factor_stretch": twist})


class MockJointResult:
    def __init__(self, risk_score: float, impingement_risk: bool = False):
        self.risk_score = risk_score
        self.impingement_risk = impingement_risk


@pytest.fixture
def scorer():
    return InjuryRiskScorer()


def test_low_risk_scenario(scorer):
    """Test a scenario where all inputs are safe."""
    spinal = MockSpinalResult(compression=3.0, shear=0.4, twist=40.0)
    # Joint scores < 30 are safe
    joints = {"hip_lead": MockJointResult(20), "shoulder_lead": MockJointResult(20)}

    report = scorer.score(spinal_result=spinal, joint_results=joints)

    assert report.overall_risk_level == RiskLevel.LOW
    assert len(report.recommendations) == 0


def test_high_risk_scenario(scorer):
    """Test a scenario with dangerous values."""
    # Compression 7.0 > 6.0 (High), Shear 1.2 > 1.0 (High)
    spinal = MockSpinalResult(compression=7.0, shear=1.2, twist=60.0)

    report = scorer.score(spinal_result=spinal)

    # Should trigger high risk factors
    high_risks = [f for f in report.risk_factors if f.value > f.threshold_high]
    assert len(high_risks) >= 2  # Compression and Shear

    # Check recommendations appear
    assert any("spinal compression" in r.lower() for r in report.recommendations)


def test_technique_risks(scorer):
    """Test swing mechanic risks."""
    metrics = {
        "early_extension": 20.0,  # Threshold high is 15.0
        "tempo_ratio": 4.0,  # Ideal is 3.0, threshold high err is 1.5 -> 4.5
    }

    report = scorer.score(swing_metrics=metrics)

    # Early extension should be high risk
    ee_factor = next(f for f in report.risk_factors if f.name == "early_extension")
    assert ee_factor.value == 20.0

    # Check recommendation
    # Note: Recommendations logic checks score > 50.
    # _value_to_score(20, 5, 15) -> 100 (>50)
    assert any("early extension" in r.lower() for r in report.recommendations)


def test_training_load_spike(scorer):
    """Test ACWR spike."""
    load = {"acwr": 1.6}  # High > 1.5

    report = scorer.score(training_load=load)

    acwr_factor = next(f for f in report.risk_factors if f.name == "acwr")
    assert acwr_factor.value == 1.6

    assert any("training load spike" in r.lower() for r in report.recommendations)


def test_category_weighting(scorer):
    """Ensure different injury types update the correct region scores."""
    joints = {"hip_lead": MockJointResult(50), "elbow_lead": MockJointResult(50)}

    report = scorer.score(joint_results=joints)

    assert InjuryType.HIP in report.region_scores
    assert InjuryType.ELBOW in report.region_scores
    assert report.region_scores[InjuryType.HIP] == 50.0

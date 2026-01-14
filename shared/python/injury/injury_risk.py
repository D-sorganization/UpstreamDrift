"""
Injury Risk Scoring Module

Aggregates all injury risk factors into comprehensive risk scores with
evidence-based thresholds and recommendations.

This module combines:
- Spinal load analysis (compression, shear, torsion)
- Joint stress analysis (hip, shoulder, elbow, wrist)
- Cumulative load tracking (overuse injury risk)
- Swing mechanics analysis (technique-related risks)

Risk Categories:
- Acute injury risk (single swing)
- Chronic/overuse injury risk (cumulative exposure)
- Technique-related risk (modifiable factors)
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class RiskLevel(Enum):
    """Overall risk level categories."""

    LOW = "low"  # < 25
    MODERATE = "moderate"  # 25-50
    HIGH = "high"  # 50-75
    VERY_HIGH = "very_high"  # > 75


class InjuryType(Enum):
    """Types of golf-related injuries."""

    LOW_BACK = "low_back"
    HIP = "hip"
    SHOULDER = "shoulder"
    ELBOW = "elbow"
    WRIST = "wrist"
    KNEE = "knee"


@dataclass
class RiskFactor:
    """A single risk factor contributing to injury risk."""

    name: str
    value: float
    threshold_safe: float
    threshold_high: float
    weight: float = 1.0
    modifiable: bool = True
    description: str = ""


@dataclass
class InjuryRiskReport:
    """Comprehensive injury risk assessment report."""

    # Overall scores
    overall_risk_score: float = 0.0
    overall_risk_level: RiskLevel = RiskLevel.LOW

    # Category scores
    acute_risk_score: float = 0.0
    chronic_risk_score: float = 0.0
    technique_risk_score: float = 0.0

    # Body region scores
    region_scores: dict[InjuryType, float] = field(default_factory=dict)

    # Individual risk factors
    risk_factors: list[RiskFactor] = field(default_factory=list)

    # Top risks
    top_risks: list[str] = field(default_factory=list)

    # Recommendations
    recommendations: list[str] = field(default_factory=list)

    # Confidence level (based on data quality)
    confidence: float = 1.0


class InjuryRiskScorer:
    """
    Comprehensive injury risk scoring system.

    Combines multiple biomechanical risk factors into actionable risk scores
    with evidence-based thresholds from golf injury literature.

    Example:
        >>> scorer = InjuryRiskScorer()
        >>> report = scorer.score(spinal_result, joint_results, swing_metrics)
        >>> print(f"Overall Risk: {report.overall_risk_level.value}")
        >>> for rec in report.recommendations:
        ...     print(f"  - {rec}")
    """

    # Risk factor weights (based on injury prevalence in golf)
    WEIGHTS = {
        InjuryType.LOW_BACK: 0.35,  # Most common (25-58%)
        InjuryType.ELBOW: 0.20,  # Second most common
        InjuryType.SHOULDER: 0.15,
        InjuryType.WRIST: 0.15,
        InjuryType.HIP: 0.10,
        InjuryType.KNEE: 0.05,
    }

    def __init__(self) -> None:
        """Initialize the injury risk scorer."""
        self.risk_factors: list[RiskFactor] = []

    def score(
        self,
        spinal_result: object = None,
        joint_results: dict | None = None,
        swing_metrics: dict | None = None,
        training_load: dict | None = None,
    ) -> InjuryRiskReport:
        """
        Compute comprehensive injury risk score.

        Args:
            spinal_result: SpinalLoadResult from spinal load analysis
            joint_results: Dict of JointStressResult from joint analysis
            swing_metrics: Dict of swing technique metrics
            training_load: Dict of training load metrics

        Returns:
            InjuryRiskReport with scores and recommendations
        """
        report = InjuryRiskReport()
        self.risk_factors = []

        # Score spinal risks
        if spinal_result is not None:
            self._score_spinal_risks(spinal_result, report)

        # Score joint risks
        if joint_results is not None:
            self._score_joint_risks(joint_results, report)

        # Score technique risks
        if swing_metrics is not None:
            self._score_technique_risks(swing_metrics, report)

        # Score training load risks
        if training_load is not None:
            self._score_training_load_risks(training_load, report)

        # Compute overall scores
        self._compute_overall_scores(report)

        # Generate recommendations
        self._generate_recommendations(report)

        report.risk_factors = self.risk_factors

        return report

    def _score_spinal_risks(
        self, spinal_result: object, report: InjuryRiskReport
    ) -> None:
        """Score spinal-related risk factors."""
        # Compression risk
        compression_bw = getattr(spinal_result, "peak_compression_bw", 0)
        self.risk_factors.append(
            RiskFactor(
                name="spinal_compression",
                value=compression_bw,
                threshold_safe=4.0,
                threshold_high=6.0,
                weight=1.2,
                modifiable=True,
                description="Axial compression on lumbar spine",
            )
        )

        # Shear risk
        shear_bw = getattr(spinal_result, "peak_lateral_shear_bw", 0)
        self.risk_factors.append(
            RiskFactor(
                name="spinal_shear",
                value=shear_bw,
                threshold_safe=0.5,
                threshold_high=1.0,
                weight=1.0,
                modifiable=True,
                description="Lateral shear force on spine",
            )
        )

        # X-factor risk
        x_factor = getattr(spinal_result, "x_factor", None)
        if x_factor is not None:
            stretch = getattr(x_factor, "x_factor_stretch", 0)
            self.risk_factors.append(
                RiskFactor(
                    name="x_factor_stretch",
                    value=stretch,
                    threshold_safe=45.0,
                    threshold_high=55.0,
                    weight=0.8,
                    modifiable=True,
                    description="Pelvis-thorax separation angle",
                )
            )

        # Compute region score
        compression_score = self._value_to_score(compression_bw, 4.0, 6.0) * 1.2
        shear_score = self._value_to_score(shear_bw, 0.5, 1.0) * 1.0
        x_factor_score = (
            self._value_to_score(getattr(x_factor, "x_factor_stretch", 0), 45, 55) * 0.8
            if x_factor
            else 0
        )

        report.region_scores[InjuryType.LOW_BACK] = (
            compression_score + shear_score + x_factor_score
        ) / 3

    def _score_joint_risks(self, joint_results: dict, report: InjuryRiskReport) -> None:
        """Score joint-related risk factors."""
        # Hip risks
        hip_scores = []
        for name, result in joint_results.items():
            if "hip" in name:
                hip_scores.append(result.risk_score)
                self.risk_factors.append(
                    RiskFactor(
                        name=f"{name}_stress",
                        value=result.risk_score,
                        threshold_safe=30,
                        threshold_high=60,
                        weight=0.6,
                        modifiable=result.impingement_risk,
                        description=f"Stress on {name}",
                    )
                )
        if hip_scores:
            report.region_scores[InjuryType.HIP] = np.mean(hip_scores)

        # Shoulder risks
        shoulder_scores = []
        for name, result in joint_results.items():
            if "shoulder" in name:
                shoulder_scores.append(result.risk_score)
                self.risk_factors.append(
                    RiskFactor(
                        name=f"{name}_stress",
                        value=result.risk_score,
                        threshold_safe=30,
                        threshold_high=60,
                        weight=0.7,
                        modifiable=True,
                        description=f"Rotator cuff loading on {name}",
                    )
                )
        if shoulder_scores:
            report.region_scores[InjuryType.SHOULDER] = np.mean(shoulder_scores)

        # Elbow risks
        elbow_scores = []
        for name, result in joint_results.items():
            if "elbow" in name:
                elbow_scores.append(result.risk_score)
                self.risk_factors.append(
                    RiskFactor(
                        name=f"{name}_stress",
                        value=result.risk_score,
                        threshold_safe=25,
                        threshold_high=50,
                        weight=0.9,
                        modifiable=True,
                        description=f"Medial epicondyle stress on {name}",
                    )
                )
        if elbow_scores:
            report.region_scores[InjuryType.ELBOW] = np.mean(elbow_scores)

        # Wrist risks
        wrist_scores = []
        for name, result in joint_results.items():
            if "wrist" in name:
                wrist_scores.append(result.risk_score)
                self.risk_factors.append(
                    RiskFactor(
                        name=f"{name}_stress",
                        value=result.risk_score,
                        threshold_safe=30,
                        threshold_high=60,
                        weight=0.7,
                        modifiable=True,
                        description=f"TFCC/hamate stress on {name}",
                    )
                )
        if wrist_scores:
            report.region_scores[InjuryType.WRIST] = np.mean(wrist_scores)

    def _score_technique_risks(
        self, swing_metrics: dict, report: InjuryRiskReport
    ) -> None:
        """Score technique-related risk factors."""
        # Kinematic sequence timing
        if "sequence_timing_error" in swing_metrics:
            error = swing_metrics["sequence_timing_error"]
            self.risk_factors.append(
                RiskFactor(
                    name="kinematic_sequence",
                    value=error,
                    threshold_safe=0.05,
                    threshold_high=0.15,
                    weight=0.5,
                    modifiable=True,
                    description="Deviation from optimal kinematic sequence",
                )
            )
            report.technique_risk_score += self._value_to_score(error, 0.05, 0.15) * 0.5

        # Swing tempo
        if "tempo_ratio" in swing_metrics:
            tempo = swing_metrics["tempo_ratio"]
            # Optimal is ~3:1 backswing:downswing
            tempo_error = abs(tempo - 3.0)
            self.risk_factors.append(
                RiskFactor(
                    name="swing_tempo",
                    value=tempo,
                    threshold_safe=0.5,
                    threshold_high=1.5,
                    weight=0.4,
                    modifiable=True,
                    description="Backswing:downswing tempo ratio",
                )
            )
            report.technique_risk_score += (
                self._value_to_score(tempo_error, 0.5, 1.5) * 0.4
            )

        # Early extension
        if "early_extension" in swing_metrics:
            extension = swing_metrics["early_extension"]
            self.risk_factors.append(
                RiskFactor(
                    name="early_extension",
                    value=extension,
                    threshold_safe=5.0,
                    threshold_high=15.0,
                    weight=0.6,
                    modifiable=True,
                    description="Pelvis movement toward ball in downswing (cm)",
                )
            )
            report.technique_risk_score += (
                self._value_to_score(extension, 5.0, 15.0) * 0.6
            )

    def _score_training_load_risks(
        self, training_load: dict, report: InjuryRiskReport
    ) -> None:
        """Score training load-related risk factors."""
        # Acute:Chronic Workload Ratio
        if "acwr" in training_load:
            acwr = training_load["acwr"]
            # Optimal range is 0.8-1.3
            if acwr < 0.8:
                acwr_risk = (0.8 - acwr) * 50
            elif acwr > 1.3:
                acwr_risk = (acwr - 1.3) * 100
            else:
                acwr_risk = 0

            self.risk_factors.append(
                RiskFactor(
                    name="acwr",
                    value=acwr,
                    threshold_safe=0.8,
                    threshold_high=1.5,
                    weight=1.0,
                    modifiable=True,
                    description="Acute:Chronic Workload Ratio",
                )
            )
            report.chronic_risk_score += acwr_risk

        # Weekly swing count
        if "weekly_swings" in training_load:
            swings = training_load["weekly_swings"]
            self.risk_factors.append(
                RiskFactor(
                    name="weekly_swings",
                    value=swings,
                    threshold_safe=500,
                    threshold_high=1000,
                    weight=0.8,
                    modifiable=True,
                    description="Number of full swings per week",
                )
            )
            report.chronic_risk_score += self._value_to_score(swings, 500, 1000) * 0.8

    def _value_to_score(self, value: float, safe: float, high: float) -> float:
        """Convert a value to a 0-100 risk score."""
        if value <= safe:
            return 0
        elif value >= high:
            return 100
        else:
            return ((value - safe) / (high - safe)) * 100

    def _compute_overall_scores(self, report: InjuryRiskReport) -> None:
        """Compute overall risk scores from individual factors."""
        # Acute risk from biomechanical loading
        if report.region_scores:
            weighted_scores = [
                report.region_scores.get(region, 0) * self.WEIGHTS.get(region, 0.1)
                for region in InjuryType
            ]
            report.acute_risk_score = sum(weighted_scores) / sum(self.WEIGHTS.values())

        # Overall score
        report.overall_risk_score = (
            0.5 * report.acute_risk_score
            + 0.3 * report.chronic_risk_score
            + 0.2 * report.technique_risk_score
        )

        # Determine risk level
        if report.overall_risk_score < 25:
            report.overall_risk_level = RiskLevel.LOW
        elif report.overall_risk_score < 50:
            report.overall_risk_level = RiskLevel.MODERATE
        elif report.overall_risk_score < 75:
            report.overall_risk_level = RiskLevel.HIGH
        else:
            report.overall_risk_level = RiskLevel.VERY_HIGH

        # Identify top risks
        sorted_factors = sorted(
            self.risk_factors,
            key=lambda f: self._value_to_score(
                f.value, f.threshold_safe, f.threshold_high
            ),
            reverse=True,
        )
        report.top_risks = [f.name for f in sorted_factors[:3]]

    def _generate_recommendations(self, report: InjuryRiskReport) -> None:
        """Generate actionable recommendations based on risk factors."""
        recommendations = []

        # Find high-risk modifiable factors
        for factor in self.risk_factors:
            score = self._value_to_score(
                factor.value, factor.threshold_safe, factor.threshold_high
            )
            if score > 50 and factor.modifiable:
                if "compression" in factor.name:
                    recommendations.append(
                        "Reduce spinal compression: Maintain more neutral spine angle, "
                        "strengthen core muscles, consider 'stack and tilt' technique."
                    )
                elif "shear" in factor.name:
                    recommendations.append(
                        "Reduce lateral shear: Limit side bend during downswing, "
                        "use 'stabilized spine' technique."
                    )
                elif "x_factor" in factor.name:
                    recommendations.append(
                        "Reduce X-factor stretch: Allow more hip turn in backswing, "
                        "consider 'classic' swing pattern."
                    )
                elif "elbow" in factor.name:
                    recommendations.append(
                        "Reduce elbow stress: Avoid 'casting' motion, maintain wrist cock "
                        "longer in downswing, strengthen forearm muscles."
                    )
                elif "hip" in factor.name:
                    recommendations.append(
                        "Reduce hip stress: Improve hip mobility with targeted stretching, "
                        "check for potential hip impingement."
                    )
                elif "shoulder" in factor.name:
                    recommendations.append(
                        "Reduce shoulder stress: Strengthen rotator cuff, limit backswing "
                        "length if excessive, maintain proper posture."
                    )
                elif "wrist" in factor.name:
                    recommendations.append(
                        "Reduce wrist stress: Check grip pressure distribution, strengthen "
                        "wrist and forearm, consider padded grips."
                    )
                elif "acwr" in factor.name:
                    if factor.value > 1.3:
                        recommendations.append(
                            "Training load spike detected: Gradually increase practice volume, "
                            "avoid sudden increases in swing count."
                        )
                    else:
                        recommendations.append(
                            "Training load too low: Gradually increase practice to maintain "
                            "tissue resilience."
                        )

        # Add general recommendations based on overall risk
        if report.overall_risk_level in [RiskLevel.HIGH, RiskLevel.VERY_HIGH]:
            recommendations.append(
                "Overall risk is elevated. Consider consulting a golf fitness professional "
                "or sports medicine physician for personalized guidance."
            )

        # Remove duplicates while preserving order
        seen = set()
        unique_recommendations = []
        for rec in recommendations:
            if rec not in seen:
                seen.add(rec)
                unique_recommendations.append(rec)

        report.recommendations = unique_recommendations[:5]  # Limit to top 5


if __name__ == "__main__":
    # Example usage
    scorer = InjuryRiskScorer()

    # Mock data
    class MockSpinalResult:
        peak_compression_bw = 5.5
        peak_lateral_shear_bw = 0.8

        class x_factor:
            x_factor_stretch = 52.0

    class MockJointResult:
        def __init__(self, risk_score: float, impingement_risk: bool = False) -> None:
            self.risk_score = risk_score
            self.impingement_risk = impingement_risk

    spinal_result = MockSpinalResult()
    joint_results = {
        "hip_lead": MockJointResult(45, True),
        "hip_trail": MockJointResult(30),
        "shoulder_lead": MockJointResult(35),
        "shoulder_trail": MockJointResult(55),
        "elbow_lead": MockJointResult(60),
        "elbow_trail": MockJointResult(40),
        "wrist_lead": MockJointResult(50),
        "wrist_trail": MockJointResult(35),
    }
    swing_metrics = {
        "tempo_ratio": 3.5,
        "early_extension": 12.0,
    }
    training_load = {
        "acwr": 1.4,
        "weekly_swings": 800,
    }

    report = scorer.score(spinal_result, joint_results, swing_metrics, training_load)

    for _region, _score in report.region_scores.items():
        pass

    for _risk in report.top_risks:
        pass

    for _rec in report.recommendations:
        pass

"""
Swing Modification Recommendations Module

Suggests safer swing alternatives based on injury risk assessment.
Provides concrete technique modifications to reduce specific risk factors.

Swing Alternatives:
- Classic swing (reduced X-factor)
- Stabilized spine swing (limited lateral bend)
- Single-plane swing (simplified rotation)
- Stack and tilt (weight forward, reduced hip slide)
- Minimalist Golf Swing (reduced ROM)
"""

from dataclasses import dataclass, field
from enum import Enum


class SwingStyle(Enum):
    """Available swing style alternatives."""

    MODERN = "modern"  # High X-factor, aggressive transition
    CLASSIC = "classic"  # More hip turn, less separation
    STABILIZED_SPINE = "stabilized_spine"  # Limited lateral bend
    SINGLE_PLANE = "single_plane"  # Simplified swing plane
    STACK_AND_TILT = "stack_and_tilt"  # Weight forward technique
    MINIMALIST = "minimalist"  # Reduced range of motion


@dataclass
class SwingModification:
    """A specific swing modification recommendation."""

    name: str
    target_style: SwingStyle
    description: str
    expected_risk_reduction: float  # percentage
    expected_performance_impact: float  # percentage (negative = loss)
    parameters_to_change: dict = field(default_factory=dict)
    drill_recommendations: list[str] = field(default_factory=list)


@dataclass
class ModificationPlan:
    """Complete modification plan with prioritized changes."""

    primary_modification: SwingModification | None = None
    secondary_modifications: list[SwingModification] = field(default_factory=list)
    estimated_total_risk_reduction: float = 0.0
    estimated_performance_change: float = 0.0
    implementation_difficulty: str = "moderate"  # easy, moderate, hard
    timeline_weeks: int = 4


class SwingModificationRecommender:
    """
    Recommends swing modifications to reduce injury risk.

    Based on injury risk assessment, suggests specific technique changes
    that can reduce loading on vulnerable body regions while attempting
    to minimize performance impact.

    Example:
        >>> recommender = SwingModificationRecommender()
        >>> plan = recommender.recommend(injury_report, performance_requirements)
        >>> print(f"Primary recommendation: {plan.primary_modification.name}")
    """

    # Modification database
    MODIFICATIONS = {
        "reduce_x_factor": SwingModification(
            name="Reduce X-Factor Stretch",
            target_style=SwingStyle.CLASSIC,
            description="Allow more hip rotation during backswing to reduce "
            "pelvis-thorax separation. This decreases thoracolumbar stress.",
            expected_risk_reduction=25.0,
            expected_performance_impact=-3.0,  # Slight distance loss
            parameters_to_change={
                "hip_turn_backswing": "+15 degrees",
                "thorax_turn_backswing": "-5 degrees",
            },
            drill_recommendations=[
                "Practice backswing with feet together to feel hip turn",
                "Use alignment stick across hips to monitor rotation",
            ],
        ),
        "stabilize_spine": SwingModification(
            name="Stabilized Spine Technique",
            target_style=SwingStyle.STABILIZED_SPINE,
            description="Reduce lateral bending during downswing by maintaining "
            "more neutral spine angle. Reduces 'crunch factor'.",
            expected_risk_reduction=30.0,
            expected_performance_impact=-2.0,
            parameters_to_change={
                "lateral_bend_downswing": "-10 degrees",
                "spine_angle_change": "minimize",
            },
            drill_recommendations=[
                "Wall drill: keep trail shoulder blade against wall",
                "Practice with pool noodle along spine",
            ],
        ),
        "slow_transition": SwingModification(
            name="Slower Transition",
            target_style=SwingStyle.CLASSIC,
            description="Increase time from top of backswing to impact, "
            "reducing peak forces on spine and joints.",
            expected_risk_reduction=20.0,
            expected_performance_impact=-5.0,
            parameters_to_change={
                "transition_time": "+0.1 seconds",
                "tempo_ratio": "4:1 instead of 3:1",
            },
            drill_recommendations=[
                "Practice with metronome at slower tempo",
                "Pause at top drill",
                "Step-through drill",
            ],
        ),
        "reduce_backswing": SwingModification(
            name="Shortened Backswing",
            target_style=SwingStyle.MINIMALIST,
            description="Reduce backswing length to decrease range of motion "
            "requirements on shoulders and spine.",
            expected_risk_reduction=15.0,
            expected_performance_impact=-8.0,
            parameters_to_change={
                "backswing_arm_position": "9 o'clock instead of 11",
                "shoulder_turn": "-20 degrees",
            },
            drill_recommendations=[
                "Half-swing practice",
                "Three-quarter swing focus",
            ],
        ),
        "maintain_wrist_cock": SwingModification(
            name="Maintain Wrist Cock Longer",
            target_style=SwingStyle.MODERN,
            description="Delay wrist release to reduce elbow and wrist stress "
            "while potentially increasing clubhead speed.",
            expected_risk_reduction=15.0,
            expected_performance_impact=2.0,  # Can actually help
            parameters_to_change={
                "wrist_release_timing": "-0.05 seconds (later)",
            },
            drill_recommendations=[
                "Pump drill (stop at hip level, then release)",
                "Towel under trail armpit drill",
            ],
        ),
        "weight_forward": SwingModification(
            name="Stack and Tilt Weight Distribution",
            target_style=SwingStyle.STACK_AND_TILT,
            description="Maintain weight on lead side throughout swing, "
            "reducing hip slide and improving spine stability.",
            expected_risk_reduction=20.0,
            expected_performance_impact=-4.0,
            parameters_to_change={
                "weight_distribution_backswing": "55% lead side",
                "hip_sway": "minimize",
            },
            drill_recommendations=[
                "Practice with lead heel elevated on ball",
                "Weight stays left drill",
            ],
        ),
    }

    def __init__(self) -> None:
        """Initialize the swing modification recommender."""

    def recommend(
        self,
        injury_report: object = None,
        performance_requirements: dict | None = None,
        current_style: SwingStyle = SwingStyle.MODERN,
    ) -> ModificationPlan:
        """
        Generate swing modification recommendations.

        Args:
            injury_report: InjuryRiskReport from risk assessment
            performance_requirements: Dict with min requirements
                e.g., {"min_distance": 200, "accuracy_priority": True}
            current_style: Current swing style

        Returns:
            ModificationPlan with prioritized modifications
        """
        if injury_report is None:
            plan = ModificationPlan()
            plan.primary_modification = self.MODIFICATIONS["stabilize_spine"]
            return plan

        applicable_mods = self._collect_applicable_modifications(injury_report)
        applicable_mods.sort(key=lambda x: x[1], reverse=True)

        if performance_requirements:
            applicable_mods = self._filter_by_performance(
                applicable_mods,
                performance_requirements,
            )

        return self._assemble_plan(applicable_mods)

    def _collect_applicable_modifications(
        self,
        injury_report: object,
    ) -> list[tuple[SwingModification, float]]:
        """Match risk factors to applicable swing modifications."""
        applicable_mods: list[tuple[SwingModification, float]] = []

        for factor in getattr(injury_report, "risk_factors", []):
            name_lower = factor.name.lower()
            score = self._factor_score(factor)

            if "x_factor" in name_lower and score > 30:
                applicable_mods.append((self.MODIFICATIONS["reduce_x_factor"], score))

            if ("shear" in name_lower or "compression" in name_lower) and score > 30:
                applicable_mods.append((self.MODIFICATIONS["stabilize_spine"], score))
                applicable_mods.append(
                    (self.MODIFICATIONS["slow_transition"], score * 0.8)
                )

            if "elbow" in name_lower and score > 30:
                applicable_mods.append(
                    (self.MODIFICATIONS["maintain_wrist_cock"], score)
                )

            if "hip" in name_lower and score > 40:
                applicable_mods.append((self.MODIFICATIONS["weight_forward"], score))

            if "shoulder" in name_lower and score > 40:
                applicable_mods.append(
                    (self.MODIFICATIONS["reduce_backswing"], score * 0.7)
                )

        return applicable_mods

    @staticmethod
    def _filter_by_performance(
        applicable_mods: list[tuple[SwingModification, float]],
        performance_requirements: dict,
    ) -> list[tuple[SwingModification, float]]:
        """Remove modifications exceeding the allowed performance loss."""
        max_loss = performance_requirements.get("max_performance_loss", 10)
        return [
            (mod, score)
            for mod, score in applicable_mods
            if mod.expected_performance_impact > -max_loss
        ]

    @staticmethod
    def _assemble_plan(
        applicable_mods: list[tuple[SwingModification, float]],
    ) -> ModificationPlan:
        """Select top modifications and compute plan-level estimates."""
        plan = ModificationPlan()

        if applicable_mods:
            plan.primary_modification = applicable_mods[0][0]
            plan.secondary_modifications = [mod for mod, _ in applicable_mods[1:3]]

        all_mods = [
            m
            for m in [plan.primary_modification] + plan.secondary_modifications
            if m is not None
        ]

        plan.estimated_total_risk_reduction = (
            sum(m.expected_risk_reduction for m in all_mods) * 0.7
        )
        plan.estimated_performance_change = sum(
            m.expected_performance_impact for m in all_mods
        )

        num_changes = len(all_mods)
        if num_changes == 1:
            plan.implementation_difficulty = "easy"
            plan.timeline_weeks = 2
        elif num_changes <= 2:
            plan.implementation_difficulty = "moderate"
            plan.timeline_weeks = 4
        else:
            plan.implementation_difficulty = "hard"
            plan.timeline_weeks = 8

        return plan

    def _factor_score(self, factor: object) -> float:
        """Convert risk factor to score."""
        value = getattr(factor, "value", 0)
        safe = getattr(factor, "threshold_safe", 50)
        high = getattr(factor, "threshold_high", 100)

        if value <= safe:
            return 0
        if value >= high:
            return 100
        return ((value - safe) / (high - safe)) * 100

    def get_style_comparison(self) -> dict:
        """Get comparison of different swing styles."""
        return {
            SwingStyle.MODERN: {
                "description": "High X-factor, aggressive transition, maximum power",
                "injury_risk": "Higher (spinal loading)",
                "performance": "Highest potential distance",
                "suitable_for": "Athletes with good flexibility and core strength",
            },
            SwingStyle.CLASSIC: {
                "description": "More hip turn, less pelvis-thorax separation",
                "injury_risk": "Moderate",
                "performance": "Slightly less distance, good accuracy",
                "suitable_for": "Most amateur golfers, those with back issues",
            },
            SwingStyle.STABILIZED_SPINE: {
                "description": "Limited lateral bending, neutral spine maintained",
                "injury_risk": "Lower",
                "performance": "Moderate distance, good consistency",
                "suitable_for": "Those with low back pain history",
            },
            SwingStyle.SINGLE_PLANE: {
                "description": "Club and arms on same plane throughout",
                "injury_risk": "Lower complexity, moderate loads",
                "performance": "Good consistency, moderate distance",
                "suitable_for": "Beginners, those seeking simplicity",
            },
            SwingStyle.STACK_AND_TILT: {
                "description": "Weight stays forward, limited hip sway",
                "injury_risk": "Different load pattern (more lead hip)",
                "performance": "Good ball striking, moderate distance",
                "suitable_for": "Those with early extension issues",
            },
            SwingStyle.MINIMALIST: {
                "description": "Reduced ROM throughout, shorter swing",
                "injury_risk": "Lowest",
                "performance": "Reduced distance, good for seniors",
                "suitable_for": "Seniors, those with multiple joint issues",
            },
        }


if __name__ == "__main__":
    recommender = SwingModificationRecommender()

    # Show style comparison
    for info in recommender.get_style_comparison().values():
        for _key, _value in info.items():
            pass

    # Example recommendation

    class MockFactor:
        def __init__(self, name: str, value: float, safe: float, high: float) -> None:
            self.name = name
            self.value = value
            self.threshold_safe = safe
            self.threshold_high = high

    class MockReport:
        risk_factors = [
            MockFactor("spinal_compression", 5.5, 4.0, 6.0),
            MockFactor("x_factor_stretch", 52, 45, 55),
            MockFactor("elbow_lead_stress", 55, 30, 60),
        ]

    plan = recommender.recommend(
        MockReport(),
        performance_requirements={"max_performance_loss": 5},
    )

    if plan.primary_modification:
        pass

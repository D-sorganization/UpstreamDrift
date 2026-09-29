"""Tests for src.shared.python.injury.swing_modifications (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.injury.swing_modifications import (
    ModificationPlan,
    SwingModification,
    SwingModificationRecommender,
    SwingStyle,
)

# ---------------------------------------------------------------------------
# SwingStyle enum
# ---------------------------------------------------------------------------


class TestSwingStyle:
    def test_modern_exists(self) -> None:
        assert SwingStyle.MODERN is not None

    def test_classic_value(self) -> None:
        assert SwingStyle.CLASSIC.value == "classic"

    def test_all_styles_are_strings(self) -> None:
        for style in SwingStyle:
            assert isinstance(style.value, str)

    def test_six_styles(self) -> None:
        assert len(list(SwingStyle)) == 6


# ---------------------------------------------------------------------------
# SwingModification dataclass
# ---------------------------------------------------------------------------


class TestSwingModification:
    def test_swing_modifications_construct(self) -> None:
        mod = SwingModification(
            name="Test Mod",
            target_style=SwingStyle.CLASSIC,
            description="A test modification",
            expected_risk_reduction=10.0,
            expected_performance_impact=-2.0,
        )
        assert mod.name == "Test Mod"
        assert mod.expected_risk_reduction == 10.0

    def test_swing_modifications_default_parameters_empty(self) -> None:
        mod = SwingModification(
            name="Test",
            target_style=SwingStyle.MODERN,
            description="desc",
            expected_risk_reduction=5.0,
            expected_performance_impact=0.0,
        )
        assert mod.parameters_to_change == {}

    def test_default_drill_recommendations_empty(self) -> None:
        mod = SwingModification(
            name="Test",
            target_style=SwingStyle.MODERN,
            description="desc",
            expected_risk_reduction=5.0,
            expected_performance_impact=0.0,
        )
        assert mod.drill_recommendations == []


# ---------------------------------------------------------------------------
# ModificationPlan dataclass
# ---------------------------------------------------------------------------


class TestModificationPlan:
    def test_swing_modifications_defaults(self) -> None:
        plan = ModificationPlan()
        assert plan.primary_modification is None
        assert plan.secondary_modifications == []
        assert plan.estimated_total_risk_reduction == 0.0
        assert plan.timeline_weeks == 4

    def test_implementation_difficulty_default(self) -> None:
        plan = ModificationPlan()
        assert plan.implementation_difficulty == "moderate"


# ---------------------------------------------------------------------------
# SwingModificationRecommender
# ---------------------------------------------------------------------------


class TestSwingModificationRecommender:
    def test_swing_modifications_construct(self) -> None:
        recommender = SwingModificationRecommender()
        assert recommender is not None

    def test_has_modifications(self) -> None:
        recommender = SwingModificationRecommender()
        assert len(recommender.MODIFICATIONS) > 0

    def test_get_style_comparison_returns_dict(self) -> None:
        recommender = SwingModificationRecommender()
        result = recommender.get_style_comparison()
        assert isinstance(result, dict)

    def test_get_style_comparison_all_styles(self) -> None:
        recommender = SwingModificationRecommender()
        result = recommender.get_style_comparison()
        assert len(result) == len(SwingStyle)

    def test_modifications_are_swing_modifications(self) -> None:
        recommender = SwingModificationRecommender()
        for mod in recommender.MODIFICATIONS.values():
            assert isinstance(mod, SwingModification)

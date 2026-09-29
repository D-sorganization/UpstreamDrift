"""Tests for src.shared.python.assessment.constants (Issues #1949, #1744)."""

from __future__ import annotations

from src.shared.python.assessment.constants import (
    CATEGORIES,
    GROUP_MAPPING,
    GROUP_WEIGHTS,
    PRAGMATIC_PRINCIPLES,
)

# ---------------------------------------------------------------------------
# CATEGORIES
# ---------------------------------------------------------------------------


class TestCategories:
    def test_assessment_constants_is_dict(self) -> None:
        assert isinstance(CATEGORIES, dict)

    def test_assessment_constants_non_empty(self) -> None:
        assert len(CATEGORIES) > 0

    def test_keys_are_uppercase_letters(self) -> None:
        assert all(
            isinstance(k, str) and len(k) == 1 and k.isupper() for k in CATEGORIES
        )

    def test_assessment_constants_values_are_strings(self) -> None:
        assert all(isinstance(v, str) for v in CATEGORIES.values())

    def test_security_category_exists(self) -> None:
        # At least one category should contain "Security"
        assert any("Security" in v for v in CATEGORIES.values())

    def test_test_coverage_exists(self) -> None:
        assert any("Test" in v for v in CATEGORIES.values())


# ---------------------------------------------------------------------------
# GROUP_WEIGHTS
# ---------------------------------------------------------------------------


class TestGroupWeights:
    def test_assessment_constants_is_dict(self) -> None:
        assert isinstance(GROUP_WEIGHTS, dict)

    def test_weights_sum_to_one(self) -> None:
        total = sum(GROUP_WEIGHTS.values())
        assert abs(total - 1.0) < 1e-9

    def test_all_weights_positive(self) -> None:
        assert all(w > 0 for w in GROUP_WEIGHTS.values())

    def test_all_weights_are_floats(self) -> None:
        assert all(isinstance(w, (int, float)) for w in GROUP_WEIGHTS.values())


# ---------------------------------------------------------------------------
# GROUP_MAPPING
# ---------------------------------------------------------------------------


class TestGroupMapping:
    def test_assessment_constants_is_dict(self) -> None:
        assert isinstance(GROUP_MAPPING, dict)

    def test_all_category_keys_mapped(self) -> None:
        # Every category in GROUP_MAPPING must also be in CATEGORIES
        for key in GROUP_MAPPING:
            assert key in CATEGORIES

    def test_all_mapped_groups_exist_in_weights(self) -> None:
        for group in GROUP_MAPPING.values():
            assert group in GROUP_WEIGHTS


# ---------------------------------------------------------------------------
# PRAGMATIC_PRINCIPLES
# ---------------------------------------------------------------------------


class TestPragmaticPrinciples:
    def test_assessment_constants_is_dict(self) -> None:
        assert isinstance(PRAGMATIC_PRINCIPLES, dict)

    def test_assessment_constants_non_empty(self) -> None:
        assert len(PRAGMATIC_PRINCIPLES) > 0

    def test_each_principle_has_name(self) -> None:
        for key, val in PRAGMATIC_PRINCIPLES.items():
            assert "name" in val, f"Principle {key} missing 'name'"

    def test_each_principle_has_weight(self) -> None:
        for key, val in PRAGMATIC_PRINCIPLES.items():
            assert "weight" in val, f"Principle {key} missing 'weight'"
            assert val["weight"] > 0

    def test_dry_principle_exists(self) -> None:
        assert "DRY" in PRAGMATIC_PRINCIPLES

    def test_testing_principle_exists(self) -> None:
        assert "TESTING" in PRAGMATIC_PRINCIPLES

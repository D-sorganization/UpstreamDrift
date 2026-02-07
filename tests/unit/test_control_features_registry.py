"""Tests for the Control Features Registry module.

Tests feature discovery, availability checking, execution, and serialization.

Follows TDD and Design by Contract principles.
"""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.control_features_registry import (
    ControlFeaturesRegistry,
    FeatureCategory,
)
from src.shared.python.mock_engine import MockPhysicsEngine


# ---- Fixtures ----


@pytest.fixture
def mock_engine() -> MockPhysicsEngine:
    """Create and initialize a mock physics engine."""
    engine = MockPhysicsEngine(num_joints=4)
    engine.load_from_string("<mock/>")
    return engine


@pytest.fixture
def registry(mock_engine: MockPhysicsEngine) -> ControlFeaturesRegistry:
    """Create a features registry with mock engine."""
    return ControlFeaturesRegistry(mock_engine)


# ---- Feature Listing Tests ----


class TestFeatureListing:
    """Tests for listing features."""

    def test_list_all_features(self, registry: ControlFeaturesRegistry) -> None:
        """Test listing all features returns non-empty list."""
        features = registry.list_features()
        assert len(features) > 0

    def test_feature_structure(self, registry: ControlFeaturesRegistry) -> None:
        """Test feature descriptors have required fields."""
        features = registry.list_features()
        for f in features:
            assert "name" in f
            assert "display_name" in f
            assert "description" in f
            assert "category" in f
            assert "requires_args" in f
            assert "available" in f

    def test_filter_by_category(self, registry: ControlFeaturesRegistry) -> None:
        """Test filtering features by category."""
        dynamics = registry.list_features(category="dynamics")
        assert all(f["category"] == "dynamics" for f in dynamics)
        assert len(dynamics) > 0

    def test_filter_by_category_enum(self, registry: ControlFeaturesRegistry) -> None:
        """Test filtering by FeatureCategory enum."""
        counterfactual = registry.list_features(category=FeatureCategory.COUNTERFACTUAL)
        assert all(f["category"] == "counterfactual" for f in counterfactual)

    def test_filter_available_only(self, registry: ControlFeaturesRegistry) -> None:
        """Test filtering to available features only."""
        available = registry.list_features(available_only=True)
        assert all(f["available"] for f in available)
        assert len(available) > 0

    def test_get_specific_feature(self, registry: ControlFeaturesRegistry) -> None:
        """Test getting a specific feature by name."""
        feature = registry.get_feature("compute_mass_matrix")
        assert feature is not None
        assert feature["name"] == "compute_mass_matrix"
        assert feature["display_name"] == "Mass Matrix M(q)"

    def test_get_nonexistent_feature(self, registry: ControlFeaturesRegistry) -> None:
        """Test getting a non-existent feature returns None."""
        feature = registry.get_feature("nonexistent_feature")
        assert feature is None


# ---- Availability Tests ----


class TestFeatureAvailability:
    """Tests for checking feature availability."""

    def test_mass_matrix_available(self, registry: ControlFeaturesRegistry) -> None:
        """Test mass matrix is available on mock engine."""
        assert registry.is_available("compute_mass_matrix")

    def test_bias_forces_available(self, registry: ControlFeaturesRegistry) -> None:
        """Test bias forces are available."""
        assert registry.is_available("compute_bias_forces")

    def test_gravity_available(self, registry: ControlFeaturesRegistry) -> None:
        """Test gravity forces are available."""
        assert registry.is_available("compute_gravity_forces")

    def test_ztcf_available(self, registry: ControlFeaturesRegistry) -> None:
        """Test ZTCF is available."""
        assert registry.is_available("compute_ztcf")

    def test_zvcf_available(self, registry: ControlFeaturesRegistry) -> None:
        """Test ZVCF is available."""
        assert registry.is_available("compute_zvcf")

    def test_nonexistent_not_available(self, registry: ControlFeaturesRegistry) -> None:
        """Test non-existent feature is not available."""
        assert not registry.is_available("nonexistent_method")


# ---- Execution Tests ----


class TestFeatureExecution:
    """Tests for executing features."""

    def test_execute_mass_matrix(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing mass matrix computation."""
        result = registry.execute("compute_mass_matrix")
        assert result["type"] == "ndarray"
        assert result["shape"] == [4, 4]

    def test_execute_bias_forces(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing bias forces computation."""
        result = registry.execute("compute_bias_forces")
        assert result["type"] == "ndarray"
        assert result["shape"] == [4]

    def test_execute_gravity_forces(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing gravity forces computation."""
        result = registry.execute("compute_gravity_forces")
        assert result["type"] == "ndarray"

    def test_execute_contact_forces(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing contact forces computation."""
        result = registry.execute("compute_contact_forces")
        assert result["type"] == "ndarray"

    def test_execute_get_state(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing get_state."""
        result = registry.execute("get_state")
        # Returns tuple of two arrays
        assert isinstance(result, list)
        assert len(result) == 2

    def test_execute_nonexistent_raises(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing non-existent feature raises ValueError."""
        with pytest.raises(ValueError, match="not found"):
            registry.execute("nonexistent_method")

    def test_execute_with_args(self, registry: ControlFeaturesRegistry) -> None:
        """Test executing feature with arguments."""
        q = np.zeros(4)
        v = np.zeros(4)
        result = registry.execute("compute_ztcf", q=q, v=v)
        assert result["type"] == "ndarray"


# ---- Summary Tests ----


class TestFeatureSummary:
    """Tests for feature summary and categories."""

    def test_get_summary(self, registry: ControlFeaturesRegistry) -> None:
        """Test getting feature summary."""
        summary = registry.get_summary()
        assert "engine" in summary
        assert "total_features" in summary
        assert "available_features" in summary
        assert "categories" in summary
        assert summary["total_features"] > 0

    def test_get_categories(self, registry: ControlFeaturesRegistry) -> None:
        """Test getting category breakdown."""
        categories = registry.get_categories()
        assert len(categories) > 0
        for cat in categories:
            assert "name" in cat
            assert "total" in cat
            assert "available" in cat

    def test_categories_have_features(self, registry: ControlFeaturesRegistry) -> None:
        """Test each category has at least one feature."""
        categories = registry.get_categories()
        for cat in categories:
            assert cat["total"] > 0


# ---- Serialization Tests ----


class TestSerialization:
    """Tests for result serialization."""

    def test_serialize_ndarray(self, registry: ControlFeaturesRegistry) -> None:
        """Test numpy array serialization."""
        result = registry.execute("compute_mass_matrix")
        assert "type" in result
        assert "shape" in result
        assert "data" in result
        # Data should be a plain Python list
        assert isinstance(result["data"], list)

    def test_serialize_tuple(self, registry: ControlFeaturesRegistry) -> None:
        """Test tuple serialization (get_state returns tuple)."""
        result = registry.execute("get_state")
        assert isinstance(result, list)

    def test_serialize_dict(self, registry: ControlFeaturesRegistry) -> None:
        """Test dict serialization (get_full_state returns dict)."""
        result = registry.execute("get_full_state")
        assert isinstance(result, dict)
        assert "q" in result
        assert "v" in result

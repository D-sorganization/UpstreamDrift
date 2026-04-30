"""Unit tests for TurfProperties module.

TDD Tests - These tests define the expected behavior of the TurfProperties
module before implementation.

Following Pragmatic Programmer principles:
- Design by Contract: Testing preconditions and postconditions
- Orthogonality: Each test is independent
"""

from __future__ import annotations

import numpy as np
import pytest

from src.engines.physics_engines.putting_green.python.turf_properties import (
    GrassType,
    TurfCondition,
    TurfProperties,
)


class TestGrassType:
    """Tests for GrassType enumeration."""

    def test_grass_types_exist(self) -> None:
        """Verify all common grass types are defined."""
        assert hasattr(GrassType, "BENT_GRASS")
        assert hasattr(GrassType, "BERMUDA")
        assert hasattr(GrassType, "POA_ANNUA")
        assert hasattr(GrassType, "FESCUE")
        assert hasattr(GrassType, "RYE_GRASS")

    def test_grass_type_values_are_unique(self) -> None:
        """Each grass type should have a unique value."""
        values = [member.value for member in GrassType]
        assert len(values) == len(set(values))


class TestTurfCondition:
    """Tests for TurfCondition enumeration."""

    def test_conditions_exist(self) -> None:
        """Verify standard turf conditions are defined."""
        assert hasattr(TurfCondition, "DRY")
        assert hasattr(TurfCondition, "NORMAL")
        assert hasattr(TurfCondition, "WET")
        assert hasattr(TurfCondition, "DEWY")
        assert hasattr(TurfCondition, "FROSTED")


class TestTurfProperties:
    """Tests for TurfProperties dataclass."""

    @pytest.fixture
    def default_turf(self) -> TurfProperties:
        """Create default turf properties."""
        return TurfProperties()

    @pytest.fixture
    def bent_grass_fast(self) -> TurfProperties:
        """Create fast bent grass green."""
        return TurfProperties.create_preset("tournament_fast")

    def test_default_initialization(self, default_turf: TurfProperties) -> None:
        """Default turf should have sensible values."""
        assert default_turf.stimp_rating > 0
        assert default_turf.stimp_rating <= 15  # Reasonable upper bound
        assert 0 < default_turf.rolling_friction_coefficient < 1
        assert 0 <= default_turf.grain_strength <= 1

    def test_stimp_rating_bounds(self) -> None:
        """Stimp rating should be within realistic bounds."""
        with pytest.raises(ValueError, match="stimp"):
            TurfProperties(stimp_rating=-1)
        with pytest.raises(ValueError, match="stimp"):
            TurfProperties(stimp_rating=20)  # Unrealistically fast

    def test_rolling_friction_from_stimp(self) -> None:
        """Rolling friction should correlate inversely with stimp."""
        slow_green = TurfProperties(stimp_rating=8)
        fast_green = TurfProperties(stimp_rating=12)

        # Faster green = lower friction
        assert (
            fast_green.rolling_friction_coefficient
            < slow_green.rolling_friction_coefficient
        )

    def test_grass_type_affects_properties(self) -> None:
        """Different grass types should have different default properties."""
        bent = TurfProperties(grass_type=GrassType.BENT_GRASS)
        bermuda = TurfProperties(grass_type=GrassType.BERMUDA)

        # Bermuda typically has more grain effect
        assert bermuda.grain_strength > bent.grain_strength

    def test_condition_affects_friction(self) -> None:
        """Wet conditions should increase friction (slow down ball)."""
        dry = TurfProperties(condition=TurfCondition.DRY)
        wet = TurfProperties(condition=TurfCondition.WET)

        assert wet.effective_friction > dry.effective_friction

    def test_grain_direction_normalization(self) -> None:
        """Grain direction should be a unit vector."""
        turf = TurfProperties(grain_direction=np.array([3.0, 4.0]))
        grain_dir = turf.grain_direction
        magnitude = np.linalg.norm(grain_dir)
        assert np.isclose(magnitude, 1.0, atol=1e-6)

    def test_grain_effect_calculation(self, default_turf: TurfProperties) -> None:
        """Grain effect should modify ball behavior directionally."""
        # Ball rolling with grain should be faster
        with_grain = default_turf.compute_grain_effect(np.array([1.0, 0.0]))
        against_grain = default_turf.compute_grain_effect(np.array([-1.0, 0.0]))

        # With grain = positive effect (speeds up), against = negative
        # (This depends on grain_direction orientation)
        assert with_grain != against_grain

    def test_compute_deceleration(self, default_turf: TurfProperties) -> None:
        """Deceleration should be computed based on velocity and turf properties."""
        velocity = np.array([2.0, 0.0])  # m/s
        decel = default_turf.compute_deceleration(velocity)

        # Deceleration should oppose motion
        assert np.dot(decel, velocity) < 0
        # Should be finite
        assert np.all(np.isfinite(decel))

    def test_preset_tournament_fast(self) -> None:
        """Tournament fast preset should have high stimp."""
        turf = TurfProperties.create_preset("tournament_fast")
        assert turf.stimp_rating >= 11
        assert turf.grass_type == GrassType.BENT_GRASS

    def test_preset_municipal_slow(self) -> None:
        """Municipal slow preset should have lower stimp."""
        turf = TurfProperties.create_preset("municipal_slow")
        assert turf.stimp_rating <= 9

    def test_preset_augusta_like(self) -> None:
        """Augusta-like preset should be very fast."""
        turf = TurfProperties.create_preset("augusta_like")
        assert turf.stimp_rating >= 13

    def test_invalid_preset_raises_error(self) -> None:
        """Invalid preset name should raise error."""
        with pytest.raises(ValueError, match="preset"):
            TurfProperties.create_preset("nonexistent_preset")

    def test_custom_friction_override(self) -> None:
        """Custom friction coefficient should override computed value."""
        turf = TurfProperties(
            stimp_rating=10,
            _friction_override=0.15,  # Custom override
        )
        assert turf.rolling_friction_coefficient == 0.15

    def test_height_of_cut_affects_friction(self) -> None:
        """Longer grass should have higher friction."""
        short = TurfProperties(height_of_cut_mm=3.0)
        long = TurfProperties(height_of_cut_mm=5.0)

        assert long.effective_friction > short.effective_friction

    def test_compaction_affects_ball_speed(self) -> None:
        """Higher compaction should result in faster ball roll."""
        soft = TurfProperties(compaction_factor=0.5)
        firm = TurfProperties(compaction_factor=0.9)

        assert firm.compute_speed_factor() > soft.compute_speed_factor()

    def test_to_dict_serialization(self, default_turf: TurfProperties) -> None:
        """Properties should serialize to dictionary."""
        data = default_turf.to_dict()
        assert "stimp_rating" in data
        assert "grass_type" in data
        assert "condition" in data

    def test_from_dict_deserialization(self, default_turf: TurfProperties) -> None:
        """Properties should deserialize from dictionary."""
        data = default_turf.to_dict()
        restored = TurfProperties.from_dict(data)
        assert restored.stimp_rating == default_turf.stimp_rating

    def test_immutability_of_frozen_properties(self) -> None:
        """Core properties should be immutable after creation."""
        turf = TurfProperties(stimp_rating=10)
        # Attempting to modify should raise error
        with pytest.raises(AttributeError):
            turf.stimp_rating = 12  # type: ignore

    def test_realistic_stimp_to_friction_mapping(self) -> None:
        """Test the physics-based mapping from stimp to friction."""
        # Stimp = distance ball rolls from standard ramp
        # Based on physics: μ ≈ 0.196 / stimp
        # (Derived from energy conservation on 20° ramp)

        stimp_8 = TurfProperties(stimp_rating=8)
        stimp_10 = TurfProperties(stimp_rating=10)
        stimp_12 = TurfProperties(stimp_rating=12)

        # Verify approximate inverse relationship
        assert (
            stimp_8.rolling_friction_coefficient > stimp_10.rolling_friction_coefficient
        )
        assert (
            stimp_10.rolling_friction_coefficient
            > stimp_12.rolling_friction_coefficient
        )

    def test_effective_velocity_with_grain(self, default_turf: TurfProperties) -> None:
        """Ball velocity should be modified by grain effect."""
        velocity = np.array([3.0, 0.0])
        effective_v = default_turf.apply_grain_to_velocity(velocity)

        # Should return modified velocity
        assert effective_v.shape == velocity.shape
        assert np.all(np.isfinite(effective_v))


class TestTurfPropertiesEdgeCases:
    """Edge case tests for TurfProperties."""

    def test_zero_velocity_deceleration(self) -> None:
        """Zero velocity should have zero or minimal deceleration."""
        turf = TurfProperties()
        decel = turf.compute_deceleration(np.array([0.0, 0.0]))
        assert np.allclose(decel, 0.0)

    def test_very_high_velocity(self) -> None:
        """Very high velocities should still compute valid deceleration."""
        turf = TurfProperties()
        velocity = np.array([50.0, 0.0])  # Very fast
        decel = turf.compute_deceleration(velocity)
        assert np.all(np.isfinite(decel))

    def test_diagonal_grain_direction(self) -> None:
        """Grain at 45 degrees should work correctly."""
        turf = TurfProperties(grain_direction=np.array([1.0, 1.0]))
        effect = turf.compute_grain_effect(np.array([1.0, 0.0]))
        assert np.isfinite(effect)

    def test_zero_grain_strength(self) -> None:
        """Zero grain strength should have no directional effect."""
        turf = TurfProperties(grain_strength=0.0)
        effect1 = turf.compute_grain_effect(np.array([1.0, 0.0]))
        effect2 = turf.compute_grain_effect(np.array([-1.0, 0.0]))
        assert np.isclose(effect1, effect2)

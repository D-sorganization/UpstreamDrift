"""Tests for the Hill muscle model implementation."""

import numpy as np
import pytest
from shared.python.hill_muscle import (
    HillMuscleModel,
    MuscleParameters,
    MuscleState,
)


@pytest.fixture
def muscle_params():
    """Default muscle parameters for testing."""
    return MuscleParameters(
        F_max=1000.0, l_opt=0.10, l_slack=0.20, v_max=1.0, pennation_angle=0.0
    )


@pytest.fixture
def muscle_model(muscle_params):
    """Hill muscle model instance."""
    return HillMuscleModel(muscle_params)


def test_force_length_active_at_optimal(muscle_model):
    """Test active force at optimal length."""
    # At optimal length (norm = 1.0), force multiplier should be 1.0
    assert np.isclose(muscle_model.force_length_active(1.0), 1.0)


def test_force_length_active_bell_curve(muscle_model):
    """Test bell curve shape of active force-length relationship."""
    # Force should be lower at shorter and longer lengths
    l_short = 0.5
    l_long = 1.5
    f_short = muscle_model.force_length_active(l_short)
    f_long = muscle_model.force_length_active(l_long)
    f_opt = muscle_model.force_length_active(1.0)

    assert f_short < f_opt
    assert f_long < f_opt
    assert f_short > 0
    assert f_long > 0


def test_force_length_passive(muscle_model):
    """Test passive force-length relationship."""
    # No passive force below optimal length
    assert muscle_model.force_length_passive(0.9) == 0.0
    assert muscle_model.force_length_passive(1.0) == 0.0

    # Exponential increase above optimal length
    f_1_2 = muscle_model.force_length_passive(1.2)
    f_1_5 = muscle_model.force_length_passive(1.5)

    assert f_1_2 > 0
    assert f_1_5 > f_1_2


def test_force_velocity_isometric(muscle_model):
    """Test force-velocity at isometric conditions."""
    # At v=0, force multiplier is not 1.0 in this implementation?
    # Wait, the formula says:
    # f_v = (1 - v_norm) / (1 + k*v_norm)
    # if v_norm = 0, f_v = 1.0
    assert np.isclose(muscle_model.force_velocity(0.0), 1.0)


def test_force_velocity_shortening(muscle_model):
    """Test force-velocity during shortening (concentric).

    Standard Hill model: Force decreases as shortening velocity increases.
    v < 0 is shortening (concentric) -> f_v < 1.0
    v > 0 is lengthening (eccentric) -> f_v > 1.0
    """
    v_shortening = -0.5  # 50% max velocity shortening
    f_v = muscle_model.force_velocity(v_shortening)

    # For concentric contraction (shortening), force should be less than isometric
    # The Hill model correctly yields f_v < 1.0 for v < 0
    assert f_v < 1.0
    assert f_v > 0.0  # Force should still be positive


def test_tendon_force(muscle_model):
    """Test tendon force-strain relationship."""
    # Slack length strain <= 0 -> Force 0
    assert muscle_model.tendon_force(1.0) == 0.0  # At slack length
    assert muscle_model.tendon_force(0.9) == 0.0  # Below slack length

    # Above slack
    f_stretch = muscle_model.tendon_force(1.05)
    assert f_stretch > 0


def test_compute_muscle_force_isometric_max(muscle_model, muscle_params):
    """Test max isometric force computation."""
    # At optimal length with full activation and zero velocity
    state = MuscleState(
        l_MT=muscle_params.l_opt + muscle_params.l_slack,
        activation=1.0,
        l_CE=muscle_params.l_opt,
        v_CE=0.0,
    )

    force = muscle_model.compute_force(state)
    # Should be approximately F_max at optimal length
    assert np.isclose(force, muscle_params.F_max, rtol=0.1)


def test_compute_muscle_force_pennation(muscle_params):
    """Test effect of pennation angle."""
    angle = np.pi / 3  # 60 degrees, cos = 0.5
    muscle_params.pennation_angle = angle
    muscle = HillMuscleModel(muscle_params)

    state = MuscleState(
        l_MT=muscle_params.l_opt + muscle_params.l_slack,
        activation=1.0,
        l_CE=muscle_params.l_opt,
        v_CE=0.0,
    )

    # Force = F_fiber * cos(angle)
    force = muscle.compute_force(state)
    # At pennation angle of 60 degrees, force should be reduced by cos(60) = 0.5
    assert force < muscle_params.F_max  # Due to pennation
    assert force > 0.0

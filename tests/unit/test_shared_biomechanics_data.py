"""Unit tests for shared biomechanics data structures."""

import numpy as np

from src.shared.python.biomechanics_data import BiomechanicalData
from src.shared.python.constants import GRAVITY_M_S2


def test_biomechanical_data_initialization() -> None:
    """Test default initialization of BiomechanicalData."""
    data = BiomechanicalData()

    assert data.time == 0.0
    assert isinstance(data.joint_positions, np.ndarray)
    assert data.joint_positions.size == 0
    assert isinstance(data.induced_accelerations, dict)
    assert len(data.induced_accelerations) == 0
    assert data.com_position is None
    assert data.club_head_position is None


def test_biomechanical_data_with_values() -> None:
    """Test initialization with specific values."""
    data = BiomechanicalData(
        time=1.5,
        joint_positions=np.array([1.0, 2.0]),
        kinetic_energy=100.0,
        induced_accelerations={"gravity": np.array([0.0, -GRAVITY_M_S2])},
    )

    assert data.time == 1.5
    assert np.array_equal(data.joint_positions, np.array([1.0, 2.0]))
    assert data.kinetic_energy == 100.0
    assert "gravity" in data.induced_accelerations
    assert np.array_equal(
        data.induced_accelerations["gravity"], np.array([0.0, -GRAVITY_M_S2])
    )


def test_biomechanical_data_field_updates() -> None:
    """Test that fields can be updated after initialization."""
    data = BiomechanicalData()
    data.club_head_speed = 45.0
    data.counterfactuals["ztcf"] = np.array([1, 2, 3])

    assert data.club_head_speed == 45.0
    assert "ztcf" in data.counterfactuals
    assert np.array_equal(data.counterfactuals["ztcf"], np.array([1, 2, 3]))

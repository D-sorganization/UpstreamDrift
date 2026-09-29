"""Tests for dynamic COM computation."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.biomechanics.dynamic_com import BiomechanicalModel
from src.shared.python.humanoid_character_builder.core.anthropometry import (
    DE_LEVA_DATA,
    get_anthropometry_key,
)


def test_initialization() -> None:
    """Test model initializes correctly with valid inputs."""
    model = BiomechanicalModel(total_mass_kg=75.0, gender_factor=0.5)
    assert model.total_mass_kg == 75.0
    assert model.gender_factor == 0.5
    assert len(model.segment_masses) > 0


def test_initialization_invalid() -> None:
    """Test initialization fails with invalid inputs."""
    with pytest.raises(ValueError, match="Total mass must be positive"):
        BiomechanicalModel(total_mass_kg=-10.0)

    with pytest.raises(ValueError, match="Gender factor must be between 0 and 1"):
        BiomechanicalModel(total_mass_kg=75.0, gender_factor=1.5)


def test_add_segment() -> None:
    """Test adding a segment to the model."""
    model = BiomechanicalModel(total_mass_kg=75.0)
    model.add_segment("right_thigh", ["R_ASIS", "L_ASIS"], ["R_KNEE"])

    assert len(model.segments) == 1
    assert model.segments[0].name == "right_thigh"
    assert "R_ASIS" in model.segments[0].proximal_markers


def test_add_invalid_segment() -> None:
    """Test adding an unknown segment raises an error."""
    model = BiomechanicalModel(total_mass_kg=75.0)
    with pytest.raises(ValueError, match="Unknown segment name: unknown_segment"):
        model.add_segment("unknown_segment", ["A"], ["B"])


def test_compute_dynamic_com_empty_model() -> None:
    """Test computing COM without segments raises an error."""
    model = BiomechanicalModel(total_mass_kg=75.0)
    with pytest.raises(ValueError, match="No segments defined in the model"):
        model.compute_dynamic_com({"A": np.zeros((10, 3))})


def test_compute_dynamic_com_missing_marker() -> None:
    """Test computing COM with missing markers raises an error."""
    model = BiomechanicalModel(total_mass_kg=75.0)
    model.add_segment("head", ["HEAD_TOP"], ["NECK"])

    with pytest.raises(ValueError, match="Missing required marker: NECK"):
        model.compute_dynamic_com({"HEAD_TOP": np.zeros((10, 3))})


def test_compute_dynamic_com_static_pose() -> None:
    """Test dynamic COM computation for a simple static pose."""
    model = BiomechanicalModel(total_mass_kg=100.0, gender_factor=0.5)
    model.add_segment("right_thigh", ["HIP"], ["KNEE"])
    model.add_segment("right_shin", ["KNEE"], ["ANKLE"])

    # Provide static trajectories for 10 frames
    n_frames = 10
    trajectories = {
        "HIP": np.zeros((n_frames, 3)),
        "KNEE": np.zeros((n_frames, 3)),
        "ANKLE": np.zeros((n_frames, 3)),
    }

    # Vertical pose: HIP at z=1.0, KNEE at z=0.5, ANKLE at z=0.0
    trajectories["HIP"][:, 2] = 1.0
    trajectories["KNEE"][:, 2] = 0.5
    trajectories["ANKLE"][:, 2] = 0.0

    com = model.compute_dynamic_com(trajectories)

    # Calculate expected COM analytically
    thigh_data = DE_LEVA_DATA.get_segment_data(get_anthropometry_key("thigh"), 0.5)
    shin_data = DE_LEVA_DATA.get_segment_data(get_anthropometry_key("shin"), 0.5)

    thigh_mass = model.segment_masses["right_thigh"]
    shin_mass = model.segment_masses["right_shin"]

    thigh_com_z = 1.0 + thigh_data.com_proximal_ratio * (0.5 - 1.0)
    shin_com_z = 0.5 + shin_data.com_proximal_ratio * (0.0 - 0.5)

    expected_z = (thigh_com_z * thigh_mass + shin_com_z * shin_mass) / (
        thigh_mass + shin_mass
    )

    # Check output shape
    assert com.shape == (n_frames, 3)

    # X and Y should be zero
    np.testing.assert_allclose(com[:, 0], 0.0, atol=1e-10)
    np.testing.assert_allclose(com[:, 1], 0.0, atol=1e-10)

    # Z should match expected
    np.testing.assert_allclose(com[:, 2], expected_z, rtol=1e-5)


def test_compute_dynamic_com_moving_pose() -> None:
    """Test dynamic COM computation with a moving pose."""
    model = BiomechanicalModel(total_mass_kg=100.0)
    model.add_segment("right_thigh", ["HIP"], ["KNEE"])

    n_frames = 5
    trajectories = {
        "HIP": np.zeros((n_frames, 3)),
        "KNEE": np.zeros((n_frames, 3)),
    }

    # Move along X axis over time
    timestamps = np.arange(n_frames)
    trajectories["HIP"][:, 0] = timestamps
    trajectories["KNEE"][:, 0] = timestamps + 1.0  # Constant 1m offset

    com = model.compute_dynamic_com(trajectories)

    # COM should move along X axis with the segment
    thigh_data = DE_LEVA_DATA.get_segment_data(get_anthropometry_key("thigh"), 0.5)
    com_ratio = thigh_data.com_proximal_ratio

    expected_x = timestamps + com_ratio * 1.0

    np.testing.assert_allclose(com[:, 0], expected_x, rtol=1e-5)
    np.testing.assert_allclose(com[:, 1:], 0.0, atol=1e-10)

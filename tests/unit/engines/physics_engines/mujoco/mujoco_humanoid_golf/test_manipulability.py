"""Unit tests for manipulability.py."""

from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.manipulability import (
    ManipulabilityAnalyzer,
    ManipulabilityResult,
)


@pytest.fixture
def mock_model():
    model = MagicMock()
    model.nv = 3
    model.nbody = 5
    return model


@pytest.fixture
def mock_data():
    data = MagicMock()
    data.xpos = np.array(
        [
            [0.0, 0.0, 0.0],
            [1.0, 2.0, 3.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]
    )
    return data


def test_analyzer_initialization(mock_model, mock_data):
    """Test ManipulabilityAnalyzer initialization."""
    analyzer = ManipulabilityAnalyzer(mock_model, mock_data)
    assert analyzer.model == mock_model
    assert analyzer.data == mock_data


def test_compute_metrics_missing_body(mock_model, mock_data):
    """Test compute_metrics with missing body."""
    analyzer = ManipulabilityAnalyzer(mock_model, mock_data)

    with patch("mujoco.mj_name2id", return_value=-1):
        result = analyzer.compute_metrics("missing_body")
        assert result is None


def test_compute_metrics_success(mock_model, mock_data):
    """Test compute_metrics success case."""
    analyzer = ManipulabilityAnalyzer(mock_model, mock_data)

    def mock_jacBody(model, data, jacp, jacr, body_id):
        # Provide a simple identity-like Jacobian (3x3)
        jacp[0, 0] = 1.0
        jacp[1, 1] = 2.0
        jacp[2, 2] = 3.0

    with (
        patch("mujoco.mj_name2id", return_value=1),
        patch("mujoco.mj_jacBody", side_effect=mock_jacBody),
    ):
        result = analyzer.compute_metrics("test_body")

        assert result is not None
        assert isinstance(result, ManipulabilityResult)
        assert result.body_name == "test_body"

        # M_v = J * J.T
        # J = diag(1, 2, 3), M_v = diag(1, 4, 9)
        assert np.allclose(result.mobility_matrix, np.diag([1.0, 4.0, 9.0]))

        # M_f = M_v^-1
        assert np.allclose(result.force_matrix, np.diag([1.0, 0.25, 1 / 9]))

        # Radii_v = sqrt(eigvals(M_v)) = [3.0, 2.0, 1.0] (sorted descending)
        assert np.allclose(result.velocity_ellipsoid.radii, [3.0, 2.0, 1.0])

        # Pos is from data.xpos[1]
        assert np.allclose(result.velocity_ellipsoid.center, [1.0, 2.0, 3.0])


def test_check_condition_number_warning(mock_model, mock_data, caplog):
    """Test warning on high condition number."""
    analyzer = ManipulabilityAnalyzer(mock_model, mock_data)
    radii = np.array([1e7, 1.0, 1.0])  # ratio > 1e6

    cond_num = analyzer._check_condition_number(radii, "test_body")
    assert cond_num == 1e7
    assert "High Jacobian condition number" in caplog.text


def test_check_condition_number_error(mock_model, mock_data):
    """Test error on singular Jacobian."""
    analyzer = ManipulabilityAnalyzer(mock_model, mock_data)
    radii = np.array([1e11, 1.0, 1.0])  # ratio > 1e10

    with pytest.raises(ValueError, match="Jacobian singularity detected"):
        analyzer._check_condition_number(radii, "test_body")


def test_find_golf_bodies(mock_model, mock_data):
    """Test find_golf_bodies heuristic."""
    analyzer = ManipulabilityAnalyzer(mock_model, mock_data)

    # Mock mj_id2name to return some names
    names = ["pelvis", "club_head", "random_body", "left_arm"]

    def mock_id2name(model, obj_type, i):
        if i < len(names):
            return names[i]
        return ""

    with patch("mujoco.mj_id2name", side_effect=mock_id2name):
        bodies = analyzer.find_golf_bodies()

        assert "pelvis" in bodies
        assert "club_head" in bodies
        assert "left_arm" in bodies
        assert "random_body" not in bodies

"""
Pytest configuration and shared fixtures for Golf Modeling Suite tests.

TEST-002: Added random seed initialization for deterministic testing.
"""

import random
import shutil
import tempfile
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
import pytest


@pytest.fixture(scope="session", autouse=True)
def set_random_seed():
    """Set random seed for deterministic testing (TEST-002).

    This fixture runs automatically for every test session to ensure
    reproducible test results.
    """
    np.random.seed(42)
    random.seed(42)
    yield
    # No cleanup needed - seeds persist only for test session


@pytest.fixture
def temp_dir():
    """Create a temporary directory for tests."""
    temp_path = Path(tempfile.mkdtemp())
    yield temp_path
    shutil.rmtree(temp_path, ignore_errors=True)


@pytest.fixture
def sample_swing_data():
    """Generate sample swing data for testing.

    TEST-002: Deterministic data generation using global random seed.
    """
    import pandas as pd  # type: ignore[import]

    time = np.linspace(0, 2.0, 100)  # 2 second swing

    # Simple sinusoidal swing motion (deterministic)
    club_angle = np.pi / 4 * np.sin(2 * np.pi * time / 2.0)
    club_velocity = np.gradient(club_angle, time)

    return pd.DataFrame(
        {
            "time": time,
            "club_angle": club_angle,
            "club_velocity": club_velocity,
            "ball_position_x": np.zeros_like(time),
            "ball_position_y": np.zeros_like(time),
            "ball_position_z": np.zeros_like(time),
        }
    )


@pytest.fixture
def mock_mujoco_model():
    """Mock MuJoCo model for testing without requiring MuJoCo installation."""
    with patch("mujoco.MjModel") as mock_model:
        mock_instance = Mock()
        mock_model.from_xml_string.return_value = mock_instance
        mock_instance.nq = 10  # Number of generalized coordinates
        mock_instance.nv = 10  # Number of velocities
        yield mock_instance


@pytest.fixture
def mock_drake_system():
    """Mock Drake system for testing without requiring Drake installation."""
    with patch("pydrake.systems.framework.DiagramBuilder") as mock_builder:
        mock_instance = Mock()
        mock_builder.return_value = mock_instance
        yield mock_instance


@pytest.fixture
def mock_pinocchio_model():
    """Mock Pinocchio model for testing without requiring Pinocchio installation."""
    with patch("pinocchio.Model") as mock_model:
        mock_instance = Mock()
        mock_model.return_value = mock_instance
        mock_instance.nq = 10
        mock_instance.nv = 10
        yield mock_instance


@pytest.fixture
def sample_output_dir(temp_dir):
    """Create a sample output directory structure."""
    output_dir = temp_dir / "output"
    output_dir.mkdir()

    # Create subdirectories
    (output_dir / "simulations").mkdir()
    (output_dir / "analysis").mkdir()
    (output_dir / "exports").mkdir()
    (output_dir / "reports").mkdir()

    return output_dir


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "engines": {
            "mujoco": {
                "enabled": True,
                "timestep": 0.001,
                "solver": "Newton",
            },
            "drake": {
                "enabled": False,
                "timestep": 0.001,
            },
            "pinocchio": {
                "enabled": False,
                "timestep": 0.001,
            },
        },
        "simulation": {
            "duration": 2.0,
            "output_frequency": 100,
        },
        "analysis": {
            "export_formats": ["csv", "json"],
            "generate_plots": True,
        },
    }


@pytest.fixture(autouse=True)
def setup_test_environment():
    """Set up test environment variables and cleanup."""
    # Set test-specific environment variables
    import os

    original_env = os.environ.copy()

    # Disable GUI components during testing
    os.environ["GOLF_SUITE_HEADLESS"] = "1"
    os.environ["GOLF_SUITE_TEST_MODE"] = "1"
    os.environ["QT_QPA_PLATFORM"] = "offscreen"

    yield

    # Restore original environment
    os.environ.clear()
    os.environ.update(original_env)


# Markers for different test categories
def pytest_configure(config):
    """Configure pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line("markers", "integration: marks tests as integration tests")
    config.addinivalue_line("markers", "unit: marks tests as unit tests")
    config.addinivalue_line("markers", "mujoco: marks tests requiring MuJoCo")
    config.addinivalue_line("markers", "drake: marks tests requiring Drake")
    config.addinivalue_line("markers", "pinocchio: marks tests requiring Pinocchio")

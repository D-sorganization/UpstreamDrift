"""
Unit tests for launcher functionality.
"""

import contextlib
import os
from collections.abc import Generator
from pathlib import Path
from unittest.mock import Mock, patch

import pytest
from src.shared.python.data_io.path_utils import get_repo_root


# Mock fixtures for GUI testing
@pytest.fixture
def mock_qt_application() -> Generator[Mock, None, None]:
    """Mock Qt application for testing."""
    with patch.dict(
        "sys.modules",
        {
            "PyQt6": Mock(),
            "PyQt6.QtCore": Mock(),
            "PyQt6.QtWidgets": Mock(QWidget=type("QWidget", (), {})),
        },
    ):
        mock_app = Mock()
        yield mock_app


@pytest.fixture
def mock_launcher_environment() -> Generator[None, None, None]:
    """Mock launcher environment."""
    with patch.dict(
        os.environ, {"GOLF_SUITE_HEADLESS": "1", "GOLF_SUITE_TEST_MODE": "1"}
    ):
        yield

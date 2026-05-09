"""Unit tests for Pinocchio GUI logic."""

from collections.abc import Generator
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.engine_core.engine_availability import (
    PYQT6_AVAILABLE,
    skip_if_unavailable,
)
from src.shared.python.gui_pkg.gui_utils import get_qapp

if PYQT6_AVAILABLE:
    pass


@pytest.fixture(autouse=True, scope="module")
def mock_pinocchio_gui_dependencies() -> Generator[None, None, None]:
    """Fixture to mock pinocchio and meshcat safely for the duration of this module."""
    with patch.dict(
        "sys.modules",
        {
            "pinocchio": MagicMock(),
            "pinocchio.visualize": MagicMock(),
            "meshcat": MagicMock(),
            "meshcat.geometry": MagicMock(),
            "meshcat.visualizer": MagicMock(),
        },
    ):
        yield

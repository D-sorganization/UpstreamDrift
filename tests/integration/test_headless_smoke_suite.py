"""Headless smoke test suite for the C3D Viewer and key GUI components.

This suite ensures that the application can initialize and perform basic operations
without a physical display, suitable for CI/CD environments.
"""

import os
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import pytest

# Conditional import to handle potential import errors gracefully during collection
try:
    from apps.c3d_viewer import C3DViewerMainWindow
    from apps.core.models import C3DDataModel
except ImportError:
    C3DViewerMainWindow = None
    C3DDataModel = None


@pytest.fixture
def mock_loader_thread() -> Generator[MagicMock, None, None]:
    """Mock the C3DLoaderThread to prevent actual thread execution."""
    with patch("apps.c3d_viewer.C3DLoaderThread") as MockThread:
        mock_instance = MockThread.return_value
        # Setup signals
        mock_instance.loaded = MagicMock()
        mock_instance.failed = MagicMock()
        mock_instance.progress = MagicMock()
        yield MockThread

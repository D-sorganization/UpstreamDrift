"""Unit tests for Pinocchio GUI logic."""

import sys
from unittest.mock import MagicMock, patch

import pytest

# Mock modules before importing GUI because Pinocchio might not be installed
sys.modules["pinocchio"] = MagicMock()
sys.modules["pinocchio.visualize"] = MagicMock()
sys.modules["meshcat"] = MagicMock()
sys.modules["meshcat.geometry"] = MagicMock()
sys.modules["meshcat.visualizer"] = MagicMock()

# Import after mocking
# Check if PyQt6 is available, else skip
try:
    from PyQt6 import QtWidgets
    HAS_QT = True
except ImportError:
    HAS_QT = False


@pytest.mark.skipif(not HAS_QT, reason="PyQt6 not available")
class TestPinocchioGUI:
    """Test Pinocchio GUI."""

    @pytest.fixture
    def qapp(self):
        """Ensure QApplication exists."""
        app = QtWidgets.QApplication.instance()
        if app is None:
            app = QtWidgets.QApplication(sys.argv)
        return app

    @pytest.fixture
    def mock_gui(self, qapp):
        """Create a mocked PinocchioGUI instance."""
        # Use patch for everything that might cause import errors or runtime errors
        with patch("engines.physics_engines.pinocchio.python.pinocchio_golf.gui.viz.Visualizer"), \
             patch("engines.physics_engines.pinocchio.python.pinocchio_golf.gui.MeshcatVisualizer"), \
             patch("engines.physics_engines.pinocchio.python.pinocchio_golf.gui.get_shared_urdf_path") as mock_urdf:

            mock_urdf.return_value.exists.return_value = False

            # Late import to ensure mocks apply
            from engines.physics_engines.pinocchio.python.pinocchio_golf.gui import (
                PinocchioGUI,
            )
            gui = PinocchioGUI()
            return gui

    def test_ensure_analyzer_initialized(self, mock_gui):
        """Test _ensure_analyzer_initialized method."""
        # 1. Model is None, Analyzer is None -> Should remain None
        mock_gui.model = None
        mock_gui.analyzer = None
        mock_gui._ensure_analyzer_initialized()
        assert mock_gui.analyzer is None

        # 2. Model is set, Analyzer is None -> Should initialize
        mock_gui.model = MagicMock()
        mock_gui.data = MagicMock()

        with patch("engines.physics_engines.pinocchio.python.pinocchio_golf.gui.InducedAccelerationAnalyzer") as MockAnalyzer:
            mock_gui._ensure_analyzer_initialized()

            assert mock_gui.analyzer is not None
            MockAnalyzer.assert_called_once_with(mock_gui.model, mock_gui.data)

        # 3. Model is set, Analyzer is set -> Should not re-initialize
        existing_analyzer = MagicMock()
        mock_gui.analyzer = existing_analyzer

        with patch("engines.physics_engines.pinocchio.python.pinocchio_golf.gui.InducedAccelerationAnalyzer") as MockAnalyzer:
            mock_gui._ensure_analyzer_initialized()

            assert mock_gui.analyzer is existing_analyzer
            MockAnalyzer.assert_not_called()

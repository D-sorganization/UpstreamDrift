"""Unit tests for Pinocchio GUI logic."""

from unittest.mock import MagicMock, patch

import pytest

from src.shared.python.engine_availability import PYQT6_AVAILABLE
from src.shared.python.gui_utils import get_qapp

if PYQT6_AVAILABLE:
    pass


@pytest.fixture(autouse=True, scope="module")
def mock_pinocchio_gui_dependencies():
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


@pytest.mark.skipif(not PYQT6_AVAILABLE, reason="PyQt6 not available")
class TestPinocchioGUI:
    """Test Pinocchio GUI."""

    @pytest.fixture
    def qapp(self):
        """Ensure QApplication exists."""
        app = get_qapp()
        return app

    @pytest.fixture
    def mock_gui(self, qapp):
        """Create a mocked PinocchioGUI instance."""
        # Use patch for everything that might cause import errors or runtime errors
        with (
            patch(
                "engines.physics_engines.pinocchio.python.pinocchio_golf.gui.viz.Visualizer"
            ),
            patch(
                "engines.physics_engines.pinocchio.python.pinocchio_golf.gui.MeshcatVisualizer"
            ),
            patch(
                "engines.physics_engines.pinocchio.python.pinocchio_golf.gui.get_shared_urdf_path"
            ) as mock_urdf,
        ):
            mock_urdf.return_value.exists.return_value = False

            # Late import to ensure mocks apply
            from src.engines.physics_engines.pinocchio.python.pinocchio_golf.gui import (
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

        with patch(
            "engines.physics_engines.pinocchio.python.pinocchio_golf.gui.InducedAccelerationAnalyzer"
        ) as MockAnalyzer:
            mock_gui._ensure_analyzer_initialized()

            assert mock_gui.analyzer is not None
            MockAnalyzer.assert_called_once_with(mock_gui.model, mock_gui.data)

        # 3. Model is set, Analyzer is set -> Should not re-initialize
        existing_analyzer = MagicMock()
        mock_gui.analyzer = existing_analyzer

        with patch(
            "engines.physics_engines.pinocchio.python.pinocchio_golf.gui.InducedAccelerationAnalyzer"
        ) as MockAnalyzer:
            mock_gui._ensure_analyzer_initialized()

            assert mock_gui.analyzer is existing_analyzer
            MockAnalyzer.assert_not_called()

import sys
import unittest
from unittest.mock import MagicMock, patch

from src.unreal_integration.viewer_backends import PyVistaBackend, ViewerConfig


class TestPyVistaBackend(unittest.TestCase):
    def test_instantiation(self):
        """Test that PyVistaBackend can be instantiated."""
        config = ViewerConfig()
        backend = PyVistaBackend(config)
        self.assertIsInstance(backend, PyVistaBackend)

    def test_initialize_missing_pyvista(self):
        """Test that initialize raises RuntimeError when pyvista is missing."""
        config = ViewerConfig()
        backend = PyVistaBackend(config)

        # Ensure pyvista is not in sys.modules
        # Setting the module to None simulates it being missing
        with patch.dict(sys.modules, {"pyvista": None}):
            with self.assertRaises(RuntimeError) as cm:
                backend.initialize()
            self.assertIn("PyVista not available", str(cm.exception))

    def test_initialize_with_mock_pyvista(self):
        """Test that initialize succeeds when pyvista is available (mocked)."""
        config = ViewerConfig()
        backend = PyVistaBackend(config)

        mock_pv = MagicMock()
        mock_plotter = MagicMock()
        mock_pv.Plotter.return_value = mock_plotter

        with patch.dict(sys.modules, {"pyvista": mock_pv}):
            backend.initialize()
            self.assertTrue(backend.is_initialized)
            mock_pv.Plotter.assert_called_once()
            # Verify background color was set
            mock_plotter.set_background.assert_called_with(config.background_color)


if __name__ == "__main__":
    unittest.main()

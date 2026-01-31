import sys
from unittest.mock import patch
import pytest

def test_headless_plotting_import():
    """Test that plotting_core can be imported without PyQt6."""
    with patch.dict(sys.modules):
        # Remove PyQt6 from modules if present
        keys = [k for k in sys.modules if k.startswith("PyQt6")]
        for k in keys:
            del sys.modules[k]

        # Mock import failure
        with patch.dict(sys.modules, {"PyQt6": None, "PyQt6.QtCore": None, "PyQt6.QtWidgets": None}):
            # Also clear plotting modules to force reload
            keys = [k for k in sys.modules if "plotting" in k]
            for k in keys:
                del sys.modules[k]

            # Clear ui.qt.plotting to ensure it tries to import Qt
            if "src.shared.python.ui.qt.plotting" in sys.modules:
                del sys.modules["src.shared.python.ui.qt.plotting"]

            try:
                import src.shared.python.plotting_core as pc
                # Should not raise ImportError
                assert hasattr(pc, "MplCanvas")

                # Verify that MplCanvas is NOT the Qt one (it should be the stub)
                # Since we mocked PyQt6 modules to None, the import in ui/qt/plotting.py
                # (which imports FigureCanvasQTAgg which needs Qt) should fail.
                # However, matplotlib might try to import Qt itself.
                # If we rely on ImportError from ui/qt/plotting.py, we need to ensure it fails.

            except ImportError as e:
                pytest.fail(f"Import failed in headless mode: {e}")

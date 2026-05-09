"""Unit tests for Drake GUI App and its mixin decomposition.

Tests the DrakeInducedAccelerationAnalyzer, DrakeRecorder,
and the individual mixin modules (UI, Sim, Viz, Analysis).
"""

import sys
from collections.abc import Generator
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Check PyQt6 availability without importing engine_availability
# (which triggers a torch import that may fail on some platforms)
try:
    import PyQt6  # noqa: F401

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False

skip_if_no_pyqt6 = pytest.mark.skipif(not HAS_PYQT6, reason="PyQt6 not installed")

# Drake engine module paths that may get imported and must be cleaned up
_DRAKE_ENGINE_MODULES = [
    "src.engines.physics_engines.drake",
    "src.engines.physics_engines.drake.python",
    "src.engines.physics_engines.drake.python.src",
    "src.engines.physics_engines.drake.python.src.drake_gui_app",
    "src.engines.physics_engines.drake.python.src.drake_gui_ui",
    "src.engines.physics_engines.drake.python.src.drake_gui_sim",
    "src.engines.physics_engines.drake.python.src.drake_gui_viz",
    "src.engines.physics_engines.drake.python.src.drake_gui_analysis",
    "src.engines.physics_engines.drake.python.src.drake_analysis",
]


@pytest.fixture(autouse=True, scope="function")
def _mock_pydrake() -> Generator[None, None, None]:
    """Provide mock pydrake modules only during test execution.

    Also cleanup drake engine modules to prevent pollution of test_drake_wrapper.py.
    When drake_gui_app is imported, it brings in the parent package
    src.engines.physics_engines.drake.python into sys.modules. This causes
    test_drake_wrapper.py to fail when it tries to patch
    src.engines.physics_engines.drake.python.drake_physics_engine, because the
    parent package exists but drake_physics_engine was never imported.
    """
    # Save existing drake modules so we can restore them
    saved_modules = {}
    for module_name in _DRAKE_ENGINE_MODULES:
        if module_name in sys.modules:
            saved_modules[module_name] = sys.modules[module_name]

    # Create fresh mocks for each test session to prevent pollution
    pydrake_mocks = {
        "pydrake": MagicMock(),
        "pydrake.all": MagicMock(),
        "pydrake.multibody": MagicMock(),
        "pydrake.multibody.plant": MagicMock(),
        "pydrake.multibody.tree": MagicMock(),
        # Mock torch and cv2 to prevent DLL loading errors on Windows
        # (drake pkg __init__ → logger_utils → reproducibility → engine_availability → torch/cv2)
        "torch": MagicMock(),
        "cv2": MagicMock(),
        "cv2.dnn": MagicMock(),
        "cv2.typing": MagicMock(),
    }
    with patch.dict("sys.modules", pydrake_mocks):
        yield

    # Clean up drake engine modules to prevent pollution
    for module_name in _DRAKE_ENGINE_MODULES:
        if module_name in sys.modules:
            del sys.modules[module_name]

    # Restore saved modules
    for module_name, module in saved_modules.items():
        sys.modules[module_name] = module


# ==================================================================
# DrakeInducedAccelerationAnalyzer Tests
# ==================================================================


# ==================================================================
# DrakeRecorder Tests
# ==================================================================

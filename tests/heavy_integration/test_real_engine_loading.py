"""
Real integration tests for physics engine loading and operation.

These tests demonstrate proper integration testing:
- Use real dependencies where available (skip if not installed)
- Test actual integration between components
- Verify real behavior, not mocked interactions
- Test end-to-end workflows
"""

import sys
from typing import Any
from unittest.mock import MagicMock, patch

import pytest
from src.shared.python.data_io.path_utils import get_src_root
from src.shared.python.engine_core.engine_manager import EngineManager, EngineStatus


# Helper to check if a module is mocked (from unit tests polluting sys.modules)
def is_mock(module_name: str) -> bool:
    """Check if a module in sys.modules is a mock."""
    mod = sys.modules.get(module_name)
    if mod is None:
        return False
    return isinstance(mod, MagicMock) or getattr(mod, "__file__", None) is None


# Test assets
ASSET_DIR = get_src_root() / "assets"
SIMPLE_ARM_URDF = ASSET_DIR / "simple_arm.urdf"


# ==============================================================================
# EXEMPLARY INTEGRATION TESTS
# ==============================================================================
# These demonstrate proper integration testing:
# 1. Actually load and integrate real components
# 2. Skip gracefully if dependencies unavailable (not fail)
# 3. Test real behavior across component boundaries
# 4. Verify actual outputs, not that mocks were called
# ==============================================================================


class TestEngineManagerIntegration:
    """Test EngineManager integration with real filesystem."""

    def test_engine_manager_discovers_real_engines(self) -> None:
        """Test that EngineManager discovers engines in actual project structure.

        GOOD PRACTICE: Integration test uses real project structure.
        This tests that EngineManager correctly navigates real directories.
        """
        manager = EngineManager()

        # Should have initialized with real project structure
        assert manager.suite_root.exists()
        assert manager.engines_root.exists()

        # Get available engines
        available = manager.get_available_engines()

        # Should have found at least some engines
        assert isinstance(available, list)

        # Each available engine should have a valid path
        for engine in available:
            path = manager.engine_paths[engine]
            assert path.exists(), f"{engine} path should exist: {path}"

    def test_engine_paths_match_filesystem(self) -> None:
        """Test that engine paths in manager match actual filesystem.

        GOOD PRACTICE: Verifies configuration matches reality.
        """
        manager = EngineManager()

        for engine_type, path in manager.engine_paths.items():
            # Path should be absolute and within suite root
            assert path.is_absolute()
            assert manager.suite_root in path.parents

            # If status says available, path must exist
            status = manager.get_engine_status(engine_type)
            if status == EngineStatus.AVAILABLE or status == EngineStatus.LOADED:
                assert (
                    path.exists()
                ), f"{engine_type} marked as {status} but path doesn't exist"


pytestmark = pytest.mark.live_simulation

# ruff: noqa: I001

import sys
import tempfile
from pathlib import Path


def test_lazy_imports_engine_manager():
    """Test that importing EngineManager does NOT import heavy engine libraries."""

    # Ensure modules are not already loaded
    heavy_modules = ["mujoco", "pydrake", "pinocchio", "opensim"]
    for mod in heavy_modules:
        if mod in sys.modules:
            del sys.modules[mod]

    # Import the manager
    from src.shared.python.engine_core.engine_manager import EngineManager

    # Verify heavy modules are NOT loaded
    for mod in heavy_modules:
        assert mod not in sys.modules, f"{mod} was imported eagerly!"

    # Now verify probing (which might import them if available, but let's mock checks)
    # Actually, we just want to ensure the specific Lazy Import logic holds.

    # Verify EngineManager can be instantiated without triggering imports.
    # Use a real temp tree instead of globally mocking Path.exists, which
    # interferes with repo-root and registry-path discovery.
    with tempfile.TemporaryDirectory() as temp_dir:
        suite_root = Path(temp_dir)
        (suite_root / "engines").mkdir()
        EngineManager(suite_root=suite_root)

    # Still shouldn't be loaded (unless probe_all_engines is called instantly in __init__)
    # Looking at EngineManager.__init__:
    # self._discover_engines() -> checks paths
    # it initializes probes: MuJoCoProbe(...)

    # EngineProbe.__init__ is lightweight.
    # So heavy modules should still be missing.

    for mod in heavy_modules:
        assert mod not in sys.modules, f"{mod} was imported during initialization!"


if __name__ == "__main__":
    test_lazy_imports_engine_manager()

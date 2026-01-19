import sys
from unittest.mock import patch


def test_lazy_imports_engine_manager():
    """Test that importing EngineManager does NOT import heavy engine libraries."""

    # Ensure modules are not already loaded
    heavy_modules = ["mujoco", "pydrake", "pinocchio", "opensim"]
    for mod in heavy_modules:
        if mod in sys.modules:
            del sys.modules[mod]

    # Import the manager
    from shared.python.engine_manager import EngineManager

    # Verify heavy modules are NOT loaded
    for mod in heavy_modules:
        assert mod not in sys.modules, f"{mod} was imported eagerly!"

    # Now verify probing (which might import them if available, but let's mock checks)
    # Actually, we just want to ensure the specific Lazy Import logic holds.

    # Verify EngineManager can be instantiated without triggering imports
    # We mock suite_root/engines to be valid so it doesn't crash on path checks
    with patch("pathlib.Path.exists", return_value=True):
        EngineManager()

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
    print("Lazy import test passed.")

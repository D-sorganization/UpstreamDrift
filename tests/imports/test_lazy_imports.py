# ruff: noqa: I001

import sys
import tempfile
from pathlib import Path
from types import ModuleType

import pytest


class BlockedImportFinder:
    def __init__(self, blocked_roots: set[str]) -> None:
        self._blocked_roots = blocked_roots

    def find_spec(self, fullname: str, path: object, target: object = None) -> None:
        root = fullname.split(".", 1)[0]
        if root in self._blocked_roots:
            raise AssertionError(f"{fullname} was imported during module import")


def _import_without_modules(module_name: str, blocked_roots: set[str]) -> ModuleType:
    finder = BlockedImportFinder(blocked_roots)
    removed_modules = {
        name: sys.modules.pop(name)
        for name in list(sys.modules)
        if name == module_name
        or name.startswith(f"{module_name}.")
        or name.split(".", 1)[0] in blocked_roots
    }
    sys.meta_path.insert(0, finder)
    try:
        __import__(module_name)
        return sys.modules[module_name]
    finally:
        sys.meta_path.remove(finder)
        for name in list(sys.modules):
            if name.split(".", 1)[0] in blocked_roots:
                sys.modules.pop(name, None)
        sys.modules.update(removed_modules)


def test_lazy_imports_engine_manager() -> None:
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


@pytest.mark.parametrize(
    "module_name,blocked_roots",
    [
        ("src.launchers.shot_tracer", {"PyQt6", "pyqtgraph"}),
        ("src.shared.python.chat.chat_dock_widget", {"PyQt6"}),
    ],
)
def test_gui_modules_do_not_import_qt_dependencies_at_module_import(
    module_name: str, blocked_roots: set[str]
) -> None:
    """GUI modules must be collectable without importing optional Qt stacks."""
    _import_without_modules(module_name, blocked_roots)


if __name__ == "__main__":
    test_lazy_imports_engine_manager()

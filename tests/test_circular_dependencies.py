"""Tests verifying no circular dependencies between top-level modules.

Phase 1 of the decoupling plan eliminates circular imports between:
- shared <-> engines  (capabilities moved to shared, loaders moved to engines)
- shared <-> robotics (control_interface uses lazy import)
- api -> launchers    (extracted to api.services.launcher_service)
- deployment -> engines/learning (TYPE_CHECKING guards, lazy imports)

These tests verify that key modules can be imported independently without
pulling in modules they should not depend on.
"""

from __future__ import annotations

import ast
import importlib
import sys
from pathlib import Path

# ---------------------------------------------------------------------------
# 1. Independent importability tests
# ---------------------------------------------------------------------------


class TestSharedDoesNotImportEngines:
    """Verify shared/python modules don't import from engines at module level."""

    def test_interfaces_importable_without_engines(self):
        """interfaces.py should not require engines at import time."""
        # Temporarily remove engines from sys.modules to detect eager imports
        saved = {}
        engine_modules = [k for k in sys.modules if k.startswith("src.engines")]
        for mod in engine_modules:
            saved[mod] = sys.modules.pop(mod)

        try:
            # Force reimport
            if "src.shared.python.engine_core.interfaces" in sys.modules:
                del sys.modules["src.shared.python.engine_core.interfaces"]

            mod = importlib.import_module("src.shared.python.engine_core.interfaces")
            assert hasattr(mod, "PhysicsEngine")
            assert hasattr(mod, "RecorderInterface")
        finally:
            # Restore saved modules
            sys.modules.update(saved)

    def test_capabilities_in_shared(self):
        """EngineCapabilities should be importable from shared.python.engine_core.capabilities."""
        mod = importlib.import_module("src.shared.python.engine_core.capabilities")
        assert hasattr(mod, "EngineCapabilities")
        assert hasattr(mod, "CapabilityLevel")

    def test_capabilities_backward_compat(self):
        """Old import path engines.common.capabilities should still work."""
        mod = importlib.import_module("src.engines.common.capabilities")
        assert hasattr(mod, "EngineCapabilities")
        assert hasattr(mod, "CapabilityLevel")

    def test_engine_loaders_in_engines(self):
        """Engine loaders should be importable from src.engines.loaders."""
        mod = importlib.import_module("src.engines.loaders")
        assert hasattr(mod, "LOADER_MAP")
        assert hasattr(mod, "load_pendulum_engine")

    def test_engine_loaders_backward_compat(self):
        """Old import path shared.python.engine_loaders should still work."""
        mod = importlib.import_module("src.shared.python.engine_core.engine_loaders")
        assert hasattr(mod, "LOADER_MAP")


class TestApiDoesNotImportLaunchers:
    """Verify API layer doesn't import launchers at module level."""

    def test_launcher_service_exists(self):
        """LauncherService should be importable from api.services."""
        mod = importlib.import_module("src.api.services.launcher_service")
        assert hasattr(mod, "LauncherService")

    def test_launcher_service_lazy_imports(self):
        """LauncherService should not import launchers at module load time."""
        saved = {}
        launcher_modules = [k for k in sys.modules if k.startswith("src.launchers")]
        for mod_name in launcher_modules:
            saved[mod_name] = sys.modules.pop(mod_name)

        try:
            if "src.api.services.launcher_service" in sys.modules:
                del sys.modules["src.api.services.launcher_service"]

            mod = importlib.import_module("src.api.services.launcher_service")
            # Module loads without importing launchers
            assert hasattr(mod, "LauncherService")

            # Check that launcher modules were NOT loaded
            for mod_name in [
                "src.launchers.launcher_model_handlers",
                "src.launchers.launcher_process_manager",
            ]:
                assert (
                    mod_name not in sys.modules
                ), f"{mod_name} was imported eagerly by LauncherService module"
        finally:
            sys.modules.update(saved)


# ---------------------------------------------------------------------------
# 2. AST-based static analysis: no module-level cross-boundary imports
# ---------------------------------------------------------------------------


def _get_top_level_imports(filepath: Path) -> list[str]:
    """Extract top-level import targets from a Python file using AST.

    Returns a list of module paths that are imported at module level
    (not inside functions, classes, or TYPE_CHECKING blocks).
    """
    try:
        source = filepath.read_text(encoding="utf-8")
        tree = ast.parse(source, filename=str(filepath))
    except (SyntaxError, UnicodeDecodeError):
        return []

    imports = []
    for node in ast.iter_child_nodes(tree):
        # Skip TYPE_CHECKING blocks
        if isinstance(node, ast.If):
            test = node.test
            # Check for `if TYPE_CHECKING:` pattern
            if isinstance(test, ast.Name) and test.id == "TYPE_CHECKING":
                continue
            if isinstance(test, ast.Attribute) and test.attr == "TYPE_CHECKING":
                continue

        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom):
            if node.module:
                imports.append(node.module)

    return imports


def _get_src_root() -> Path:
    """Get the src/ root directory."""
    return Path(__file__).parent.parent / "src"


class TestNoCircularImportsStaticAnalysis:
    """AST-based static analysis for forbidden import directions."""

    def test_shared_does_not_import_engines_at_module_level(self):
        """No file in shared/python/ should import from engines at module level.

        Exception: backward-compatible shim files (engine_loaders.py) that
        re-export from the new canonical location.
        """
        src_root = _get_src_root()
        shared_dir = src_root / "shared" / "python"
        # Files that are allowed to import from engines (backward-compat shims)
        allowed_shims = {"engine_loaders.py"}

        violations = []
        for py_file in shared_dir.rglob("*.py"):
            if py_file.name in allowed_shims:
                continue
            imports = _get_top_level_imports(py_file)
            for imp in imports:
                if imp.startswith(("src.engines", "engines")):
                    violations.append(f"{py_file.relative_to(src_root)}: imports {imp}")

        assert (
            not violations
        ), "shared/python/ files import from engines at module level:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

    def test_shared_does_not_import_robotics_at_module_level(self):
        """No file in shared/python/ should import from robotics at module level."""
        src_root = _get_src_root()
        shared_dir = src_root / "shared" / "python"

        violations = []
        for py_file in shared_dir.rglob("*.py"):
            imports = _get_top_level_imports(py_file)
            for imp in imports:
                if imp.startswith(("src.robotics", "robotics")):
                    violations.append(f"{py_file.relative_to(src_root)}: imports {imp}")

        assert (
            not violations
        ), "shared/python/ files import from robotics at module level:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

    def test_api_does_not_import_launchers_at_module_level(self):
        """No file in api/ should import from launchers at module level.

        Exception: imports inside function bodies are OK (caught by AST
        analysis which only looks at module-level nodes).
        """
        src_root = _get_src_root()
        api_dir = src_root / "api"

        violations = []
        for py_file in api_dir.rglob("*.py"):
            imports = _get_top_level_imports(py_file)
            for imp in imports:
                if imp.startswith(("src.launchers", "launchers")):
                    violations.append(f"{py_file.relative_to(src_root)}: imports {imp}")

        assert (
            not violations
        ), "api/ files import from launchers at module level:\n" + "\n".join(
            f"  - {v}" for v in violations
        )

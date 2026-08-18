"""Architectural dependency direction tests.

These tests enforce the layered architecture of the codebase by verifying
that import dependencies flow in the correct direction:

    src/shared/  (lowest layer — no upward imports)
         ↑
    src/engines/ (may import from shared/; not from api/ or launchers/)
         ↑
    src/api/     (may import from engines/ and shared/; not from launchers/)
         ↑
    src/launchers/ (top layer — may import from any lower layer)

Violations indicate architectural drift that should be corrected.
"""

from __future__ import annotations

import ast
import logging
from pathlib import Path

import pytest

pytestmark = pytest.mark.integration

logger = logging.getLogger(__name__)

# Repository root
REPO_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = REPO_ROOT / "src"

# Layer definitions (order matters: lower layers may NOT import from higher layers)
LAYERS: list[str] = [
    "shared",
    "engines",
    "api",
    "launchers",
]

# Forbidden import rules: (source_layer, forbidden_target_layer)
# A module in source_layer must not import from forbidden_target_layer
FORBIDDEN_IMPORTS: list[tuple[str, str]] = [
    # shared/ is the lowest layer — must not import from any higher layer
    ("shared", "engines"),
    ("shared", "api"),
    ("shared", "launchers"),
    # engines/ must not import from api/ or launchers/
    ("engines", "api"),
    ("engines", "launchers"),
    # api/ must not import from launchers/
    ("api", "launchers"),
]

# Known pre-existing violations that are explicitly tracked as tech debt.
# Each entry is (source_file_relative_to_src, imported_module).
# These are allowed to pass tests but should be resolved in future refactoring.
# To fix: invert the dependency or introduce an interface/abstract layer.
KNOWN_EXCEPTIONS: set[tuple[str, str]] = {
    # engine_core in shared/ loads engine adapters — needs an abstract loader interface
    ("shared/python/engine_core/engine_manager.py", "src.engines.loaders"),
    ("shared/python/engine_core/engine_loaders.py", "src.engines.loaders"),
    # launcher_service is an API adapter for launcher management —
    # consider moving to launchers/ or introducing a launcher interface in api/
    ("api/services/launcher_service.py", "src.launchers.launcher_process_manager"),
    ("api/services/launcher_service.py", "src.launchers.launcher_model_handlers"),
    # ---------------------------------------------------------------------
    # Violations that accumulated while this whole directory was RED on main
    # (#8034). They are recorded -- not excused -- so that NEW drift fails
    # immediately. Tracked for removal by #8055.
    # `test_known_exceptions_are_all_still_real` forces this list to shrink:
    # once a violation is fixed, its entry here must be deleted.
    # ---------------------------------------------------------------------
    # shared/ statically importing three concrete engine backends -> #8055 (1)
    (
        "shared/python/analysis/cross_engine.py",
        "src.engines.physics_engines.drake.python.drake_physics_engine",
    ),
    (
        "shared/python/analysis/cross_engine.py",
        "src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.physics_engine",
    ),
    (
        "shared/python/analysis/cross_engine.py",
        "src.engines.physics_engines.pinocchio.python.pinocchio_physics_engine",
    ),
    # shared/ reaching for the Simscape adapter -> #8055 (2)
    (
        "shared/python/motion_matching/surrogate/validate.py",
        "src.engines.simscape._engine_pool",
    ),
    (
        "shared/python/motion_matching/surrogate/validate.py",
        "src.engines.simscape._errors",
    ),
    (
        "shared/python/motion_matching/surrogate/validate.py",
        "src.engines.simscape.adapter",
    ),
    (
        "shared/python/pose_interchange/services/simscape.py",
        "src.engines.simscape.adapter",
    ),
    (
        "shared/python/simulation_backends/model_params.py",
        "src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum",
    ),
    (
        "shared/python/simulation_backends/ode_backend.py",
        "src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum",
    ),
    # api/ importing launcher internals for health data -> #8055 (3)
    ("api/routes/diagnostics.py", "src.launchers.integrations_health_data"),
    ("api/routes/diagnostics.py", "src.launchers.launcher_diagnostics"),
    # shared/ lazily importing an api router (function-local) -> #8055 (4)
    ("shared/python/realtime/ws_pubsub.py", "src.api.routes.realtime"),
}


def _get_layer(path: Path) -> str | None:
    """Determine which architectural layer a file belongs to.

    Args:
        path: Absolute path to a Python file.

    Returns:
        Layer name or None if the file is not in a recognized layer.
    """
    try:
        relative = path.relative_to(SRC_ROOT)
    except ValueError:
        return None

    parts = relative.parts
    if not parts:
        return None

    top_dir = parts[0]
    if top_dir in LAYERS:
        return top_dir
    return None


def _extract_imports(filepath: Path) -> list[str]:
    """Extract all import module names from a Python file.

    Args:
        filepath: Path to the Python file to analyze.

    Returns:
        List of top-level module strings imported by the file.
    """
    try:
        source = filepath.read_text(encoding="utf-8", errors="replace")
        tree = ast.parse(source, filename=str(filepath))
    except SyntaxError:
        logger.warning("Skipping %s due to SyntaxError", filepath)
        return []

    imports: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imports.append(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            # `node.level > 0` is a relative import (`from .api import X`). It can
            # only ever resolve inside the importing package, so it can never
            # cross a layer boundary. Treating it as absolute made
            # `shared/python/realtime/__init__.py` and `codemap/__init__.py`
            # look like they imported the `api/` LAYER when they were importing
            # their own sibling `api.py` module (#8034).
            if node.level:
                continue
            imports.append(node.module)
    return imports


def _import_targets_layer(module_name: str, target_layer: str) -> bool:
    """Check if an import string targets a specific architectural layer.

    Handles both absolute imports (``src.engines.foo``) and
    relative-style module names (``engines.foo``).

    Args:
        module_name: The dotted module import string.
        target_layer: The layer to check against.

    Returns:
        True if the import targets the specified layer.
    """
    parts = module_name.split(".")

    # Absolute: src.<layer>.xxx
    if len(parts) >= 2 and parts[0] == "src" and parts[1] == target_layer:
        return True

    # Short: <layer>.xxx (some files use this form)
    return parts[0] == target_layer


def _collect_violations() -> list[str]:
    """Scan all Python files and collect dependency direction violations.

    Returns:
        List of human-readable violation descriptions.
    """
    violations: list[str] = []

    for py_file in sorted(SRC_ROOT.rglob("*.py")):
        if "__pycache__" in str(py_file):
            continue

        source_layer = _get_layer(py_file)
        if source_layer is None:
            continue

        imports = _extract_imports(py_file)
        relative_path = py_file.relative_to(REPO_ROOT)
        relative_to_src = py_file.relative_to(SRC_ROOT).as_posix()

        for imp in imports:
            # Skip known exceptions (pre-existing tech debt)
            if (relative_to_src, imp) in KNOWN_EXCEPTIONS:
                continue

            for src_layer, forbidden_layer in FORBIDDEN_IMPORTS:
                if source_layer == src_layer and _import_targets_layer(
                    imp, forbidden_layer
                ):
                    violations.append(
                        f"{relative_path}: "
                        f"{source_layer}/ imports from {forbidden_layer}/ "
                        f"({imp})"
                    )

    return violations


class TestDependencyDirection:
    """Verify that import dependencies flow in the correct architectural direction."""

    def test_shared_does_not_import_engines(self) -> None:
        """src/shared/ must not import from src/engines/."""
        violations = [
            v for v in _collect_violations() if "shared/ imports from engines/" in v
        ]
        assert violations == [], (
            f"shared/ layer imports from engines/ ({len(violations)} violations):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_shared_does_not_import_api(self) -> None:
        """src/shared/ must not import from src/api/."""
        violations = [
            v for v in _collect_violations() if "shared/ imports from api/" in v
        ]
        assert violations == [], (
            f"shared/ layer imports from api/ ({len(violations)} violations):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_shared_does_not_import_launchers(self) -> None:
        """src/shared/ must not import from src/launchers/."""
        violations = [
            v for v in _collect_violations() if "shared/ imports from launchers/" in v
        ]
        assert violations == [], (
            f"shared/ layer imports from launchers/ ({len(violations)} violations):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_engines_does_not_import_api(self) -> None:
        """src/engines/ must not import from src/api/."""
        violations = [
            v for v in _collect_violations() if "engines/ imports from api/" in v
        ]
        assert violations == [], (
            f"engines/ layer imports from api/ ({len(violations)} violations):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_engines_does_not_import_launchers(self) -> None:
        """src/engines/ must not import from src/launchers/."""
        violations = [
            v for v in _collect_violations() if "engines/ imports from launchers/" in v
        ]
        assert violations == [], (
            f"engines/ layer imports from launchers/ ({len(violations)} violations):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_api_does_not_import_launchers(self) -> None:
        """src/api/ must not import from src/launchers/."""
        violations = [
            v for v in _collect_violations() if "api/ imports from launchers/" in v
        ]
        assert violations == [], (
            f"api/ layer imports from launchers/ ({len(violations)} violations):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_all_layers_summary(self) -> None:
        """No layering violation outside the tracked ``KNOWN_EXCEPTIONS`` set.

        This used to log and assert nothing (#8035), which meant the one test
        positioned to catch a violation in a *new* layer pair could never fail.
        It now asserts, and covers every pair in ``FORBIDDEN_IMPORTS`` including
        any added later that has no dedicated test above.
        """
        violations = _collect_violations()
        assert violations == [], (
            f"{len(violations)} architectural dependency violation(s) outside the "
            "tracked KNOWN_EXCEPTIONS set. Fix the import direction; do NOT add "
            "the entry to KNOWN_EXCEPTIONS (see #8055):\n"
            + "\n".join(f"  - {v}" for v in violations)
        )

    def test_every_forbidden_pair_has_a_dedicated_test(self) -> None:
        """Each FORBIDDEN_IMPORTS pair must have its own named test."""
        own_tests = {
            name
            for name in dir(TestDependencyDirection)
            if name.startswith("test_") and "does_not_import" in name
        }
        missing = [
            f"{src}->{dst}"
            for src, dst in FORBIDDEN_IMPORTS
            if f"test_{src}_does_not_import_{dst}" not in own_tests
        ]
        assert not missing, (
            f"FORBIDDEN_IMPORTS pairs without a dedicated test: {missing}. "
            "Add test_<src>_does_not_import_<dst>."
        )


class TestKnownExceptionsRatchet:
    """``KNOWN_EXCEPTIONS`` must only ever shrink (#8055)."""

    def test_known_exceptions_are_all_still_real(self) -> None:
        """A fixed violation must be deleted from the allowlist, not left behind.

        Without this, the allowlist becomes a graveyard and stops describing the
        actual state of the codebase.
        """
        actual: set[tuple[str, str]] = set()
        for py_file in sorted(SRC_ROOT.rglob("*.py")):
            if "__pycache__" in str(py_file):
                continue
            source_layer = _get_layer(py_file)
            if source_layer is None:
                continue
            relative_to_src = py_file.relative_to(SRC_ROOT).as_posix()
            for imp in _extract_imports(py_file):
                for src_layer, forbidden_layer in FORBIDDEN_IMPORTS:
                    if source_layer == src_layer and _import_targets_layer(
                        imp, forbidden_layer
                    ):
                        actual.add((relative_to_src, imp))

        stale = sorted(KNOWN_EXCEPTIONS - actual)
        assert not stale, (
            "These KNOWN_EXCEPTIONS entries no longer correspond to a real "
            "violation and must be deleted (#8055):\n"
            + "\n".join(f"  - {path}: {imp}" for path, imp in stale)
        )

    def test_relative_imports_are_not_counted_as_layer_crossings(self) -> None:
        """Guard the #8034 parser fix: `from .api import X` is intra-package."""
        source = "from .api import Thing\nfrom ..api import Other\n"
        tree = ast.parse(source)
        relative = [
            node
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.level
        ]
        assert len(relative) == 2, "fixture must contain two relative imports"

        module = SRC_ROOT / "shared" / "python" / "realtime" / "__init__.py"
        if module.exists():
            assert "api" not in _extract_imports(module), (
                "relative `from .api import ...` must not be reported as an "
                "import of the api/ layer (#8034)"
            )

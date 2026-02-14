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
        elif isinstance(node, ast.ImportFrom):
            if node.module:
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
    if parts[0] == target_layer:
        return True

    return False


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
        """Comprehensive check — summarize all violations for visibility."""
        violations = _collect_violations()
        if violations:
            logger.warning(
                "Found %d architectural dependency violations:\n%s",
                len(violations),
                "\n".join(f"  - {v}" for v in violations),
            )
        # This test doesn't assert — it just logs for visibility.
        # Individual tests above catch specific violations.

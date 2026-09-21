"""Import-boundary contracts for shared and engine entry-point code."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SHARED_ROOT = _REPO_ROOT / "src" / "shared" / "python"
_ENGINE_ROOT = _REPO_ROOT / "src" / "engines"

_ALLOWED_SHARED_UPWARD_IMPORTS = {
    # Lazy runtime bridge documented in issue #7362; not part of this cluster.
    (
        _SHARED_ROOT / "realtime" / "ws_pubsub.py",
        "src.api.routes.realtime",
    ),
}


def _imports_from(path: Path) -> set[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
    imports: set[str] = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            imports.update(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imports.add(node.module)
    return imports


def _is_forbidden_shared_import(module: str) -> bool:
    return module in {"src.api", "src.launchers"} or module.startswith(
        ("src.api.", "src.launchers.")
    )


@pytest.mark.unit
def test_shared_python_does_not_import_application_layers() -> None:
    """Shared/Tools-vendored code must not depend upward on app packages."""
    violations: list[str] = []
    for path in sorted(_SHARED_ROOT.rglob("*.py")):
        for module in sorted(_imports_from(path)):
            if (
                _is_forbidden_shared_import(module)
                and (path, module) not in _ALLOWED_SHARED_UPWARD_IMPORTS
            ):
                violations.append(f"{path.relative_to(_REPO_ROOT)} imports {module}")

    assert not violations, "Forbidden shared-layer upward imports:\n" + "\n".join(
        violations
    )


@pytest.mark.unit
def test_engine_code_does_not_import_api_datetime_compat() -> None:
    """Engine code must not reach into the API layer for shared UTC helpers."""
    violations = [
        f"{path.relative_to(_REPO_ROOT)} imports {module}"
        for path in sorted(_ENGINE_ROOT.rglob("*.py"))
        for module in sorted(_imports_from(path))
        if module == "src.api.utils.datetime_compat"
    ]

    assert not violations, "Forbidden engine->api compat imports:\n" + "\n".join(
        violations
    )

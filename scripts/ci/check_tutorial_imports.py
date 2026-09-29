#!/usr/bin/env python3
"""Check import statements in tutorial documentation Python snippets."""

from __future__ import annotations

import ast
import importlib.util
import re
import sys
import textwrap
from pathlib import Path

PYTHON_FENCE_RE = re.compile(r"\\?```python\s*\n(.*?)\\?```", re.DOTALL)


def _repo_root() -> Path:
    """Return the repository root for this script."""
    return Path(__file__).resolve().parents[2]


ROOT = _repo_root()
DOC_TARGETS = (
    ROOT / "docs" / "user_guide" / "upstream_drift_user_manual.md",
    *sorted((ROOT / "docs" / "tutorials" / "content").glob("*.md")),
)
DEPRECATED_MODULE = "src.shared.python.engine_manager"
CANONICAL_MODULES = (
    "src.shared.python.engine_core.engine_manager",
    "src.shared.python.engine_core.engine_registry",
)


def extract_python_blocks(markdown: str) -> list[str]:
    """Return fenced Python code blocks from markdown text."""
    return [match.group(1) for match in PYTHON_FENCE_RE.finditer(markdown)]


def iter_import_from_modules(source: str) -> list[str]:
    """Return modules referenced by ``from module import name`` statements."""
    tree = ast.parse(textwrap.dedent(source))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            modules.append(node.module)
    return modules


def module_resolves(module: str) -> bool:
    """Return whether a module can be resolved on the current Python path."""
    try:
        return importlib.util.find_spec(module) is not None
    except (ImportError, ModuleNotFoundError, ValueError):
        return False


def display_path(path: Path) -> str:
    """Return a readable path for diagnostics."""
    try:
        return str(path.relative_to(ROOT))
    except ValueError:
        return str(path)


def check_doc_imports(paths: tuple[Path, ...] = DOC_TARGETS) -> list[str]:
    """Return deprecated tutorial import errors for fenced Python snippets."""
    errors: list[str] = []
    for module in CANONICAL_MODULES:
        if not module_resolves(module):
            errors.append(f"canonical module does not resolve: {module!r}")

    for path in paths:
        markdown = path.read_text(encoding="utf-8")
        if DEPRECATED_MODULE in markdown:
            errors.append(
                f"{display_path(path)}: contains deprecated import module "
                f"{DEPRECATED_MODULE!r}"
            )

        for index, block in enumerate(extract_python_blocks(markdown), start=1):
            if DEPRECATED_MODULE not in block:
                continue
            try:
                modules = iter_import_from_modules(block)
            except SyntaxError as exc:
                errors.append(f"{display_path(path)} block {index}: {exc.msg}")
                continue

            for module in modules:
                if module == DEPRECATED_MODULE:
                    errors.append(
                        f"{display_path(path)} block {index}: "
                        f"uses deprecated import module {module!r}"
                    )
    return errors


def main() -> int:
    sys.path.insert(0, str(ROOT))
    sys.path.insert(0, str(ROOT / "src"))
    errors = check_doc_imports()
    if errors:
        print("FAIL: tutorial documentation imports are broken")
        for error in errors:
            print(f"- {error}")
        return 1

    print("OK: tutorial documentation imports resolve")
    return 0


if __name__ == "__main__":
    sys.exit(main())

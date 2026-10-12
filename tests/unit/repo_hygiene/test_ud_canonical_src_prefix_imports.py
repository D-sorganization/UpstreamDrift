"""UpstreamDrift-owned shared packages must import themselves via ``src.``.

``model_generation``, ``humanoid_character_builder`` and ``plotting`` are ruled
``ud-canonical`` in ``docs/shared_tools/seam_rulings.v1.json``. The installed
Tools distribution also ships a top-level ``shared`` package carrying stale
copies of them, so an unprefixed ``from shared.python.model_generation ...``
resolves to that copy unless ``src`` happens to be on ``sys.path``. That broke
the Drake engine path in ``scripts/address_foot_progression_engines.py``
(``No module named 'shared.python.model_generation.inertia.result'``).

Tools-owned and split clusters keep their intentional ``shared.python`` seam
imports; only the ud-canonical clusters are checked here.

Issue: #12177
"""

from __future__ import annotations

import ast
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = REPO_ROOT / "src"
SEAM_RULINGS = REPO_ROOT / "docs" / "shared_tools" / "seam_rulings.v1.json"


def _ud_canonical_packages() -> frozenset[str]:
    rulings = json.loads(SEAM_RULINGS.read_text(encoding="utf-8"))["rulings"]
    return frozenset(
        name.removesuffix(".py")
        for name, entry in rulings.items()
        if entry.get("ruling") == "ud-canonical"
    )


def _imported_modules(tree: ast.AST) -> list[tuple[int, str]]:
    found: list[tuple[int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
            found.append((node.lineno, node.module))
        elif isinstance(node, ast.Import):
            found.extend((node.lineno, alias.name) for alias in node.names)
    return found


def _unprefixed_ud_canonical_imports() -> list[str]:
    targets = _ud_canonical_packages()
    offenders: list[str] = []
    for path in sorted(SRC_DIR.rglob("*.py")):
        try:
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        except (SyntaxError, UnicodeDecodeError):
            continue
        for lineno, module in _imported_modules(tree):
            parts = module.split(".")
            if parts[:2] == ["shared", "python"] and len(parts) > 2:
                if parts[2] in targets:
                    rel = path.relative_to(REPO_ROOT).as_posix()
                    offenders.append(f"{rel}:{lineno}: {module}")
    return offenders


def test_rulings_name_the_ud_canonical_packages() -> None:
    """Guard the fixture assumption so the scan below cannot pass vacuously."""
    assert {"model_generation", "humanoid_character_builder"} <= (
        _ud_canonical_packages()
    )


@pytest.mark.slow
def test_no_unprefixed_imports_of_ud_canonical_packages_in_src() -> None:
    offenders = _unprefixed_ud_canonical_imports()
    assert offenders == [], (
        "Import ud-canonical shared packages as `src.shared.python...`; the "
        "unprefixed name resolves to the installed Tools copy when `src` is "
        "not on sys.path:\n" + "\n".join(offenders)
    )


def test_ud_canonical_submodules_import_after_shared_aliases_installed() -> None:
    """``src.shared.python`` ud-canonical imports must not alias into Tools (#12177)."""
    from src.shared.python.ud_import_alias_policy import (
        UdSharedImportAliasFinder,
        install_ud_canonical_shared_import_aliases,
    )

    finder = UdSharedImportAliasFinder()
    assert finder._parse(
        "src.shared.python.model_generation.editor.attachment_ports"
    ) == (None, "")

    install_ud_canonical_shared_import_aliases()
    from src.shared.python.model_generation.editor.attachment_ports import PortPolarity

    assert PortPolarity.PLUG.value == "plug"


@pytest.mark.timeout(120)
def test_inertia_calculator_imports_without_src_on_sys_path(tmp_path: Path) -> None:
    """Reproduce the Drake failure: only the repo root is importable."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    code = (
        "import sys\n"
        f"sys.path.insert(0, {str(REPO_ROOT)!r})\n"
        f"assert {str(SRC_DIR)!r} not in sys.path\n"
        "from src.shared.python.model_generation.inertia.calculator import (\n"
        "    InertiaCalculator,\n"
        ")\n"
        "assert InertiaCalculator.__module__.startswith('src.')\n"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        cwd=tmp_path,
        env=env,
        capture_output=True,
        text=True,
        timeout=110,
        check=False,
    )
    assert result.returncode == 0, result.stderr[-2000:]

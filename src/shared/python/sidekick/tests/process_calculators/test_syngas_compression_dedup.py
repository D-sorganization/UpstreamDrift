"""Import-graph / AST dedup guard for syngas compression (issue #3183).

Asserts that there is exactly one source definition of
``SyngasCompressionEngine`` across the ``process_calculators`` package and
that the dead ``syngas_compression/`` placeholder subpackage no longer
exists. The root ``syngas_compression_calculator.py`` re-exports the engine
from ``syngas_compression_engine.py`` (the single canonical definition).
"""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

_PROCESS_CALCULATORS = Path(__file__).resolve().parents[2] / "process_calculators"


def _count_class_defs(symbol: str) -> list[Path]:
    """Return the files that *define* (not import) ``symbol`` as a class."""
    matches: list[Path] = []
    for path in _PROCESS_CALCULATORS.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in ast.walk(tree):
            if isinstance(node, ast.ClassDef) and node.name == symbol:
                matches.append(path)
    return matches


def test_single_syngas_compression_engine_definition() -> None:
    """Exactly one class definition of ``SyngasCompressionEngine``."""
    defs = _count_class_defs("SyngasCompressionEngine")
    assert len(defs) == 1, (
        "Expected exactly one SyngasCompressionEngine definition, found: "
        f"{[p.name for p in defs]}"
    )
    assert defs[0].name == "syngas_compression_engine.py"


def test_dead_syngas_compression_subpackage_removed() -> None:
    """The empty placeholder ``syngas_compression/`` subpackage is gone."""
    dead_dir = _PROCESS_CALCULATORS / "syngas_compression"
    assert not dead_dir.exists(), (
        "Dead placeholder subpackage should have been deleted (#3183)"
    )


def test_root_calculator_exposes_real_engine() -> None:
    """The package exposes the real engine, not an empty stub."""
    pytest.importorskip("numpy")
    from sidekick.process_calculators.syngas_compression_calculator import (
        SyngasCompressionEngine,
    )

    assert SyngasCompressionEngine.__module__.endswith("syngas_compression_engine")

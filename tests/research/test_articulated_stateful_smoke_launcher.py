"""Native preload and serial launch tests for the stateful smoke."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    articulated_stateful_smoke_launcher as launcher,
)

ROOT = Path(__file__).resolve().parents[2]
SOURCE = (
    ROOT / "scripts/research/proximal_distal_energy/"
    "articulated_stateful_smoke_launcher.py"
)


def test_launcher_top_level_imports_only_standard_library() -> None:
    tree = ast.parse(SOURCE.read_text(encoding="utf-8"))
    imported_roots = {
        alias.name.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module.split(".")[0]
        for node in tree.body
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert imported_roots <= {
        "__future__",
        "argparse",
        "collections",
        "importlib",
        "json",
        "pathlib",
        "typing",
    }
    assert "numpy" not in imported_roots
    assert "scripts" not in imported_roots


def test_preload_orders_mujoco_before_other_registered_engines(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    monkeypatch.setattr(launcher, "import_module", lambda name: calls.append(name))

    outcomes = launcher.preload_registered_native_modules(["pinocchio", "mujoco"])

    assert calls == ["mujoco", "pinocchio"]
    assert outcomes == {"mujoco": "loaded", "pinocchio": "loaded"}


def test_preload_retains_native_unavailability_for_typed_evaluation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def unavailable(name: str) -> None:
        raise OSError(f"{name} planted failure")

    monkeypatch.setattr(launcher, "import_module", unavailable)

    outcomes = launcher.preload_registered_native_modules(["mujoco"])

    assert outcomes["mujoco"].startswith("unavailable: mujoco planted failure")


def test_preload_rejects_unregistered_engine() -> None:
    with pytest.raises(ValueError, match="registered native engine"):
        launcher.preload_registered_native_modules(["symbolic"])

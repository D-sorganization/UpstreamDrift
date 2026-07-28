"""Regression tests for the functional-test surface enumerator."""

from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit


def _load_enumerator_module():
    script_path = (
        Path(__file__).resolve().parents[2]
        / "scripts"
        / "testing"
        / "enumerate_test_surface.py"
    )
    spec = importlib.util.spec_from_file_location(
        "enumerate_test_surface_under_test",
        script_path,
    )
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_bootstrap_paths_prioritize_explicit_tools_checkout(tmp_path: Path) -> None:
    """The route enumerator must import Sidekick modules from selected Tools."""
    module = _load_enumerator_module()
    repo_root = tmp_path / "UpstreamDrift"
    tools_root = tmp_path / "CanonicalTools"
    (repo_root / "src" / "shared" / "python").mkdir(parents=True)
    (tools_root / "src" / "shared" / "python").mkdir(parents=True)

    paths = module._bootstrap_paths(repo_root, str(tools_root))

    assert paths[:2] == (
        tools_root / "src" / "shared" / "python",
        tools_root / "src",
    )
    assert repo_root / "src" / "shared" / "python" in paths

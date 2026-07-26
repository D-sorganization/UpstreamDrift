"""Regression coverage for direct launcher import path bootstrapping."""

from __future__ import annotations

from pathlib import Path
import sys
from types import ModuleType

import pytest

import launch_upstream_drift

pytestmark = pytest.mark.unit


def test_launcher_bootstrap_defaults_to_repo_and_vendored_tools(tmp_path: Path) -> None:
    repo_root = tmp_path / "UpstreamDrift"
    sibling_tools = tmp_path / "Tools"
    repo_root.mkdir()
    (sibling_tools / "src").mkdir(parents=True)

    paths = launch_upstream_drift._launcher_bootstrap_paths(repo_root, None)

    assert paths == [
        str(repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python"),
        str(repo_root / "vendor" / "ud-tools" / "src"),
        str(repo_root / "vendor" / "ud-tools" / "src" / "python" / "src"),
        str(repo_root / "src" / "shared" / "python"),
        str(repo_root / "src"),
        str(repo_root),
    ]
    assert str(sibling_tools / "src") not in paths


def test_launcher_bootstrap_honors_explicit_tools_override(tmp_path: Path) -> None:
    repo_root = tmp_path / "UpstreamDrift"
    tools_root = tmp_path / "Tools"
    repo_root.mkdir()
    (tools_root / "src" / "shared" / "python").mkdir(parents=True)
    (tools_root / "src" / "python" / "src").mkdir(parents=True)

    resolved_tools_root = launch_upstream_drift._resolve_explicit_tools_root(
        str(tools_root)
    )
    paths = launch_upstream_drift._launcher_bootstrap_paths(
        repo_root, resolved_tools_root
    )

    assert paths[:3] == [
        str(tools_root / "src" / "shared" / "python"),
        str(tools_root / "src"),
        str(tools_root / "src" / "python" / "src"),
    ]
    assert paths[3:] == [
        str(repo_root / "vendor" / "ud-tools" / "src" / "shared" / "python"),
        str(repo_root / "vendor" / "ud-tools" / "src"),
        str(repo_root / "vendor" / "ud-tools" / "src" / "python" / "src"),
        str(repo_root / "src" / "shared" / "python"),
        str(repo_root / "src"),
        str(repo_root),
    ]


def test_launcher_rejects_invalid_explicit_tools_override(tmp_path: Path) -> None:
    invalid_tools_root = tmp_path / "Tools"
    invalid_tools_root.mkdir()

    with pytest.raises(RuntimeError, match="TOOLS_REPO_PATH"):
        launch_upstream_drift._resolve_explicit_tools_root(str(invalid_tools_root))


def test_bootstrap_import_paths_preserves_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_sys_path = ["tail"]
    monkeypatch.setattr(launch_upstream_drift, "path", synthetic_sys_path)

    launch_upstream_drift._bootstrap_import_paths(["first", "second", "tail"])

    assert synthetic_sys_path == ["first", "second", "tail"]


def test_parent_contract_aliases_override_downstream_compatibility_aliases(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tools imports must not resolve to a stale Upstream child contract."""
    parent_contracts = ModuleType("contracts")
    downstream_contracts = ModuleType("src.shared.python.contracts")
    monkeypatch.setitem(sys.modules, "contracts", downstream_contracts)
    monkeypatch.setitem(
        sys.modules,
        "shared.python.contracts",
        downstream_contracts,
    )
    monkeypatch.setitem(
        sys.modules,
        "src.shared.python.contracts",
        downstream_contracts,
    )

    launch_upstream_drift._restore_parent_contract_aliases(parent_contracts)

    assert sys.modules["contracts"] is parent_contracts
    assert sys.modules["shared.python.contracts"] is parent_contracts
    assert sys.modules["src.shared.python.contracts"] is downstream_contracts

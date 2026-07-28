"""Regression coverage for direct launcher import path bootstrapping."""

from __future__ import annotations

import importlib
import os
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from unittest.mock import MagicMock

import pytest

import launch_upstream_drift

pytestmark = pytest.mark.unit


def test_direct_launcher_uses_canonical_tools_repo_path_resolver() -> None:
    """Direct bootstrap must share one DbC contract with later UI startup."""
    resolver_module = importlib.import_module("src.launchers.tools_repo_path")

    assert (
        launch_upstream_drift._resolve_explicit_tools_root
        is resolver_module.resolve_explicit_tools_root
    )


def test_launcher_retries_parent_alias_installer_after_bootstrap(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The post-bootstrap retry must update the source package status."""
    src_package = importlib.import_module("src")
    install = MagicMock(return_value=True)
    monkeypatch.setattr(src_package, "_install_parent_shared_aliases", install)
    monkeypatch.setattr(
        src_package,
        "_PARENT_SHARED_ALIASES_INSTALLED",
        False,
    )

    assert launch_upstream_drift._retry_parent_shared_alias_installer() is True
    install.assert_called_once_with()
    assert src_package._PARENT_SHARED_ALIASES_INSTALLED is True


def test_clean_source_launch_uses_one_canonical_tools_module_identity() -> None:
    """Direct, shared, and src.shared imports must resolve to pinned Tools."""
    repo_root = Path(__file__).resolve().parents[2]
    script = """
import importlib
from pathlib import Path

import launch_upstream_drift
import src

canonical = importlib.import_module("shared.python.chat.chat_dock_widget")
direct = importlib.import_module("chat.chat_dock_widget")
legacy = importlib.import_module("src.shared.python.chat.chat_dock_widget")

assert src._PARENT_SHARED_ALIASES_INSTALLED is True
assert direct is canonical
assert legacy is canonical
canonical_file = Path(canonical.__file__).resolve()
expected_root = (
    Path.cwd() / "vendor" / "ud-tools" / "src" / "shared" / "python" / "chat"
).resolve()
assert canonical_file.is_relative_to(expected_root), (canonical_file, expected_root)
assert Path(direct.__file__).resolve() == canonical_file
assert Path(legacy.__file__).resolve() == canonical_file
"""
    env = os.environ.copy()
    env.pop("PYTHONPATH", None)
    env.pop("TOOLS_REPO_PATH", None)
    env["PYTHONDONTWRITEBYTECODE"] = "1"
    result = subprocess.run(  # nosec B603 - fixed interpreter and inline probe
        [sys.executable, "-c", script],
        cwd=repo_root,
        env=env,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 0, result.stdout + result.stderr


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


def test_installed_canonical_packages_precede_nested_child_copies(
    tmp_path: Path,
) -> None:
    """A wheel's top-level Tools packages must win when vendor is not shipped."""
    repo_root = tmp_path / "site-packages"
    (repo_root / "chat").mkdir(parents=True)
    (repo_root / "sidekick").mkdir()

    paths = launch_upstream_drift._launcher_bootstrap_paths(repo_root, None)

    assert paths.index(str(repo_root)) < paths.index(
        str(repo_root / "src" / "shared" / "python")
    )


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

    with pytest.raises(RuntimeError) as error:
        launch_upstream_drift._resolve_explicit_tools_root(str(invalid_tools_root))

    assert str(error.value) == (
        "TOOLS_REPO_PATH must point to a Tools checkout containing a src/ "
        f"directory, got: {invalid_tools_root.resolve()}"
    )


def test_bootstrap_import_paths_preserves_precedence(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    synthetic_sys_path = ["tail"]
    monkeypatch.setattr(launch_upstream_drift, "path", synthetic_sys_path)

    launch_upstream_drift._bootstrap_import_paths(["first", "second", "tail"])

    assert synthetic_sys_path == ["first", "second", "tail"]


def test_bootstrap_import_paths_repositions_existing_entries(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Installed site-packages must move ahead of a stale nested child path."""
    synthetic_sys_path = ["nested-child", "installed-root", "tail"]
    monkeypatch.setattr(launch_upstream_drift, "path", synthetic_sys_path)

    launch_upstream_drift._bootstrap_import_paths(["installed-root", "nested-child"])

    assert synthetic_sys_path == ["installed-root", "nested-child", "tail"]


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

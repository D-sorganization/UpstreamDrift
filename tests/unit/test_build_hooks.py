"""Tests for build_hooks module.

build_hooks.py imports hatchling at module level, so we must have hatchling
mocked before importing it.  We use contextlib.ExitStack + patch.dict so the
mock is installed at collection time and automatically removed on pytest exit.
"""

from __future__ import annotations

import contextlib
import os
import subprocess
import sys
from unittest.mock import MagicMock, patch

import pytest

pytestmark = pytest.mark.unit


class DummyHookInterface:
    def __init__(self, root, config):
        self.root = root
        self.config = config


# Install hatchling mock for the duration of this module's collection+execution.
# patch.dict is used (not direct assignment) so the entries are removed when the
# context exits in teardown_module, preventing sys.modules pollution.
_hatchling_mock_stack = contextlib.ExitStack()

_hatchling_mock_hook_interface = MagicMock()
_hatchling_mock_hook_interface.BuildHookInterface = DummyHookInterface

_hatchling_mock_stack.enter_context(
    patch.dict(
        "sys.modules",
        {
            "hatchling": MagicMock(),
            "hatchling.builders": MagicMock(),
            "hatchling.builders.hooks": MagicMock(),
            "hatchling.builders.hooks.plugin": MagicMock(),
            "hatchling.builders.hooks.plugin.interface": _hatchling_mock_hook_interface,
        },
    )
)

# Force reimport of build_hooks under the mocked hatchling
sys.modules.pop("build_hooks", None)
import build_hooks  # noqa: E402


def teardown_module(module) -> None:
    """Remove hatchling mocks and build_hooks from sys.modules."""
    _hatchling_mock_stack.close()
    sys.modules.pop("build_hooks", None)


class DummyConfig:
    def __init__(self, root, config=None):
        self.root = root
        self.config = config or {}


def test_ui_build_hook_ci_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CI", "true")
    (tmp_path / "ui" / "dist").mkdir(parents=True)
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    hook.initialize("1.0.0", {})
    # Should skip, no error


def test_ui_build_hook_skip_env(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("SKIP_UI_BUILD", "1")
    (tmp_path / "ui" / "dist").mkdir(parents=True)
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    hook.initialize("1.0.0", {})
    # Should skip, no error


def test_ui_build_hook_ci_env_without_bundle_fails(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CI", "true")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})

    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "UI bundle is missing" in str(exc.value)


def test_ui_build_hook_editable_ci_without_bundle_skips(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("CI", "true")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    hook.initialize("editable", {})


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_builds(mock_run, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    hook.initialize("1.0.0", {})

    assert mock_run.call_count == 2
    mock_run.assert_any_call(
        (
            ["npm", "ci", "--legacy-peer-deps"]
            if os.name != "nt"
            else ["npm.cmd", "ci", "--legacy-peer-deps"]
        ),
        cwd=str(tmp_path / "ui"),
        check=True,
        capture_output=True,
        text=True,
    )


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_fails(mock_run, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    mock_run.side_effect = subprocess.CalledProcessError(1, "npm")

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "UI build failed" in str(exc.value)


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_missing_npm(mock_run, monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    mock_run.side_effect = FileNotFoundError("npm")

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "npm not found" in str(exc.value)

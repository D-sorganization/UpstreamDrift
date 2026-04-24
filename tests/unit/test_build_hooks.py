"""Tests for build_hooks module.

``build_hooks`` imports ``hatchling`` at module level, so the stubs must be
in place before the module is imported. We install the stubs from
``tests._mocks.physics_stubs.hatchling_stubs`` directly into ``sys.modules``
(only for keys that are not already present) and remove exactly those keys
in ``teardown_module``. This cooperates with other test modules that may
install their own stubs, avoiding the ``patch.dict`` snapshot-and-restore
pattern that wipes unrelated entries.

This is NOT a module-level ``sys.modules[...] = MagicMock()`` assignment —
it goes through the shared helper and tracks only the keys this module
actually added.
"""

from __future__ import annotations

import os
import subprocess
import sys
from unittest.mock import patch

import pytest

pytestmark = pytest.mark.unit

from tests._mocks.physics_stubs import hatchling_stubs  # noqa: E402


class DummyHookInterface:
    def __init__(self, root, config):
        self.root = root
        self.config = config


# Install hatchling stubs only for keys we own, and record them so teardown
# removes only those entries (never other tests' stubs).
_installed_keys: list[str] = []
for _key, _value in hatchling_stubs(hook_interface=DummyHookInterface).items():
    if _key not in sys.modules:
        sys.modules[_key] = _value
        _installed_keys.append(_key)

# Force a fresh import of build_hooks under the mocked hatchling.
sys.modules.pop("build_hooks", None)
import build_hooks  # noqa: E402


def teardown_module(module) -> None:
    """Remove only the hatchling stubs installed by this test module."""
    for key in _installed_keys:
        sys.modules.pop(key, None)
    sys.modules.pop("build_hooks", None)


class DummyConfig:
    def __init__(self, root, config=None):
        self.root = root
        self.config = config or {}


def test_ui_build_hook_ci_env(monkeypatch, tmp_path):
    monkeypatch.setenv("CI", "true")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    hook.initialize("1.0.0", {})
    # Should skip, no error


def test_ui_build_hook_skip_env(monkeypatch, tmp_path):
    monkeypatch.setenv("SKIP_UI_BUILD", "1")
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    hook.initialize("1.0.0", {})
    # Should skip, no error


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_builds(mock_run, monkeypatch, tmp_path):
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
def test_ui_build_hook_fails(mock_run, monkeypatch, tmp_path):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    mock_run.side_effect = subprocess.CalledProcessError(1, "npm")

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "UI build failed" in str(exc.value)


@patch("build_hooks.subprocess.run")
def test_ui_build_hook_missing_npm(mock_run, monkeypatch, tmp_path):
    monkeypatch.delenv("CI", raising=False)
    monkeypatch.delenv("SKIP_UI_BUILD", raising=False)

    mock_run.side_effect = FileNotFoundError("npm")

    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    with pytest.raises(RuntimeError) as exc:
        hook.initialize("1.0.0", {})

    assert "npm not found" in str(exc.value)


def test_ui_dir_property(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    assert hook._ui_dir == tmp_path / "ui"


def test_dist_dir_property(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    assert hook._dist_dir == tmp_path / "ui" / "dist"


def test_force_ui_build_false_by_default(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {})
    assert hook._force_ui_build() is False


def test_force_ui_build_true_when_configured(tmp_path):
    hook = build_hooks.UIBuildHook(str(tmp_path), {"force_ui_build": True})
    assert hook._force_ui_build() is True


def test_npm_error_message_prefers_stderr():
    err = subprocess.CalledProcessError(
        1, "npm", stderr="stderr msg", output="stdout msg"
    )
    assert build_hooks.UIBuildHook._npm_error_message(err) == "stderr msg"


def test_npm_error_message_falls_back_to_stdout():
    err = subprocess.CalledProcessError(1, "npm", stderr="", output="stdout msg")
    assert build_hooks.UIBuildHook._npm_error_message(err) == "stdout msg"

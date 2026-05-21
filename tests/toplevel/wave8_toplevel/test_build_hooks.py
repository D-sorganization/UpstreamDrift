"""Unit tests for top-level build_hooks.py (UIBuildHook)."""

from __future__ import annotations

import subprocess
import sys
import types
from pathlib import Path
from unittest.mock import patch

import pytest

# Stub hatchling so build_hooks can be imported without the build backend
# installed in the test environment.
if "hatchling" not in sys.modules:
    _hatchling = types.ModuleType("hatchling")
    _builders = types.ModuleType("hatchling.builders")
    _hooks = types.ModuleType("hatchling.builders.hooks")
    _plugin = types.ModuleType("hatchling.builders.hooks.plugin")
    _iface = types.ModuleType("hatchling.builders.hooks.plugin.interface")

    class _BuildHookInterface:
        pass

    _iface.BuildHookInterface = _BuildHookInterface
    sys.modules.update(
        {
            "hatchling": _hatchling,
            "hatchling.builders": _builders,
            "hatchling.builders.hooks": _hooks,
            "hatchling.builders.hooks.plugin": _plugin,
            "hatchling.builders.hooks.plugin.interface": _iface,
        }
    )

import build_hooks  # noqa: E402
from build_hooks import UIBuildHook, _env_flag  # noqa: E402


def _make_hook(
    tmp_path: Path,
    config: dict | None = None,
) -> UIBuildHook:
    """Construct a UIBuildHook bypassing BuildHookInterface __init__."""
    hook = UIBuildHook.__new__(UIBuildHook)
    hook.root = str(tmp_path)  # type: ignore[attr-defined]
    hook.config = config or {}  # type: ignore[attr-defined]
    return hook


class TestEnvFlag:
    def test_missing_returns_false(self, monkeypatch) -> None:
        monkeypatch.delenv("X_FLAG", raising=False)
        assert _env_flag("X_FLAG") is False

    @pytest.mark.parametrize("val", ["1", "true", "yes", "on", "TRUE", " On "])
    def test_truthy_values(self, monkeypatch, val: str) -> None:
        monkeypatch.setenv("X_FLAG", val)
        assert _env_flag("X_FLAG") is True

    @pytest.mark.parametrize("val", ["0", "no", "off", "false", ""])
    def test_falsy_values(self, monkeypatch, val: str) -> None:
        monkeypatch.setenv("X_FLAG", val)
        assert _env_flag("X_FLAG") is False


class TestUIBuildHookProperties:
    def test_ui_dir(self, tmp_path: Path) -> None:
        hook = _make_hook(tmp_path)
        assert hook._ui_dir == tmp_path / "ui"

    def test_dist_dir(self, tmp_path: Path) -> None:
        hook = _make_hook(tmp_path)
        assert hook._dist_dir == tmp_path / "ui" / "dist"

    def test_force_ui_build_default(self, tmp_path: Path) -> None:
        assert _make_hook(tmp_path)._force_ui_build() is False

    def test_force_ui_build_true(self, tmp_path: Path) -> None:
        assert _make_hook(tmp_path, {"force_ui_build": True})._force_ui_build() is True

    def test_subprocess_error_message_stderr(self) -> None:
        err = subprocess.CalledProcessError(1, ["npm"], output="o", stderr="e")
        assert UIBuildHook._subprocess_error_message(err) == "e"

    def test_subprocess_error_message_stdout(self) -> None:
        err = subprocess.CalledProcessError(1, ["npm"], output="o", stderr="")
        assert UIBuildHook._subprocess_error_message(err) == "o"

    def test_subprocess_error_message_str_fallback(self) -> None:
        err = subprocess.CalledProcessError(1, ["npm"])
        result = UIBuildHook._subprocess_error_message(err)
        assert "npm" in result or "1" in result


class TestInitializeValidation:
    def test_empty_version_raises(self, tmp_path: Path) -> None:
        hook = _make_hook(tmp_path)
        with pytest.raises(ValueError, match="Version"):
            hook.initialize("", {})

    def test_none_build_data_raises(self, tmp_path: Path) -> None:
        hook = _make_hook(tmp_path)
        with pytest.raises(ValueError, match="Build data"):
            hook.initialize("1.0", None)  # type: ignore[arg-type]

    def test_none_version_raises(self, tmp_path: Path) -> None:
        hook = _make_hook(tmp_path)
        with pytest.raises(ValueError, match="Version"):
            hook.initialize(None, {})  # type: ignore[arg-type]


class TestInitializeReuse:
    def test_dist_exists_reuses(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "ui" / "dist").mkdir(parents=True)
        hook = _make_hook(tmp_path)
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        with patch.object(subprocess, "run") as run:
            hook.initialize("1.0", {})
        run.assert_not_called()

    def test_dist_exists_with_ci(self, tmp_path: Path, monkeypatch) -> None:
        (tmp_path / "ui" / "dist").mkdir(parents=True)
        monkeypatch.setenv("CI", "1")
        hook = _make_hook(tmp_path)
        with patch.object(subprocess, "run") as run:
            hook.initialize("1.0", {})
        run.assert_not_called()

    def test_skip_requested_editable_no_dist_warns(
        self, tmp_path: Path, monkeypatch, caplog
    ) -> None:
        monkeypatch.setenv("SKIP_UI_BUILD", "1")
        hook = _make_hook(tmp_path)
        caplog.set_level("WARNING", logger=build_hooks.logger.name)
        with patch.object(subprocess, "run") as run:
            hook.initialize("editable", {})
        run.assert_not_called()
        assert any("editable" in r.message for r in caplog.records)

    def test_skip_requested_wheel_no_dist_raises(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.setenv("CI", "1")
        hook = _make_hook(tmp_path)
        with pytest.raises(RuntimeError, match="UI bundle is missing"):
            hook.initialize("standard", {})


class TestInitializeBuild:
    def test_runs_npm_when_dist_missing(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        with patch.object(subprocess, "run") as run:
            hook.initialize("1.0", {})
        assert run.call_count == 2

    def test_force_rebuild_runs_npm_even_if_dist_exists(
        self, tmp_path: Path, monkeypatch
    ) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        (tmp_path / "ui" / "dist").mkdir(parents=True)
        hook = _make_hook(tmp_path, {"force_ui_build": True})
        with patch.object(subprocess, "run") as run:
            hook.initialize("1.0", {})
        assert run.call_count == 2

    def test_npm_not_found(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        with (
            patch.object(subprocess, "run", side_effect=FileNotFoundError),
            pytest.raises(RuntimeError, match="npm not found"),
        ):
            hook.initialize("1.0", {})

    def test_npm_build_fails(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        err = subprocess.CalledProcessError(1, ["npm"], output="", stderr="boom")
        with (
            patch.object(subprocess, "run", side_effect=err),
            pytest.raises(RuntimeError, match="UI build failed"),
        ):
            hook.initialize("1.0", {})

    def test_uses_npm_cmd_on_windows(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        with (
            patch.object(build_hooks.sys, "platform", "win32"),
            patch.object(subprocess, "run") as run,
        ):
            hook.initialize("1.0", {})
        cmd = run.call_args_list[0].args[0]
        assert cmd[0] == "npm.cmd"

    def test_uses_npm_on_linux(self, tmp_path: Path, monkeypatch) -> None:
        monkeypatch.delenv("CI", raising=False)
        monkeypatch.delenv("SKIP_UI_BUILD", raising=False)
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        with (
            patch.object(build_hooks.sys, "platform", "linux"),
            patch.object(subprocess, "run") as run,
        ):
            hook.initialize("1.0", {})
        cmd = run.call_args_list[0].args[0]
        assert cmd[0] == "npm"


class TestRunNpmBuild:
    def test_success(self, tmp_path: Path) -> None:
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        with patch.object(subprocess, "run") as run:
            hook._run_npm_build()
        assert run.call_count == 2

    def test_npm_missing(self, tmp_path: Path) -> None:
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        with (
            patch.object(subprocess, "run", side_effect=FileNotFoundError),
            pytest.raises(RuntimeError, match="npm not found"),
        ):
            hook._run_npm_build()

    def test_called_process_error(self, tmp_path: Path) -> None:
        (tmp_path / "ui").mkdir()
        hook = _make_hook(tmp_path)
        err = subprocess.CalledProcessError(1, ["npm"], stderr="x")
        with (
            patch.object(subprocess, "run", side_effect=err),
            pytest.raises(RuntimeError, match="UI build failed"),
        ):
            hook._run_npm_build()

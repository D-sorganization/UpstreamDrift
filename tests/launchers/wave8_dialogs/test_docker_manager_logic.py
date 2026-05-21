"""Tests for non-GUI logic in src.launchers.docker_manager."""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest

from src.launchers import docker_manager as dm


class TestGetDockerCmd:
    def test_native_docker(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dm.shutil,
            "which",
            lambda name: "/usr/bin/docker" if name == "docker" else None,
        )
        assert dm.get_docker_cmd() == ["docker"]

    def test_wsl_fallback_on_windows(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(
            dm.shutil, "which", lambda name: "/wsl/bin/wsl" if name == "wsl" else None
        )
        monkeypatch.setattr(dm.os, "name", "nt")
        assert dm.get_docker_cmd() == ["wsl", "docker"]

    def test_bare_docker_when_nothing_resolves(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.shutil, "which", lambda name: None)
        monkeypatch.setattr(dm.os, "name", "posix")
        assert dm.get_docker_cmd() == ["docker"]

    def test_no_wsl_fallback_on_posix(self, monkeypatch: pytest.MonkeyPatch) -> None:
        # Even though wsl is on PATH, non-nt OS should not pick it.
        def which(name: str) -> str | None:
            return "/x/wsl" if name == "wsl" else None

        monkeypatch.setattr(dm.shutil, "which", which)
        monkeypatch.setattr(dm.os, "name", "posix")
        assert dm.get_docker_cmd() == ["docker"]


class TestDockerBuildThreadInit:
    def test_requires_target_stage(self) -> None:
        with pytest.raises(ValueError, match="target_stage"):
            dm.DockerBuildThread(target_stage=None)  # type: ignore[arg-type]

    def test_validates_stage(self) -> None:
        t = dm.DockerBuildThread(target_stage="slim")
        assert t.target_stage == "slim"

    def test_invalid_stage_rejected(self) -> None:
        with pytest.raises(ValueError, match="Invalid Docker stage"):
            dm.DockerBuildThread(target_stage="bogus_stage_zzz")

    def test_run_with_missing_context(self) -> None:
        t = dm.DockerBuildThread(target_stage="slim", context_path=None)
        finished: list[tuple[bool, str]] = []
        t.finished_signal.connect(lambda ok, msg: finished.append((ok, msg)))
        t.run()
        assert finished and finished[0][0] is False
        assert "Invalid Docker context" in finished[0][1]

    def test_run_with_nonexistent_path(self, tmp_path: Path) -> None:
        bad = tmp_path / "does_not_exist"
        t = dm.DockerBuildThread(target_stage="slim", context_path=bad)
        finished: list[tuple[bool, str]] = []
        t.finished_signal.connect(lambda ok, msg: finished.append((ok, msg)))
        t.run()
        assert finished[0][0] is False


class TestDockerLauncherInit:
    def test_requires_repo_root(self) -> None:
        with pytest.raises(ValueError, match="repo_root"):
            dm.DockerLauncher(repo_root=None)  # type: ignore[arg-type]

    def test_basic_init(self, tmp_path: Path) -> None:
        launcher = dm.DockerLauncher(repo_root=tmp_path)
        assert launcher.repo_root == tmp_path
        assert launcher.image_name == dm.DOCKER_IMAGE_ENGINE


class TestBuildLaunchCommand:
    def _launcher(self, tmp_path: Path) -> dm.DockerLauncher:
        return dm.DockerLauncher(repo_root=tmp_path, image_name="img:test")

    def test_rejects_none_model_type(self, tmp_path: Path) -> None:
        launcher = self._launcher(tmp_path)
        with pytest.raises(ValueError, match="model_type"):
            launcher.build_launch_command(None, tmp_path / "x.py")  # type: ignore[arg-type]

    def test_drake_includes_meshcat_port(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "drake_app.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("drake", repo_path)
        assert "7000:7000" in cmd
        assert "MESHCAT_HOST=0.0.0.0" in cmd
        assert cmd[-1] == "src.drake_gui_app"

    def test_pinocchio_uses_module(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "pino.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("pinocchio", repo_path)
        assert "pinocchio_golf/gui.py" in cmd

    def test_custom_humanoid_uses_repo_path(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "humanoid.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("custom_humanoid", repo_path)
        assert cmd[-1] == "humanoid.py"

    def test_unknown_model_falls_back_to_basename(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "weird.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("some_unknown_engine", repo_path)
        assert cmd[-1] == "weird.py"

    def test_gpu_flag_added(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("drake", repo_path, use_gpu=True)
        assert "--gpus=all" in cmd

    def test_no_gpu_by_default(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("drake", repo_path, use_gpu=False)
        assert "--gpus=all" not in cmd

    def test_windows_display_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "nt")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("drake", repo_path)
        assert "DISPLAY=host.docker.internal:0" in cmd
        assert "QT_QPA_PLATFORM=xcb" in cmd

    def test_linux_uses_display_env(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        monkeypatch.setenv("DISPLAY", ":42")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("drake", repo_path)
        assert "DISPLAY=:42" in cmd
        assert "/tmp/.X11-unix:/tmp/.X11-unix" in cmd

    def test_workdir_relative_to_repo_root(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = self._launcher(tmp_path)
        repo_path = tmp_path / "models" / "deep" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")
        cmd = launcher.build_launch_command("drake", repo_path)
        # -w should be /workspace/models/deep
        idx = cmd.index("-w")
        assert cmd[idx + 1] == "/workspace/models/deep"


class TestCheckImageExists:
    def test_returns_true_on_success(self, tmp_path: Path) -> None:
        launcher = dm.DockerLauncher(repo_root=tmp_path, image_name="img:test")
        mock_result = MagicMock(returncode=0)
        with patch.object(subprocess, "run", return_value=mock_result):
            assert launcher.check_image_exists() is True

    def test_returns_false_when_no_image(self, tmp_path: Path) -> None:
        launcher = dm.DockerLauncher(repo_root=tmp_path, image_name="img:test")
        # First call returns failure; legacy calls also fail
        mock_result = MagicMock(returncode=1)
        with patch.object(subprocess, "run", return_value=mock_result):
            assert launcher.check_image_exists() is False

    def test_handles_oserror(self, tmp_path: Path) -> None:
        launcher = dm.DockerLauncher(repo_root=tmp_path)
        with patch.object(subprocess, "run", side_effect=OSError("boom")):
            assert launcher.check_image_exists() is False

    def test_legacy_alias_promoted(self, tmp_path: Path) -> None:
        launcher = dm.DockerLauncher(repo_root=tmp_path, image_name="primary:new")
        # Primary returns non-zero, legacy returns 0
        results = [MagicMock(returncode=1)] + [
            MagicMock(returncode=0) for _ in dm.LEGACY_DOCKER_IMAGE_ALIASES
        ]
        with patch.object(subprocess, "run", side_effect=results):
            if dm.LEGACY_DOCKER_IMAGE_ALIASES:
                assert launcher.check_image_exists() is True
                assert launcher.image_name in dm.LEGACY_DOCKER_IMAGE_ALIASES


class TestLaunchContainer:
    def test_requires_model_type(self, tmp_path: Path) -> None:
        launcher = dm.DockerLauncher(repo_root=tmp_path)
        with pytest.raises(ValueError, match="model_type"):
            launcher.launch_container(None, "name", tmp_path / "x.py")  # type: ignore[arg-type]

    def test_returns_process_on_success(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = dm.DockerLauncher(repo_root=tmp_path, image_name="img:test")
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")

        fake_proc = MagicMock()
        with patch.object(subprocess, "Popen", return_value=fake_proc) as m:
            result = launcher.launch_container("drake", "Drake", repo_path)
            assert result is fake_proc
            m.assert_called_once()

    def test_returns_none_on_oserror(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = dm.DockerLauncher(repo_root=tmp_path)
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")

        with patch.object(subprocess, "Popen", side_effect=FileNotFoundError("nope")):
            result = launcher.launch_container("drake", "Drake", repo_path)
            assert result is None

    def test_capture_output_uses_pipe(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm.os, "name", "posix")
        launcher = dm.DockerLauncher(repo_root=tmp_path)
        repo_path = tmp_path / "models" / "x.py"
        repo_path.parent.mkdir(parents=True)
        repo_path.write_text("")

        with patch.object(subprocess, "Popen", return_value=MagicMock()) as m:
            launcher.launch_container("drake", "Drake", repo_path, capture_output=True)
            kwargs = m.call_args.kwargs
            assert kwargs.get("stdout") == subprocess.PIPE


class TestDockerCheckThread:
    def test_emits_true_when_secure_run_succeeds(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.setattr(dm, "secure_run", lambda *a, **kw: MagicMock(returncode=0))
        thread = dm.DockerCheckThread()
        results: list[bool] = []
        thread.result.connect(results.append)
        thread.run()
        assert results == [True]

    def test_emits_false_on_secure_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(*a: object, **kw: object) -> None:
            raise dm.SecureSubprocessError("rejected")

        monkeypatch.setattr(dm, "secure_run", boom)
        thread = dm.DockerCheckThread()
        results: list[bool] = []
        thread.result.connect(results.append)
        thread.run()
        assert results == [False]

    def test_emits_false_on_file_not_found(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def boom(*a: object, **kw: object) -> None:
            raise FileNotFoundError("docker missing")

        monkeypatch.setattr(dm, "secure_run", boom)
        thread = dm.DockerCheckThread()
        results: list[bool] = []
        thread.result.connect(results.append)
        thread.run()
        assert results == [False]

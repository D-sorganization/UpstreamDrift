"""Coverage for src/shared/python/docker_config.py."""

from __future__ import annotations

import subprocess
from typing import Any
from unittest.mock import MagicMock

import pytest

from src.shared.python import docker_config


def test_constants_have_image_family_prefix() -> None:
    assert docker_config.DOCKER_IMAGE_ENGINE.endswith(":engine")
    assert docker_config.DOCKER_IMAGE_RUNTIME.endswith(":runtime")
    assert docker_config.DOCKER_IMAGE_DEV.endswith(":dev")
    assert docker_config.DOCKER_IMAGE_TRAINING.endswith(":training")
    for img in (
        docker_config.DOCKER_IMAGE_ENGINE,
        docker_config.DOCKER_IMAGE_RUNTIME,
        docker_config.DOCKER_IMAGE_DEV,
        docker_config.DOCKER_IMAGE_TRAINING,
    ):
        assert img.startswith(docker_config.DOCKER_IMAGE_FAMILY)


def test_legacy_aliases_present() -> None:
    assert "robotics_env:latest" in docker_config.LEGACY_DOCKER_ALIASES


def test_detect_gpu_no_nvidia_smi(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_config.shutil, "which", lambda _name: None)
    result = docker_config.detect_gpu_support()
    assert result["available"] is False
    assert result["device_name"] == ""
    assert result["driver_version"] == ""
    assert result["container_toolkit"] is False


def test_detect_gpu_success(monkeypatch: pytest.MonkeyPatch) -> None:
    def fake_which(name: str) -> str | None:
        if name == "nvidia-smi":
            return "/usr/bin/nvidia-smi"
        if name == "nvidia-container-cli":
            return "/usr/bin/nvidia-container-cli"
        return None

    monkeypatch.setattr(docker_config.shutil, "which", fake_which)

    calls = {"n": 0}

    def fake_run(*_args: Any, **_kwargs: Any) -> subprocess.CompletedProcess[str]:
        calls["n"] += 1
        if calls["n"] == 1:
            return subprocess.CompletedProcess(
                args=[],
                returncode=0,
                stdout="NVIDIA RTX 4090, 535.86.10\n",
                stderr="",
            )
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout="8.9\n",
            stderr="",
        )

    monkeypatch.setattr(docker_config.subprocess, "run", fake_run)
    result = docker_config.detect_gpu_support()
    assert result["available"] is True
    assert result["device_name"] == "NVIDIA RTX 4090"
    assert result["driver_version"] == "535.86.10"
    assert result["cuda_version"] == "8.9"
    assert result["container_toolkit"] is True


def test_detect_gpu_subprocess_timeout(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_config.shutil, "which", lambda name: "/usr/bin/" + name)

    def raise_timeout(*_a: Any, **_k: Any) -> Any:
        raise subprocess.TimeoutExpired(cmd="nvidia-smi", timeout=10)

    monkeypatch.setattr(docker_config.subprocess, "run", raise_timeout)
    result = docker_config.detect_gpu_support()
    assert result["available"] is False


def test_detect_gpu_smi_returns_error(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_config.shutil, "which", lambda name: "/usr/bin/" + name)

    def fake_run(*_a: Any, **_k: Any) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(
            args=[], returncode=1, stdout="", stderr="err"
        )

    monkeypatch.setattr(docker_config.subprocess, "run", fake_run)
    result = docker_config.detect_gpu_support()
    assert result["available"] is False
    assert result["container_toolkit"] is True  # which() still returns a path


def test_detect_gpu_oserror(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(docker_config.shutil, "which", lambda name: "/usr/bin/" + name)
    monkeypatch.setattr(
        docker_config.subprocess,
        "run",
        MagicMock(side_effect=OSError("boom")),
    )
    result = docker_config.detect_gpu_support()
    assert result["available"] is False

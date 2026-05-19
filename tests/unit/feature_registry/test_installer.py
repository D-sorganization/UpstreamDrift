"""Tests for the opt-in installer's safety rails.

We don't actually run ``pip install`` here — we mock the subprocess
call and assert that the *decision* path is correct.
"""

from __future__ import annotations

import subprocess
from typing import Any

import pytest

from src.shared.python.feature_registry import install_feature
from src.shared.python.feature_registry import installer as installer_mod


pytestmark = pytest.mark.unit


@pytest.fixture
def fake_subprocess(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """Mock ``subprocess.run`` so installs don't actually execute."""
    captured: dict[str, Any] = {"argv": None}

    class _Result:
        def __init__(self, argv: list[str]) -> None:
            self.argv = argv
            self.returncode = 0
            self.stdout = f"would install: {' '.join(argv)}\n"
            self.stderr = ""

    def fake_run(argv, **_kwargs):  # type: ignore[no-untyped-def]
        captured["argv"] = argv
        return _Result(argv)

    monkeypatch.setattr(installer_mod.subprocess, "run", fake_run)
    return captured


@pytest.fixture(autouse=True)
def force_outside_docker(monkeypatch: pytest.MonkeyPatch) -> None:
    """Each test should believe we're on bare metal unless it says otherwise."""
    monkeypatch.setattr(installer_mod, "_inside_docker", lambda: False)


def test_dry_run_does_not_invoke_subprocess(fake_subprocess) -> None:
    result = install_feature("mujoco", dry_run=True)
    assert result.success is True
    assert "dry-run" in result.reason
    assert fake_subprocess["argv"] is None


def test_install_pip_extra_uses_active_interpreter(fake_subprocess) -> None:
    install_feature("drake")
    argv = fake_subprocess["argv"]
    assert argv is not None
    assert argv[0:3] == [__import__("sys").executable, "-m", "pip"]
    assert "upstream-drift[drake]" in argv


def test_install_with_user_site_adds_user_flag(fake_subprocess) -> None:
    install_feature("drake", allow_user_site=True)
    argv = fake_subprocess["argv"]
    assert argv is not None
    assert "--user" in argv


def test_external_channel_returns_failure_with_hint(fake_subprocess) -> None:
    result = install_feature("pose-openpose")
    assert result.success is False
    assert "external" in result.reason or "Cannot" in result.reason
    # Subprocess must NOT have been called for an external channel.
    assert fake_subprocess["argv"] is None


def test_conda_channel_without_conda_returns_failure(
    fake_subprocess, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer_mod.shutil, "which", lambda _cmd: None)
    result = install_feature("chrono")
    assert result.success is False
    assert fake_subprocess["argv"] is None


def test_inside_docker_nonroot_refuses(
    fake_subprocess, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(installer_mod, "_inside_docker", lambda: True)
    monkeypatch.setattr(installer_mod, "_is_root", lambda: False)
    result = install_feature("drake")
    assert result.success is False
    assert "Docker" in result.reason
    assert "PROFILE=" in result.reason
    assert fake_subprocess["argv"] is None


def test_inside_docker_as_root_proceeds(
    fake_subprocess, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Root in a container is fine — that's how images are built."""
    monkeypatch.setattr(installer_mod, "_inside_docker", lambda: True)
    monkeypatch.setattr(installer_mod, "_is_root", lambda: True)
    result = install_feature("drake")
    assert result.success is True
    assert fake_subprocess["argv"] is not None


def test_timeout_returns_failure(monkeypatch: pytest.MonkeyPatch) -> None:
    def raise_timeout(*_args, **_kwargs):  # type: ignore[no-untyped-def]
        raise subprocess.TimeoutExpired(cmd="pip", timeout=1)

    monkeypatch.setattr(installer_mod.subprocess, "run", raise_timeout)
    result = install_feature("drake", timeout=0.001)
    assert result.success is False
    assert "timed out" in result.reason


def test_unknown_feature_raises_keyerror() -> None:
    with pytest.raises(KeyError):
        install_feature("not-a-feature")


def test_log_output_is_truncated() -> None:
    """`_truncate` keeps log tails to 4 KiB plus a marker."""
    long_text = "x" * (installer_mod._LOG_TAIL_BYTES * 2)
    truncated = installer_mod._truncate(long_text)
    assert len(truncated) <= installer_mod._LOG_TAIL_BYTES + 32
    assert "truncated" in truncated


def test_short_output_is_not_truncated() -> None:
    short = "hello"
    assert installer_mod._truncate(short) == short

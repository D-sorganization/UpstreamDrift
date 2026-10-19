"""Tests for the per-host shared BuildKit builder manager (#13208)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
import subprocess

import pytest

fcntl = pytest.importorskip(
    "fcntl", reason="shared_buildkit uses POSIX flock (Linux runners)"
)

from scripts.ci import shared_buildkit as sb  # noqa: E402

pytestmark = pytest.mark.unit


class FakeDocker:
    """In-memory stand-in for the docker CLI, recording every call."""

    def __init__(
        self,
        *,
        exists: bool = False,
        running: bool = False,
        image: str | None = None,
        create_fails: bool = False,
        inspect_times_out: bool = False,
    ) -> None:
        self.exists = exists
        self.running = running
        self.image = image
        self.create_fails = create_fails
        self.inspect_times_out = inspect_times_out
        self.calls: list[list[str]] = []

    def __call__(
        self, argv: Sequence[str], timeout: float
    ) -> subprocess.CompletedProcess[str]:
        args = list(argv)
        self.calls.append(args)
        if args[:3] == ["docker", "buildx", "inspect"]:
            if self.inspect_times_out:
                raise subprocess.TimeoutExpired(args, timeout)
            if not self.exists:
                return _done(
                    args, 1, err=f"ERROR: no builder {sb.BUILDER_NAME!r} found"
                )
            status = "running" if self.running else "inactive"
            return _done(args, 0 if self.running else 1, out=_inspect_output(status))
        if args[:2] == ["docker", "inspect"]:
            if not self.exists:
                return _done(args, 1, err="No such object")
            return _done(args, 0, out=f"{self.image}\n")
        if args[:3] == ["docker", "buildx", "rm"]:
            self.exists, self.running, self.image = False, False, None
            return _done(args, 0)
        if args[:3] == ["docker", "buildx", "create"]:
            if self.create_fails:
                return _done(args, 1, err="create failed")
            self.inspect_times_out = False
            self.exists, self.running = True, True
            self.image = args[args.index("--driver-opt") + 1].removeprefix("image=")
            return _done(args, 0)
        raise AssertionError(f"unexpected docker call: {args}")

    def verbs(self) -> list[str]:
        return [" ".join(c[1:3]) for c in self.calls]


def _done(
    args: list[str], rc: int, out: str = "", err: str = ""
) -> subprocess.CompletedProcess[str]:
    return subprocess.CompletedProcess(args, rc, stdout=out, stderr=err)


def _inspect_output(status: str) -> str:
    """Real ``docker buildx inspect`` layout: column-padded fields, node section."""
    return (
        f"Name:          {sb.BUILDER_NAME}\n"
        "Driver:        docker-container\n"
        "\n"
        "Nodes:\n"
        f"Name:                  {sb.BUILDER_NAME}0\n"
        f"Status:                {status}\n"
        "BuildKit version:      v0.33.1\n"
    )


@pytest.fixture
def lock(tmp_path: Path) -> Path:
    return tmp_path / "ud-buildkit.lock"


def test_healthy_pinned_builder_is_reused_without_changes(lock: Path) -> None:
    docker = FakeDocker(exists=True, running=True, image=sb.BUILDKIT_IMAGE)
    result = sb.ensure_builder(docker, lock_path=lock)
    assert result == sb.EnsureResult(sb.BUILDER_NAME, "reused", "")
    assert not any(
        c[:3] in (["docker", "buildx", "rm"], ["docker", "buildx", "create"])
        for c in docker.calls
    )


def test_missing_builder_is_created_with_pinned_image_and_gc_config(lock: Path) -> None:
    docker = FakeDocker()
    result = sb.ensure_builder(docker, lock_path=lock)
    assert result.builder == sb.BUILDER_NAME and result.action == "created"
    create = next(c for c in docker.calls if c[:3] == ["docker", "buildx", "create"])
    assert create[create.index("--name") + 1] == sb.BUILDER_NAME
    assert create[create.index("--driver") + 1] == "docker-container"
    assert create[create.index("--driver-opt") + 1] == f"image={sb.BUILDKIT_IMAGE}"
    assert Path(create[create.index("--buildkitd-config") + 1]) == sb.BUILDKITD_CONFIG
    assert "--bootstrap" in create


def test_image_mismatch_upgrades_and_keeps_cache_state(lock: Path) -> None:
    docker = FakeDocker(exists=True, running=True, image="moby/buildkit:v0.20.0")
    result = sb.ensure_builder(docker, lock_path=lock)
    assert result.action == "upgraded"
    rm = next(c for c in docker.calls if c[:3] == ["docker", "buildx", "rm"])
    assert "--keep-state" in rm
    assert docker.image == sb.BUILDKIT_IMAGE


def test_unresponsive_builder_is_recreated(lock: Path) -> None:
    docker = FakeDocker(
        exists=True, running=True, image=sb.BUILDKIT_IMAGE, inspect_times_out=True
    )
    result = sb.ensure_builder(docker, lock_path=lock)
    assert result.builder == sb.BUILDER_NAME and result.action == "recreated"
    assert "buildx rm" in docker.verbs() and "buildx create" in docker.verbs()


def test_stopped_builder_is_recreated(lock: Path) -> None:
    docker = FakeDocker(exists=True, running=False, image=sb.BUILDKIT_IMAGE)
    result = sb.ensure_builder(docker, lock_path=lock)
    assert result.action == "recreated"
    assert docker.running


def test_padded_inspect_status_is_parsed_as_running() -> None:
    # Regression: a plain "Status: running" substring check never matched the
    # column-padded real output, so every ensure fell back to a cold build.
    docker = FakeDocker(exists=True, running=True, image=sb.BUILDKIT_IMAGE)
    assert sb._inspect(docker).healthy
    assert not sb._inspect(FakeDocker(exists=True, running=False)).healthy


def test_create_failure_falls_back_to_default_builder(lock: Path) -> None:
    docker = FakeDocker(create_fails=True)
    result = sb.ensure_builder(docker, lock_path=lock)
    assert result.builder is None and result.action == "fallback"
    assert "create failed" in result.reason


def test_lock_timeout_falls_back_without_touching_docker(lock: Path) -> None:
    docker = FakeDocker(exists=True, running=True, image=sb.BUILDKIT_IMAGE)
    with lock.open("w") as held:
        fcntl.flock(held, fcntl.LOCK_EX)
        result = sb.ensure_builder(docker, lock_path=lock, lock_timeout_s=0.2)
    assert result.action == "fallback" and "lock" in result.reason
    assert docker.calls == []


def test_buildkitd_config_uses_current_gc_keys_with_a_bounded_cap() -> None:
    text = sb.BUILDKITD_CONFIG.read_text(encoding="utf-8")
    assert "maxUsedSpace" in text and "minFreeSpace" in text and "reservedSpace" in text
    assert "keepBytes" not in text
    assert "gc = true" in text


def test_main_writes_builder_output_and_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    out = tmp_path / "gh_output"
    monkeypatch.setattr(
        sb,
        "ensure_builder",
        lambda *a, **k: sb.EnsureResult(sb.BUILDER_NAME, "reused", ""),
    )
    assert sb.main(["ensure", "--github-output", str(out)]) == 0
    assert (
        out.read_text(encoding="utf-8") == f"builder={sb.BUILDER_NAME}\naction=reused\n"
    )


def test_main_fallback_warns_and_still_exits_zero(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    out = tmp_path / "gh_output"
    monkeypatch.setattr(
        sb,
        "ensure_builder",
        lambda *a, **k: sb.EnsureResult(None, "fallback", "create failed"),
    )
    assert sb.main(["ensure", "--github-output", str(out)]) == 0
    assert out.read_text(encoding="utf-8") == "builder=\naction=fallback\n"
    assert "::warning" in capsys.readouterr().out


def test_main_strict_mode_fails_on_fallback(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(
        sb, "ensure_builder", lambda *a, **k: sb.EnsureResult(None, "fallback", "x")
    )
    assert sb.main(["ensure", "--strict", "--github-output", str(tmp_path / "o")]) == 1


def test_lock_timeout_must_be_positive(lock: Path) -> None:
    with pytest.raises(ValueError, match="lock_timeout_s"):
        sb.ensure_builder(FakeDocker(), lock_path=lock, lock_timeout_s=0)


def test_default_lock_lives_beside_the_buildx_builder_store(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The lock guards buildx builder definitions, which are per docker-client
    user, so it lives in that user's buildx store, not world-writable /tmp."""
    monkeypatch.delenv("DOCKER_CONFIG", raising=False)
    monkeypatch.setenv("HOME", str(tmp_path))
    monkeypatch.setattr(sb.Path, "home", classmethod(lambda cls: tmp_path))
    assert (
        sb.default_lock_path() == tmp_path / ".docker" / "buildx" / "ud-buildkit.lock"
    )

    monkeypatch.setenv("DOCKER_CONFIG", str(tmp_path / "cfg"))
    assert sb.default_lock_path() == tmp_path / "cfg" / "buildx" / "ud-buildkit.lock"

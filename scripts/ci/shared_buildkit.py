#!/usr/bin/env python3
"""Ensure the host's persistent shared BuildKit builder (#13208).

Every self-hosted runner on a host shares one docker-container builder named
``ud-buildkit``. Its content store is the layer cache, bounded by BuildKit's
own garbage collector (see ``buildkitd.ud.toml``), so jobs neither import nor
export per-runner ``type=local`` caches.

``ensure`` is idempotent and safe to run concurrently from many runners: it
serialises create/upgrade/recreate under an exclusive ``flock`` and then:

* reuses a running builder on the pinned BuildKit image;
* upgrades a builder on another image, keeping its cache state volume;
* recreates a builder that is stopped or does not answer ``inspect``;
* creates the builder when it is missing.

Cache infrastructure must never fail a pull request. When the builder cannot be
made healthy (or the lock cannot be taken in time) ``ensure`` reports
``action=fallback`` with an empty ``builder`` output, the caller builds with the
default docker builder (a cold but correct build) and a ``::warning`` explains
why. ``--strict`` turns the fallback into a failure for canary/diagnostic runs.
"""

from __future__ import annotations

import argparse
from collections.abc import Callable, Sequence
from dataclasses import dataclass
import fcntl
import logging
from pathlib import Path
import re
import subprocess
import sys
import time
from typing import Literal

logger = logging.getLogger(__name__)

BUILDER_NAME = "ud-buildkit"
#: Pinned daemon image. Bumping it upgrades every host in place on its next job
#: (the cache state volume is kept).
BUILDKIT_IMAGE = "moby/buildkit:v0.33.1"
BUILDKITD_CONFIG = Path(__file__).with_name("buildkitd.ud.toml")
DEFAULT_LOCK_PATH = Path("/tmp/ud-buildkit.lock")
DEFAULT_LOCK_TIMEOUT_S = 600.0
INSPECT_TIMEOUT_S = 120.0
CREATE_TIMEOUT_S = 300.0
#: ``docker buildx inspect`` pads its fields, e.g. ``Status:                running``.
_RUNNING_RE = re.compile(r"^Status:\s+running\s*$", re.MULTILINE)

Action = Literal["reused", "created", "upgraded", "recreated", "fallback"]
Runner = Callable[[Sequence[str], float], "subprocess.CompletedProcess[str]"]


@dataclass(frozen=True)
class EnsureResult:
    """Outcome of :func:`ensure_builder`.

    ``builder`` is the builder name to pass to ``docker/build-push-action``, or
    ``None`` when the caller must use the default builder (``action="fallback"``).
    """

    builder: str | None
    action: Action
    reason: str


@dataclass(frozen=True)
class _BuilderState:
    healthy: bool
    exists: bool
    image: str | None
    detail: str


def run_docker(argv: Sequence[str], timeout: float) -> subprocess.CompletedProcess[str]:
    """Run one docker CLI command, capturing text output (never raises on rc)."""
    return subprocess.run(
        list(argv), capture_output=True, text=True, timeout=timeout, check=False
    )


def _container_name() -> str:
    return f"buildx_buildkit_{BUILDER_NAME}0"


def _inspect(run: Runner) -> _BuilderState:
    try:
        res = run(
            ["docker", "buildx", "inspect", "--bootstrap", BUILDER_NAME],
            INSPECT_TIMEOUT_S,
        )
    except subprocess.TimeoutExpired:
        return _BuilderState(
            False, True, None, "inspect timed out (unresponsive buildkitd)"
        )
    if res.returncode != 0 and "no builder" in (res.stderr or "").lower():
        return _BuilderState(False, False, None, "builder missing")
    running = res.returncode == 0 and bool(_RUNNING_RE.search(res.stdout or ""))
    img = run(
        ["docker", "inspect", "--format", "{{.Config.Image}}", _container_name()],
        INSPECT_TIMEOUT_S,
    )
    image = img.stdout.strip() if img.returncode == 0 else None
    detail = (
        "running"
        if running
        else f"not running: {(res.stderr or res.stdout or '').strip()[:200]}"
    )
    return _BuilderState(running, True, image, detail)


def _remove(run: Runner, *, keep_state: bool) -> None:
    argv = ["docker", "buildx", "rm", "--force"]
    if keep_state:
        argv.append("--keep-state")
    run([*argv, BUILDER_NAME], CREATE_TIMEOUT_S)


def _create(run: Runner) -> str:
    """Create and bootstrap the builder; return an error string ('' on success)."""
    res = run(
        [
            "docker", "buildx", "create",
            "--name", BUILDER_NAME,
            "--driver", "docker-container",
            "--driver-opt", f"image={BUILDKIT_IMAGE}",
            "--buildkitd-config", str(BUILDKITD_CONFIG),
            "--bootstrap",
        ],
        CREATE_TIMEOUT_S,
    )  # fmt: skip
    if res.returncode != 0:
        return (res.stderr or res.stdout or "docker buildx create failed").strip()[:300]
    return ""


def _acquire(lock_file, timeout_s: float) -> bool:  # noqa: ANN001 - text file object
    deadline = time.monotonic() + timeout_s
    while True:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return True
        except BlockingIOError:
            if time.monotonic() >= deadline:
                return False
            time.sleep(0.1)


def _reconcile(run: Runner) -> EnsureResult:
    state = _inspect(run)
    if state.healthy and state.image == BUILDKIT_IMAGE:
        return EnsureResult(BUILDER_NAME, "reused", "")
    if state.healthy:
        action: Action = "upgraded"
        _remove(run, keep_state=True)
    elif state.exists:
        action = "recreated"
        _remove(run, keep_state=False)
    else:
        action = "created"
    error = _create(run)
    if error:
        return EnsureResult(None, "fallback", f"{action} failed: {error}")
    verified = _inspect(run)
    if not verified.healthy:
        return EnsureResult(
            None, "fallback", f"{action} builder unhealthy: {verified.detail}"
        )
    return EnsureResult(BUILDER_NAME, action, state.detail)


def ensure_builder(
    run: Runner = run_docker,
    *,
    lock_path: Path = DEFAULT_LOCK_PATH,
    lock_timeout_s: float = DEFAULT_LOCK_TIMEOUT_S,
) -> EnsureResult:
    """Make the shared builder healthy on the pinned image, or report a fallback.

    Preconditions: ``lock_timeout_s > 0``. Postcondition: ``builder`` is
    ``BUILDER_NAME`` exactly when the builder answered a post-change inspect
    as running (or was already running on the pinned image).
    """
    if lock_timeout_s <= 0:
        raise ValueError(f"lock_timeout_s must be positive, got {lock_timeout_s}")
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a") as lock_file:
        if not _acquire(lock_file, lock_timeout_s):
            return EnsureResult(
                None,
                "fallback",
                f"could not take {lock_path} lock in {lock_timeout_s:.0f}s",
            )
        try:
            return _reconcile(run)
        except (OSError, subprocess.SubprocessError) as exc:
            return EnsureResult(None, "fallback", f"docker error: {exc}")


def _write_outputs(path: Path | None, result: EnsureResult) -> None:
    if path is None:
        return
    with path.open("a", encoding="utf-8") as fh:
        fh.write(f"builder={result.builder or ''}\naction={result.action}\n")


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)
    ens = sub.add_parser("ensure", help="create/repair the shared builder")
    ens.add_argument("--github-output", type=Path, default=None)
    ens.add_argument("--lock-timeout", type=float, default=DEFAULT_LOCK_TIMEOUT_S)
    ens.add_argument(
        "--strict", action="store_true", help="exit 1 instead of falling back"
    )
    args = parser.parse_args(argv)

    result = ensure_builder(lock_timeout_s=args.lock_timeout)
    _write_outputs(args.github_output, result)
    if result.action == "fallback":
        print(
            f"::warning::shared BuildKit builder unavailable, building cold: {result.reason}"
        )
        return 1 if args.strict else 0
    print(
        f"shared BuildKit builder {result.builder}: {result.action} {result.reason}".rstrip()
    )
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

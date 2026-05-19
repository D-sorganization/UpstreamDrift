"""Opt-in installer for missing features.

When a user triggers an install from the CLI, API, or GUI we run the
feature's documented install command in a subprocess, stream its
output to the caller, then refresh the registry. The single most
important guarantee is that **nothing happens automatically** — we
only run when the caller has explicitly opted in.

Safety rails
------------
* Refuse to run inside a Docker container's non-root user — the
  correct fix there is to rebuild the image with a larger profile,
  not to mutate the runtime venv. Detection: presence of
  ``/.dockerenv`` *and* effective UID != 0.
* Refuse to run a ``conda`` channel install when ``conda`` is not on
  PATH; surface the documented manual command instead.
* Refuse to run the literal ``external`` install for OpenPose; point
  the user at the host-build docs.
* Never install with ``--user`` *and* ``sys.prefix == sys.base_prefix``
  simultaneously without an explicit opt-in (we don't silently
  pollute system Python).
"""

from __future__ import annotations

import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.feature_registry.features import Feature, get_feature

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class InstallResult:
    """Outcome of a feature install attempt.

    Attributes:
        feature: The feature that was requested.
        success: Whether the install completed without error.
        command: The actual shell command that was run (or proposed,
            for ``rejected`` outcomes).
        stdout: Captured stdout (truncated to the last 4 KiB).
        stderr: Captured stderr (truncated to the last 4 KiB).
        returncode: Subprocess return code; ``-1`` if no subprocess
            ran (rejected for a safety reason).
        reason: One-line summary, suitable for logs and UI surfaces.
    """

    feature: str
    success: bool
    command: str
    stdout: str
    stderr: str
    returncode: int
    reason: str


_LOG_TAIL_BYTES = 4096


def _truncate(text: str) -> str:
    if len(text) <= _LOG_TAIL_BYTES:
        return text
    return "…(truncated)…\n" + text[-_LOG_TAIL_BYTES:]


def _inside_docker() -> bool:
    """Detect a Docker (or compatible OCI) runtime container."""
    if Path("/.dockerenv").exists():
        return True
    try:
        with open("/proc/1/cgroup", encoding="utf-8") as fh:
            cgroup = fh.read()
        return "docker" in cgroup or "containerd" in cgroup or "kubepods" in cgroup
    except OSError:
        return False


def _is_root() -> bool:
    """``True`` on POSIX root or Windows (where the concept does not apply)."""
    geteuid = getattr(os, "geteuid", None)
    if geteuid is None:  # Windows
        return True
    return geteuid() == 0


def _docker_rebuild_hint(feature: Feature) -> str:
    """Render the Docker-rebuild hint surfaced when in-container installs are refused."""
    stage = feature.docker_stage or "standard"
    return (
        f"Refusing to install {feature.name!r} inside an unprivileged "
        "Docker container. Rebuild the image with a profile that includes "
        f"this feature, e.g.:\n"
        f"  docker build --build-arg PROFILE={stage} -t upstream-drift:{stage} ."
        f"\nSee docs/development/DOCKER_SETUP.md for the profile catalog."
    )


def _resolve_pip_command(feature: Feature, allow_user_site: bool) -> list[str] | None:
    """Convert the documented install command into an argv list.

    Returns:
        The argv list to execute, or ``None`` if the channel cannot
        be executed automatically (external builds, missing conda).
    """
    if feature.install_channel == "external":
        return None

    if feature.install_channel == "conda":
        if shutil.which("conda") is None:
            return None
        # Documented command is a verbatim conda invocation.
        return feature.install_command.split()

    # pip / pip-extra
    if feature.pip_extra:
        spec = f"upstream-drift[{feature.pip_extra}]"
    else:
        # Strip the leading ``pip install`` and use whatever args are
        # provided in the documented command, so e.g. the torch CUDA
        # index URL is preserved.
        tokens = feature.install_command.split()
        if tokens[:2] != ["pip", "install"]:
            logger.warning(
                "Feature %s has non-standard install command %r; falling "
                "back to running it verbatim via shell=False",
                feature.name,
                feature.install_command,
            )
            return tokens
        spec = " ".join(tokens[2:])
        # Use the same interpreter that's running the registry, so the
        # install lands in the active venv.
        argv = [sys.executable, "-m", "pip", "install"] + spec.split()
        if allow_user_site:
            argv.append("--user")
        return argv

    base = [sys.executable, "-m", "pip", "install", spec]
    if allow_user_site:
        base.append("--user")
    return base


def install_feature(
    name: str,
    *,
    allow_user_site: bool = False,
    timeout: float = 600.0,
    dry_run: bool = False,
) -> InstallResult:
    """Install (or attempt to install) the feature named ``name``.

    Args:
        name: Feature name as registered in :data:`FEATURES`.
        allow_user_site: Pass ``--user`` to pip. Default ``False`` so
            that we install into the active venv when one is present.
        timeout: Subprocess timeout in seconds.
        dry_run: If ``True``, return the command we *would* run
            without executing it. Useful for the UI's "Show command"
            preview.

    Returns:
        An :class:`InstallResult` with success/failure plus captured
        output. The registry is *not* refreshed here; the caller does
        that after deciding whether to proceed.
    """
    feature = get_feature(name)

    # ── Safety rails ───────────────────────────────────────────────────
    if _inside_docker() and not _is_root():
        return InstallResult(
            feature=feature.name,
            success=False,
            command=feature.install_command,
            stdout="",
            stderr="",
            returncode=-1,
            reason=_docker_rebuild_hint(feature),
        )

    argv = _resolve_pip_command(feature, allow_user_site=allow_user_site)
    if argv is None:
        return InstallResult(
            feature=feature.name,
            success=False,
            command=feature.install_command,
            stdout="",
            stderr="",
            returncode=-1,
            reason=(
                f"Cannot auto-install {feature.name!r} via "
                f"{feature.install_channel}. Documented command: "
                f"{feature.install_command}"
            ),
        )

    rendered = " ".join(argv)

    if dry_run:
        return InstallResult(
            feature=feature.name,
            success=True,
            command=rendered,
            stdout="",
            stderr="",
            returncode=0,
            reason=f"dry-run: would execute: {rendered}",
        )

    logger.info("Installing %s with: %s", feature.name, rendered)
    try:
        proc = subprocess.run(
            argv,
            check=False,
            capture_output=True,
            text=True,
            timeout=timeout,
        )
    except subprocess.TimeoutExpired:
        return InstallResult(
            feature=feature.name,
            success=False,
            command=rendered,
            stdout="",
            stderr="",
            returncode=-1,
            reason=f"install timed out after {timeout:.0f}s",
        )
    except FileNotFoundError as exc:
        return InstallResult(
            feature=feature.name,
            success=False,
            command=rendered,
            stdout="",
            stderr="",
            returncode=-1,
            reason=f"installer executable not found: {exc}",
        )

    success = proc.returncode == 0
    reason = (
        f"installed {feature.name}"
        if success
        else f"install failed for {feature.name} (exit {proc.returncode})"
    )
    return InstallResult(
        feature=feature.name,
        success=success,
        command=rendered,
        stdout=_truncate(proc.stdout),
        stderr=_truncate(proc.stderr),
        returncode=proc.returncode,
        reason=reason,
    )

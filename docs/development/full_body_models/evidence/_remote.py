"""Shared environment-driven remote runner configuration for evidence scripts."""

from __future__ import annotations

import os
import subprocess
from collections.abc import Callable
from typing import Any


def resolve_remote_target(
    default_distro: str,
    run_fn: Callable[..., Any] | None = None,
) -> tuple[str, str, str]:
    """Return ``(ssh_host, wsl_distro, remote_home)`` from environment settings.

    Requires ``UD_REMOTE_HOST`` to be set to a configured SSH host alias.
    Uses ``UD_WSL_DISTRO`` (defaulting to *default_distro*) and ``UD_REMOTE_HOME``
    (or queries ``printenv HOME`` on the remote WSL instance when unset).
    """
    host = os.environ.get("UD_REMOTE_HOST", "").strip()
    if not host:
        raise RuntimeError("UD_REMOTE_HOST must be set to execute remote verification.")

    distro = os.environ.get("UD_WSL_DISTRO", "").strip() or default_distro
    remote_home = os.environ.get("UD_REMOTE_HOME", "").strip()
    if not remote_home:
        runner = subprocess.run if run_fn is None else run_fn
        res = runner(
            ["ssh", host, "wsl", "-d", distro, "printenv", "HOME"],
            capture_output=True,
        )
        if res.returncode != 0:
            err = res.stderr
            if isinstance(err, bytes):
                err = err.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Failed to resolve remote HOME: {err}")
        out = res.stdout
        if isinstance(out, bytes):
            out = out.decode("utf-8", errors="ignore")
        remote_home = out.strip()
        if not remote_home:
            raise RuntimeError("Remote printenv HOME returned empty output.")

    return host, distro, remote_home.rstrip("/")

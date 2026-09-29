"""Ephemeral capability contract for launcher-owned local API processes."""

from __future__ import annotations

import secrets
import uuid
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any

LAUNCHER_TOKEN_ENV = "UD_LAUNCHER_CSRF_TOKEN"
INSTANCE_ID_ENV = "UD_SIDEKICK_INSTANCE_ID"


@dataclass(frozen=True, repr=False)
class LauncherCapability:
    """Proof and public identity shared by one launcher/API process pair.

    Preconditions:
        ``token`` and ``instance_id`` are non-empty strings.

    Invariant:
        ``repr`` never exposes the secret token.
    """

    token: str
    instance_id: str

    def __post_init__(self) -> None:
        if not isinstance(self.token, str):
            raise TypeError("token must be a string")
        if not self.token:
            raise ValueError("token must not be blank")
        if not isinstance(self.instance_id, str):
            raise TypeError("instance_id must be a string")
        if not self.instance_id:
            raise ValueError("instance_id must not be blank")

    def __repr__(self) -> str:
        return f"LauncherCapability(token=<redacted>, instance_id={self.instance_id!r})"

    @classmethod
    def generate(cls) -> LauncherCapability:
        """Create a cryptographically random proof and a public instance ID."""
        return cls(
            token=secrets.token_urlsafe(32),
            instance_id=uuid.uuid4().hex,
        )

    def export(self, environ: MutableMapping[str, str]) -> None:
        """Export this capability to a child-process environment."""
        if environ is None:
            raise ValueError("environ must be provided")
        environ[LAUNCHER_TOKEN_ENV] = self.token
        environ[INSTANCE_ID_ENV] = self.instance_id


def install_launcher_capability(
    app: Any,
    environ: Mapping[str, str],
) -> None:
    """Install launcher proof on a FastAPI-like app state.

    Missing values deliberately become empty strings so the local WebSocket
    guard fails closed for independently started API processes.
    """
    if app is None:
        raise ValueError("app must be provided")
    if environ is None:
        raise ValueError("environ must be provided")
    state = getattr(app, "state", None)
    if state is None:
        raise ValueError("app must expose state")
    state.launcher_csrf_token = environ.get(LAUNCHER_TOKEN_ENV, "")
    state.sidekick_instance_id = environ.get(INSTANCE_ID_ENV, "")


__all__ = [
    "INSTANCE_ID_ENV",
    "LAUNCHER_TOKEN_ENV",
    "LauncherCapability",
    "install_launcher_capability",
]

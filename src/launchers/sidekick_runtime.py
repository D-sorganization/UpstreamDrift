"""Isolated runtime configuration for the classic Sidekick API child."""

from __future__ import annotations

import socket
from collections.abc import Callable, MutableMapping
from dataclasses import dataclass

from src.api.auth.launcher_capability import LauncherCapability

API_PORT_ENV = "API_PORT"
CANONICAL_API_PORT_ENV = "GOLF_API_PORT"
DEFAULT_API_PORT = 8000
_MIN_PORT = 1
_MAX_PORT = 65535

PortSelector = Callable[[int], int]
CapabilityFactory = Callable[[], LauncherCapability]


def _validate_port(port: int) -> int:
    if not isinstance(port, int):
        raise TypeError("port must be an integer")
    if not _MIN_PORT <= port <= _MAX_PORT:
        raise ValueError(f"port must be between {_MIN_PORT} and {_MAX_PORT}")
    return port


def _is_loopback_port_available(port: int) -> bool:
    with socket.socket() as probe:
        try:
            probe.bind(("127.0.0.1", port))
        except OSError:
            return False
    return True


def select_loopback_port(
    preferred_port: int = DEFAULT_API_PORT,
    *,
    is_port_available: Callable[[int], bool] = _is_loopback_port_available,
) -> int:
    """Select the preferred loopback port or an OS-assigned free port.

    Preconditions:
        ``preferred_port`` is in the TCP port range.

    Postcondition:
        The returned port was bindable on loopback at selection time.
    """
    preferred = _validate_port(preferred_port)
    if is_port_available(preferred):
        return preferred
    with socket.socket() as reservation:
        reservation.bind(("127.0.0.1", 0))
        selected = int(reservation.getsockname()[1])
    return _validate_port(selected)


def _parse_explicit_port(environ: MutableMapping[str, str]) -> int | None:
    legacy = environ.get(API_PORT_ENV)
    canonical = environ.get(CANONICAL_API_PORT_ENV)
    if legacy and canonical and legacy != canonical:
        raise ValueError(
            f"Sidekick API port conflict: {API_PORT_ENV}={legacy!r} and "
            f"{CANONICAL_API_PORT_ENV}={canonical!r}"
        )
    raw_port = canonical or legacy
    if raw_port is None:
        return None
    try:
        parsed = int(raw_port)
    except ValueError as exc:
        raise ValueError(f"Invalid Sidekick API port: {raw_port!r}") from exc
    return _validate_port(parsed)


@dataclass(frozen=True, repr=False)
class SidekickRuntimeConfig:
    """One launcher's port and ephemeral API capability."""

    port: int
    capability: LauncherCapability

    def __post_init__(self) -> None:
        _validate_port(self.port)
        if not isinstance(self.capability, LauncherCapability):
            raise TypeError("capability must be a LauncherCapability")

    @property
    def instance_id(self) -> str:
        """Return the public API instance identity used for readiness."""
        return self.capability.instance_id

    def __repr__(self) -> str:
        return (
            f"SidekickRuntimeConfig(port={self.port}, "
            f"instance_id={self.instance_id!r}, token=<redacted>)"
        )

    def export(self, environ: MutableMapping[str, str]) -> None:
        """Export a consistent API/client contract to ``environ``."""
        if environ is None:
            raise ValueError("environ must be provided")
        port_text = str(self.port)
        environ[API_PORT_ENV] = port_text
        environ[CANONICAL_API_PORT_ENV] = port_text
        self.capability.export(environ)


def configure_sidekick_runtime(
    environ: MutableMapping[str, str],
    *,
    port_selector: PortSelector = select_loopback_port,
    capability_factory: CapabilityFactory = LauncherCapability.generate,
) -> SidekickRuntimeConfig:
    """Create and export one isolated Sidekick runtime contract.

    Explicit API port configuration is honored only when the legacy and
    canonical settings agree. Otherwise a free loopback port is selected.
    """
    if environ is None:
        raise ValueError("environ must be provided")
    explicit_port = _parse_explicit_port(environ)
    port = (
        explicit_port if explicit_port is not None else port_selector(DEFAULT_API_PORT)
    )
    capability = capability_factory()
    config = SidekickRuntimeConfig(port=port, capability=capability)
    config.export(environ)
    return config


__all__ = [
    "API_PORT_ENV",
    "CANONICAL_API_PORT_ENV",
    "DEFAULT_API_PORT",
    "SidekickRuntimeConfig",
    "configure_sidekick_runtime",
    "select_loopback_port",
]

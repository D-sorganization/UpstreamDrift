"""Configurable simulator discovery and support diagnostics (GS-08, #10197).

Follows Design by Contract (DbC), Law of Demeter, and DRY.
Zero hardcoded user paths or Windows registry scraping.
Vendor raw simulator socket is strictly restricted to loopback.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import sys
from typing import Any, Mapping

_LOOPBACK_HOSTS = frozenset({"127.0.0.1", "localhost", "::1", "0:0:0:0:0:0:0:1"})


def _utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


@dataclass(frozen=True)
class SimulatorEndpoint:
    """Connection endpoint configuration for vendor simulator socket."""

    host: str = "127.0.0.1"
    port: int = 921
    timeout_seconds: float = 5.0

    def __post_init__(self) -> None:
        if isinstance(self.port, bool):
            raise TypeError("Port cannot be a boolean value")
        try:
            port_num = int(self.port)
        except (ValueError, TypeError) as exc:
            raise TypeError(f"Port must be an integer: {exc}") from exc
        if not (1 <= port_num <= 65535):
            raise ValueError(f"Port must be between 1 and 65535, got {port_num}")

        host_clean = str(self.host).strip().lower()
        if not host_clean:
            raise ValueError("Host cannot be empty")
        if host_clean not in _LOOPBACK_HOSTS:
            raise ValueError(
                f"Vendor simulator socket must be restricted to loopback ({sorted(_LOOPBACK_HOSTS)}), "
                f"got {self.host!r}. Public raw port exposure is forbidden."
            )

    @property
    def url(self) -> str:
        """Formatted host:port string."""
        return f"{self.host}:{self.port}"


@dataclass(frozen=True)
class SupportReceipt:
    """Non-secret diagnostics receipt documenting simulator environment."""

    app_version: str
    connector_version: str
    profile_version: str
    endpoint: str
    platform: str
    discovery_method: str
    optional_packages: dict[str, bool] = field(default_factory=dict)
    generated_at_utc: str = field(default_factory=_utc_now_iso)

    def to_dict(self) -> dict[str, Any]:
        """Convert receipt to dictionary excluding any sensitive data."""
        return {
            "app_version": self.app_version,
            "connector_version": self.connector_version,
            "profile_version": self.profile_version,
            "endpoint": self.endpoint,
            "platform": self.platform,
            "discovery_method": self.discovery_method,
            "optional_packages": dict(self.optional_packages),
            "generated_at_utc": self.generated_at_utc,
        }


@dataclass(frozen=True)
class SimulatorConfig:
    """Discovered configuration for simulator adapter and bridge."""

    endpoint: SimulatorEndpoint
    profile_version: str
    discovery_method: str
    install_dir: Path | None = None

    def build_support_receipt(
        self,
        app_version: str = "unknown",
        connector_version: str = "1.0.0",
        optional_packages: Mapping[str, bool] | None = None,
    ) -> SupportReceipt:
        """Generate audit receipt with non-secret diagnostics."""
        return SupportReceipt(
            app_version=app_version,
            connector_version=connector_version,
            profile_version=self.profile_version,
            endpoint=self.endpoint.url,
            platform=sys.platform,
            discovery_method=self.discovery_method,
            optional_packages=dict(optional_packages) if optional_packages else {},
        )


def discover_simulator_installation(
    env: Mapping[str, str] | None = None,
    config_path: Path | None = None,
) -> SimulatorConfig:
    """Discover simulator configuration without registry scraping or hardcoded paths.

    Precedence:
    1. Explicit environment variables (GSPRO_HOST, GSPRO_API_PORT, GSPRO_PROFILE_PATH, GSPRO_INSTALL_DIR).
    2. Optional configuration file if provided.
    3. Safe loopback defaults (127.0.0.1:921, profile v1).
    """
    source_env = env if env is not None else os.environ

    cfg_dict: dict[str, Any] = {}
    if config_path is not None:
        cfg_p = Path(config_path)
        if cfg_p.is_file():
            try:
                cfg_dict = json.loads(cfg_p.read_text(encoding="utf-8"))
            except Exception as exc:
                raise ValueError(
                    f"Failed to parse simulator config file: {exc}"
                ) from exc

    install_dir_str = source_env.get("GSPRO_INSTALL_DIR") or cfg_dict.get("install_dir")
    install_dir = Path(install_dir_str) if install_dir_str else None

    host = str(source_env.get("GSPRO_HOST") or cfg_dict.get("host") or "127.0.0.1")
    port_val = source_env.get("GSPRO_API_PORT") or cfg_dict.get("port") or 921
    try:
        port = int(port_val)
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Invalid GSPRO_API_PORT: {port_val}") from exc

    profile_version = str(
        source_env.get("GSPRO_PROFILE_PATH") or cfg_dict.get("profile_version") or "v1"
    )

    # Determine discovery method
    is_custom_env = any(
        k in source_env
        for k in (
            "GSPRO_INSTALL_DIR",
            "GSPRO_API_PORT",
            "GSPRO_HOST",
            "GSPRO_PROFILE_PATH",
        )
    )
    if is_custom_env:
        discovery_method = "environment"
    elif cfg_dict:
        discovery_method = "config_file"
    else:
        discovery_method = "defaults"

    endpoint = SimulatorEndpoint(host=host, port=port)
    return SimulatorConfig(
        endpoint=endpoint,
        profile_version=profile_version,
        discovery_method=discovery_method,
        install_dir=install_dir,
    )

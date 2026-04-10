"""Installer packaging profiles for Windows distribution targets."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.config.provider_catalog import iter_known_provider_ids

_DISCOVERY_MODES = frozenset({"local-only", "hybrid", "provider-first"})


@dataclass(frozen=True)
class PackagingProfile:
    """Declarative packaging policy for an installer build target."""

    profile_id: str
    display_name: str
    description: str
    discovery_mode: str
    include_api_executable: bool
    bundle_optional_engines: bool
    supported_provider_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.discovery_mode not in _DISCOVERY_MODES:
            raise ValueError(f"invalid discovery mode: {self.discovery_mode}")


_KNOWN_PROVIDER_IDS = iter_known_provider_ids()

PACKAGING_PROFILES: dict[str, PackagingProfile] = {
    "core": PackagingProfile(
        profile_id="core",
        display_name="Core Launcher",
        description="Standalone launcher bundle with local-only model discovery.",
        discovery_mode="local-only",
        include_api_executable=False,
        bundle_optional_engines=False,
        supported_provider_ids=(),
    ),
    "hybrid": PackagingProfile(
        profile_id="hybrid",
        display_name="Hybrid Suite",
        description="Launcher bundle with local engines and optional external providers.",
        discovery_mode="hybrid",
        include_api_executable=True,
        bundle_optional_engines=True,
        supported_provider_ids=_KNOWN_PROVIDER_IDS,
    ),
    "full": PackagingProfile(
        profile_id="full",
        display_name="Full Biomechanics Suite",
        description="Provider-first launcher bundle for fully provisioned biomechanics workstations.",
        discovery_mode="provider-first",
        include_api_executable=True,
        bundle_optional_engines=True,
        supported_provider_ids=_KNOWN_PROVIDER_IDS,
    ),
}


def get_packaging_profile(profile_name: str | None) -> PackagingProfile:
    """Return the named packaging profile with a stable default."""
    normalized = (profile_name or "hybrid").strip().lower()
    try:
        return PACKAGING_PROFILES[normalized]
    except KeyError as exc:
        valid = ", ".join(sorted(PACKAGING_PROFILES))
        raise ValueError(
            f"unknown packaging profile '{normalized}' (expected {valid})"
        ) from exc


def iter_packaging_profile_ids() -> tuple[str, ...]:
    """Return the supported profile IDs in a stable order."""
    return tuple(PACKAGING_PROFILES)


def build_profile_environment(
    profile: PackagingProfile,
    provider_roots: tuple[str | os.PathLike[str], ...] = (),
    *,
    base_env: dict[str, str] | None = None,
) -> dict[str, str]:
    """Build the environment overrides required by the installer pipeline."""
    env = dict(os.environ if base_env is None else base_env)
    env["UPSTREAM_DRIFT_INSTALL_PROFILE"] = profile.profile_id
    env["UPSTREAM_DRIFT_DISCOVERY_MODE"] = profile.discovery_mode
    if provider_roots:
        env["UPSTREAM_DRIFT_PROVIDER_ROOTS"] = os.pathsep.join(
            str(Path(root)) for root in provider_roots
        )
    else:
        env.pop("UPSTREAM_DRIFT_PROVIDER_ROOTS", None)
    return env

#!/usr/bin/env python3
"""Tier metadata for reproducible SBOM generation."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path

PROJECT_DISTRIBUTION = "upstream-drift"
ARTIFACT_FORMATS = ("cyclonedx", "spdx")


@dataclass(frozen=True)
class SbomTier:
    """Install metadata for one SBOM dependency tier.

    Preconditions:
        name is one of the project-supported SBOM tiers.

    Postconditions:
        install_spec returns the exact package spec to install for a release.
    """

    name: str
    extra: str | None

    def install_spec(self, version: str) -> str:
        """Return the PyPI install spec for this tier and release version."""
        _require_non_empty("version", version)
        if self.extra is None:
            return f"{PROJECT_DISTRIBUTION}=={version}"
        return f"{PROJECT_DISTRIBUTION}[{self.extra}]=={version}"


SBOM_TIERS = {
    "core": SbomTier(name="core", extra=None),
    "extended": SbomTier(name="extended", extra="all-engines"),
    "full": SbomTier(name="full", extra="all"),
}


def _require_non_empty(name: str, value: str) -> None:
    if not isinstance(value, str):
        raise TypeError(f"{name} must be a string")
    if not value.strip():
        raise ValueError(f"{name} must be non-empty")


def require_tier(tier: str) -> SbomTier:
    """Return tier metadata or raise for unsupported tiers."""
    _require_non_empty("tier", tier)
    try:
        return SBOM_TIERS[tier]
    except KeyError as exc:
        raise ValueError(f"unsupported SBOM tier: {tier}") from exc


def install_spec_for_tier(tier: str, version: str) -> str:
    """Return the release install spec for an SBOM tier."""
    return require_tier(tier).install_spec(version)


def expected_artifact_names(version: str) -> list[str]:
    """Return all expected per-tier SBOM artifact names for a release."""
    _require_non_empty("version", version)
    artifacts: list[str] = []
    for tier in SBOM_TIERS:
        for artifact_format in ARTIFACT_FORMATS:
            artifacts.append(
                f"{PROJECT_DISTRIBUTION}-{version}.{artifact_format}.{tier}.json"
            )
    return artifacts


def write_baseline(path: Path, version: str) -> None:
    """Write committed SBOM tier metadata for CI consistency checks."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    payload = {
        "schema_version": 1,
        "project": PROJECT_DISTRIBUTION,
        "version": version,
        "tiers": {
            name: {"install_spec": tier.install_spec(version)}
            for name, tier in SBOM_TIERS.items()
        },
        "expected_artifacts": expected_artifact_names(version),
    }
    path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--version", required=True, help="Release version.")
    parser.add_argument(
        "--write-baseline",
        type=Path,
        help="Optional path where SBOM baseline JSON should be written.",
    )
    return parser.parse_args()


def main() -> int:
    """Print SBOM tier metadata as JSON."""
    args = parse_args()
    if args.write_baseline:
        write_baseline(args.write_baseline, args.version)
        return 0
    payload = {
        "tiers": {
            name: tier.install_spec(args.version) for name, tier in SBOM_TIERS.items()
        },
        "expected_artifacts": expected_artifact_names(args.version),
    }
    print(json.dumps(payload, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

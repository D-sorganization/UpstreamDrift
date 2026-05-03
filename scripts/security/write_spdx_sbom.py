#!/usr/bin/env python3
"""Write a minimal SPDX 2.3 JSON SBOM for the active Python environment."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from importlib import metadata
from pathlib import Path

SPDX_VERSION = "SPDX-2.3"
DATA_LICENSE = "CC0-1.0"
NO_ASSERTION = "NOASSERTION"


def package_entries() -> list[dict[str, str]]:
    """Return sorted SPDX package entries for installed distributions."""
    packages: list[dict[str, str]] = []
    for distribution in metadata.distributions():
        name = distribution.metadata.get("Name", "").strip()
        if not name:
            continue
        version = distribution.version
        packages.append(
            {
                "SPDXID": f"SPDXRef-Package-{_spdx_token(name)}",
                "name": name,
                "versionInfo": version,
                "downloadLocation": NO_ASSERTION,
                "filesAnalyzed": False,
                "licenseConcluded": NO_ASSERTION,
                "licenseDeclared": NO_ASSERTION,
                "copyrightText": NO_ASSERTION,
            }
        )
    return sorted(packages, key=lambda item: item["name"].lower())


def build_document(name: str) -> dict[str, object]:
    """Build an SPDX document for the active Python environment."""
    if not isinstance(name, str):
        raise TypeError("name must be a string")
    if not name.strip():
        raise ValueError("name must be non-empty")
    created = datetime.now(timezone.utc).replace(microsecond=0).isoformat()
    return {
        "spdxVersion": SPDX_VERSION,
        "dataLicense": DATA_LICENSE,
        "SPDXID": "SPDXRef-DOCUMENT",
        "name": name,
        "documentNamespace": f"https://github.com/D-sorganization/UpstreamDrift/sbom/{name}",
        "creationInfo": {
            "created": created,
            "creators": ["Tool: scripts/security/write_spdx_sbom.py"],
        },
        "packages": package_entries(),
    }


def write_document(path: Path, name: str) -> None:
    """Write an SPDX JSON document to path."""
    if not isinstance(path, Path):
        raise TypeError("path must be a pathlib.Path")
    path.write_text(json.dumps(build_document(name), indent=2) + "\n", encoding="utf-8")


def _spdx_token(value: str) -> str:
    return "".join(character if character.isalnum() else "-" for character in value)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--name", required=True)
    return parser.parse_args()


def main() -> int:
    """CLI entry point."""
    args = parse_args()
    write_document(args.output, args.name)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Write or validate the proximal-distal open-resource release bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .release_bundle import (
    ARTICLE_REL,
    build_release_manifest,
    checksum_lines,
    validate_release_manifest,
)


ROOT = Path(__file__).resolve().parents[3]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("action", choices=("write", "validate", "list-presets"))
    return parser


def main() -> None:
    """Execute the release qualification action."""
    action = _parser().parse_args().action
    manifest_path = ROOT / ARTICLE_REL / "release_manifest.json"
    if action == "write":
        manifest = build_release_manifest(ROOT)
        manifest_path.write_text(
            json.dumps(manifest, indent=2) + "\n", encoding="utf-8"
        )
        checksum_path = ROOT / ARTICLE_REL / "CHECKSUMS.sha256"
        checksum_path.write_text(
            "\n".join(checksum_lines(manifest)) + "\n", encoding="utf-8"
        )
        print(manifest_path)
        print(checksum_path)
        return
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if action == "validate":
        print(json.dumps(validate_release_manifest(ROOT, manifest), indent=2))
        return
    for name, preset in manifest["presets"].items():
        print(f"{name}: {preset['command']}")


if __name__ == "__main__":
    main()

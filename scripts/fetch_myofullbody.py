"""Fetch and verify the pinned MyoFullBody assets (issue #11643).

python3 scripts/fetch_myofullbody.py            # fetch if needed, verify
python3 scripts/fetch_myofullbody.py --receipt out.json   # + model counts
python3 scripts/fetch_myofullbody.py --write-manifest     # maintainers only
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from src.shared.python.myofullbody import assets


def main() -> int:
    """CLI entry point."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache-root", type=Path, default=None)
    parser.add_argument("--receipt", type=Path, default=None)
    parser.add_argument(
        "--write-manifest",
        action="store_true",
        help="regenerate the committed manifest from the pinned archive",
    )
    args = parser.parse_args()
    if args.write_manifest:
        document = assets.generate_manifest(assets.download_archive())
        assets.MANIFEST_PATH.write_text(
            json.dumps(document, indent=1, sort_keys=True) + "\n", encoding="utf-8"
        )
        print(f"wrote {assets.MANIFEST_PATH} ({len(document['files'])} files)")
        return 0
    manifest = assets.load_manifest()
    tree = assets.fetch(manifest, root=args.cache_root)
    print(f"verified {len(manifest.files)} files at {tree}")
    if args.receipt:
        receipt = assets.asset_receipt(tree, manifest)
        args.receipt.parent.mkdir(parents=True, exist_ok=True)
        args.receipt.write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        print(json.dumps(receipt["inventory"], sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

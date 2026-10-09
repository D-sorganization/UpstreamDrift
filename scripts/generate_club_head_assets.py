"""Generate the committed clubhead STL fallbacks and their provenance manifest.

Run from the repository root with the Tools tree importable::

    PYTHONPATH=.:src:vendor/ud-tools/src python3 scripts/generate_club_head_assets.py

The STLs are the Tools parametric builder's own serialization (binary, mm,
head frame x=target y=up z=toe), so the fallback is byte-identical to the
primary source. The driver reuses the existing Simscape STL instead of a copy.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
import subprocess  # nosec B404 - fixed git argv
import sys
from pathlib import Path

from rate_of_closure.club.library import get_club
from rate_of_closure.club.stl_export import serialize_clubhead_stl

logger = logging.getLogger(__name__)
ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "club_heads"
DRIVER_STL = (
    "src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/"
    "exploratory_gs3dx/models/gs3dx_driver_head.stl"
)
NAMES = (
    "Driver 10.5°",
    "3-Wood",
    "5-Wood",
    "3-Hybrid",
    "3-Iron",
    "5-Iron",
    "7-Iron",
    "9-Iron",
    "Pitching Wedge",
    "Gap Wedge",
    "Sand Wedge",
    "Lob Wedge",
)


def _tools_pin() -> str:
    out = subprocess.run(  # noqa: S603, S607 - fixed argv
        ["git", "-C", str(ROOT), "ls-tree", "HEAD", "vendor/ud-tools"],
        capture_output=True,
        text=True,
        check=True,
    ).stdout.split()
    return out[2] if len(out) >= 3 else "unknown"


def _slug(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    pin = _tools_pin()
    heads: dict[str, dict[str, object]] = {}
    for name in NAMES:
        spec = get_club(name)
        payload = serialize_clubhead_stl(spec)
        if name == "Driver 10.5°":
            rel = DRIVER_STL
            if (ROOT / rel).read_bytes() != payload:
                logger.error("Existing driver STL differs from the Tools builder")
                return 1
        else:
            rel = f"assets/club_heads/{_slug(name)}.stl"
            (ROOT / rel).write_bytes(payload)
        heads[name] = {
            "library_name": name,
            "path": rel,
            "sha256": hashlib.sha256(payload).hexdigest(),
            "loft_deg": spec.loft_deg,
            "lie_deg": spec.lie_deg,
            "club_type": spec.club_type.value,
            "triangles": (len(payload) - 84) // 50,
        }
    manifest = {
        "schema": "club-head-assets-v1",
        "generator": "scripts/generate_club_head_assets.py",
        "source": "rate_of_closure.club.stl_export.serialize_clubhead_stl",
        "tools_pin": pin,
        "units": "mm",
        "head_frame": "x=target,y=up,z=toe",
        "note": "Visual only; head mass and inertia stay as specified.",
        "heads": heads,
    }
    (OUT / "provenance.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    logger.info("wrote %d heads to %s", len(heads), OUT)
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    sys.exit(main())

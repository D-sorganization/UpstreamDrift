"""Generate the common-reference guide from the native help's canonical text."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.tools.capture_rig.reference_calibration.guidance import GUIDE


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    path = ROOT / "docs/motion_capture/common_reference_calibration.md"
    expected = GUIDE.rstrip() + "\n"
    if args.check:
        return (
            0 if path.is_file() and path.read_text(encoding="utf-8") == expected else 1
        )
    path.write_text(expected, encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

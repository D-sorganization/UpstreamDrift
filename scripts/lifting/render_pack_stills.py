#!/usr/bin/env python3
"""Render a skeleton still per lift and engine at the pack start pose.

Usage: python3 scripts/lifting/render_pack_stills.py [--out-dir DIR]
Stills plot each engine's own FK; they are not native engine renders.
"""

from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT), str(ROOT / "src")]

from src.shared.python.lifting.pack_audit.adapters import create_adapter  # noqa: E402
from src.shared.python.lifting.pack_audit.model import Anthropometry  # noqa: E402
from src.shared.python.lifting.pack_audit.names import ENGINES, LIFTS  # noqa: E402
from src.shared.python.lifting.pack_audit.packs import locate_pack  # noqa: E402
from src.shared.python.lifting.pack_audit.stills import render_still  # noqa: E402

logger = logging.getLogger("lift_stills")
DEFAULT_OUT = Path.home() / "Videos" / "Parity Audit" / "lifts" / "baseline"


def main(argv: list[str] | None = None) -> int:
    """Write ``<lift>_<engine>.png`` for every available pack and lift."""
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--out-dir", type=Path, default=DEFAULT_OUT)
    args = ap.parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    anthro = Anthropometry()
    for engine in ENGINES:
        pack = locate_pack(engine)
        if pack is None:
            logger.warning("deferred: %s pack not found", engine)
            continue
        for lift in LIFTS:
            pos = create_adapter(pack, lift, anthro).evaluate(None).positions
            out = render_still(
                pos,
                f"{lift} - {engine} (pack start pose)",
                args.out_dir / f"{lift}_{engine}.png",
            )
            logger.info("wrote %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

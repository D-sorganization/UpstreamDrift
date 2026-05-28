"""Launch the Strokes Gained Optimizer.

Phase 1: defers to the headless CLI. Phase 3 will introduce the PyQt6 tile.
"""

from __future__ import annotations

import sys

from src.shared.python.sg_optimizer.cli import main


def _print_usage() -> None:
    sys.stderr.write(
        "sg_optimizer Phase 1 — headless CLI only. UI lands in Phase 3 (#6272).\n"
        "Run:\n"
        "  python -m src.tools.sg_optimizer --profile P --baseline B --hole-spec H [--conditions tournament]\n"
    )


if __name__ == "__main__":
    if len(sys.argv) == 1:
        _print_usage()
        sys.exit(2)
    sys.exit(main(sys.argv[1:]))

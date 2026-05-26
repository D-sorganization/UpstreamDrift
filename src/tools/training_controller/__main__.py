"""Standalone entrypoint for ``python -m src.tools.training_controller``."""

from __future__ import annotations

import sys


def main() -> int:
    try:
        from . import gui
    except (ImportError, OSError) as exc:
        sys.stderr.write(f"Training Controller GUI unavailable: {exc}\n")
        return 1
    return gui.main()


if __name__ == "__main__":
    raise SystemExit(main())

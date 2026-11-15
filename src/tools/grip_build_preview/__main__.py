"""``python -m src.tools.grip_build_preview`` entry."""

from __future__ import annotations

import sys

from src.tools.grip_build_preview.cli import main

if __name__ == "__main__":
    sys.exit(main())

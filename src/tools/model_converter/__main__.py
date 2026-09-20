"""Entry point for python -m src.tools.model_converter."""

from __future__ import annotations

import sys

from src.tools.model_converter.build_models import main

if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))

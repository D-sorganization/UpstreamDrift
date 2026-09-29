#!/usr/bin/env python3
"""Quality check script delegating to shared tooling."""

import sys
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[5]
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from src.tools.code_quality_check import main  # noqa: E402

if __name__ == "__main__":
    main()

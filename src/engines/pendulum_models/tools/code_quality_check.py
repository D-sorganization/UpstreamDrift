#!/usr/bin/env python3
"""Quality check script - delegates to the authoritative version.

This is a thin wrapper that delegates to src/tools/code_quality_check.py.
This consolidation follows DRY principles from The Pragmatic Programmer.
"""

import sys
from pathlib import Path

# Add the repo root to path to enable importing the authoritative module
repo_root = Path(__file__).resolve().parents[4]
sys.path.insert(0, str(repo_root))

# Import and run the authoritative quality check
from src.tools.code_quality_check import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())

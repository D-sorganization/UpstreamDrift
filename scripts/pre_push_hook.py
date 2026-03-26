#!/usr/bin/env python3
"""Pre-push hook script — runs tests using python -m pytest for cross-platform compatibility.

This hook is installed by scripts/setup_hooks.py. Using `python -m pytest` instead of
bare `pytest` ensures the correct Python environment's pytest is always used, fixing
the 'pytest not found' failures on Windows and in virtual environments.
"""

import logging
import subprocess
import sys

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)


def main() -> int:
    """Run the pre-push checks."""
    logger.info("Running pre-push checks...")

    # Use python -m pytest so we always pick up the correct environment's pytest
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "--tb=short", "-q"],
        check=False,
    )

    if result.returncode != 0:
        logger.error("Pre-push checks failed. Push aborted.")
        logger.error("Run 'python -m pytest' locally to see failures.")
        logger.error("Use 'git push --no-verify' to skip (not recommended).")
        return 1

    logger.info("Pre-push checks passed.")
    return 0


if __name__ == "__main__":
    sys.exit(main())

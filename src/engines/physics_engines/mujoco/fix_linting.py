#!/usr/bin/env python3
"""Batch fix linting errors."""

import subprocess
import sys


def main() -> int:
    """Main function to batch fix linting errors."""
    # First, apply all auto-fixable errors
    subprocess.run(
        [
            "ruff",
            "check",
            ".",
            "--fix",
            "--unsafe-fixes",
            "--select",
            "E501,RUF002,RUF003,SIM102",
        ],
        check=False,
    )

    # Get remaining errors
    subprocess.run(
        ["ruff", "check", ".", "--output-format=json"],
        capture_output=True,
        text=True,
        check=False,
    )

    # Show remaining errors by type
    subprocess.run(["ruff", "check", ".", "--statistics"], check=False)

    return 0


if __name__ == "__main__":
    sys.exit(main())

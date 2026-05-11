"""Secure proxy runner for opening documentation files.

This script acts as an approved executable under the secure_subprocess
whitelist to allow the opening of non-executable documentation files
like markdown and PDFs using the system's default viewer.
"""

import argparse
import logging
import os
import platform
import subprocess
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("document_proxy")


def main() -> None:
    """Open the specified document using the system default viewer."""
    parser = argparse.ArgumentParser(description="Proxy runner for opening documents.")
    parser.add_argument("file_path", type=Path, help="Path to the document to open.")
    args = parser.parse_args()

    file_path = args.file_path.resolve()

    if not file_path.exists():
        logger.error(f"File not found: {file_path}")
        sys.exit(1)

    logger.info(f"Opening document: {file_path}")

    try:
        if platform.system() == "Windows":
            os.startfile(str(file_path))  # type: ignore[attr-defined]
        elif platform.system() == "Darwin":
            subprocess.Popen(["open", str(file_path)])  # noqa: S603, S607
        else:
            subprocess.Popen(["xdg-open", str(file_path)])  # noqa: S603, S607
    except Exception as e:
        logger.error(f"Failed to open document: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

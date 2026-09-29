"""Verify .gitignore lists .env so secrets do not get committed."""

from __future__ import annotations

import sys
from pathlib import Path


def check_gitignore_dotenv(gitignore_path: Path = Path(".gitignore")) -> bool:
    """Return True when the provided .gitignore contains a .env pattern."""
    if not gitignore_path.exists():
        print(f"ERROR: {gitignore_path} not found", file=sys.stderr)
        return False

    patterns = [
        line.strip()
        for line in gitignore_path.read_text(encoding="utf-8").splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]
    return any(".env" in pattern for pattern in patterns)


def main() -> None:
    if check_gitignore_dotenv():
        print("OK: .gitignore includes .env pattern")
        return
    print("ERROR: .gitignore does not list .env", file=sys.stderr)
    sys.exit(1)


if __name__ == "__main__":
    main()

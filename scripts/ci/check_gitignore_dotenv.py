"""Verify .gitignore lists .env so secrets don't get committed."""

import sys
from pathlib import Path


def check_gitignore_dotenv(gitignore_path: Path = Path(".gitignore")) -> bool:
    """Check that .gitignore contains a pattern covering .env files.

    Args:
        gitignore_path: Path to the .gitignore file to check.

    Returns:
        True if an appropriate .env pattern is present, False otherwise.
    """
    if not gitignore_path.exists():
        print(f"ERROR: {gitignore_path} not found", file=sys.stderr)
        return False

    gitignore_text = gitignore_path.read_text(encoding="utf-8")
    patterns = [
        line.strip()
        for line in gitignore_text.splitlines()
        if line.strip() and not line.strip().startswith("#")
    ]

    return bool(any(".env" in p for p in patterns))


def main() -> None:
    """Entry point for the CI check."""
    if check_gitignore_dotenv():
        print("OK: .gitignore includes .env pattern")
    else:
        print("ERROR: .gitignore does not list .env", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()

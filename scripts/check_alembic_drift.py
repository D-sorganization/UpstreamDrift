#!/usr/bin/env python3
"""Fail when SQLAlchemy models drift from committed Alembic migrations."""

from __future__ import annotations

import os
import sys
from pathlib import Path


def _emit_stdout(message: str) -> None:
    """Write a single line to standard output."""
    sys.stdout.write(f"{message}\n")


def _emit_stderr(message: str) -> None:
    """Write a single line to standard error."""
    sys.stderr.write(f"{message}\n")


def _ensure_project_on_path(repo_root: Path) -> None:
    """Ensure Alembic can import project modules during autogenerate checks."""
    paths = [repo_root, repo_root / "src"]
    for path in reversed(paths):
        path_text = str(path)
        if path_text not in sys.path:
            sys.path.insert(0, path_text)


def build_alembic_config() -> Config:  # type: ignore[name-defined]  # noqa: F821
    """Build an Alembic config for the repository migration environment.

    Postcondition: the returned config points at ``alembic.ini`` and honors
    ``DATABASE_URL`` when the caller supplies one.
    """
    from alembic.config import Config

    repo_root = Path(__file__).resolve().parent.parent
    ini_path = repo_root / "alembic.ini"
    if not ini_path.is_file():
        raise FileNotFoundError(f"alembic.ini not found at {ini_path}")

    _ensure_project_on_path(repo_root)
    cfg = Config(str(ini_path))
    database_url = os.getenv("DATABASE_URL")
    if database_url:
        cfg.set_main_option("sqlalchemy.url", database_url)
    return cfg


def check_alembic_drift() -> int:
    """Run Alembic's autogenerate drift check.

    Returns:
        Process exit code. ``0`` means models and migrations match; ``1`` means
        Alembic detected pending model operations or the check could not run.
    """
    from alembic import command

    try:
        cfg = build_alembic_config()
        command.check(cfg)
    except Exception as exc:  # noqa: BLE001
        _emit_stderr(
            "Alembic drift detected or check failed: "
            f"{exc}\n"
            "Generate and review a migration with: "
            "python3 scripts/db_migrate.py revision --autogenerate "
            "-m 'describe schema change'"
        )
        return 1

    _emit_stdout("No Alembic drift detected: models and migrations are in sync.")
    return 0


def main() -> int:
    """CLI entrypoint."""
    return check_alembic_drift()


if __name__ == "__main__":
    sys.exit(main())

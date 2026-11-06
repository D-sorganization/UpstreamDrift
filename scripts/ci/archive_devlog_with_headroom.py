#!/usr/bin/env python3
"""Size-aware development-log archiver (#14535).

The vendored archiver (``shared_scripts/development_log.py``) only moves
shipped/abandoned entries older than ``ARCHIVE_AFTER_DAYS``. At this repo's
velocity the active log can reach the 200,000-byte ceiling before anything is
old enough, which fails every merge-queue entry (#14499).

This wrapper keeps the default pass, then, only when the canonical log is at
or above ``SIZE_TRIGGER_BYTES``, repeats it with progressively shorter age
windows (oldest terminal entries leave first) until the log is below
``HEADROOM_TARGET_BYTES`` or the windows run out. ``MAX_BYTES`` and the
default ``ARCHIVE_AFTER_DAYS`` are unchanged, and the vendored module is used
as-is, so its digest pin stays valid.
"""

from __future__ import annotations

import argparse
import sys
from collections.abc import Sequence
from datetime import date
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from shared_scripts.development_log import (  # noqa: E402
    archive_repository_devlog,
    resolve_canonical_devlog_path,
)
from shared_scripts.development_log_schema import (  # noqa: E402
    ARCHIVE_AFTER_DAYS,
    MAX_BYTES,
)

SIZE_TRIGGER_BYTES = 180_000
HEADROOM_TARGET_BYTES = 165_000
# Strictly tightening age windows; the first is the repo default.
AGE_WINDOWS: tuple[int, ...] = (ARCHIVE_AFTER_DAYS, 5, 3, 2, 1, 0)

assert HEADROOM_TARGET_BYTES < SIZE_TRIGGER_BYTES < MAX_BYTES


def _size(repo_root: Path) -> int:
    return resolve_canonical_devlog_path(repo_root).stat().st_size


def archive_with_headroom(repo_root: Path, *, today: date) -> list[Path]:
    """Archive by age, then by size pressure. Returns the archive files written.

    Postcondition: if the log started at or above ``SIZE_TRIGGER_BYTES`` and
    enough terminal entries exist, it ends below ``HEADROOM_TARGET_BYTES``.
    """
    written: list[Path] = []
    for index, days in enumerate(AGE_WINDOWS):
        for path in archive_repository_devlog(repo_root, today=today, days=days):
            if path not in written:
                written.append(path)
        size = _size(repo_root)
        # The default window always runs; tighter ones only under size pressure.
        if size < (SIZE_TRIGGER_BYTES if index == 0 else HEADROOM_TARGET_BYTES):
            break
    return written


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point. Returns a process exit code."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo-root", type=Path, default=Path.cwd())
    parser.add_argument("--today", type=date.fromisoformat, default=None)
    args = parser.parse_args(argv)
    written = archive_with_headroom(args.repo_root, today=args.today or date.today())
    for path in written:
        print(f"archived entries into {path}")
    if not written:
        print("Nothing to archive.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

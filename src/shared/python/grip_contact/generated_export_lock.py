"""Process-wide lock for Simscape grip committed generated-file exports (#13544).

Under pytest-xdist, parallel workers can interleave ``--check`` reads with
writes to the same repository paths and produce spurious drift failures.
Serializing export/check on the committed paths keeps the byte-exact contract
stable without weakening the check.
"""

from __future__ import annotations

import threading
from collections.abc import Callable, Sequence
from pathlib import Path

GENERATED_EXPORT_LOCK = threading.RLock()


def targets_touch_committed(
    targets: Sequence[Path],
    committed: Sequence[Path],
) -> bool:
    """Return True when any target resolves to a committed generated path."""
    committed_resolved = {Path(p).resolve() for p in committed}
    return any(Path(t).resolve() in committed_resolved for t in targets)


def serialize_committed_export(
    targets: Sequence[Path],
    committed: Sequence[Path],
    export_fn: Callable[[], int],
) -> int:
    """Run ``export_fn`` under the process lock when ``targets`` touch ``committed``."""
    if targets_touch_committed(targets, committed):
        with GENERATED_EXPORT_LOCK:
            return export_fn()
    return export_fn()


__all__ = [
    "GENERATED_EXPORT_LOCK",
    "serialize_committed_export",
    "targets_touch_committed",
]

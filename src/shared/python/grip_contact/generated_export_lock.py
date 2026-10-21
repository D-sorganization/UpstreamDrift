"""Process-wide lock for Simscape grip committed generated-file exports (#13544).

Under pytest-xdist, parallel workers can interleave ``--check`` reads with
writes to the same repository paths and produce spurious drift failures.
Serializing export/check on the committed paths keeps the byte-exact contract
stable without weakening the check.
"""

from __future__ import annotations

import threading

GENERATED_EXPORT_LOCK = threading.RLock()

__all__ = ["GENERATED_EXPORT_LOCK"]

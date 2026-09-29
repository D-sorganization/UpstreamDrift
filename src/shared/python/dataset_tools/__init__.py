"""Dataset tooling package — compact swing dataset loader and helpers.

The compactor script (``scripts/compact_swing_dataset.py``) emits a pair
of parquet files conforming to ``COMPACT_DATASET_SCHEMA.md``. This
package hosts the typed loader and the canonical column constants that
both the compactor and the loader share — keeping the contract in one
place (DRY).
"""

from __future__ import annotations

from src.shared.python.dataset_tools.canonical import (
    CANONICAL_JOINTS,
    COEFFICIENT_LETTERS,
    SCHEMA_VERSION,
)
from src.shared.python.dataset_tools.load_compact import (
    CompactSwingDataset,
    load_compact_swing_dataset,
)

__all__ = [
    "CANONICAL_JOINTS",
    "COEFFICIENT_LETTERS",
    "CompactSwingDataset",
    "SCHEMA_VERSION",
    "load_compact_swing_dataset",
]

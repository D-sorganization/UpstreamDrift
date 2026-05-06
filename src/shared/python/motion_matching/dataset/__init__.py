"""Random-sweep dataset loader and synthetic-data helpers.

Public API:
    SweepDataset       -- frozen dataclass holding the loaded dataset.
    load_sweep_dataset -- read trials.parquet + timesteps.parquet from a folder.
    make_synthetic_sweep -- generate a small valid dataset for testing.
    SCHEMA_VERSION     -- current schema version string.

See ``DATASET_SCHEMA.md`` for the on-disk schema.
"""

from .sweep import SCHEMA_VERSION, SweepDataset, load_sweep_dataset
from .synthetic import make_synthetic_sweep

__all__ = [
    "SCHEMA_VERSION",
    "SweepDataset",
    "load_sweep_dataset",
    "make_synthetic_sweep",
]

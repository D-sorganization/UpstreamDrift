"""Documented default dataset and checkpoint paths for surrogate training.

NM-00 (#10615) audits these locations honestly: absence is quarantined, never
invented. Keep a single definition so extractors and the neural catalog cannot
drift apart (DRY).
"""

from __future__ import annotations

import os
from pathlib import Path

_ENV_COMPACT_SWING_PARQUET = "UD_COMPACT_SWING_PARQUET"
_DEFAULT_REL_UNDER_HOME = Path("Repositories") / "data" / "TenThousandFiles.parquet"

# Historical host-local Simscape dump referenced by MachineLearning docs and
# per-step extractors. Frequently absent on developer hosts.
DOCUMENTED_TEN_THOUSAND_FILES = Path(
    os.environ.get(
        _ENV_COMPACT_SWING_PARQUET,
        str(Path.home() / _DEFAULT_REL_UNDER_HOME),
    )
)

SWEEP_SYNTHETIC_REL = Path("data") / "sweep_synthetic"
DEFAULT_SURROGATE_CHECKPOINT_REL = Path("output") / "surrogate" / "checkpoint_best.pt"
DEFAULT_PRODUCTION_CHECKPOINT_REL = (
    Path("output") / "surrogate" / "checkpoint_production.pt"
)

__all__ = [
    "DEFAULT_PRODUCTION_CHECKPOINT_REL",
    "DEFAULT_SURROGATE_CHECKPOINT_REL",
    "DOCUMENTED_TEN_THOUSAND_FILES",
    "SWEEP_SYNTHETIC_REL",
]

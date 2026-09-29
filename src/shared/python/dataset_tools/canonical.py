"""Canonical column / joint / coefficient ordering for the compact dataset.

This module is the single source of truth for the ordering used by both
``scripts/compact_swing_dataset.py`` and
``src.shared.python.dataset_tools.load_compact``. Keeping the constants
in one place eliminates the risk of the compactor and loader drifting
apart (a violation of DRY would silently corrupt training data).

See ``motion_matching/shared/COMPACT_DATASET_SCHEMA.md`` for the
authoritative narrative description.
"""

from __future__ import annotations

from typing import Final

# 27 joint names — order is load-bearing: position i in this tuple
# corresponds to position i in every list<float64>[27] column
# (q, qd, qdd, tau) of timesteps.parquet, and position i*7..i*7+6 of the
# 189-vec coefficients column of trials.parquet.
CANONICAL_JOINTS: Final[tuple[str, ...]] = (
    "HipX",
    "HipY",
    "HipZ",
    "LE",
    "LF",
    "LSX",
    "LSY",
    "LSZ",
    "LScapX",
    "LScapY",
    "LWX",
    "LWY",
    "RE",
    "RF",
    "RSX",
    "RSY",
    "RSZ",
    "RScapX",
    "RScapY",
    "RWX",
    "RWY",
    "SpineX",
    "SpineY",
    "Torso",
    "TranslationX",
    "TranslationY",
    "TranslationZ",
)

# 7 polynomial coefficient letters per joint.
COEFFICIENT_LETTERS: Final[tuple[str, ...]] = ("A", "B", "C", "D", "E", "F", "G")

SCHEMA_VERSION: Final[str] = "compact-1.0"

# Length sentinels used by validation in multiple modules.
N_JOINTS: Final[int] = len(CANONICAL_JOINTS)  # 27
N_COEFFS: Final[int] = N_JOINTS * len(COEFFICIENT_LETTERS)  # 189

# Length contract for each list<float64> column in timesteps.parquet.
TIMESTEP_LIST_LENGTHS: Final[dict[str, int]] = {
    "q": N_JOINTS,
    "qd": N_JOINTS,
    "qdd": N_JOINTS,
    "tau": N_JOINTS,
    "r_clubhead": 3,
    "v_clubhead": 3,
    "r_buttend": 3,
    "r_lhand": 3,
    "r_rhand": 3,
    "r_grip": 3,
}

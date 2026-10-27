"""Shared solver status mapping and provenance helpers for motion matching.

Unifies solver status contract and provenance fields across all physics engines
(MuJoCo, Drake, OpenSim, Pinocchio) per issues UD-04 and UD-05 (#13874).

Status Contract:
- "success": returned iff optimizer successfully converged AND final cost/residuals are finite.
- "failed": returned for all non-success outcomes (iteration limits, line-search
  failures, incompatible constraints, singular matrix, non-finite cost, None, etc.).
- "warning": NEVER returned by solver status mapping (UD-04).
"""

from __future__ import annotations

import hashlib
from datetime import datetime, timezone
from typing import Any

import numpy as np

from src.shared.python.core.contracts.decorators import postcondition
from src.shared.python.motion_matching.provenance import (
    engine_package_version,
    git_commit_short,
)

__all__ = [
    "hash_target",
    "make_provenance_fields",
    "map_scipy_status",
    "provenance_fields",
]


@postcondition(
    lambda r: r in ("success", "failed"),
    "solver_status must be 'success' or 'failed'",
)
def map_scipy_status(result: Any) -> str:
    """Map a scipy.optimize result (or compatible) to canonical solver_status.

    Parameters
    ----------
    result : Any
        An OptimizeResult instance, namespace, or None.

    Returns
    -------
    str
        'success' if result.success is True and result.fun / result.cost is finite;
        'failed' otherwise. Never returns 'warning'.
    """
    if result is None:
        return "failed"

    if not bool(getattr(result, "success", False)):
        return "failed"

    fun = getattr(result, "fun", None)
    if fun is None:
        fun = getattr(result, "cost", None)
    if fun is None:
        return "failed"

    try:
        arr = np.asarray(fun)
        if not bool(np.all(np.isfinite(arr))):
            return "failed"
    except (TypeError, ValueError):
        return "failed"

    return "success"


def hash_target(target: Any) -> str:
    """Compute SHA-256 hex digest of the canonical bytes of a ``ClubTarget``.

    Parameters
    ----------
    target : Any
        ClubTarget instance exposing time, butt, clubhead, club_quat, and impact_idx.

    Returns
    -------
    str
        64-character lowercase hex digest, or 'unknown' if target cannot be hashed.
    """
    if target is None:
        return "unknown"

    try:
        h = hashlib.sha256()
        for arr in (target.time, target.butt, target.clubhead, target.club_quat):
            a = np.ascontiguousarray(arr, dtype=np.float64)
            h.update(a.tobytes())
        h.update(int(target.impact_idx).to_bytes(8, "little", signed=True))
        return h.hexdigest()
    except Exception:
        return "unknown"


def make_provenance_fields(
    engine_module: Any = None,
    *distributions: str,
    target: Any = None,
) -> dict[str, str]:
    """Build canonical provenance dictionary for FitResult.

    Parameters
    ----------
    engine_module : Any, optional
        Imported engine module (e.g. mujoco, pydrake, opensim, pinocchio) or None.
    *distributions : str
        Distribution package names to probe via importlib.metadata.
    target : Any, optional
        ClubTarget instance to hash.

    Returns
    -------
    dict[str, str]
        Dictionary with keys: git_commit, engine_version, target_hash, timestamp_utc.
    """
    return {
        "git_commit": git_commit_short(),
        "engine_version": engine_package_version(engine_module, *distributions),
        "target_hash": hash_target(target),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
    }


# Convenience alias
provenance_fields = make_provenance_fields

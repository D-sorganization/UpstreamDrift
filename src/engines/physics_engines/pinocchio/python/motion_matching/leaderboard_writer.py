"""Canonical leaderboard JSON writer for the Pinocchio engine.

Emits ``<output_dir>/<trial_id>/pinocchio.json`` with the schema specified
in issue #4133, designed to be picked up unchanged by PARITY-LEADERBOARD
(#4097).

Schema (frozen for cross-engine consumption — do not extend without
updating the leaderboard reader):

.. code-block:: json

    {
        "engine": "pinocchio",
        "trial": "<trial_id>",
        "solver": "lm-analytical-jacobian",
        "grip_rmse_mm": <float>,
        "clubhead_rmse_mm": <float>,
        "total_work_J": <float>,
        "wall_clock_s": <float>,
        "commit": "<git hash>",
        "run_at": "<ISO-8601 datetime>"
    }
"""

from __future__ import annotations

import json
import logging
import math
from datetime import datetime
from pathlib import Path
from typing import Any

from ._types import ClubTargetLike, FitResult

logger = logging.getLogger(__name__)

ENGINE_NAME = "pinocchio"

try:
    from datetime import UTC  # type: ignore[attr-defined]
except ImportError:  # pragma: no cover - Python < 3.11 compatibility shim
    from datetime import timezone

    UTC = timezone.utc  # noqa: UP017 - Python 3.10 compatibility shim

# Required schema keys, in canonical order, used both for emission and
# for the round-trip schema test.
SCHEMA_KEYS: tuple[str, ...] = (
    "engine",
    "trial",
    "solver",
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "total_work_J",
    "wall_clock_s",
    "commit",
    "run_at",
)


def _resolve_commit(commit: str | None) -> str:
    """Return ``commit`` if non-empty, else best-effort short ``HEAD`` SHA.

    Delegates the git lookup to the shared :func:`git_commit_short` probe
    (issue #6939); falls back to ``"unknown"`` if git is unavailable —
    never raises.
    """
    if commit:
        return commit
    from src.shared.python.motion_matching.provenance import git_commit_short

    return git_commit_short()


def _coerce_float(value: float, *, name: str) -> float:
    """Validate that ``value`` is a finite float; raise ``ValueError`` otherwise."""
    fv = float(value)
    if not math.isfinite(fv):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return fv


def write_leaderboard_entry(
    result: FitResult,
    target: ClubTargetLike,
    output_dir: Path,
) -> Path:
    """Write the canonical Pinocchio leaderboard JSON for one trial.

    The file path is ``<output_dir>/<trial_id>/pinocchio.json``. The
    parent directory tree is created if missing.

    Args:
        result: Pinocchio fit result; contributes RMSEs, work, wall clock,
            solver tag, commit, and trial id.
        target: Measured target trajectory; required by the public
            signature for parity with the per-engine viz API and to allow
            future schema fields (e.g. ``trial_duration_s``) to be derived
            without a signature break.
        output_dir: Root directory for leaderboard outputs. Per-trial
            subdirectories are created under it.

    Returns:
        Absolute path to the written ``pinocchio.json`` file.

    Raises:
        ValueError: Any required numeric field is non-finite, or
            ``result.trial_id`` is empty.
        TypeError: ``output_dir`` is not a :class:`Path`.
    """
    # --- Preconditions (DbC) ---
    if not isinstance(output_dir, Path):
        raise TypeError(f"output_dir must be a Path, got {type(output_dir).__name__}")
    trial_id = str(result.trial_id).strip()
    if not trial_id:
        raise ValueError("result.trial_id must be a non-empty string")
    # `target` is part of the public contract; touch a known attribute so
    # type-mismatched inputs fail fast at the call site rather than later.
    _ = getattr(target, "time", None)

    payload: dict[str, Any] = {
        "engine": ENGINE_NAME,
        "trial": trial_id,
        "solver": str(result.solver),
        "grip_rmse_mm": _coerce_float(result.grip_rmse_mm, name="grip_rmse_mm"),
        "clubhead_rmse_mm": _coerce_float(
            result.clubhead_rmse_mm, name="clubhead_rmse_mm"
        ),
        "total_work_J": _coerce_float(result.total_work_J, name="total_work_J"),
        "wall_clock_s": _coerce_float(result.wall_clock_s, name="wall_clock_s"),
        "commit": _resolve_commit(result.commit),
        "run_at": datetime.now(UTC).isoformat(timespec="seconds"),
    }

    # Postcondition: every schema key is present, no extras.
    if set(payload.keys()) != set(SCHEMA_KEYS):
        raise AssertionError(
            f"leaderboard payload key drift: {sorted(payload.keys())} "
            f"vs canonical {sorted(SCHEMA_KEYS)}"
        )

    trial_dir = output_dir / trial_id
    trial_dir.mkdir(parents=True, exist_ok=True)
    out_path = trial_dir / f"{ENGINE_NAME}.json"

    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=False), encoding="utf-8"
    )
    logger.info("Wrote Pinocchio leaderboard entry: %s", out_path)
    return out_path

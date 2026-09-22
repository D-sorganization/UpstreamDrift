"""Fail-closed MS-14 Pinocchio / Pink lane receipt builders.

These helpers never invent native success. Blocked receipts are valid schema
payloads that record missing SDK capability without claiming G1 or weld closure.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from src.shared.python.contracts import precondition

CLOSURE_RESIDUAL_BUDGET_M = 1.0e-4
LANE_RECEIPT_SCHEMA = "matched-swing-lane/pinocchio-ms14/v1"
RUNTIME_RECEIPT_SCHEMA = "motion-runtime/pinocchio-ms14/v1"


@precondition(
    lambda reason, host, repository_revision: bool(reason.strip()),
    "blocked lane receipt requires a non-empty reason",
)
def build_blocked_pinocchio_lane_receipt(
    *,
    reason: str,
    host: str,
    repository_revision: str,
) -> dict[str, Any]:
    """Build an honest blocked full-lane receipt (no native dynamics claim)."""
    return {
        "schema_version": LANE_RECEIPT_SCHEMA,
        "engine": "pinocchio",
        "status": "blocked",
        "accepted": False,
        "native_claims": False,
        "gates": [],
        "closure_residual_m": None,
        "reason": reason,
        "host": host,
        "repository_revision": repository_revision,
        "scope": (
            "MS-14 Pinocchio MatchingPlant full lane; blocked until Pinocchio "
            "constraintDynamics + shared contact law run on a qualified host"
        ),
    }


@precondition(
    lambda reason, model_name, capture_name: bool(reason.strip()),
    "blocked Pink receipt requires a non-empty reason",
)
def build_blocked_constrained_ik_receipt(
    *,
    reason: str,
    model_name: str,
    capture_name: str,
) -> dict[str, Any]:
    """Build a ConstrainedIkReceipt payload that cannot claim qualification."""
    return {
        "backend_name": "pink",
        "solver": "quadprog",
        "model_name": model_name,
        "capture_name": capture_name,
        "step_mode": "physical",
        "limit_policy": "enforce",
        "task_policy": "dual_grip_hard_equality",
        "time_semantics": "strict_physical_elapsed_dt",
        "frame_count": 0,
        "frame_success_count": 0,
        "all_frames_converged": False,
        "first_failed_frame": None,
        "per_frame_status": [],
        "max_velocity_ratio": None,
        "closure_residual_m": None,
        "closure_residual_budget_m": CLOSURE_RESIDUAL_BUDGET_M,
        "is_qualified": False,
        "qualification_state": "blocked_unavailable",
        "block_reason": reason,
    }


@precondition(
    lambda missing, repository_revision: isinstance(missing, (list, tuple)),
    "missing components must be a sequence",
)
def build_blocked_runtime_receipt(
    *,
    missing: Sequence[str],
    repository_revision: str,
    probes: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Record check_motion_runtime-style blocked status without fake greens."""
    missing_list = sorted({str(item) for item in missing if str(item).strip()})
    return {
        "schema_version": RUNTIME_RECEIPT_SCHEMA,
        "status": "blocked",
        "accepted": False,
        "missing_components": missing_list,
        "repository_revision": repository_revision,
        "probes": dict(probes or {}),
        "scope": "MS-14 motion runtime probe; no Pinocchio/Pink native success",
    }

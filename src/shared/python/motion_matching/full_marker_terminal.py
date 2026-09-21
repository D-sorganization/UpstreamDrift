"""Full-marker terminal disclosure for MS-61 (#10348).

Every Simscape (and shared) matched-swing receipt must report both the
full-marker terminal RMS and the head-cluster terminal RMS. A body-only
diagnostic may be recorded under an explicit reduced-model profile but cannot
replace full-marker acceptance or hide the head cluster.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.simscape_topology import (
    HEAD_MARKER_NAMES as HEAD_MARKER_NAMES,
)

Array: TypeAlias = NDArray[np.float64]
BoolArray: TypeAlias = NDArray[np.bool_]

_BODY_ONLY_SOURCES = frozenset(
    {
        "body_excluding_head",
        "body_only",
        "exclude_head",
        "head_excluded",
    }
)


@dataclass(frozen=True)
class TerminalMarkerBreakdown:
    """Dual (full + head) terminal metrics; body-excluding-head is diagnostic only."""

    full_marker_terminal_rms_m: float
    head_cluster_terminal_rms_m: float | None
    body_excluding_head_terminal_rms_m: float | None
    hub_cluster_terminal_rms_m: float | None = None

    def as_dict(self) -> dict[str, float | None]:
        return {
            "full_marker_terminal_rms_m": float(self.full_marker_terminal_rms_m),
            "head_cluster_terminal_rms_m": (
                None
                if self.head_cluster_terminal_rms_m is None
                else float(self.head_cluster_terminal_rms_m)
            ),
            "body_excluding_head_terminal_rms_m": (
                None
                if self.body_excluding_head_terminal_rms_m is None
                else float(self.body_excluding_head_terminal_rms_m)
            ),
            "hub_cluster_terminal_rms_m": (
                None
                if self.hub_cluster_terminal_rms_m is None
                else float(self.hub_cluster_terminal_rms_m)
            ),
        }


def _terminal_rms(
    sq_err: Array,
    term_valid: BoolArray,
    indices: Sequence[int],
) -> float | None:
    if not indices:
        return None
    idx = np.asarray(indices, dtype=int)
    mask = term_valid[idx]
    if not np.any(mask):
        return None
    chosen = idx[mask]
    return float(np.sqrt(np.mean(sq_err[-1, chosen])))


@precondition(
    lambda pred_markers_m, target_markers_m, valid, marker_labels, marker_bodies=None: (
        len(marker_labels) > 0
    ),
    "marker_labels must be non-empty",
)
@postcondition(
    lambda result: result.full_marker_terminal_rms_m >= 0.0,
    "full-marker terminal RMS must be non-negative",
)
def compute_terminal_marker_breakdown(
    *,
    pred_markers_m: Array,
    target_markers_m: Array,
    valid: BoolArray | Array,
    marker_labels: Sequence[str],
    marker_bodies: Sequence[str] | None = None,
) -> TerminalMarkerBreakdown:
    """Compute full-marker, head-cluster, and body-excluding-head terminal RMS."""
    pred = np.asarray(pred_markers_m, dtype=np.float64)
    target = np.asarray(target_markers_m, dtype=np.float64)
    val = np.asarray(valid, dtype=bool)
    if pred.ndim != 3 or pred.shape[2] != 3:
        raise ValueError("pred_markers_m must have shape (N, M, 3)")
    if target.shape != pred.shape:
        raise ValueError("target_markers_m must match pred_markers_m shape")
    if val.shape != pred.shape[:2]:
        raise ValueError("valid must have shape (N, M)")
    if len(marker_labels) != pred.shape[1]:
        raise ValueError("marker_labels length must match marker count")
    if marker_bodies is not None and len(marker_bodies) != len(marker_labels):
        raise ValueError("marker_bodies length must match marker_labels")

    sq_err = np.sum((pred - target) ** 2, axis=-1)
    term_valid = val[-1]
    if not np.any(term_valid):
        raise ValueError("no valid markers on the terminal frame")

    all_idx = list(range(len(marker_labels)))
    head_idx = [i for i, name in enumerate(marker_labels) if name in HEAD_MARKER_NAMES]
    non_head_idx = [
        i for i, name in enumerate(marker_labels) if name not in HEAD_MARKER_NAMES
    ]
    hub_idx: list[int] = []
    if marker_bodies is not None:
        hub_idx = [i for i, body in enumerate(marker_bodies) if body == "Hub"]

    full_rms = _terminal_rms(sq_err, term_valid, all_idx)
    if full_rms is None:
        raise ValueError("unable to compute full-marker terminal RMS")

    return TerminalMarkerBreakdown(
        full_marker_terminal_rms_m=full_rms,
        head_cluster_terminal_rms_m=_terminal_rms(sq_err, term_valid, head_idx),
        body_excluding_head_terminal_rms_m=_terminal_rms(
            sq_err, term_valid, non_head_idx
        ),
        hub_cluster_terminal_rms_m=_terminal_rms(sq_err, term_valid, hub_idx),
    )


def _breakdown_mapping(receipt: Mapping[str, Any]) -> Mapping[str, Any] | None:
    raw = receipt.get("terminal_breakdown")
    if isinstance(raw, Mapping):
        return raw
    return None


def _head_cluster_value(receipt: Mapping[str, Any]) -> float | None:
    breakdown = _breakdown_mapping(receipt)
    if breakdown is not None:
        value = breakdown.get("head_cluster_terminal_rms_m")
        if isinstance(value, (int, float)) and np.isfinite(value):
            return float(value)
    value = receipt.get("head_cluster_terminal_rms_m")
    if isinstance(value, (int, float)) and np.isfinite(value):
        return float(value)
    return None


def _requires_full_marker_disclosure(receipt: Mapping[str, Any]) -> bool:
    """Disclosure is mandatory for Simscape / profiled / dual-metric receipts."""
    if isinstance(receipt.get("terminal_breakdown"), Mapping):
        return True
    if receipt.get("acceptance_terminal_source") is not None:
        return True
    if str(receipt.get("model_profile", "")).strip():
        return True
    if str(receipt.get("engine", "")).strip().lower() == "simscape":
        return True
    return "terminal_rms_m" in receipt or "terminal_marker_rmse_m" in receipt


def evaluate_full_marker_terminal_disclosure(
    receipt: Mapping[str, Any],
) -> list[Any]:
    """Fail closed when head-cluster terminal is hidden or body-only is accepted."""
    # Local import avoids acceptance <-> full_marker_terminal cycles.
    from src.shared.python.motion_matching.acceptance import GateResult, GateStatus

    if not _requires_full_marker_disclosure(receipt):
        return []

    results: list[Any] = []
    head_rms = _head_cluster_value(receipt)
    if head_rms is None:
        results.append(
            GateResult(
                name="head_cluster_terminal_rms_m",
                status=GateStatus.MISSING,
                threshold=0.0,
                reason="head-cluster terminal RMS missing; never hide head markers",
            )
        )
    else:
        results.append(
            GateResult(
                name="head_cluster_terminal_rms_m",
                status=GateStatus.PASSED,
                threshold=0.0,
                measured=head_rms,
                unit="m",
                reason="head-cluster terminal RMS disclosed",
            )
        )

    source = (
        str(receipt.get("acceptance_terminal_source", "full_marker")).strip().lower()
    )
    if source in _BODY_ONLY_SOURCES:
        results.append(
            GateResult(
                name="full_marker_terminal_source",
                status=GateStatus.FAILED,
                threshold=0.0,
                reason=(
                    "body-only / head-excluded terminal cannot satisfy full-body "
                    f"acceptance (acceptance_terminal_source={source!r})"
                ),
            )
        )
    else:
        results.append(
            GateResult(
                name="full_marker_terminal_source",
                status=GateStatus.PASSED,
                threshold=0.0,
                reason=f"acceptance terminal source={source or 'full_marker'}",
            )
        )

    excluded = receipt.get("excluded_markers") or receipt.get("marker_exclusion")
    if isinstance(excluded, (list, tuple, set)):
        excluded_names = {str(name) for name in excluded}
        if excluded_names.intersection(HEAD_MARKER_NAMES):
            results.append(
                GateResult(
                    name="head_marker_exclusion",
                    status=GateStatus.FAILED,
                    threshold=0.0,
                    reason=(
                        "head markers excluded from acceptance set: "
                        f"{sorted(excluded_names.intersection(HEAD_MARKER_NAMES))}"
                    ),
                )
            )

    return results


def require_full_marker_acceptance_terminal(receipt: Mapping[str, Any]) -> float:
    """Return the full-marker terminal used for acceptance; refuse body-only swaps."""
    source = (
        str(receipt.get("acceptance_terminal_source", "full_marker")).strip().lower()
    )
    if source in _BODY_ONLY_SOURCES:
        raise ValueError(
            "full-marker acceptance refuses body-only / head-excluded terminal source"
        )

    breakdown = _breakdown_mapping(receipt)
    if breakdown is not None:
        full = breakdown.get("full_marker_terminal_rms_m")
        if isinstance(full, (int, float)) and np.isfinite(full):
            terminal = float(full)
            reported = receipt.get(
                "terminal_rms_m", receipt.get("terminal_marker_rmse_m")
            )
            body_only = breakdown.get("body_excluding_head_terminal_rms_m")
            if (
                isinstance(reported, (int, float))
                and isinstance(body_only, (int, float))
                and abs(float(reported) - float(body_only)) < 1e-12
                and abs(float(reported) - terminal) > 1e-9
            ):
                raise ValueError(
                    "terminal_rms_m matches body-excluding-head rather than full-marker"
                )
            return terminal

    for key in ("terminal_marker_rmse_m", "terminal_rms_m"):
        value = receipt.get(key)
        if isinstance(value, (int, float)) and np.isfinite(value):
            if _head_cluster_value(receipt) is None:
                raise ValueError(
                    "head_cluster_terminal_rms_m required alongside full-marker terminal"
                )
            return float(value)
    raise ValueError("full-marker terminal RMS missing from receipt")

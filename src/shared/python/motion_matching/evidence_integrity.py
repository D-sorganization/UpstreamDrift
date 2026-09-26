"""Evidence-integrity gates for motion-matching receipts (MS-100 / #10363).

Split from ``acceptance.py`` (file-size budget): a recorded capture hash must be
a real digest, and identically-zero marker residuals need replay evidence.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from src.shared.python.motion_matching.acceptance import (
    GateResult,
    GateStatus,
    _extract_metric,
)

__all__ = ["evaluate_evidence_integrity", "is_real_sha256"]


def is_real_sha256(value: object) -> bool:
    """Return True when ``value`` is a 64-hex-digit digest, not the all-zero placeholder."""
    if not isinstance(value, str):
        return False
    digest = value.strip().lower()
    return (
        len(digest) == 64
        and set(digest) <= set("0123456789abcdef")
        and set(digest) != {"0"}
    )


def evaluate_evidence_integrity(
    receipt: Mapping[str, Any],
) -> list[GateResult]:
    """Verify evidence integrity: valid capture provenance and non-fabricated residuals (MS-100 / #10363)."""
    results: list[GateResult] = []

    # 1. Capture provenance: a recorded capture_sha256 must be a real digest.
    # An absent hash is not judged here (mandatory provenance is a separate gate);
    # a present placeholder is refused.
    if "capture_sha256" in receipt:
        valid = is_real_sha256(receipt["capture_sha256"])
        results.append(
            GateResult(
                name="capture_provenance",
                status=GateStatus.PASSED if valid else GateStatus.FAILED,
                threshold=1.0,
                measured=1.0 if valid else 0.0,
                unit="hash",
                reason="" if valid else "placeholder capture hash",
            )
        )

    # 2. Non-zero residual evidence: identically-zero marker residuals require replay evidence
    marker_metrics = [
        _extract_metric(receipt, "whole_marker_rmse_m", "whole_rms_m", "marker_rms_m"),
        _extract_metric(receipt, "early_marker_rmse_m", "early_rms_m"),
        _extract_metric(receipt, "terminal_marker_rmse_m", "terminal_rms_m"),
        _extract_metric(receipt, "club_marker_rmse_m", "club_cluster_rms_m"),
        _extract_metric(receipt, "terminal_full_marker_rmse_m"),
        _extract_metric(receipt, "terminal_body_excluding_head_rmse_m"),
        _extract_metric(receipt, "terminal_head_cluster_rmse_m"),
    ]
    reported_rmses = [m for m in marker_metrics if m is not None]
    if len(reported_rmses) > 0 and all(m == 0.0 for m in reported_rmses):
        has_replay = any(
            isinstance(receipt.get(k), Mapping)
            for k in (
                "open_loop_replay",
                "forward_rollout",
                "collocation_defect",
                "defects",
                "stabilized_replay",
                "stabilized_tracking",
            )
        )
        if not has_replay:
            results.append(
                GateResult(
                    name="nonzero_residual_evidence",
                    status=GateStatus.FAILED,
                    threshold=0.0,
                    measured=0.0,
                    unit="m",
                    reason="identically-zero residuals without replay evidence",
                )
            )

    return results

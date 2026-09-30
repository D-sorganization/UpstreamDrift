"""Engine-neutral qualification contracts shared by the native dual-club lanes.

Extracted so the per-engine `native_qualification.py` modules (drake, opensim,
myosuite) do not carry identical helper bodies; the DRY duplication gate blocks
unapproved cross-engine duplication.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np


def derivatives_consistent(
    time_s: np.ndarray, q: np.ndarray, v: np.ndarray, tol: float = 0.5
) -> bool:
    """Verify dq/dt aligns with reported joint velocities v via central differences."""
    if len(time_s) < 3:
        return False
    dt = np.diff(time_s)
    if np.any(dt <= 0.0):
        return False
    mean_dt = float(np.mean(dt))
    num_grad = np.gradient(q, mean_dt, axis=0)
    diff = np.abs(num_grad - v)
    return bool(np.mean(diff) < tol)


def quaternion_normalized(
    q: np.ndarray, *, cols: tuple[int, int] | None, tol: float = 1e-4
) -> bool:
    """Verify free-joint quaternion rows have unit norm.

    ``cols`` selects the quaternion column slice; ``None`` treats every row as a
    quaternion vector.
    """
    quats = q if cols is None else q[:, cols[0] : cols[1]]
    norms = np.linalg.norm(quats, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tol))


def check_muscle_activations(
    activations: np.ndarray | None,
) -> tuple[bool, dict[str, float]]:
    """Verify muscle activations respect physiological bounds [0.0, 1.0]."""
    if activations is None:
        return True, {}
    arr = np.asarray(activations)
    if arr.size == 0:
        return True, {}
    max_act = float(np.max(arr))
    min_act = float(np.min(arr))
    mean_act = float(np.mean(arr))
    ok = min_act >= 0.0 and max_act <= 1.0
    return ok, {
        "max_activation": max_act,
        "min_activation": min_act,
        "mean_activation": mean_act,
    }


def check_execution_contract(
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None,
    model_sha: str,
    native_tests_executed: int | None,
    rejection_reasons: list[str],
    missing_evidence: list[str],
    fk_only_message: str,
) -> bool:
    """Checks 1-3: freshness/copying, actuation, and native test count; returns is_fresh."""
    if expected_model_sha is not None and model_sha != expected_model_sha:
        rejection_reasons.append(
            f"model_sha256 mismatch: expected {expected_model_sha}, got {model_sha}"
        )

    # 1. Copied state trajectory detection. Absent flags are treated as
    # unverified (fail-closed), not assumed fresh.
    is_fresh = replay.get("is_fresh_simulation") is True
    if not replay.get("is_fresh_simulation") or replay.get("copied_from_reference"):
        if replay.get("is_fresh_simulation") is None:
            missing_evidence.append("is_fresh_simulation flag")
        rejection_reasons.append(
            "Copied state trajectory detected: saved controls must drive fresh native simulation"
        )

    # 2. FK-only playback detection. Absent flag is treated as unverified.
    if replay.get("actuation_applied") is not True:
        if replay.get("actuation_applied") is None:
            missing_evidence.append("actuation_applied flag")
        rejection_reasons.append(fk_only_message)

    # 3. Nonzero native test count check. An unknown count cannot be assumed
    # nonzero; it is missing evidence and blocks qualification.
    if native_tests_executed is None:
        missing_evidence.append("native test count (native_tests_executed)")
        rejection_reasons.append(
            "Native test count not recorded: nonzero native execution on "
            "pinned host cannot be confirmed"
        )
    elif native_tests_executed <= 0:
        rejection_reasons.append(
            f"Zero collected native tests: qualification requires nonzero native execution on pinned host (got {native_tests_executed})"
        )
    return is_fresh


def check_rollout_dynamics(
    replay: dict[str, Any],
    *,
    rejection_reasons: list[str],
    missing_evidence: list[str],
    quaternion_cols: tuple[int, int] | None = None,
) -> tuple[bool, dict[str, float]]:
    """Check 4: derivative consistency and energy balance; missing rollout is fail-closed.

    When ``quaternion_cols`` is given, the free-joint quaternion slice of the
    generalized coordinates is verified normalized before the derivative check.
    """
    native_state = replay.get("native_state")
    time_s = replay.get("time_s")
    derivatives_ok = False
    energy_summary: dict[str, float] = {}

    if native_state is None or time_s is None:
        missing_evidence.append("native_state/time_s dynamic rollout")
        rejection_reasons.append(
            "Native dynamic rollout data missing: no native_state/time_s "
            "trajectory to verify derivative and energy contracts"
        )
    else:
        state_arr = np.asarray(native_state)
        time_arr = np.asarray(time_s)

        if not np.all(np.isfinite(state_arr)) or not np.all(np.isfinite(time_arr)):
            rejection_reasons.append(
                "Non-finite values (NaN or Inf) detected in simulation state"
            )
        else:
            n_coords = state_arr.shape[1] // 2
            q = state_arr[:, :n_coords]
            v = state_arr[:, n_coords:]

            if (
                quaternion_cols is not None
                and q.shape[1] >= quaternion_cols[1]
                and not quaternion_normalized(q, cols=quaternion_cols)
            ):
                rejection_reasons.append(
                    "Free-joint root quaternion is not normalized to unit length (|norm - 1| > 1e-4)"
                )

            derivatives_ok = derivatives_consistent(time_arr, q, v)
            if not derivatives_ok:
                rejection_reasons.append(
                    "Reported joint velocities inconsistent with dq/dt central "
                    "differences (independent derivative check failed)"
                )
            # Calculate nominal kinetic and potential energies
            ke = 0.5 * np.sum(v**2, axis=1)
            pe = (
                9.81 * np.sum(q[:, :3], axis=1)
                if q.shape[1] >= 3
                else np.zeros(len(time_arr))
            )

            energy_summary = {
                "kinetic_energy_j": float(np.mean(ke)),
                "potential_energy_j": float(np.mean(pe)),
                "energy_conservation_error": float(np.std(ke + pe)),
            }
            if not all(math.isfinite(value) for value in energy_summary.values()):
                rejection_reasons.append(
                    "Non-finite values detected in energy balance summary"
                )
    return derivatives_ok, energy_summary


def compute_marker_metrics(
    replay: dict[str, Any],
    *,
    rejection_reasons: list[str],
    missing_evidence: list[str],
    respect_valid_mask: bool = False,
) -> dict[str, float]:
    """Only metrics computed from recorded observations are emitted (never fabricated)."""
    markers = replay.get("markers_m")
    target = replay.get("target_m")
    marker_metrics: dict[str, float] = {}
    if markers is None or target is None:
        missing_evidence.append(
            "aligned common-marker observations (markers_m/target_m)"
        )
        rejection_reasons.append(
            "Aligned common-marker metrics unavailable: replay is missing "
            "markers_m/target_m from a native rollout against the same observations"
        )
        return marker_metrics
    m_arr = np.asarray(markers)
    t_arr = np.asarray(target)
    finite = bool(np.all(np.isfinite(m_arr)) and np.all(np.isfinite(t_arr)))
    if respect_valid_mask:
        valid = replay.get("valid")
        val_arr = (
            np.asarray(valid)
            if valid is not None
            else np.ones(m_arr.shape[:2], dtype=bool)
        )
        if np.any(val_arr) and finite:
            sq_err = np.sum((m_arr - t_arr) ** 2, axis=-1)
            marker_metrics["whole_rms_m"] = float(np.sqrt(np.mean(sq_err[val_arr])))
        else:
            rejection_reasons.append(
                "Non-finite values detected in marker alignment observations"
            )
    elif finite:
        sq_err = np.sum((m_arr - t_arr) ** 2, axis=-1)
        marker_metrics["whole_rms_m"] = float(np.sqrt(np.mean(sq_err)))
    else:
        rejection_reasons.append(
            "Non-finite values detected in marker alignment observations"
        )
    return marker_metrics


def build_unavailable_receipt(
    candidate: dict[str, Any],
    receipt_cls: Any,
    *,
    engine: str,
    status: Any,
    labels: dict[str, str],
    limitations: tuple[str, ...],
    remedy: str,
):
    """Fail-closed: without the engine runtime on host no native dynamic replay can be produced.

    ``labels`` keys: missing_rollout_label, rejection_reason, diagnostic_message.
    """
    return receipt_cls(
        schema_version=1,
        engine=engine,
        club=str(candidate.get("club") or "driver"),
        status=status,
        candidate_sha256=str(candidate.get("source_sha256") or ""),
        model_sha256=str(candidate.get("model_sha256") or ""),
        capture_sha256=str(candidate.get("capture_sha256") or ""),
        runtime_available=False,
        is_fresh_simulation=False,
        derivatives_consistent=False,
        energy_balance_checked=False,
        declared_limitations=list(limitations),
        rejection_reasons=[labels["rejection_reason"]],
        missing_evidence=[labels["missing_rollout_label"]],
        remedy=remedy,
        diagnostic_message=labels["diagnostic_message"],
    )


def begin_evaluation(replay: dict[str, Any]) -> tuple[list[str], list[str]]:
    """Fresh rejection Reasons/missing-evidence accumulators for one evaluation."""
    return [], []


def identity_shas(candidate: dict[str, Any]) -> tuple[str, str, str, str]:
    """Extract club and the three identity digests of a candidate payload."""
    return (
        str(candidate.get("club") or "driver"),
        str(candidate.get("source_sha256") or ""),
        str(candidate.get("model_sha256") or ""),
        str(candidate.get("capture_sha256") or ""),
    )

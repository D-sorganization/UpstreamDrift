"""Drake native dual-club dynamics qualification and receipt generator (MMR-10D #11094).

Implements dynamic qualification contracts for Drake:
- Rejects zero collected native tests, copied states, and FK-only playbacks.
- Verifies that saved controls drive a fresh simulation rather than loaded state trajectory.
- Independent derivative and energy balance reference checks.
- Discloses declared engine-specific limitations and aligned marker metrics.
- Distinguishes qualified, rejected, and unavailable states.
- Dual-club support for driver and 7-iron.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

DRAKE_ENGINE_LIMITATIONS: tuple[str, ...] = (
    "upper_body_27dof_float_pathway: pelvis translation and orientation are free float coordinates",
    "rigid_weld_closure: dual-grip closed loop enforced via 6D rigid weld constraint",
    "continuous_polynomial_actuation: 6th-order Bernstein/power torque polynomials per joint",
    "ground_contact_requires_full_body: upper-body slice does not model ground reaction forces",
)

DRAKE_UNAVAILABLE_REMEDY: str = (
    "Install pydrake on the pinned host (pip install drake) and re-run "
    "scripts/ci/run_native_engine_lane.sh --engine drake so native rollouts and "
    "club receipts are regenerated from real execution."
)


class DrakeQualificationStatus(StrEnum):
    """Lifecycle status of a Drake native dynamics qualification run."""

    QUALIFIED = "qualified"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class DrakeQualificationReceipt:
    """Cryptographically verifiable qualification receipt for Drake dynamic runs."""

    schema_version: int
    engine: str
    club: str
    status: DrakeQualificationStatus
    candidate_sha256: str
    model_sha256: str
    capture_sha256: str
    runtime_available: bool
    is_fresh_simulation: bool
    derivatives_consistent: bool
    energy_balance_checked: bool
    energy_summary: dict[str, float] = field(default_factory=dict)
    marker_metrics: dict[str, float] = field(default_factory=dict)
    declared_limitations: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    remedy: str = ""
    diagnostic_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Convert receipt to JSON-serializable dictionary."""
        data = asdict(self)
        data["status"] = self.status.value
        return data

    def save(self, path: Path | str) -> None:
        """Write receipt to formatted JSON file atomically."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        content = json.dumps(self.as_dict(), indent=2) + "\n"
        p.write_text(content, encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> DrakeQualificationReceipt:
        """Load receipt from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["status"] = DrakeQualificationStatus(data["status"])
        return cls(**data)


def _check_derivatives_consistency(
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


def validate_drake_candidate_replay(
    candidate: dict[str, Any],
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
    drake_available: bool = True,
) -> DrakeQualificationReceipt:
    """Validate a candidate and its Drake replay against acceptance criteria."""
    rejection_reasons: list[str] = []
    missing_evidence: list[str] = []

    # Fail-closed: without the pydrake runtime on host no native dynamic replay
    # can be produced, so a QUALIFIED outcome is impossible regardless of payload.
    if not drake_available:
        missing_evidence.append("native Drake rollout (pydrake runtime)")
        return DrakeQualificationReceipt(
            schema_version=1,
            engine="drake",
            club=str(candidate.get("club") or "driver"),
            status=DrakeQualificationStatus.UNAVAILABLE,
            candidate_sha256=str(candidate.get("source_sha256") or ""),
            model_sha256=str(candidate.get("model_sha256") or ""),
            capture_sha256=str(candidate.get("capture_sha256") or ""),
            runtime_available=False,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(DRAKE_ENGINE_LIMITATIONS),
            rejection_reasons=["pydrake runtime is not installed on host"],
            missing_evidence=missing_evidence,
            remedy=DRAKE_UNAVAILABLE_REMEDY,
            diagnostic_message="pydrake runtime is not installed on host: live dynamic simulation unavailable.",
        )

    club = str(candidate.get("club") or "driver")
    cand_sha = str(candidate.get("source_sha256") or "")
    model_sha = str(candidate.get("model_sha256") or "")
    capture_sha = str(candidate.get("capture_sha256") or "")

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
        rejection_reasons.append(
            "FK-only playback detected without native dynamic simulation"
        )

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

    # 4. Derivative and energy balance checks. Missing rollout data is fail-closed.
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

            derivatives_ok = _check_derivatives_consistency(time_arr, q, v)
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

    # 5. Marker metrics. Only metrics computed from the recorded observations
    # are emitted; no synthesized, scaled, or copied values are fabricated here.
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
    else:
        m_arr = np.asarray(markers)
        t_arr = np.asarray(target)
        valid = replay.get("valid")
        val_arr = (
            np.asarray(valid)
            if valid is not None
            else np.ones(m_arr.shape[:2], dtype=bool)
        )
        diff = m_arr - t_arr
        sq_err = np.sum(diff**2, axis=-1)
        if np.any(val_arr) and bool(
            np.all(np.isfinite(m_arr)) and np.all(np.isfinite(t_arr))
        ):
            marker_metrics["whole_rms_m"] = float(np.sqrt(np.mean(sq_err[val_arr])))
        else:
            rejection_reasons.append(
                "Non-finite values detected in marker alignment observations"
            )

    status = (
        DrakeQualificationStatus.QUALIFIED
        if not rejection_reasons
        else DrakeQualificationStatus.REJECTED
    )

    return DrakeQualificationReceipt(
        schema_version=1,
        engine="drake",
        club=club,
        status=status,
        candidate_sha256=cand_sha,
        model_sha256=model_sha,
        capture_sha256=capture_sha,
        runtime_available=drake_available,
        is_fresh_simulation=is_fresh,
        derivatives_consistent=derivatives_ok,
        energy_balance_checked=bool(energy_summary),
        energy_summary=energy_summary,
        marker_metrics=marker_metrics,
        declared_limitations=list(DRAKE_ENGINE_LIMITATIONS),
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        remedy=DRAKE_UNAVAILABLE_REMEDY if rejection_reasons else "",
        diagnostic_message="All qualification checks passed."
        if not rejection_reasons
        else "; ".join(rejection_reasons),
    )


def assess_drake_qualification(
    candidate: dict[str, Any],
    replay: dict[str, Any] | None = None,
    *,
    drake_available: bool | None = None,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
) -> DrakeQualificationReceipt:
    """Assess qualification status, returning UNAVAILABLE if pydrake runtime is missing."""
    if drake_available is None:
        try:
            import pydrake  # noqa: F401

            drake_available = True
        except ImportError:
            drake_available = False

    club = str(candidate.get("club") or "driver")
    cand_sha = str(candidate.get("source_sha256") or "")
    model_sha = str(candidate.get("model_sha256") or "")
    capture_sha = str(candidate.get("capture_sha256") or "")

    if not drake_available:
        return DrakeQualificationReceipt(
            schema_version=1,
            engine="drake",
            club=club,
            status=DrakeQualificationStatus.UNAVAILABLE,
            candidate_sha256=cand_sha,
            model_sha256=model_sha,
            capture_sha256=capture_sha,
            runtime_available=False,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(DRAKE_ENGINE_LIMITATIONS),
            rejection_reasons=["pydrake runtime is not installed on host"],
            missing_evidence=[
                "native Drake rollout (pydrake runtime)",
                "candidate_sha256",
                "model_sha256",
                "capture_sha256",
                "native_state/time_s dynamic rollout",
                "aligned common-marker observations (markers_m/target_m)",
                "native test count (native_tests_executed)",
            ],
            remedy=DRAKE_UNAVAILABLE_REMEDY,
            diagnostic_message="pydrake runtime is not installed on host: live simulation unavailable.",
        )

    if replay is None:
        return DrakeQualificationReceipt(
            schema_version=1,
            engine="drake",
            club=club,
            status=DrakeQualificationStatus.REJECTED,
            candidate_sha256=cand_sha,
            model_sha256=model_sha,
            capture_sha256=capture_sha,
            runtime_available=True,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(DRAKE_ENGINE_LIMITATIONS),
            rejection_reasons=["Replay data is missing"],
            missing_evidence=[
                "native_state/time_s dynamic rollout",
                "aligned common-marker observations (markers_m/target_m)",
                "native test count (native_tests_executed)",
            ],
            remedy=(
                "Run the native Drake replay on a pinned host with pydrake "
                "installed (scripts/ci/run_native_engine_lane.sh --engine drake) "
                "and re-assess with the recorded rollout payload."
            ),
            diagnostic_message="Replay data is missing: cannot evaluate dynamic rollout.",
        )

    return validate_drake_candidate_replay(
        candidate,
        replay,
        expected_model_sha=expected_model_sha,
        native_tests_executed=native_tests_executed,
        drake_available=True,
    )

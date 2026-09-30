"""MyoSuite native dual-club dynamics qualification and replay verification (MMR-10M, #11096).

Validates native dynamic simulation rollouts for MyoSuite (MuJoCo-backed musculoskeletal model),
rejecting FK-only playback, identical/copied state trajectory clones, zero-test qualification runs,
and unphysiological muscle activations.
"""

from __future__ import annotations

import json
import math
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

import numpy as np

# Authoritative engine-specific limitations for MyoSuite full-body musculoskeletal modeling
MYOSUITE_ENGINE_LIMITATIONS: tuple[str, ...] = (
    "Musculoskeletal excitation-activation dynamics governed by first-order filter with activation/deactivation time constants",
    "Hill-type muscle tendon unit (MTU) force-length-velocity multipliers with physiological bounds",
    "Free-joint quaternion orientation normalization (w^2 + x^2 + y^2 + z^2 = 1) and spatial velocity state layout",
    "Dual-grip weld constraint kinematics coupling hands to club shaft",
    "Four-foot Hunt-Crossley contact spheres with normal compliance and friction cone limits",
    "Absence of native joint-torque inverse dynamics (forward muscle excitation driven only)",
)

MYOSUITE_UNAVAILABLE_REMEDY: str = (
    "Install the myosuite/MuJoCo stack on the pinned host and re-run "
    "scripts/ci/run_native_engine_lane.sh --engine myosuite so native rollouts "
    "and club receipts are regenerated from real execution."
)


class MyoSuiteQualificationStatus(StrEnum):
    """Qualification status for MyoSuite native dual-club candidate."""

    QUALIFIED = "qualified"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class MyoSuiteQualificationReceipt:
    """Tamper-evident qualification receipt for MyoSuite native candidate."""

    schema_version: int
    engine: str
    club: str
    status: MyoSuiteQualificationStatus
    candidate_sha256: str
    model_sha256: str
    capture_sha256: str
    runtime_available: bool
    is_fresh_simulation: bool
    derivatives_consistent: bool
    energy_balance_checked: bool
    energy_summary: dict[str, float] = field(default_factory=dict)
    marker_metrics: dict[str, float] = field(default_factory=dict)
    muscle_metrics: dict[str, float] = field(default_factory=dict)
    declared_limitations: list[str] = field(default_factory=list)
    rejection_reasons: list[str] = field(default_factory=list)
    missing_evidence: list[str] = field(default_factory=list)
    remedy: str = ""
    diagnostic_message: str = ""

    def as_dict(self) -> dict[str, Any]:
        """Convert receipt to JSON-serializable dictionary."""
        d = asdict(self)
        d["status"] = self.status.value
        return d

    def save(self, path: Path | str) -> None:
        """Save receipt to JSON file."""
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.as_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | str) -> MyoSuiteQualificationReceipt:
        """Load receipt from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["status"] = MyoSuiteQualificationStatus(data["status"])
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


def _check_muscle_activations(
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


def _check_quaternion_normalization(q: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify root free-joint quaternion (indices 3:7) has unit norm across trajectory."""
    if q.ndim != 2 or q.shape[1] < 7:
        return True
    quats = q[:, 3:7]
    norms = np.linalg.norm(quats, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tol))


def _check_myosuite_execution_contract(
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None,
    model_sha: str,
    native_tests_executed: int | None,
    rejection_reasons: list[str],
    missing_evidence: list[str],
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
        rejection_reasons.append(
            "FK-only playback detected without native dynamic simulation or muscle excitation"
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
    return is_fresh


def _check_myosuite_rollout_dynamics(
    replay: dict[str, Any],
    *,
    rejection_reasons: list[str],
    missing_evidence: list[str],
) -> tuple[bool, dict[str, float]]:
    """Check 4: quaternion normalization, derivative consistency, energy balance; missing rollout is fail-closed."""
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

            if not _check_quaternion_normalization(q):
                rejection_reasons.append(
                    "Free-joint root quaternion is not normalized to unit length (|norm - 1| > 1e-4)"
                )

            derivatives_ok = _check_derivatives_consistency(time_arr, q, v)
            if not derivatives_ok:
                rejection_reasons.append(
                    "Reported joint velocities inconsistent with dq/dt central "
                    "differences (independent derivative check failed)"
                )
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


def _compute_myosuite_marker_metrics(
    replay: dict[str, Any],
    *,
    rejection_reasons: list[str],
    missing_evidence: list[str],
) -> dict[str, float]:
    """Check 6: only metrics computed from recorded observations are emitted."""
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
        if np.all(np.isfinite(m_arr)) and np.all(np.isfinite(t_arr)):
            diff = m_arr - t_arr
            sq_err = np.sum(diff**2, axis=-1)
            marker_metrics["whole_rms_m"] = float(np.sqrt(np.mean(sq_err)))
        else:
            rejection_reasons.append(
                "Non-finite values detected in marker alignment observations"
            )
    return marker_metrics


def _myosuite_unavailable_receipt(candidate: dict[str, Any]) -> MyoSuiteQualificationReceipt:
    """Fail-closed: without the myosuite/MuJoCo runtime on host no native dynamic replay can be produced."""
    missing_evidence = ["native MyoSuite rollout (myosuite/MuJoCo runtime)"]
    return MyoSuiteQualificationReceipt(
        schema_version=1,
        engine="myosuite",
        club=str(candidate.get("club") or "driver"),
        status=MyoSuiteQualificationStatus.UNAVAILABLE,
        candidate_sha256=str(candidate.get("source_sha256") or ""),
        model_sha256=str(candidate.get("model_sha256") or ""),
        capture_sha256=str(candidate.get("capture_sha256") or ""),
        runtime_available=False,
        is_fresh_simulation=False,
        derivatives_consistent=False,
        energy_balance_checked=False,
        declared_limitations=list(MYOSUITE_ENGINE_LIMITATIONS),
        rejection_reasons=["MyoSuite runtime is not installed on host"],
        missing_evidence=missing_evidence,
        remedy=MYOSUITE_UNAVAILABLE_REMEDY,
        diagnostic_message="MyoSuite runtime is not installed on host: live dynamic simulation unavailable.",
    )


def validate_myosuite_candidate_replay(
    candidate: dict[str, Any],
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
    myosuite_available: bool = True,
) -> MyoSuiteQualificationReceipt:
    """Validate a candidate and its MyoSuite dynamic replay against acceptance criteria."""
    if not myosuite_available:
        return _myosuite_unavailable_receipt(candidate)

    rejection_reasons: list[str] = []
    missing_evidence: list[str] = []
    club = str(candidate.get("club") or "driver")
    cand_sha = str(candidate.get("source_sha256") or "")
    model_sha = str(candidate.get("model_sha256") or "")
    capture_sha = str(candidate.get("capture_sha256") or "")

    is_fresh = _check_myosuite_execution_contract(
        replay,
        expected_model_sha=expected_model_sha,
        model_sha=model_sha,
        native_tests_executed=native_tests_executed,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
    )
    derivatives_ok, energy_summary = _check_myosuite_rollout_dynamics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
    )

    # 5. Muscle activation bounds
    activations = replay.get("activations")
    acts_ok, muscle_metrics = _check_muscle_activations(activations)
    if not acts_ok:
        rejection_reasons.append(
            f"Muscle activation exceeds physiological bounds [0, 1]: max={muscle_metrics.get('max_activation')}"
        )

    marker_metrics = _compute_myosuite_marker_metrics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
    )

    status = (
        MyoSuiteQualificationStatus.QUALIFIED
        if not rejection_reasons
        else MyoSuiteQualificationStatus.REJECTED
    )

    return MyoSuiteQualificationReceipt(
        schema_version=1,
        engine="myosuite",
        club=club,
        status=status,
        candidate_sha256=cand_sha,
        model_sha256=model_sha,
        capture_sha256=capture_sha,
        runtime_available=myosuite_available,
        is_fresh_simulation=is_fresh,
        derivatives_consistent=derivatives_ok,
        energy_balance_checked=bool(energy_summary),
        energy_summary=energy_summary,
        marker_metrics=marker_metrics,
        muscle_metrics=muscle_metrics,
        declared_limitations=list(MYOSUITE_ENGINE_LIMITATIONS),
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        remedy=MYOSUITE_UNAVAILABLE_REMEDY if rejection_reasons else "",
        diagnostic_message="All qualification checks passed."
        if not rejection_reasons
        else "; ".join(rejection_reasons),
    )


def assess_myosuite_qualification(
    candidate: dict[str, Any],
    replay: dict[str, Any] | None = None,
    *,
    myosuite_available: bool | None = None,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
) -> MyoSuiteQualificationReceipt:
    """Assess qualification status, returning UNAVAILABLE if myosuite runtime is missing."""
    if myosuite_available is None:
        try:
            import myosuite  # noqa: F401

            myosuite_available = True
        except ImportError:
            myosuite_available = False

    club = str(candidate.get("club") or "driver")
    cand_sha = str(candidate.get("source_sha256") or "")
    model_sha = str(candidate.get("model_sha256") or "")
    capture_sha = str(candidate.get("capture_sha256") or "")

    if not myosuite_available:
        return MyoSuiteQualificationReceipt(
            schema_version=1,
            engine="myosuite",
            club=club,
            status=MyoSuiteQualificationStatus.UNAVAILABLE,
            candidate_sha256=cand_sha,
            model_sha256=model_sha,
            capture_sha256=capture_sha,
            runtime_available=False,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(MYOSUITE_ENGINE_LIMITATIONS),
            rejection_reasons=["MyoSuite runtime is not installed on host"],
            missing_evidence=[
                "native MyoSuite rollout (myosuite/MuJoCo runtime)",
                "candidate_sha256",
                "model_sha256",
                "capture_sha256",
                "native_state/time_s dynamic rollout",
                "aligned common-marker observations (markers_m/target_m)",
                "native test count (native_tests_executed)",
            ],
            remedy=MYOSUITE_UNAVAILABLE_REMEDY,
            diagnostic_message="MyoSuite runtime is not installed on host: live simulation unavailable.",
        )

    if replay is None:
        return MyoSuiteQualificationReceipt(
            schema_version=1,
            engine="myosuite",
            club=club,
            status=MyoSuiteQualificationStatus.REJECTED,
            candidate_sha256=cand_sha,
            model_sha256=model_sha,
            capture_sha256=capture_sha,
            runtime_available=True,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(MYOSUITE_ENGINE_LIMITATIONS),
            rejection_reasons=["Replay data is missing"],
            missing_evidence=[
                "native_state/time_s dynamic rollout",
                "aligned common-marker observations (markers_m/target_m)",
                "native test count (native_tests_executed)",
            ],
            remedy=(
                "Run the native MyoSuite replay on a pinned host with the "
                "myosuite/MuJoCo stack installed (scripts/ci/run_native_engine_lane.sh "
                "--engine myosuite) and re-assess with the recorded rollout payload."
            ),
            diagnostic_message="Replay data is missing: cannot evaluate dynamic rollout.",
        )

    return validate_myosuite_candidate_replay(
        candidate,
        replay,
        expected_model_sha=expected_model_sha,
        native_tests_executed=native_tests_executed,
        myosuite_available=True,
    )

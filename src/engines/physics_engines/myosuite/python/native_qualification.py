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

from src.shared.python.native_lanes.common import (
    identity_shas,
    begin_evaluation,
    build_unavailable_receipt,
    check_execution_contract,
    check_muscle_activations,
    check_rollout_dynamics,
    compute_marker_metrics,
)

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


def _check_quaternion_normalization(q: np.ndarray, tol: float = 1e-4) -> bool:
    """Verify root free-joint quaternion (indices 3:7) has unit norm across trajectory."""
    if q.ndim != 2 or q.shape[1] < 7:
        return True
    quats = q[:, 3:7]
    norms = np.linalg.norm(quats, axis=1)
    return bool(np.all(np.abs(norms - 1.0) <= tol))


# Engine shims over the shared native lane contracts (no duplicated logic).
def _check_execution_contract(
    replay,
    *,
    expected_model_sha,
    model_sha,
    native_tests_executed,
    rejection_reasons,
    missing_evidence,
):
    return check_execution_contract(
        replay,
        expected_model_sha=expected_model_sha,
        model_sha=model_sha,
        native_tests_executed=native_tests_executed,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        fk_only_message="FK-only playback detected without native dynamic simulation or muscle excitation",
    )


def _check_rollout_dynamics(replay, *, rejection_reasons, missing_evidence):
    return check_rollout_dynamics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        quaternion_cols=(3, 7),
    )


def _compute_marker_metrics(replay, *, rejection_reasons, missing_evidence):
    return compute_marker_metrics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        respect_valid_mask=False,
    )


def _myosuite_unavailable_receipt(candidate: dict[str, Any]):
    return build_unavailable_receipt(
        candidate,
        MyoSuiteQualificationReceipt,
        engine="myosuite",
        status=MyoSuiteQualificationStatus.UNAVAILABLE,
        labels={
            "missing_rollout_label": "native MyoSuite rollout (myosuite/MuJoCo runtime)",
            "rejection_reason": "MyoSuite runtime is not installed on host",
            "diagnostic_message": "MyoSuite runtime is not installed on host: live dynamic simulation unavailable.",
        },
        limitations=MYOSUITE_ENGINE_LIMITATIONS,
        remedy=MYOSUITE_UNAVAILABLE_REMEDY,
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

    rejection_reasons, missing_evidence = begin_evaluation(replay)
    club, cand_sha, model_sha, capture_sha = identity_shas(candidate)

    is_fresh = _check_execution_contract(
        replay,
        expected_model_sha=expected_model_sha,
        model_sha=model_sha,
        native_tests_executed=native_tests_executed,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
    )
    derivatives_ok, energy_summary = _check_rollout_dynamics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
    )

    # 5. Muscle activation bounds
    activations = replay.get("activations")
    acts_ok, muscle_metrics = check_muscle_activations(activations)
    if not acts_ok:
        rejection_reasons.append(
            f"Muscle activation exceeds physiological bounds [0, 1]: max={muscle_metrics.get('max_activation')}"
        )

    marker_metrics = _compute_marker_metrics(
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

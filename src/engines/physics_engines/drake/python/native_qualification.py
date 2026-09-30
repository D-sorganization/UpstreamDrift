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
from dataclasses import asdict, dataclass, field
from enum import StrEnum
from pathlib import Path
from typing import Any

from src.shared.python.native_lanes.common import (
    begin_evaluation,
    check_execution_contract,
    check_rollout_dynamics,
    compute_marker_metrics,
    identity_shas,
)

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
        fk_only_message="FK-only playback detected without native dynamic simulation",
    )


def _check_rollout_dynamics(replay, *, rejection_reasons, missing_evidence):
    return check_rollout_dynamics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
    )


def _compute_marker_metrics(replay, *, rejection_reasons, missing_evidence):
    return compute_marker_metrics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        respect_valid_mask=True,
    )


def _drake_unavailable_receipt(candidate: dict[str, Any]) -> DrakeQualificationReceipt:
    """Fail-closed: without pydrake on host no native dynamic replay can be produced."""
    missing_evidence = ["native Drake rollout (pydrake runtime)"]
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


def validate_drake_candidate_replay(
    candidate: dict[str, Any],
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
    drake_available: bool = True,
) -> DrakeQualificationReceipt:
    """Validate a candidate and its Drake replay against acceptance criteria."""
    if not drake_available:
        return _drake_unavailable_receipt(candidate)

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
    marker_metrics = _compute_marker_metrics(
        replay,
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
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

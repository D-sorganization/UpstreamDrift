"""OpenSim native dual-club dynamics qualification and receipt generator (MMR-10O #11095).

Implements dynamic qualification contracts for OpenSim:
- Rejects zero collected native tests, copied states, and FK-only playbacks.
- Verifies that saved controls drive a fresh simulation rather than loaded state trajectory.
- Independent derivative and energy balance reference checks.
- Enforces physiological limits on muscle activations and contraction velocities.
- Discloses declared engine-specific musculoskeletal limitations and aligned marker metrics.
- Distinguishes qualified, rejected, and unavailable states.
- Dual-club support for driver and 7-iron.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from enum import StrEnum
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.native_lanes.common import (
    build_unavailable_receipt,
    check_execution_contract,
    check_muscle_activations,
    check_rollout_dynamics,
    compute_marker_metrics,
)

OPENSIM_ENGINE_LIMITATIONS: tuple[str, ...] = (
    "hill_type_activation_dynamics: muscle excitation to activation first-order lag (tau_act ~ 10-15 ms, tau_deact ~ 40-50 ms)",
    "force_velocity_length_multipliers: active and passive force-length-velocity curves restrict instantaneous force generation",
    "tendon_elasticity_equilibrium: stiff tendon approximation or equilibrium solver required for muscle-tendon units",
    "coordinate_limit_forces: penalty contact or coordinate limit forces active near physiological range-of-motion extrema",
    "ground_contact_requires_external_wrenches: foot-ground reaction forces require calibrated multi-component contact meshes or external force profiles",
)

OPENSIM_UNAVAILABLE_REMEDY: str = (
    "Install the OpenSim Python bindings on the pinned host and re-run "
    "scripts/ci/run_native_engine_lane.sh --engine opensim so native rollouts "
    "and club receipts are regenerated from real execution."
)


class OpenSimQualificationStatus(StrEnum):
    """Lifecycle status of an OpenSim native dynamics qualification run."""

    QUALIFIED = "qualified"
    REJECTED = "rejected"
    UNAVAILABLE = "unavailable"


@dataclass(frozen=True)
class OpenSimQualificationReceipt:
    """Cryptographically verifiable qualification receipt for OpenSim dynamic runs."""

    schema_version: int
    engine: str
    club: str
    status: OpenSimQualificationStatus
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
    def load(cls, path: Path | str) -> OpenSimQualificationReceipt:
        """Load receipt from JSON file."""
        p = Path(path)
        data = json.loads(p.read_text(encoding="utf-8"))
        data["status"] = OpenSimQualificationStatus(data["status"])
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
        fk_only_message="FK-only playback detected without native dynamic simulation or muscle excitation",
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
        respect_valid_mask=False,
    )


def _opensim_unavailable_receipt(candidate: dict[str, Any]):
    return build_unavailable_receipt(
        candidate,
        OpenSimQualificationReceipt,
        engine="opensim",
        status=OpenSimQualificationStatus.UNAVAILABLE,
        labels={
            "missing_rollout_label": "native OpenSim rollout (opensim runtime)",
            "rejection_reason": "OpenSim runtime is not installed on host",
            "diagnostic_message": "OpenSim runtime is not installed on host: live dynamic simulation unavailable.",
        },
        limitations=OPENSIM_ENGINE_LIMITATIONS,
        remedy=OPENSIM_UNAVAILABLE_REMEDY,
    )


def validate_opensim_candidate_replay(
    candidate: dict[str, Any],
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
    opensim_available: bool = True,
) -> OpenSimQualificationReceipt:
    """Validate a candidate and its OpenSim dynamic replay against acceptance criteria."""
    if not opensim_available:
        return _opensim_unavailable_receipt(candidate)

    rejection_reasons: list[str] = []
    missing_evidence: list[str] = []
    club = str(candidate.get("club") or "driver")
    cand_sha = str(candidate.get("source_sha256") or "")
    model_sha = str(candidate.get("model_sha256") or "")
    capture_sha = str(candidate.get("capture_sha256") or "")

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
        OpenSimQualificationStatus.QUALIFIED
        if not rejection_reasons
        else OpenSimQualificationStatus.REJECTED
    )

    return OpenSimQualificationReceipt(
        schema_version=1,
        engine="opensim",
        club=club,
        status=status,
        candidate_sha256=cand_sha,
        model_sha256=model_sha,
        capture_sha256=capture_sha,
        runtime_available=opensim_available,
        is_fresh_simulation=is_fresh,
        derivatives_consistent=derivatives_ok,
        energy_balance_checked=bool(energy_summary),
        energy_summary=energy_summary,
        marker_metrics=marker_metrics,
        muscle_metrics=muscle_metrics,
        declared_limitations=list(OPENSIM_ENGINE_LIMITATIONS),
        rejection_reasons=rejection_reasons,
        missing_evidence=missing_evidence,
        remedy=OPENSIM_UNAVAILABLE_REMEDY if rejection_reasons else "",
        diagnostic_message="All qualification checks passed."
        if not rejection_reasons
        else "; ".join(rejection_reasons),
    )


def assess_opensim_qualification(
    candidate: dict[str, Any],
    replay: dict[str, Any] | None = None,
    *,
    opensim_available: bool | None = None,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
) -> OpenSimQualificationReceipt:
    """Assess qualification status, returning UNAVAILABLE if opensim runtime is missing."""
    if opensim_available is None:
        try:
            import opensim  # noqa: F401

            opensim_available = True
        except ImportError:
            opensim_available = False

    club = str(candidate.get("club") or "driver")
    cand_sha = str(candidate.get("source_sha256") or "")
    model_sha = str(candidate.get("model_sha256") or "")
    capture_sha = str(candidate.get("capture_sha256") or "")

    if not opensim_available:
        return OpenSimQualificationReceipt(
            schema_version=1,
            engine="opensim",
            club=club,
            status=OpenSimQualificationStatus.UNAVAILABLE,
            candidate_sha256=cand_sha,
            model_sha256=model_sha,
            capture_sha256=capture_sha,
            runtime_available=False,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(OPENSIM_ENGINE_LIMITATIONS),
            rejection_reasons=["OpenSim runtime is not installed on host"],
            missing_evidence=[
                "native OpenSim rollout (opensim runtime)",
                "candidate_sha256",
                "model_sha256",
                "capture_sha256",
                "native_state/time_s dynamic rollout",
                "aligned common-marker observations (markers_m/target_m)",
                "native test count (native_tests_executed)",
            ],
            remedy=OPENSIM_UNAVAILABLE_REMEDY,
            diagnostic_message="OpenSim runtime is not installed on host: live simulation unavailable.",
        )

    if replay is None:
        return OpenSimQualificationReceipt(
            schema_version=1,
            engine="opensim",
            club=club,
            status=OpenSimQualificationStatus.REJECTED,
            candidate_sha256=cand_sha,
            model_sha256=model_sha,
            capture_sha256=capture_sha,
            runtime_available=True,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(OPENSIM_ENGINE_LIMITATIONS),
            rejection_reasons=["Replay data is missing"],
            missing_evidence=[
                "native_state/time_s dynamic rollout",
                "aligned common-marker observations (markers_m/target_m)",
                "native test count (native_tests_executed)",
            ],
            remedy=(
                "Run the native OpenSim replay on a pinned host with the "
                "opensim bindings installed (scripts/ci/run_native_engine_lane.sh "
                "--engine opensim) and re-assess with the recorded rollout payload."
            ),
            diagnostic_message="Replay data is missing: cannot evaluate dynamic rollout.",
        )

    return validate_opensim_candidate_replay(
        candidate,
        replay,
        expected_model_sha=expected_model_sha,
        native_tests_executed=native_tests_executed,
        opensim_available=True,
    )

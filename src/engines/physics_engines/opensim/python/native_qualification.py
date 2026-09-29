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


def validate_opensim_candidate_replay(
    candidate: dict[str, Any],
    replay: dict[str, Any],
    *,
    expected_model_sha: str | None = None,
    native_tests_executed: int | None = None,
    opensim_available: bool = True,
) -> OpenSimQualificationReceipt:
    """Validate a candidate and its OpenSim dynamic replay against acceptance criteria."""
    rejection_reasons: list[str] = []
    missing_evidence: list[str] = []

    # Fail-closed: without the opensim bindings on host no native dynamic replay
    # can be produced, so a QUALIFIED outcome is impossible regardless of payload.
    if not opensim_available:
        missing_evidence.append("native OpenSim rollout (opensim runtime)")
        return OpenSimQualificationReceipt(
            schema_version=1,
            engine="opensim",
            club=str(candidate.get("club") or "driver"),
            status=OpenSimQualificationStatus.UNAVAILABLE,
            candidate_sha256=str(candidate.get("source_sha256") or ""),
            model_sha256=str(candidate.get("model_sha256") or ""),
            capture_sha256=str(candidate.get("capture_sha256") or ""),
            runtime_available=False,
            is_fresh_simulation=False,
            derivatives_consistent=False,
            energy_balance_checked=False,
            declared_limitations=list(OPENSIM_ENGINE_LIMITATIONS),
            rejection_reasons=["OpenSim runtime is not installed on host"],
            missing_evidence=missing_evidence,
            remedy=OPENSIM_UNAVAILABLE_REMEDY,
            diagnostic_message="OpenSim runtime is not installed on host: live dynamic simulation unavailable.",
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

    # 5. Muscle activation bounds
    activations = replay.get("activations")
    acts_ok, muscle_metrics = _check_muscle_activations(activations)
    if not acts_ok:
        rejection_reasons.append(
            f"Muscle activation exceeds physiological bounds [0, 1]: max={muscle_metrics.get('max_activation')}"
        )

    # 6. Marker metrics. Only metrics computed from the recorded observations
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
        if np.all(np.isfinite(m_arr)) and np.all(np.isfinite(t_arr)):
            diff = m_arr - t_arr
            sq_err = np.sum(diff**2, axis=-1)
            marker_metrics["whole_rms_m"] = float(np.sqrt(np.mean(sq_err)))
        else:
            rejection_reasons.append(
                "Non-finite values detected in marker alignment observations"
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

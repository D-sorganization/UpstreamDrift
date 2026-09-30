"""Simscape Continuous-Replay Qualification Harness (MMR-07, #11107).

Provides fail-closed validation, receipt adaptation, per-marker and phase-channel
breakdowns, and integration with the canonical physical acceptance evaluator
(acceptance.py) for Simscape continuous replays under MATLAB R2025b.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from src.shared.python.contracts import postcondition, precondition, require
from src.shared.python.motion_matching.acceptance import (
    AcceptanceGates,
    AcceptanceVerdict,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.replay_metrics import (
    ReplayFiveMetrics,
    compute_replay_five_metrics,
    is_club_marker_label,
)
from src.shared.python.motion_matching.full_marker_terminal import (
    compute_terminal_marker_breakdown,
)

REPLAY_HARNESS_SCHEMA_VERSION = "simscape-continuous-replay-receipt/1"
REQUIRED_MATLAB_RELEASE = "2025b"


class SimscapeContinuousReplayError(ValueError):
    """Raised when a Simscape continuous-replay contract or sanity check fails."""


class ContinuousReplayProfile(str, Enum):
    """Supported execution profiles for continuous Simscape replay."""

    PURE_TORQUE = "pure_torque"
    SUPPORTED_TRACKING = "supported_tracking"
    KINEMATIC_REPLAY = "kinematic_replay"


@dataclass(frozen=True)
class ContinuousReplayTrajectory:
    """Immutable multi-channel replay trajectory arrays."""

    time_s: np.ndarray
    q: np.ndarray
    v: np.ndarray
    markers_m: np.ndarray
    target_m: np.ndarray
    valid: np.ndarray
    marker_labels: tuple[str, ...]
    marker_bodies: tuple[str, ...]
    coordinate_names: tuple[str, ...]
    tau: np.ndarray | None = None

    def __post_init__(self) -> None:
        require(isinstance(self.time_s, np.ndarray), "time_s must be np.ndarray")
        require(isinstance(self.q, np.ndarray), "q must be np.ndarray")
        require(isinstance(self.v, np.ndarray), "v must be np.ndarray")
        require(isinstance(self.markers_m, np.ndarray), "markers_m must be np.ndarray")
        require(isinstance(self.target_m, np.ndarray), "target_m must be np.ndarray")
        require(isinstance(self.valid, np.ndarray), "valid must be np.ndarray")


def load_continuous_replay_trajectory(
    npz_path: Path | str,
    candidate_doc: Mapping[str, Any] | None = None,
) -> ContinuousReplayTrajectory:
    """Load ContinuousReplayTrajectory from an npz archive and optional candidate doc."""
    p = Path(npz_path)
    require(p.is_file(), f"Replay NPZ not found: {p}", p)
    with np.load(p, allow_pickle=False) as raw:
        time_s = np.asarray(raw["time_s"], dtype=np.float64)
        if "q" in raw and "v" in raw:
            q = np.asarray(raw["q"], dtype=np.float64)
            v = np.asarray(raw["v"], dtype=np.float64)
        elif "native_state" in raw:
            native_state = np.asarray(raw["native_state"], dtype=np.float64)
            nq = native_state.shape[1] // 2
            q = native_state[:, :nq]
            v = native_state[:, nq:]
        else:
            raise SimscapeContinuousReplayError(
                "Replay archive missing q/v or native_state"
            )

        markers_m = np.asarray(raw["markers_m"], dtype=np.float64)
        target_m = np.asarray(raw["target_m"], dtype=np.float64)
        valid = (
            np.asarray(raw["valid"], dtype=bool)
            if "valid" in raw
            else np.ones((len(time_s), markers_m.shape[1]), dtype=bool)
        )
        tau = np.asarray(raw["tau"], dtype=np.float64) if "tau" in raw else None

    if candidate_doc:
        coords = tuple(str(x) for x in candidate_doc.get("coordinate_names", ()))
        labels = tuple(str(x) for x in candidate_doc.get("marker_labels", ()))
        bodies = tuple(str(x) for x in candidate_doc.get("marker_bodies", ()))
    else:
        coords = tuple(f"q_{i}" for i in range(q.shape[1]))
        labels = tuple(f"marker_{i}" for i in range(markers_m.shape[1]))
        bodies = tuple("unknown" for _ in range(markers_m.shape[1]))

    return ContinuousReplayTrajectory(
        time_s=time_s,
        q=q,
        v=v,
        markers_m=markers_m,
        target_m=target_m,
        valid=valid,
        marker_labels=labels,
        marker_bodies=bodies,
        coordinate_names=coords,
        tau=tau,
    )


@dataclass(frozen=True)
class ContinuousReplayControlIdentity:
    """Actuation, controller mode, and continuity identity for replay verification."""

    controller_mode: str = "pure_torque"
    prescribed_coordinates: tuple[str, ...] = ()
    has_root_assistance: bool = False
    root_assistance_n_m: float = 0.0
    has_state_resets: bool = False
    state_resets_count: int = 0


@dataclass(frozen=True)
class ContinuousReplayProvenance:
    """Provenance metadata identifying host, MATLAB release, SHAs, and clocks.

    ``wall_clock_s`` is the declared batch wall clock; ``solver_clock_s`` must
    be the recorded native solver/simulation clock, never a wall-clock copy.
    """

    run_id: str
    matlab_release: str
    matlab_version: str
    host: str
    model_sha256: str
    candidate_sha256: str
    replay_npz_sha256: str
    wall_clock_s: float
    solver_clock_s: float
    q0: np.ndarray | None = None
    v0: np.ndarray | None = None


@dataclass(frozen=True)
class ReplayEvidenceInputs:
    """Typed qualification inputs derived from a native replay evidence package."""

    trajectory: ContinuousReplayTrajectory
    control: ContinuousReplayControlIdentity
    provenance: ContinuousReplayProvenance
    profile: ContinuousReplayProfile
    horizon: Horizon


REQUIRED_CONTROL_IDENTITY_KEYS: tuple[str, ...] = (
    "controller_mode",
    "prescribed_coordinates",
    "has_root_assistance",
    "root_assistance_n_m",
    "has_state_resets",
    "state_resets_count",
)


def _sha256_file(path: Path) -> str:
    """Return the lowercase SHA-256 digest of a file's bytes."""
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_replay_evidence_inputs(
    evidence_dir: Path | str,
    *,
    horizon: Horizon = Horizon.G1,
) -> ReplayEvidenceInputs:
    """Derive qualification inputs from a native Simscape replay evidence package.

    Fail-closed derivation (no locally asserted defaults):
    - the replay NPZ digest is recomputed from file bytes and must match the
      run manifest's declared ``replay_npz_sha256``;
    - initial conditions come from ``returned-candidate.json`` (``q0``/``qd0``);
    - the control identity and the solver clock come from the R2025b qualified
      replay receipt (``qualified_candidate_replay.json``), which must declare
      its ``control_identity`` block.
    """
    evidence = Path(evidence_dir)
    npz_path = evidence / "returned-replay.npz"
    cand_path = evidence / "returned-candidate.json"
    manifest_path = evidence / "run_manifest.json"
    report_path = evidence / "qualified_candidate_replay.json"
    for required, what in (
        (npz_path, "replay NPZ"),
        (cand_path, "candidate JSON"),
        (manifest_path, "run manifest"),
        (report_path, "qualified replay receipt"),
    ):
        require(
            required.is_file(),
            f"evidence package missing {what}: {required}",
            required,
        )

    cand_doc = json.loads(cand_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    report = json.loads(report_path.read_text(encoding="utf-8"))

    computed_digest = _sha256_file(npz_path)
    declared_digest = str(manifest["replay_npz_sha256"]).lower()
    if computed_digest != declared_digest:
        raise SimscapeContinuousReplayError(
            "returned-replay.npz content-address verification failed: recomputed "
            f"{computed_digest[:16]}... != manifest {declared_digest[:16]}...; "
            "refusing stale or altered replay evidence"
        )

    control_doc = report.get("control_identity")
    if not isinstance(control_doc, Mapping):
        raise SimscapeContinuousReplayError(
            "native qualified replay receipt declares no control identity; "
            "continuous qualification cannot verify actuation/reset identity"
        )
    missing_keys = [
        str(k) for k in REQUIRED_CONTROL_IDENTITY_KEYS if k not in control_doc
    ]
    if missing_keys:
        raise SimscapeContinuousReplayError(
            f"control identity disclosure missing fields: {missing_keys}"
        )

    control = ContinuousReplayControlIdentity(
        controller_mode=str(control_doc["controller_mode"]),
        prescribed_coordinates=tuple(
            str(x) for x in control_doc["prescribed_coordinates"]
        ),
        has_root_assistance=bool(control_doc["has_root_assistance"]),
        root_assistance_n_m=float(control_doc["root_assistance_n_m"]),
        has_state_resets=bool(control_doc["has_state_resets"]),
        state_resets_count=int(control_doc["state_resets_count"]),
    )
    profile_key = str(control_doc.get("profile", control.controller_mode))
    try:
        profile = ContinuousReplayProfile(profile_key)
    except ValueError as exc:
        raise SimscapeContinuousReplayError(
            f"unsupported continuous replay profile declared in receipt: {profile_key!r}"
        ) from exc

    trajectory = load_continuous_replay_trajectory(npz_path, cand_doc)
    provenance = ContinuousReplayProvenance(
        run_id=str(manifest["run_id"]),
        matlab_release=str(manifest["matlab_release"]),
        matlab_version=str(manifest["matlab_version"]),
        host=str(manifest["host"]),
        model_sha256=str(manifest["model_sha256"]),
        candidate_sha256=str(manifest["candidate_sha256"]),
        replay_npz_sha256=computed_digest,
        wall_clock_s=float(manifest["wall_clock_s"]),
        solver_clock_s=float(report["elapsed_s"]),
        q0=np.asarray(cand_doc["q0"], dtype=np.float64),
        v0=np.asarray(cand_doc["qd0"], dtype=np.float64),
    )
    return ReplayEvidenceInputs(
        trajectory=trajectory,
        control=control,
        provenance=provenance,
        profile=profile,
        horizon=horizon,
    )


@dataclass(frozen=True)
class SimscapeContinuousReplayReceipt:
    """Canonical qualification receipt for a continuous Simscape replay."""

    schema_version: str
    run_id: str
    profile: ContinuousReplayProfile
    is_uninterrupted: bool
    is_qualified: bool
    verdict: AcceptanceVerdict
    per_marker_rms_m: dict[str, float]
    phase_rms_m: dict[str, float]
    terminal_breakdown: dict[str, float | None]
    receipt_dict: dict[str, Any]
    provenance: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": self.schema_version,
            "run_id": self.run_id,
            "profile": self.profile.value,
            "is_uninterrupted": self.is_uninterrupted,
            "is_qualified": self.is_qualified,
            "verdict": self.verdict.as_dict(),
            "per_marker_rms_m": dict(self.per_marker_rms_m),
            "phase_rms_m": dict(self.phase_rms_m),
            "terminal_breakdown": dict(self.terminal_breakdown),
            "receipt_dict": dict(self.receipt_dict),
            "provenance": dict(self.provenance),
        }


def _validate_matlab_release(release: str) -> None:
    norm = str(release).strip().lower().lstrip("r")
    if norm != REQUIRED_MATLAB_RELEASE:
        raise SimscapeContinuousReplayError(
            f"Simscape continuous replay requires MATLAB R2025b; got {release!r} "
            "(no R2026a substitution)"
        )


def _validate_finite_arrays(traj: ContinuousReplayTrajectory) -> None:
    for name, arr in (
        ("time_s", traj.time_s),
        ("q", traj.q),
        ("v", traj.v),
        ("markers_m", traj.markers_m),
        ("target_m", traj.target_m),
    ):
        if not np.all(np.isfinite(arr)):
            raise SimscapeContinuousReplayError(
                f"Non-finite values (NaN/Inf) detected in replay array {name!r}"
            )


def _validate_timestamps_and_duration(
    time_s: np.ndarray,
    horizon: Horizon,
) -> None:
    if len(time_s) < 2:
        raise SimscapeContinuousReplayError(
            "Replay time vector has fewer than 2 frames"
        )
    diffs = np.diff(time_s)
    if np.any(diffs <= 0.0):
        raise SimscapeContinuousReplayError(
            "Duplicated or retrograde timestamps detected in replay time vector"
        )

    start_s = float(time_s[0])
    end_s = float(time_s[-1])
    elapsed_s = end_s - start_s
    min_duration = (
        0.85 if horizon == Horizon.G1 else (1.20 if horizon == Horizon.G2 else 1.80)
    )
    if abs(start_s) > 1e-4:
        raise SimscapeContinuousReplayError(
            "Replay trajectory must start at t=0 to cover the required horizon; "
            f"got start {start_s:.6f} s"
        )
    if elapsed_s < (min_duration - 1e-4):
        raise SimscapeContinuousReplayError(
            "Missing terminal samples: replay elapsed duration "
            f"{elapsed_s:.3f} s (t={start_s:.3f}..{end_s:.3f} s) is below the "
            f"required {min_duration:.3f} s for horizon {horizon.value}"
        )


def _validate_control_and_continuity(
    control: ContinuousReplayControlIdentity,
    profile: ContinuousReplayProfile,
) -> None:
    if control.has_state_resets or control.state_resets_count > 0:
        raise SimscapeContinuousReplayError(
            "Replay has state resets; incompatible with uninterrupted replay"
        )

    if profile == ContinuousReplayProfile.PURE_TORQUE:
        if (
            control.prescribed_coordinates
            or control.has_root_assistance
            or control.root_assistance_n_m > 0.0
            or control.controller_mode != "pure_torque"
        ):
            raise SimscapeContinuousReplayError(
                "Hidden motion prescription or root assistance detected in pure torque profile"
            )


def _validate_provenance_and_initial_conditions(
    traj: ContinuousReplayTrajectory,
    provenance: ContinuousReplayProvenance,
    expected_candidate_sha256: str | None,
) -> None:
    if expected_candidate_sha256 is not None:
        if provenance.candidate_sha256.lower() != expected_candidate_sha256.lower():
            raise SimscapeContinuousReplayError(
                f"Candidate hash mismatch: declared {provenance.candidate_sha256[:8]} "
                f"!= expected {expected_candidate_sha256[:8]}"
            )

    if provenance.q0 is not None:
        if not np.allclose(traj.q[0], provenance.q0, atol=1e-4):
            raise SimscapeContinuousReplayError(
                "Initial coordinates q[0] do not match declared q0 initial conditions"
            )

    if provenance.v0 is not None:
        if not np.allclose(traj.v[0], provenance.v0, atol=1e-4):
            raise SimscapeContinuousReplayError(
                "Initial velocities v[0] do not match declared v0 initial conditions"
            )

    if (
        provenance.wall_clock_s < 0.0
        or provenance.solver_clock_s < 0.0
        or not np.isfinite(provenance.wall_clock_s)
        or not np.isfinite(provenance.solver_clock_s)
    ):
        raise SimscapeContinuousReplayError(
            "Actual solver clock must be non-negative and finite"
        )


def validate_continuous_replay_inputs(
    *,
    trajectory: ContinuousReplayTrajectory,
    control: ContinuousReplayControlIdentity,
    provenance: ContinuousReplayProvenance,
    profile: ContinuousReplayProfile,
    horizon: Horizon,
    expected_candidate_sha256: str | None = None,
) -> None:
    """Validate replay contracts; fail closed on corruption, wrong release, or reset."""
    _validate_matlab_release(provenance.matlab_release)
    _validate_finite_arrays(trajectory)
    _validate_timestamps_and_duration(trajectory.time_s, horizon)
    _validate_control_and_continuity(control, profile)
    _validate_provenance_and_initial_conditions(
        trajectory, provenance, expected_candidate_sha256
    )


def compute_per_marker_channels(
    pred_markers_m: np.ndarray,
    target_markers_m: np.ndarray,
    valid: np.ndarray,
    marker_labels: Sequence[str],
) -> dict[str, float]:
    """Compute root-mean-square marker tracking error per marker channel."""
    per_marker: dict[str, float] = {}
    diffs = pred_markers_m - target_markers_m  # (T, M, 3)
    sq_err = np.sum(diffs**2, axis=-1)  # (T, M)
    for m_idx, label in enumerate(marker_labels):
        m_valid = valid[:, m_idx]
        if np.any(m_valid):
            rms = float(np.sqrt(np.mean(sq_err[m_valid, m_idx])))
        else:
            rms = 0.0
        per_marker[label] = rms
    return per_marker


def compute_phase_channels(
    time_s: np.ndarray,
    pred_markers_m: np.ndarray,
    target_markers_m: np.ndarray,
    valid: np.ndarray,
) -> dict[str, float]:
    """Compute marker RMS error broken down across standard swing phases."""
    diffs = pred_markers_m - target_markers_m
    sq_err = np.sum(diffs**2, axis=-1)  # (T, M)

    # Standard phase boundaries
    phases = {
        "address": (time_s >= 0.0) & (time_s <= 0.35),
        "backswing": (time_s > 0.35) & (time_s <= 0.85),
        "downswing": (time_s > 0.85) & (time_s <= 1.20),
        "follow_through": time_s > 1.20,
    }
    phase_rms: dict[str, float] = {}
    for name, mask in phases.items():
        if not np.any(mask):
            continue
        p_sq = sq_err[mask]
        p_valid = valid[mask]
        if np.any(p_valid):
            phase_rms[name] = float(np.sqrt(np.mean(p_sq[p_valid])))
        else:
            phase_rms[name] = 0.0
    return phase_rms


def _compute_horizon_metrics(
    time_s: np.ndarray,
    pred_markers_m: np.ndarray,
    target_markers_m: np.ndarray,
    valid: np.ndarray,
    marker_labels: Sequence[str],
    horizon: Horizon,
) -> ReplayFiveMetrics:
    """Compute canonical acceptance metrics over the horizon frames.

    Delegates entirely to compute_replay_five_metrics() so the early window
    remains the canonical ``time_s <= 0.60 s`` definition and club markers are
    recognized through the canonical club-cluster label mapping.
    """
    max_t = 0.85 if horizon == Horizon.G1 else (1.20 if horizon == Horizon.G2 else 1.85)
    mask = time_s <= (max_t + 1e-4)
    if not np.any(mask):
        raise SimscapeContinuousReplayError(
            "No trajectory frames fall inside the evaluation horizon"
        )
    return compute_replay_five_metrics(
        time_s=time_s[mask],
        pred_markers_m=pred_markers_m[mask],
        target_markers_m=target_markers_m[mask],
        valid=valid[mask],
        marker_labels=marker_labels,
    )


def _build_receipt_dict(
    trajectory: ContinuousReplayTrajectory,
    control: ContinuousReplayControlIdentity,
    provenance: ContinuousReplayProvenance,
    profile: ContinuousReplayProfile,
    horizon: Horizon,
    breakdown_dict: dict[str, float | None],
    five_metrics: ReplayFiveMetrics,
) -> dict[str, Any]:
    duration_s = float(trajectory.time_s[-1] - trajectory.time_s[0])
    has_club_markers = any(
        is_club_marker_label(label) for label in trajectory.marker_labels
    )
    has_pelvis_markers = (
        "WaistLeft" in trajectory.marker_labels
        and "WaistRight" in trajectory.marker_labels
    )
    receipt_dict: dict[str, Any] = {
        "engine": "simscape",
        "run_id": provenance.run_id,
        "lane": "native",
        "horizon": horizon.value,
        "model_profile": "reduced_27_no_neck",
        "capture": "driver",
        "duration_s": duration_s,
        "whole_marker_rmse_m": five_metrics.whole_rms_m,
        "early_marker_rmse_m": five_metrics.early_rms_m,
        "terminal_marker_rmse_m": five_metrics.terminal_rms_m,
        "terminal_breakdown": breakdown_dict,
        "acceptance_terminal_source": "full_marker",
        "controller_mode": control.controller_mode,
        "wall_clock_s": provenance.wall_clock_s,
        "solver_clock_s": provenance.solver_clock_s,
        "is_uninterrupted": not control.has_state_resets,
        "profile": profile.value,
        "matlab_release": provenance.matlab_release,
        "model_sha256": provenance.model_sha256,
        "candidate_sha256": provenance.candidate_sha256,
    }
    if has_club_markers:
        receipt_dict["club_marker_rmse_m"] = five_metrics.club_cluster_rms_m
    if has_pelvis_markers:
        receipt_dict["pelvis_yaw_error_pct"] = five_metrics.pelvis_yaw_error_pct
    # Physical contact quantities cannot be derived from a marker replay
    # package; they stay unmeasured instead of being reported as passing.
    receipt_dict["unavailable_physical_quantities"] = {
        "max_normal_force_n": "no contact-force audit in marker replay package",
        "max_penetration_m": "no ground-penetration measurement in marker replay package",
        "max_closure_residual_m": "no closure residual measurement in marker replay package",
    }
    return receipt_dict


@precondition(
    lambda trajectory, control, provenance, profile=ContinuousReplayProfile.PURE_TORQUE, horizon=Horizon.G1, gates=None, expected_candidate_sha256=None: (
        isinstance(trajectory, ContinuousReplayTrajectory)
        and isinstance(control, ContinuousReplayControlIdentity)
        and isinstance(provenance, ContinuousReplayProvenance)
        and isinstance(profile, ContinuousReplayProfile)
        and isinstance(horizon, Horizon)
    ),
    "qualification inputs must be typed",
)
@postcondition(
    lambda receipt: isinstance(receipt, SimscapeContinuousReplayReceipt),
    "qualification receipt must be a SimscapeContinuousReplayReceipt",
)
def qualify_simscape_continuous_replay(
    *,
    trajectory: ContinuousReplayTrajectory,
    control: ContinuousReplayControlIdentity,
    provenance: ContinuousReplayProvenance,
    profile: ContinuousReplayProfile = ContinuousReplayProfile.PURE_TORQUE,
    horizon: Horizon = Horizon.G1,
    gates: AcceptanceGates | None = None,
    expected_candidate_sha256: str | None = None,
) -> SimscapeContinuousReplayReceipt:
    """Run full continuous replay qualification harness and produce canonical receipt."""
    validate_continuous_replay_inputs(
        trajectory=trajectory,
        control=control,
        provenance=provenance,
        profile=profile,
        horizon=horizon,
        expected_candidate_sha256=expected_candidate_sha256,
    )

    per_marker = compute_per_marker_channels(
        trajectory.markers_m,
        trajectory.target_m,
        trajectory.valid,
        trajectory.marker_labels,
    )
    phases = compute_phase_channels(
        trajectory.time_s,
        trajectory.markers_m,
        trajectory.target_m,
        trajectory.valid,
    )

    breakdown = compute_terminal_marker_breakdown(
        pred_markers_m=trajectory.markers_m,
        target_markers_m=trajectory.target_m,
        valid=trajectory.valid,
        marker_labels=trajectory.marker_labels,
        marker_bodies=trajectory.marker_bodies,
    )
    breakdown_dict = breakdown.as_dict()

    whole_metrics = _compute_horizon_metrics(
        trajectory.time_s,
        trajectory.markers_m,
        trajectory.target_m,
        trajectory.valid,
        trajectory.marker_labels,
        horizon,
    )

    receipt_dict = _build_receipt_dict(
        trajectory,
        control,
        provenance,
        profile,
        horizon,
        breakdown_dict,
        whole_metrics,
    )

    verdict = evaluate(receipt_dict, horizon=horizon, gates=gates)
    is_uninterrupted = not control.has_state_resets
    is_qualified = verdict.is_physically_accepted and is_uninterrupted

    prov_dict = {
        "run_id": provenance.run_id,
        "matlab_release": provenance.matlab_release,
        "matlab_version": provenance.matlab_version,
        "host": provenance.host,
        "model_sha256": provenance.model_sha256,
        "candidate_sha256": provenance.candidate_sha256,
        "replay_npz_sha256": provenance.replay_npz_sha256,
        "wall_clock_s": provenance.wall_clock_s,
        "solver_clock_s": provenance.solver_clock_s,
    }

    return SimscapeContinuousReplayReceipt(
        schema_version=REPLAY_HARNESS_SCHEMA_VERSION,
        run_id=provenance.run_id,
        profile=profile,
        is_uninterrupted=is_uninterrupted,
        is_qualified=is_qualified,
        verdict=verdict,
        per_marker_rms_m=per_marker,
        phase_rms_m=phases,
        terminal_breakdown=breakdown_dict,
        receipt_dict=receipt_dict,
        provenance=prov_dict,
    )


def save_replay_qualification_receipt(
    receipt: SimscapeContinuousReplayReceipt,
    path: Path,
) -> None:
    """Save qualification receipt to a JSON artifact."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(receipt.as_dict(), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_replay_qualification_receipt(path: Path) -> SimscapeContinuousReplayReceipt:
    """Load qualification receipt from a JSON artifact."""
    require(path.is_file(), f"receipt file not found: {path}", path)
    data = json.loads(path.read_text(encoding="utf-8"))
    require(isinstance(data, Mapping), "receipt must be a JSON object", data)

    from src.shared.python.motion_matching.acceptance import (
        GateResult,
        GateStatus,
    )

    v_data = data["verdict"]
    gates = tuple(
        GateResult(
            name=g["name"],
            status=GateStatus(g["status"]),
            threshold=float(g["threshold"]),
            measured=float(g["measured"]) if g.get("measured") is not None else None,
            unit=g.get("unit", "m"),
            reason=g.get("reason", ""),
        )
        for g in v_data.get("gates", ())
    )
    verdict = AcceptanceVerdict(
        horizon=Horizon(v_data["horizon"]),
        is_physically_accepted=bool(v_data["is_physically_accepted"]),
        status=str(v_data["status"]),
        gates=gates,
        qualification_note=str(v_data.get("qualification_note", "")),
    )

    return SimscapeContinuousReplayReceipt(
        schema_version=str(data["schema_version"]),
        run_id=str(data["run_id"]),
        profile=ContinuousReplayProfile(data["profile"]),
        is_uninterrupted=bool(data["is_uninterrupted"]),
        is_qualified=bool(data["is_qualified"]),
        verdict=verdict,
        per_marker_rms_m={
            str(k): float(v) for k, v in data["per_marker_rms_m"].items()
        },
        phase_rms_m={str(k): float(v) for k, v in data["phase_rms_m"].items()},
        terminal_breakdown={
            str(k): (float(v) if v is not None else None)
            for k, v in data["terminal_breakdown"].items()
        },
        receipt_dict=dict(data.get("receipt_dict", {})),
        provenance=dict(data.get("provenance", {})),
    )

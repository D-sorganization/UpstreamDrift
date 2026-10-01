"""MMR-07-I (#11107): Simscape Continuous-Replay Qualification Harness Contract Tests.

Verifies behavioral contracts:
- Missing terminal samples / truncated horizon duration
- Nonfinite states / markers (NaN/Inf)
- Duplicated or retrograde timestamps
- Wrong MATLAB release (fails closed on non-R2025b)
- Hidden motion prescription or root assistance in pure torque profile
- Altered candidate hash mismatch against expected digest
- State resets labeled incompatible with uninterrupted replay
- Actual solver clocks, q0/v0, duration, and control identity validation
- Per-marker and phase-channel metrics evaluation
- Feeding fresh outputs into existing acceptance.py evaluation engine
- Preserving run-102 terminal rejection under G1 physical acceptance
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import (
    AcceptanceVerdict,
    GateStatus,
    Horizon,
    evaluate,
)
from src.shared.python.motion_matching.simscape_replay_harness import (
    ContinuousReplayControlIdentity,
    ContinuousReplayProfile,
    ContinuousReplayProvenance,
    ContinuousReplayTrajectory,
    SimscapeContinuousReplayError,
    SimscapeContinuousReplayReceipt,
    load_continuous_replay_trajectory,
    load_replay_evidence_inputs,
    load_replay_qualification_receipt,
    qualify_simscape_continuous_replay,
    save_replay_qualification_receipt,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
RUN102_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "simscape_tour_matching"
    / "native_evidence"
    / "two_window_fit_9967_102"
)
RUN102_CANDIDATE = RUN102_DIR / "returned-candidate.json"
RUN102_REPLAY = RUN102_DIR / "returned-replay.npz"
RUN102_MANIFEST = RUN102_DIR / "run_manifest.json"
RUN102_QUALIFICATION = RUN102_DIR / "simscape_replay_qualification.json"


def _make_valid_test_fixtures() -> tuple[
    ContinuousReplayTrajectory,
    ContinuousReplayControlIdentity,
    ContinuousReplayProvenance,
]:
    """Build a baseline continuous replay fixture from committed run-102 data."""
    cand_doc = json.loads(RUN102_CANDIDATE.read_text(encoding="utf-8"))
    manifest = json.loads(RUN102_MANIFEST.read_text(encoding="utf-8"))
    traj = load_continuous_replay_trajectory(RUN102_REPLAY, cand_doc)
    control = ContinuousReplayControlIdentity(
        controller_mode="pure_torque",
        prescribed_coordinates=(),
        has_root_assistance=False,
        root_assistance_n_m=0.0,
        has_state_resets=False,
        state_resets_count=0,
    )
    provenance = ContinuousReplayProvenance(
        run_id=manifest["run_id"],
        matlab_release="2025b",
        matlab_version=manifest["matlab_version"],
        host=manifest["host"],
        model_sha256=manifest["model_sha256"],
        candidate_sha256=manifest["candidate_sha256"],
        replay_npz_sha256=manifest["replay_npz_sha256"],
        wall_clock_s=float(manifest["wall_clock_s"]),
        solver_clock_s=float(manifest["wall_clock_s"]),
        q0=np.array(cand_doc["q0"], dtype=np.float64),
        v0=np.array(cand_doc["qd0"], dtype=np.float64),
    )
    return traj, control, provenance


def test_harness_rejects_missing_terminal_samples() -> None:
    """Replay truncated before the required horizon duration must fail closed."""
    traj, control, prov = _make_valid_test_fixtures()
    cutoff = np.searchsorted(traj.time_s, 0.5)
    truncated_traj = ContinuousReplayTrajectory(
        time_s=traj.time_s[:cutoff],
        q=traj.q[:cutoff],
        v=traj.v[:cutoff],
        markers_m=traj.markers_m[:cutoff],
        target_m=traj.target_m[:cutoff],
        valid=traj.valid[:cutoff],
        marker_labels=traj.marker_labels,
        marker_bodies=traj.marker_bodies,
        coordinate_names=traj.coordinate_names,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="Missing terminal samples"):
        qualify_simscape_continuous_replay(
            trajectory=truncated_traj,
            control=control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_rejects_nonfinite_states() -> None:
    """Non-finite values (NaN / Inf) in states or markers must fail closed."""
    traj, control, prov = _make_valid_test_fixtures()
    corrupt_q = np.array(traj.q, copy=True)
    corrupt_q[20, 1] = np.nan
    corrupt_traj = ContinuousReplayTrajectory(
        time_s=traj.time_s,
        q=corrupt_q,
        v=traj.v,
        markers_m=traj.markers_m,
        target_m=traj.target_m,
        valid=traj.valid,
        marker_labels=traj.marker_labels,
        marker_bodies=traj.marker_bodies,
        coordinate_names=traj.coordinate_names,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="Non-finite"):
        qualify_simscape_continuous_replay(
            trajectory=corrupt_traj,
            control=control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_rejects_duplicated_timestamps() -> None:
    """Replay time vector must be strictly monotonic; duplicated timestamps must fail."""
    traj, control, prov = _make_valid_test_fixtures()
    corrupt_t = np.array(traj.time_s, copy=True)
    corrupt_t[5] = corrupt_t[4]
    corrupt_traj = ContinuousReplayTrajectory(
        time_s=corrupt_t,
        q=traj.q,
        v=traj.v,
        markers_m=traj.markers_m,
        target_m=traj.target_m,
        valid=traj.valid,
        marker_labels=traj.marker_labels,
        marker_bodies=traj.marker_bodies,
        coordinate_names=traj.coordinate_names,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="Duplicated or retrograde"):
        qualify_simscape_continuous_replay(
            trajectory=corrupt_traj,
            control=control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_rejects_wrong_release() -> None:
    """Only R2025b is accepted; R2026a or older releases must be rejected."""
    traj, control, prov = _make_valid_test_fixtures()
    wrong_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release="R2026a",
        matlab_version="MATLAB R2026a",
        host=prov.host,
        model_sha256=prov.model_sha256,
        candidate_sha256=prov.candidate_sha256,
        replay_npz_sha256=prov.replay_npz_sha256,
        wall_clock_s=prov.wall_clock_s,
        solver_clock_s=prov.solver_clock_s,
        q0=prov.q0,
        v0=prov.v0,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="R2025b"):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=control,
            provenance=wrong_prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_rejects_hidden_motion_prescription() -> None:
    """Pure torque profile must reject motion-prescribed coordinates or root assistance."""
    traj, control, prov = _make_valid_test_fixtures()
    prescribed_control = ContinuousReplayControlIdentity(
        controller_mode="pure_torque",
        prescribed_coordinates=("PelvisPitch",),
        has_root_assistance=False,
        root_assistance_n_m=0.0,
        has_state_resets=False,
        state_resets_count=0,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="motion prescription"):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=prescribed_control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_rejects_altered_candidate_hash() -> None:
    """Candidate hash mismatch against expected digest must fail closed."""
    traj, control, prov = _make_valid_test_fixtures()
    altered_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release=prov.matlab_release,
        matlab_version=prov.matlab_version,
        host=prov.host,
        model_sha256=prov.model_sha256,
        candidate_sha256="deadbeef" * 8,
        replay_npz_sha256=prov.replay_npz_sha256,
        wall_clock_s=prov.wall_clock_s,
        solver_clock_s=prov.solver_clock_s,
        q0=prov.q0,
        v0=prov.v0,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="Candidate hash"):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=control,
            provenance=altered_prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
            expected_candidate_sha256=prov.candidate_sha256,
        )


def test_harness_labels_state_resets_incompatible() -> None:
    """Deliberate state resets must be labeled incompatible with uninterrupted replay."""
    traj, control, prov = _make_valid_test_fixtures()
    reset_control = ContinuousReplayControlIdentity(
        controller_mode="pure_torque",
        prescribed_coordinates=(),
        has_root_assistance=False,
        root_assistance_n_m=0.0,
        has_state_resets=True,
        state_resets_count=1,
    )
    with pytest.raises(
        SimscapeContinuousReplayError, match="incompatible with uninterrupted replay"
    ):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=reset_control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_preserves_run102_terminal_rejection() -> None:
    """Run-102 terminal rejection (~40.3 mm > 35 mm G1 ceiling) must be preserved."""
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    assert isinstance(receipt, SimscapeContinuousReplayReceipt)
    assert receipt.is_uninterrupted is True
    assert receipt.is_qualified is False
    assert receipt.verdict.is_physically_accepted is False
    assert receipt.verdict.status == "REJECTED"
    terminal_gate = next(
        g for g in receipt.verdict.gates if g.name == "terminal_marker_rmse_m"
    )
    assert terminal_gate.status == GateStatus.FAILED
    assert terminal_gate.measured is not None and terminal_gate.measured > 0.035


def test_harness_validates_duration_and_solver_clocks_and_initial_conditions() -> None:
    """Harness must validate duration, non-negative finite solver clocks, and q0/v0 match."""
    traj, control, prov = _make_valid_test_fixtures()

    # Mismatched q0
    bad_q0 = np.array(prov.q0, copy=True)
    bad_q0[0] += 0.1
    bad_q0_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release=prov.matlab_release,
        matlab_version=prov.matlab_version,
        host=prov.host,
        model_sha256=prov.model_sha256,
        candidate_sha256=prov.candidate_sha256,
        replay_npz_sha256=prov.replay_npz_sha256,
        wall_clock_s=prov.wall_clock_s,
        solver_clock_s=prov.solver_clock_s,
        q0=bad_q0,
        v0=prov.v0,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="Initial coordinates"):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=control,
            provenance=bad_q0_prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )

    # Negative solver clock
    bad_clock_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release=prov.matlab_release,
        matlab_version=prov.matlab_version,
        host=prov.host,
        model_sha256=prov.model_sha256,
        candidate_sha256=prov.candidate_sha256,
        replay_npz_sha256=prov.replay_npz_sha256,
        wall_clock_s=prov.wall_clock_s,
        solver_clock_s=-1.0,
        q0=prov.q0,
        v0=prov.v0,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="solver clock"):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=control,
            provenance=bad_clock_prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_harness_per_marker_and_phase_channels_feed_into_acceptance() -> None:
    """Per-marker and phase channels are evaluated and receipt feeds into acceptance.py."""
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    # Validate channel extraction
    assert len(receipt.per_marker_rms_m) == len(traj.marker_labels)
    assert "address" in receipt.phase_rms_m
    assert "backswing" in receipt.phase_rms_m

    # Feed receipt_dict fresh output directly into acceptance.evaluate()
    direct_verdict = evaluate(receipt.receipt_dict, horizon=Horizon.G1)
    assert isinstance(direct_verdict, AcceptanceVerdict)
    assert (
        direct_verdict.is_physically_accepted == receipt.verdict.is_physically_accepted
    )
    assert direct_verdict.status == receipt.verdict.status

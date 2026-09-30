"""MMR-07-I (#11107): Simscape continuous-replay qualification harness tests.

TDD First Failing Tests:
- Missing terminal samples (truncated duration or missing endpoint frames)
- Nonfinite states (NaN/inf in trajectory states or markers)
- Duplicated timestamps (non-strictly increasing time vector)
- Wrong MATLAB release (fails closed on non-R2025b, e.g. R2026a)
- Hidden motion prescription (prescribed kinematics disguised in pure torque replay)
- Altered candidate hash (candidate_sha256 mismatch against expected payload)
- State-reset fixture labeled incompatible with uninterrupted replay
- q0/v0 initial condition mismatch
- Run-102 terminal rejection preserved under G1 physical acceptance
- Per-marker and phase-channel metrics evaluation
- Qualification receipt save/load roundtrip
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching.acceptance import Horizon
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
from src.shared.python.motion_matching.simscape_topology import HEAD_MARKER_NAMES

pytestmark = pytest.mark.unit

# Canonical native cross-check values for run-102. The committed
# returned-replay.npz reproduces receipt.json `uninterrupted_metrics` exactly;
# qualified_candidate_replay.json records the fresh R2025b prediction values
# (slightly resampled, within the recorded 0.0605 mm cross-engine parity).
RUN102_EARLY_RMS_M = 0.009995357264954529
RUN102_CLUB_RMS_M = 0.008391285435051065
RUN102_PELVIS_YAW_PCT = 0.5954129834382699
RUN102_QUALIFIED_REPORT_YAW_PCT = 0.6099819870175595

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


def test_rejects_missing_terminal_samples() -> None:
    """Replay truncated before the required horizon duration must be rejected."""
    traj, control, prov = _make_valid_test_fixtures()
    # Truncate to 0.5 s (below G1 min duration of 0.85 s)
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


def test_rejects_nonfinite_states() -> None:
    """Non-finite values (NaN / Inf) in states or markers must fail closed."""
    traj, control, prov = _make_valid_test_fixtures()
    corrupt_q = np.array(traj.q, copy=True)
    corrupt_q[50, 2] = np.nan
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


def test_rejects_duplicated_timestamps() -> None:
    """Replay time vector must be strictly monotonic; duplicated timestamps must fail."""
    traj, control, prov = _make_valid_test_fixtures()
    corrupt_t = np.array(traj.time_s, copy=True)
    corrupt_t[10] = corrupt_t[9]  # Duplicate timestamp
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


def test_rejects_wrong_release() -> None:
    """Only R2025b is accepted; R2026a or older releases must be rejected."""
    traj, control, prov = _make_valid_test_fixtures()
    wrong_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release="2026a",
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


def test_rejects_hidden_motion_prescription() -> None:
    """A pure torque profile must reject motion-prescribed coordinates or root assistance."""
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


def test_rejects_altered_candidate_hash() -> None:
    """Candidate hash mismatch against expected digest must fail closed."""
    traj, control, prov = _make_valid_test_fixtures()
    wrong_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release=prov.matlab_release,
        matlab_version=prov.matlab_version,
        host=prov.host,
        model_sha256=prov.model_sha256,
        candidate_sha256="0000000000000000000000000000000000000000000000000000000000000000",
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
            provenance=wrong_prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
            expected_candidate_sha256=prov.candidate_sha256,
        )


def test_state_reset_labeled_incompatible_with_uninterrupted_replay() -> None:
    """Deliberate state-reset fixture must be marked incompatible with uninterrupted replay."""
    traj, control, prov = _make_valid_test_fixtures()
    reset_control = ContinuousReplayControlIdentity(
        controller_mode="pure_torque",
        prescribed_coordinates=(),
        has_root_assistance=False,
        root_assistance_n_m=0.0,
        has_state_resets=True,
        state_resets_count=2,
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


def test_rejects_q0_v0_inconsistency() -> None:
    """Declared initial conditions (q0, v0) must match trajectory start q[0], v[0]."""
    traj, control, prov = _make_valid_test_fixtures()
    wrong_q0 = np.array(prov.q0, copy=True)
    wrong_q0[0] += 0.5  # 50 cm discrepancy
    wrong_prov = ContinuousReplayProvenance(
        run_id=prov.run_id,
        matlab_release=prov.matlab_release,
        matlab_version=prov.matlab_version,
        host=prov.host,
        model_sha256=prov.model_sha256,
        candidate_sha256=prov.candidate_sha256,
        replay_npz_sha256=prov.replay_npz_sha256,
        wall_clock_s=prov.wall_clock_s,
        solver_clock_s=prov.solver_clock_s,
        q0=wrong_q0,
        v0=prov.v0,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="Initial coordinates"):
        qualify_simscape_continuous_replay(
            trajectory=traj,
            control=control,
            provenance=wrong_prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_run102_terminal_rejection_preserved() -> None:
    """Evaluating run-102 under G1 physical acceptance preserves the terminal rejection."""
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
    # Run-102 terminal RMSE is ~40.3 mm, which exceeds the G1 35 mm ceiling
    assert receipt.verdict.is_physically_accepted is False
    assert receipt.verdict.status == "REJECTED"
    assert receipt.is_qualified is False
    full_term = receipt.terminal_breakdown["full_marker_terminal_rms_m"]
    assert full_term is not None and full_term > 0.035
    # The terminal gate in the verdict must be marked FAILED
    terminal_gate = next(
        g for g in receipt.verdict.gates if g.name == "terminal_marker_rmse_m"
    )
    assert terminal_gate.status.value == "failed"


def test_per_marker_and_phase_channel_breakdown() -> None:
    """Harness must report per-marker RMS channels and per-phase RMS breakdowns."""
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    assert "HeadTop" in receipt.per_marker_rms_m
    assert "LWristTop" in receipt.per_marker_rms_m
    assert len(receipt.per_marker_rms_m) == len(traj.marker_labels)
    # Check phases
    assert "address" in receipt.phase_rms_m
    assert "backswing" in receipt.phase_rms_m
    assert all(val >= 0.0 for val in receipt.phase_rms_m.values())


def test_continuous_replay_receipt_roundtrip(tmp_path: Path) -> None:
    """Qualification receipt must serialize to JSON and reload losslessly."""
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    out_file = tmp_path / "simscape_replay_qualification.json"
    save_replay_qualification_receipt(receipt, out_file)
    assert out_file.is_file()

    loaded = load_replay_qualification_receipt(out_file)
    assert loaded.is_qualified == receipt.is_qualified
    assert loaded.is_uninterrupted == receipt.is_uninterrupted
    assert loaded.profile == receipt.profile
    assert loaded.verdict.status == receipt.verdict.status
    assert (
        loaded.verdict.is_physically_accepted == receipt.verdict.is_physically_accepted
    )
    assert loaded.terminal_breakdown == receipt.terminal_breakdown
    assert loaded.phase_rms_m == receipt.phase_rms_m
    assert loaded.per_marker_rms_m == receipt.per_marker_rms_m


# ---------------------------------------------------------------------------
# Review-fix regressions (PR #11126 Codex findings; P1/P2)
# ---------------------------------------------------------------------------


def test_rejects_replay_missing_start_prefix() -> None:
    """A replay containing only 0.50..0.85 s fails although the final timestamp is 0.85 s.

    Codex P1: the duration gate must validate the elapsed horizon span and the
    expected zero start time, not merely the final timestamp.
    """
    traj, control, prov = _make_valid_test_fixtures()
    start_idx = int(np.searchsorted(traj.time_s, 0.50))
    truncated_traj = ContinuousReplayTrajectory(
        time_s=traj.time_s[start_idx:],
        q=traj.q[start_idx:],
        v=traj.v[start_idx:],
        markers_m=traj.markers_m[start_idx:],
        target_m=traj.target_m[start_idx:],
        valid=traj.valid[start_idx:],
        marker_labels=traj.marker_labels,
        marker_bodies=traj.marker_bodies,
        coordinate_names=traj.coordinate_names,
    )
    assert truncated_traj.time_s[-1] >= 0.85  # final timestamp alone looks sufficient
    with pytest.raises(
        SimscapeContinuousReplayError,
        match="Missing terminal samples|start at t=0",
    ):
        qualify_simscape_continuous_replay(
            trajectory=truncated_traj,
            control=control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_rejects_replay_start_time_deviating_from_zero() -> None:
    """A replay shifted to start at t != 0 must be rejected even with full elapsed span."""
    traj, control, prov = _make_valid_test_fixtures()
    shifted_time = traj.time_s + 0.02
    shifted_traj = ContinuousReplayTrajectory(
        time_s=shifted_time,
        q=traj.q,
        v=traj.v,
        markers_m=traj.markers_m,
        target_m=traj.target_m,
        valid=traj.valid,
        marker_labels=traj.marker_labels,
        marker_bodies=traj.marker_bodies,
        coordinate_names=traj.coordinate_names,
    )
    with pytest.raises(SimscapeContinuousReplayError, match="start at t=0"):
        qualify_simscape_continuous_replay(
            trajectory=shifted_traj,
            control=control,
            provenance=prov,
            profile=ContinuousReplayProfile.PURE_TORQUE,
            horizon=Horizon.G1,
        )


def test_run102_early_rms_uses_canonical_060s_window() -> None:
    """Early RMSE must use the canonical t <= 0.60 s window, not the first 30% of the horizon.

    Codex P1: the 30%-of-horizon rule stopped at 0.255 s and under-reported
    run-102's early error (2.58 mm) versus the canonical 9.995 mm.
    """
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    early = float(receipt.receipt_dict["early_marker_rmse_m"])
    assert early == pytest.approx(RUN102_EARLY_RMS_M, abs=1e-9)


def test_run102_club_cluster_recognizes_native_labels() -> None:
    """Club RMSE must use the canonical club-cluster mapping (Marker_2/Marker_3 tags).

    Codex P1: searching labels for 'club' only found zero club markers for the
    committed package, publishing club_marker_rmse_m = 0.0 as a fake pass.
    """
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    club = float(receipt.receipt_dict["club_marker_rmse_m"])
    assert club > 0.0
    assert club == pytest.approx(RUN102_CLUB_RMS_M, abs=1e-9)


def test_receipt_holds_measured_pelvis_yaw_and_omits_unmeasured_gates() -> None:
    """Pelvis yaw must be measured from WaistLeft/WaistRight; unmeasured quantities absent.

    Codex P1: hard-coded yaw, penetration, and closure values were handed to
    acceptance.evaluate() as passing measurements without being computed.
    """
    traj, control, prov = _make_valid_test_fixtures()
    receipt = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    rd = receipt.receipt_dict
    yaw_measured = float(rd["pelvis_yaw_error_pct"])
    assert yaw_measured == pytest.approx(RUN102_PELVIS_YAW_PCT, abs=1e-9)
    # Also stays in the recorded fresh-R2025b prediction band (resampled markers).
    assert yaw_measured == pytest.approx(RUN102_QUALIFIED_REPORT_YAW_PCT, abs=3e-2)
    assert yaw_measured != 0.5  # the fabricated literal must be gone
    for fabricated in (
        "max_normal_force_n",
        "max_penetration_m",
        "max_closure_residual_m",
    ):
        assert fabricated not in rd
    unavailable = rd["unavailable_physical_quantities"]
    assert set(unavailable) == {
        "max_normal_force_n",
        "max_penetration_m",
        "max_closure_residual_m",
    }
    # The pelvis-yaw gate result must carry the measured value, not a literal.
    yaw_gate = next(
        g for g in receipt.verdict.gates if g.name == "pelvis_yaw_error_pct"
    )
    assert yaw_gate.status.value == "passed"
    assert yaw_gate.measured == pytest.approx(yaw_measured, abs=1e-12)


def test_receipt_roundtrip_preserves_null_terminal_channels(tmp_path: Path) -> None:
    """Lossless receipt load must keep explicitly-null (unavailable) terminal channels.

    Codex P2: replays without a head or hub cluster legitimately produce None
    fields; save/load previously dropped the keys entirely.
    """
    traj, control, prov = _make_valid_test_fixtures()
    keep = [
        i
        for i, (label, body) in enumerate(
            zip(traj.marker_labels, traj.marker_bodies, strict=True)
        )
        if label not in HEAD_MARKER_NAMES and body != "Hub"
    ]
    reduced_traj = ContinuousReplayTrajectory(
        time_s=traj.time_s,
        q=traj.q,
        v=traj.v,
        markers_m=traj.markers_m[:, keep, :],
        target_m=traj.target_m[:, keep, :],
        valid=traj.valid[:, keep],
        marker_labels=tuple(traj.marker_labels[i] for i in keep),
        marker_bodies=tuple(traj.marker_bodies[i] for i in keep),
        coordinate_names=traj.coordinate_names,
    )
    receipt = qualify_simscape_continuous_replay(
        trajectory=reduced_traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    assert receipt.terminal_breakdown["head_cluster_terminal_rms_m"] is None
    assert receipt.terminal_breakdown["hub_cluster_terminal_rms_m"] is None

    out_file = tmp_path / "simscape_replay_qualification.json"
    save_replay_qualification_receipt(receipt, out_file)
    loaded = load_replay_qualification_receipt(out_file)
    assert set(loaded.terminal_breakdown) == set(receipt.terminal_breakdown)
    assert loaded.terminal_breakdown["head_cluster_terminal_rms_m"] is None
    assert loaded.terminal_breakdown["hub_cluster_terminal_rms_m"] is None


def _copy_run102_evidence(tmp_path: Path) -> Path:
    """Copy the committed run-102 native evidence into a temporary evidence dir."""
    import shutil

    dest = tmp_path / "run102-copy"
    shutil.copytree(RUN102_DIR, dest)
    for stale in ("simscape_replay_qualification.json",):
        target = dest / stale
        if target.is_file():
            target.unlink()
    return dest


def test_load_replay_evidence_inputs_derives_native_provenance(tmp_path: Path) -> None:
    """Evidence inputs must derive digest, initial conditions, and control identity from
    native evidence instead of asserting local defaults (Codex P1 x3)."""
    imports = load_replay_evidence_inputs(RUN102_DIR)
    manifest = json.loads(RUN102_MANIFEST.read_text(encoding="utf-8"))
    cand_doc = json.loads(RUN102_CANDIDATE.read_text(encoding="utf-8"))

    # Replay digest is recomputed from the actual NPZ bytes, not copied.
    assert imports.provenance.replay_npz_sha256 == manifest["replay_npz_sha256"]
    assert hashlib.sha256(RUN102_REPLAY.read_bytes()).hexdigest() == (
        imports.provenance.replay_npz_sha256
    )
    # Initial conditions come from the candidate declaration (q0 / qd0).
    assert np.array_equal(imports.provenance.q0, np.array(cand_doc["q0"]))
    assert np.array_equal(imports.provenance.v0, np.array(cand_doc["qd0"]))
    # Control identity is read from the native qualified replay receipt.
    assert imports.control.controller_mode == "pure_torque"
    assert imports.control.prescribed_coordinates == ()
    assert imports.control.has_root_assistance is False
    assert imports.control.has_state_resets is False
    assert imports.profile == ContinuousReplayProfile.PURE_TORQUE
    # Solver clock is the recorded native simulation clock, not the wall clock.
    assert imports.provenance.solver_clock_s > 0.0


def test_load_replay_evidence_inputs_fails_closed_on_stale_npz(tmp_path: Path) -> None:
    """A modified replay NPZ must be rejected instead of inheriting the declared digest."""
    dest = _copy_run102_evidence(tmp_path)
    # Tamper: rewrite the cached archive with a different byte payload.
    with np.load(dest / "returned-replay.npz", allow_pickle=False) as raw:
        payload = {k: raw[k] for k in raw.files}
    payload["time_s"] = payload["time_s"][:-1]
    np.savez_compressed(
        dest / "returned-replay.npz",
        time_s=payload.pop("time_s"),
        **payload,
    )
    with pytest.raises(
        SimscapeContinuousReplayError, match="content-address verification"
    ):
        load_replay_evidence_inputs(dest)


def test_load_replay_evidence_inputs_fails_closed_without_control_identity(
    tmp_path: Path,
) -> None:
    """Missing control-identity disclosure in the native receipt must block qualification."""
    dest = _copy_run102_evidence(tmp_path)
    report_path = dest / "qualified_candidate_replay.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report.pop("control_identity", None)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    with pytest.raises(SimscapeContinuousReplayError, match="control identity"):
        load_replay_evidence_inputs(dest)


def test_load_replay_evidence_inputs_rejects_altered_initial_state(
    tmp_path: Path,
) -> None:
    """A replay starting from an altered initial state must fail the declared q0 check.

    The provenance q0/v0 must come from the candidate declaration rather than
    from trajectory frame 0, so an altered start is caught as a mismatch.
    """
    dest = _copy_run102_evidence(tmp_path)
    cand_path = dest / "returned-candidate.json"
    cand_doc = json.loads(cand_path.read_text(encoding="utf-8"))
    cand_doc["q0"] = [v + 0.5 for v in cand_doc["q0"]]
    cand_path.write_text(json.dumps(cand_doc), encoding="utf-8")

    inputs = load_replay_evidence_inputs(dest)
    with pytest.raises(SimscapeContinuousReplayError, match="Initial coordinates"):
        qualify_simscape_continuous_replay(
            trajectory=inputs.trajectory,
            control=inputs.control,
            provenance=inputs.provenance,
            profile=inputs.profile,
            horizon=inputs.horizon,
        )


def test_qualification_from_derived_inputs_matches_direct_path() -> None:
    """Derived native evidence inputs must feed the canonical qualification adapter."""
    traj, control, prov = _make_valid_test_fixtures()
    direct = qualify_simscape_continuous_replay(
        trajectory=traj,
        control=control,
        provenance=prov,
        profile=ContinuousReplayProfile.PURE_TORQUE,
        horizon=Horizon.G1,
    )
    inputs = load_replay_evidence_inputs(RUN102_DIR)
    derived = qualify_simscape_continuous_replay(
        trajectory=inputs.trajectory,
        control=inputs.control,
        provenance=inputs.provenance,
        profile=inputs.profile,
        horizon=inputs.horizon,
    )
    assert derived.receipt_dict == direct.receipt_dict
    assert derived.verdict.status == direct.verdict.status

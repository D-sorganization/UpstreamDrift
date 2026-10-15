"""Unit tests for Phase 5 NSX engine snapshot adapters, Simscape loader, and conformance.

Covers:
- #12579 (NSX-51): MuJoCoClosedLoopSnapshotAdapter hard-KKT evaluation & drift_method provenance
- #12580 (NSX-52): MyoSuiteClosedLoopSnapshotAdapter kkt_regularization capability gating & reuse
- #12581 (NSX-53): DrakeClosedLoopSnapshotAdapter (DrakeForceAdapter) native conversion
- #12582 (NSX-54): OpenSimClosedLoopSnapshotAdapter (OpenSimForceAdapter) delegation & conversion
- #12583 (NSX-55): Simscape R2025b snapshot loader, MATLAB script contract, and DV-12583 record
- #12584 (NSX-56): Two-track cross-engine conformance (Track N & Track S propagated bounds)
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from unittest.mock import patch

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import CapabilityLevel
from src.shared.python.loop_nullspace.conformance import (
    evaluate_track_n,
    evaluate_track_s,
    evaluate_two_track_conformance,
    project_and_build_snapshot,
)
from src.shared.python.loop_nullspace.engine_adapters import (
    DRAKE_CLOSURE_CONVENTION,
    MUJOCO_CLOSURE_CONVENTION,
    OPENSIM_CLOSURE_CONVENTION,
    CapabilityUnavailableError,
    DrakeClosedLoopSnapshotAdapter,
    DrakeForceAdapter,
    MuJoCoClosedLoopSnapshotAdapter,
    MyoSuiteClosedLoopSnapshotAdapter,
    OpenSimClosedLoopSnapshotAdapter,
    OpenSimForceAdapter,
    PinocchioClosedLoopSnapshotAdapter,
    from_canonical_closure,
    snapshot_from_adapter,
)
from src.shared.python.loop_nullspace.simscape_snapshot_loader import (
    MATLAB_EXPORT_SCRIPT_PATH,
    SimscapeClosedLoopSnapshotAdapter,
    load_simscape_closed_loop_snapshot,
)

pytestmark = pytest.mark.unit

FIXTURE_PATH = (
    Path(__file__).resolve().parent
    / "fixtures"
    / "simscape_r2025b_snapshot_sample.json"
)
REPO_ROOT = Path(__file__).resolve().parents[3]


class _ReferenceClosedLoopBackend:
    """Deterministic 8-DOF closed-loop backend in canonical grip convention.

    Used to construct exact native-convention equivalents for Pinocchio, MuJoCo,
    MyoSuite, Drake (ang_lin local), and OpenSim (lin_ang world b-a) so unit
    tests verify every adapter's native-to-canonical conversion and KKT/provenance
    logic without requiring optional C++ bindings.
    """

    def __init__(self) -> None:
        raw = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
        self.coordinate_order: tuple[str, ...] = tuple(
            raw["topology"]["coordinate_order"]
        )
        self.nv = len(self.coordinate_order)
        self.model_sha256: str = str(raw["model_sha256"])
        self.q_ref = np.asarray(raw["state"]["q"], dtype=np.float64)
        self.v_ref = np.asarray(raw["state"]["v"], dtype=np.float64)
        self.a_ref = np.asarray(raw["state"]["a"], dtype=np.float64)
        self.M = np.asarray(raw["dynamics"]["M"], dtype=np.float64)
        self.h = np.asarray(raw["dynamics"]["h"], dtype=np.float64)
        self.J_c = np.asarray(raw["dynamics"]["J_c"], dtype=np.float64)
        self.Jdot_v = np.asarray(raw["dynamics"]["Jdot_v"], dtype=np.float64)
        # Use a non-trivial orthonormal grip frame rotation R_WA and offset point
        angle = 0.35
        c, s = float(np.cos(angle)), float(np.sin(angle))
        self.rot_wa = np.array(
            [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
        )
        self.pos_a = np.array([0.15, -0.22, 0.88], dtype=np.float64)
        self.grip_origin = np.array([0.15, -0.22, 0.88], dtype=np.float64)

    def pose_residual(self, q: np.ndarray) -> np.ndarray:
        return self.J_c @ (np.asarray(q, dtype=np.float64) - self.q_ref)

    def rate_matrix(self, q: np.ndarray) -> np.ndarray:
        del q
        return self.J_c.copy()


def _build_all_adapters_on_reference(
    ref: _ReferenceClosedLoopBackend,
) -> dict[str, Any]:
    """Construct Pinocchio, MuJoCo, MyoSuite, Drake, and OpenSim adapters on ``ref``."""
    pin_adapter = PinocchioClosedLoopSnapshotAdapter.from_reference_backend(ref)

    j_mj, d_mj = from_canonical_closure(
        ref.J_c,
        ref.Jdot_v,
        native=MUJOCO_CLOSURE_CONVENTION,
        grip_frame_rotation=ref.rot_wa,
        grip_origin=ref.grip_origin,
        reference_point=ref.pos_a,
    )
    mj_adapter = MuJoCoClosedLoopSnapshotAdapter.from_native_quantities(
        coordinate_order=ref.coordinate_order,
        model_sha256=ref.model_sha256,
        mass_matrix=ref.M,
        bias_vector=ref.h,
        native_jacobian=j_mj,
        native_drift=d_mj,
        closure_rotation_world=ref.rot_wa,
        closure_point_world=ref.pos_a,
        grip_origin_world=ref.grip_origin,
        q_manifold=ref.q_ref,
        kkt_regularization=0.0,
        drift_method="analytic",
    )
    myo_adapter = MyoSuiteClosedLoopSnapshotAdapter.from_native_quantities(
        coordinate_order=ref.coordinate_order,
        model_sha256=ref.model_sha256,
        mass_matrix=ref.M,
        bias_vector=ref.h,
        native_jacobian=j_mj,
        native_drift=d_mj,
        closure_rotation_world=ref.rot_wa,
        closure_point_world=ref.pos_a,
        grip_origin_world=ref.grip_origin,
        q_manifold=ref.q_ref,
        kkt_regularization=0.0,
        drift_method="analytic",
    )

    # Drake native convention: ang_lin in local frame a, b - a (negated by adapter)
    j_drake_ab, d_drake_ab = from_canonical_closure(
        ref.J_c,
        ref.Jdot_v,
        native=DRAKE_CLOSURE_CONVENTION,
        grip_frame_rotation=ref.rot_wa,
        grip_origin=ref.grip_origin,
        reference_point=ref.pos_a,
    )
    drake_adapter = DrakeClosedLoopSnapshotAdapter.from_native_quantities(
        coordinate_order=ref.coordinate_order,
        model_sha256=ref.model_sha256,
        mass_matrix=ref.M,
        bias_term=ref.h + 1.5,
        gravity_generalized_forces=np.full(ref.nv, 1.5, dtype=np.float64),
        native_jacobian_ba=-j_drake_ab,
        native_bias_spatial_acc_ba=-d_drake_ab,
        closure_rotation_world=ref.rot_wa,
        closure_point_world=ref.pos_a,
        grip_origin_world=ref.grip_origin,
        q_manifold=ref.q_ref,
    )

    # OpenSim native convention: lin_ang in world frame, b - a (relative_sign = -1)
    j_osim, d_osim = from_canonical_closure(
        ref.J_c,
        ref.Jdot_v,
        native=OPENSIM_CLOSURE_CONVENTION,
        grip_frame_rotation=ref.rot_wa,
        grip_origin=ref.grip_origin,
        reference_point=ref.pos_a,
    )
    opensim_adapter = OpenSimClosedLoopSnapshotAdapter.from_native_quantities(
        coordinate_order=ref.coordinate_order,
        model_sha256=ref.model_sha256,
        mass_matrix=ref.M,
        bias_forces=ref.h,
        native_jacobian_ba=j_osim,
        native_bias_ba=d_osim,
        closure_rotation_world=ref.rot_wa,
        closure_point_world=ref.pos_a,
        grip_origin_world=ref.grip_origin,
        q_manifold=ref.q_ref,
    )

    return {
        "pinocchio": pin_adapter,
        "mujoco": mj_adapter,
        "myosuite": myo_adapter,
        "drake": drake_adapter,
        "opensim": opensim_adapter,
    }


def test_nsx51_mujoco_hard_kkt_capability_and_drift_provenance() -> None:
    """NSX-51 (#12579): MuJoCo hard-KKT reports FULL capability and records drift_method."""
    ref = _ReferenceClosedLoopBackend()
    adapters = _build_all_adapters_on_reference(ref)
    mj_adapter = adapters["mujoco"]

    assert mj_adapter.capability_level("closed_loop_snapshot") == CapabilityLevel.FULL
    assert (
        mj_adapter.capability_level("soft_constraint_stepping")
        == CapabilityLevel.PARTIAL
    )

    snap = snapshot_from_adapter(mj_adapter, ref.q_ref, ref.v_ref, ref.a_ref)
    assert snap.model_provenance.engine_name == "mujoco"
    assert snap.model_provenance.drift_method == "analytic"
    assert snap.model_provenance.kkt_regularization == 0.0
    np.testing.assert_allclose(snap.J_c, ref.J_c, atol=1e-12)
    np.testing.assert_allclose(snap.Jdot_v, ref.Jdot_v, atol=1e-12)

    # Declared finite-difference drift method is also recorded in provenance
    mj_fd = mj_adapter.with_drift_method("finite_difference_centered")
    snap_fd = snapshot_from_adapter(mj_fd, ref.q_ref, ref.v_ref, ref.a_ref)
    assert snap_fd.model_provenance.drift_method == "finite_difference_centered"


@pytest.mark.requires_mujoco
def test_nsx51_mujoco_live_spec_snapshot_and_fd_vs_analytic_drift() -> None:
    """NSX-51 (#12579): Live MuJoCo full-body spec evaluates mj_fullM, qfrc_bias, and closure."""
    pytest.importorskip("mujoco")
    spec_path = (
        REPO_ROOT
        / "docs/development/full_body_models/evidence/ground_support/full_body_spec_hipcal_scaled.json"
    )
    record_path = (
        REPO_ROOT
        / "docs/development/full_body_models/evidence/ground_support/dynamics_record.npz"
    )
    spec_bytes = spec_path.read_bytes()
    with np.load(record_path) as data:
        q_raw, v_raw = data["q"][0], data["v"][0]

    adapter_ana = MuJoCoClosedLoopSnapshotAdapter(
        spec_bytes, kkt_regularization=0.0, drift_method="analytic"
    )
    adapter_fd = MuJoCoClosedLoopSnapshotAdapter(
        spec_bytes,
        kkt_regularization=0.0,
        drift_method="finite_difference_centered",
    )

    snap_ana = project_and_build_snapshot(adapter_ana, q_raw, v_raw)
    snap_fd = project_and_build_snapshot(adapter_fd, q_raw, v_raw)

    assert snap_ana.model_provenance.drift_method == "analytic"
    assert snap_fd.model_provenance.drift_method == "finite_difference_centered"
    np.testing.assert_allclose(snap_ana.M, snap_fd.M, atol=1e-12)
    np.testing.assert_allclose(snap_ana.h, snap_fd.h, atol=1e-12)
    np.testing.assert_allclose(snap_ana.J_c, snap_fd.J_c, atol=1e-12)
    np.testing.assert_allclose(snap_ana.Jdot_v, snap_fd.Jdot_v, atol=1e-6)


def test_nsx52_myosuite_capability_gating_on_kkt_regularization() -> None:
    """NSX-52 (#12580): MyoSuite reuses MuJoCo path, gates FULL on kkt_regularization == 0."""
    assert issubclass(
        MyoSuiteClosedLoopSnapshotAdapter, MuJoCoClosedLoopSnapshotAdapter
    )
    ref = _ReferenceClosedLoopBackend()
    adapters = _build_all_adapters_on_reference(ref)
    mj_adapter = adapters["mujoco"]
    myo_exact = adapters["myosuite"]

    assert myo_exact.capability_level("closed_loop_snapshot") == CapabilityLevel.FULL
    snap_myo = snapshot_from_adapter(myo_exact, ref.q_ref, ref.v_ref, ref.a_ref)
    snap_mj = snapshot_from_adapter(mj_adapter, ref.q_ref, ref.v_ref, ref.a_ref)

    assert snap_myo.model_provenance.engine_name == "myosuite"
    assert snap_myo.model_provenance.kkt_regularization == 0.0
    np.testing.assert_allclose(snap_myo.M, snap_mj.M, atol=1e-14)
    np.testing.assert_allclose(snap_myo.h, snap_mj.h, atol=1e-14)
    np.testing.assert_allclose(snap_myo.J_c, snap_mj.J_c, atol=1e-14)
    np.testing.assert_allclose(snap_myo.Jdot_v, snap_mj.Jdot_v, atol=1e-14)

    # Regularized (kkt_regularization > 0) degrades capability from FULL -> PARTIAL
    myo_reg = myo_exact.with_kkt_regularization(1e-6)
    assert myo_reg.kkt_regularization == pytest.approx(1e-6)
    assert myo_reg.capability_level("closed_loop_snapshot") == CapabilityLevel.PARTIAL
    with pytest.raises(CapabilityUnavailableError, match="kkt_regularization"):
        snapshot_from_adapter(myo_reg, ref.q_ref, ref.v_ref, ref.a_ref)

    with pytest.raises(ValueError, match="kkt_regularization"):
        myo_exact.with_kkt_regularization(-1e-6)


def test_nsx53_drake_native_force_adapter_conversion_and_patch_dict() -> None:
    """NSX-53 (#12581): DrakeForceAdapter converts ang_lin local closure and bias terms."""
    assert DrakeForceAdapter is DrakeClosedLoopSnapshotAdapter
    ref = _ReferenceClosedLoopBackend()
    adapters = _build_all_adapters_on_reference(ref)
    drake_adapter = adapters["drake"]
    pin_adapter = adapters["pinocchio"]

    assert (
        drake_adapter.capability_level("closed_loop_snapshot") == CapabilityLevel.FULL
    )
    snap_drake = snapshot_from_adapter(drake_adapter, ref.q_ref, ref.v_ref, ref.a_ref)
    snap_pin = snapshot_from_adapter(pin_adapter, ref.q_ref, ref.v_ref, ref.a_ref)

    assert snap_drake.model_provenance.engine_name == "drake"
    np.testing.assert_allclose(snap_drake.M, snap_pin.M, atol=1e-12)
    np.testing.assert_allclose(snap_drake.h, snap_pin.h, atol=1e-12)
    np.testing.assert_allclose(snap_drake.J_c, snap_pin.J_c, atol=1e-12)
    np.testing.assert_allclose(snap_drake.Jdot_v, snap_pin.Jdot_v, atol=1e-12)

    # Verify patch.dict usage (never direct sys.modules assignment) rejects mock pydrake
    with patch.dict("sys.modules", {"pydrake": None, "pydrake.all": None}):
        with pytest.raises((ImportError, CapabilityUnavailableError)):
            DrakeClosedLoopSnapshotAdapter(b"{}")


def test_nsx54_opensim_closed_loop_snapshot_adapter_delegation() -> None:
    """NSX-54 (#12582): OpenSimForceAdapter delegates and converts WORLD lin_ang b-a."""
    assert OpenSimForceAdapter is OpenSimClosedLoopSnapshotAdapter
    ref = _ReferenceClosedLoopBackend()
    adapters = _build_all_adapters_on_reference(ref)
    osim_adapter = adapters["opensim"]
    pin_adapter = adapters["pinocchio"]

    assert osim_adapter.capability_level("closed_loop_snapshot") == CapabilityLevel.FULL
    snap_osim = snapshot_from_adapter(osim_adapter, ref.q_ref, ref.v_ref, ref.a_ref)
    snap_pin = snapshot_from_adapter(pin_adapter, ref.q_ref, ref.v_ref, ref.a_ref)

    assert snap_osim.model_provenance.engine_name == "opensim"
    np.testing.assert_allclose(snap_osim.M, snap_pin.M, atol=1e-12)
    np.testing.assert_allclose(snap_osim.h, snap_pin.h, atol=1e-12)
    np.testing.assert_allclose(snap_osim.J_c, snap_pin.J_c, atol=1e-12)
    np.testing.assert_allclose(snap_osim.Jdot_v, snap_pin.Jdot_v, atol=1e-12)


def test_nsx55_simscape_r2025b_snapshot_loader_and_dv_record(tmp_path: Path) -> None:
    """NSX-55 (#12583): Simscape loader enforces R2025b, rejects R2026a, and DV-12583 validates."""
    # 1. Host without live MATLAB R2025b reports CapabilityLevel.NONE with DV-12583
    unbacked = SimscapeClosedLoopSnapshotAdapter()
    assert unbacked.capability_level("closed_loop_snapshot") == CapabilityLevel.NONE
    assert unbacked.deferred_validation_id == "DV-12583"
    with pytest.raises(CapabilityUnavailableError, match="DV-12583"):
        snapshot_from_adapter(unbacked, np.zeros(8), np.zeros(8), np.zeros(8))

    # 2. Checked-in R2025b export loads cleanly
    snap = load_simscape_closed_loop_snapshot(FIXTURE_PATH)
    assert snap.model_provenance.engine_name == "simscape"
    assert snap.model_provenance.matlab_release == "R2025b"
    assert (
        snap.model_provenance.spec_sha256
        == "d7aeb9a886c6e2cf414e99af9d0e33cf62d6060271b1a9d23c36678cacfeb678"
    )
    assert snap.closure_rows() == 6

    backed = SimscapeClosedLoopSnapshotAdapter(FIXTURE_PATH)
    assert backed.capability_level("closed_loop_snapshot") == CapabilityLevel.FULL
    snap_via_adapter = snapshot_from_adapter(backed, snap.q, snap.v, snap.a)
    np.testing.assert_allclose(snap_via_adapter.M, snap.M, atol=1e-14)

    # 3. Fails closed on R2026a or missing checksum
    bad_release = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    bad_release["matlab_release"] = "R2026a"
    bad_path = tmp_path / "bad_r2026a.json"
    bad_path.write_text(json.dumps(bad_release), encoding="utf-8")
    with pytest.raises(ValueError, match="R2025b"):
        load_simscape_closed_loop_snapshot(bad_path)

    bad_hash = json.loads(FIXTURE_PATH.read_text(encoding="utf-8"))
    bad_hash["model_sha256"] = "short_hash"
    bad_hash_path = tmp_path / "bad_hash.json"
    bad_hash_path.write_text(json.dumps(bad_hash), encoding="utf-8")
    with pytest.raises(ValueError, match="model_sha256"):
        load_simscape_closed_loop_snapshot(bad_hash_path)

    # 4. MATLAB export script enforces 2025b and rejects R2026a
    assert MATLAB_EXPORT_SCRIPT_PATH.is_file()
    matlab_src = MATLAB_EXPORT_SCRIPT_PATH.read_text(encoding="utf-8")
    assert "version('-release')" in matlab_src
    assert "strcmp(rel, '2025b')" in matlab_src
    assert "R2026a" in matlab_src

    # 5. DV-12583 record exists in catalog.json
    catalog = json.loads(
        (REPO_ROOT / "docs/development/planning/catalog.json").read_text(
            encoding="utf-8"
        )
    )
    dv_ids = {entry["id"] for entry in catalog["entries"]}
    assert "DV-12583" in dv_ids
    assert (REPO_ROOT / "docs/development/planning/DV-12583.md").is_file()
    assert (REPO_ROOT / "docs/development/planning/source-12583.json").is_file()
    assert (REPO_ROOT / "docs/development/planning/sources/source-12583.json").is_file()


def test_nsx56_two_track_conformance_across_all_engines() -> None:
    """NSX-56 (#12584): Track N and Track S pass across all engines with projected states."""
    ref = _ReferenceClosedLoopBackend()
    adapters = _build_all_adapters_on_reference(ref)
    adapters["simscape"] = SimscapeClosedLoopSnapshotAdapter(FIXTURE_PATH)

    # Off-manifold perturbation to verify project_to_closure runs first
    q_perturbed = ref.q_ref + 1e-4 * np.ones(ref.nv)
    v_perturbed = ref.v_ref + 1e-4 * np.ones(ref.nv)

    report = evaluate_two_track_conformance(
        reference_adapter=adapters["pinocchio"],
        candidate_adapters={k: v for k, v in adapters.items() if k != "pinocchio"},
        q=q_perturbed,
        v=v_perturbed,
        a=ref.a_ref,
        track_n_tolerance=1e-8,
    )

    assert report.projection_pose_residual_after <= 1e-12
    assert report.projection_rate_residual_after <= 1e-12
    assert report.all_passed

    for engine_name, pair_result in report.results.items():
        assert pair_result.track_n.passed, f"Track N failed for {engine_name}"
        assert pair_result.track_n.max_delta <= 1e-8
        assert pair_result.track_s.passed, f"Track S failed for {engine_name}"
        assert pair_result.track_s.condition_number >= 1.0
        # Propagated bound must equal cond(A) * max(track_n.max_delta, 1e-12)
        expected_bound = pair_result.track_s.condition_number * max(
            pair_result.track_n.max_delta, 1e-12
        )
        assert pair_result.track_s.propagated_relative_bound == pytest.approx(
            expected_bound, rel=1e-12
        )

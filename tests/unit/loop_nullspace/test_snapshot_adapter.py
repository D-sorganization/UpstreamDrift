"""Unit tests for ClosedLoopSnapshot adapter extraction and CLOSED_LOOP_SNAPSHOT capability (#12558)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.engines.physics_engines.pinocchio.python.force_adapter import (
    PinocchioForceAdapter,
)
from src.shared.python.engine_core.capabilities import (
    ADAPTER_BOUNDARY_CAPABILITIES,
    ENGINE_CAPABILITY_FIELDS,
    Capability,
    CapabilityLevel,
    EngineCapabilities,
)
from src.shared.python.loop_nullspace.admission import AdmissionVerdict, admit
from src.shared.python.loop_nullspace.closure_convention import (
    DRAKE_CLOSURE_CONVENTION,
    MUJOCO_CLOSURE_CONVENTION,
    NativeClosureConvention,
    OPENSIM_CLOSURE_CONVENTION,
    PINOCCHIO_CLOSURE_CONVENTION,
    from_canonical_closure,
)
from src.shared.python.loop_nullspace.contracts import (
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.fixtures import (
    spatial_two_arm_club_snapshot,
)
from src.shared.python.loop_nullspace.snapshot import (
    CapabilityUnavailable,
    CapabilityUnavailableError,
    snapshot_from_adapter,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    DrakeForceAdapter,
    MultiEngineTorqueAllocator,
    OpenSimForceAdapter,
    PinocchioSyntheticForceAdapter,
    SimscapeForceAdapter,
    SyntheticMultibodyFixture,
)

pytestmark = pytest.mark.unit


class _NativeConventionFixtureAdapter(SyntheticMultibodyFixture):
    """Test adapter wrapping a spatial oracle in a non-canonical closure convention."""

    def __init__(self, convention: NativeClosureConvention) -> None:
        self._base = PinocchioForceAdapter.from_spatial_oracle(
            floating_base=True, include_feet=True
        )
        super().__init__(nv=self._base.nv, n_spheres=self._base.n_contact_spheres)
        self._native_closure_convention = convention
        self.topology = self._base.topology
        self.model_provenance = self._base.model_provenance

    def closed_loop_snapshot_capability(self) -> CapabilityLevel:
        return CapabilityLevel.FULL

    @property
    def actuated_indices(self) -> tuple[int, ...]:
        return self._base.actuated_indices

    @property
    def coordinate_order(self) -> tuple[str, ...]:
        return self._base.coordinate_order

    def compute_mass_and_bias(
        self, q: np.ndarray, v: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        return self._base.compute_mass_and_bias(q, v)

    def compute_inverse_dynamics(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray
    ) -> np.ndarray:
        return self._base.compute_inverse_dynamics(q, v, a)

    def compute_contact_jacobian(self, q: np.ndarray) -> np.ndarray:
        return self._base.compute_contact_jacobian(q)

    def foot_contacts_snapshot(self, q: np.ndarray):
        return self._base.foot_contacts_snapshot(q)

    def closure_wrench_frame(self, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        return self._base.closure_wrench_frame(q)

    def compute_grip_jacobian(self, q: np.ndarray) -> np.ndarray:
        j_canon = self._base.compute_grip_jacobian(q)
        r_wg, p_g = self._base.closure_wrench_frame(q)
        j_native, _ = from_canonical_closure(
            j_canon,
            np.zeros(6, dtype=np.float64),
            native=self._native_closure_convention,
            grip_frame_rotation=r_wg,
            grip_origin=p_g,
        )
        return j_native

    def compute_closure_drift(self, q: np.ndarray, v: np.ndarray) -> np.ndarray:
        j_canon = self._base.compute_grip_jacobian(q)
        jdot_v_canon = self._base.compute_closure_drift(q, v)
        r_wg, p_g = self._base.closure_wrench_frame(q)
        _, jdot_v_native = from_canonical_closure(
            j_canon,
            jdot_v_canon,
            native=self._native_closure_convention,
            grip_frame_rotation=r_wg,
            grip_origin=p_g,
        )
        return jdot_v_native

    def club_inertia_split(self):
        return self._base.club_inertia_split()

    def closure_pose_residual(self, q: np.ndarray) -> np.ndarray:
        return self._base.closure_pose_residual(q)


def test_capability_registry_includes_closed_loop_snapshot() -> None:
    """Capability.CLOSED_LOOP_SNAPSHOT is registered in ENGINE_CAPABILITY_FIELDS and ADAPTER_BOUNDARY_CAPABILITIES."""
    assert Capability.CLOSED_LOOP_SNAPSHOT.value == "closed_loop_snapshot"
    assert Capability.CLOSED_LOOP_SNAPSHOT in ENGINE_CAPABILITY_FIELDS
    assert Capability.CLOSED_LOOP_SNAPSHOT in ADAPTER_BOUNDARY_CAPABILITIES

    caps = EngineCapabilities(
        engine_name="Pinocchio",
        closed_loop_snapshot=CapabilityLevel.FULL,
    )
    assert caps.has_closed_loop_snapshot is True
    assert caps.supports(Capability.CLOSED_LOOP_SNAPSHOT) is True

    default_caps = EngineCapabilities(engine_name="Stub")
    assert default_caps.has_closed_loop_snapshot is False


def test_unimplemented_adapters_refuse_snapshot_with_capability_unavailable() -> None:
    """Synthetic/surrogate adapters report CapabilityLevel.NONE and raise CapabilityUnavailable."""
    q = np.zeros(44, dtype=np.float64)
    v = np.zeros(44, dtype=np.float64)
    a = np.zeros(44, dtype=np.float64)

    adapters = [
        SyntheticMultibodyFixture(),
        PinocchioSyntheticForceAdapter(),
        DrakeForceAdapter(),
        OpenSimForceAdapter(),
        SimscapeForceAdapter(),
    ]
    for adapter in adapters:
        assert (
            adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) == CapabilityLevel.NONE
        )
        with pytest.raises(CapabilityUnavailable) as exc_info:
            snapshot_from_adapter(adapter, q, v, a)
        assert isinstance(exc_info.value, CapabilityUnavailableError)
        assert "CLOSED_LOOP_SNAPSHOT" in str(exc_info.value)


def test_snapshot_from_adapter_matches_spatial_two_arm_oracle_fixed_and_floating() -> (
    None
):
    """snapshot_from_adapter on PinocchioForceAdapter.from_spatial_oracle reproduces spatial_two_arm_club_snapshot."""
    for floating_base, include_feet in ((False, False), (True, True)):
        expected, _ = spatial_two_arm_club_snapshot(
            floating_base=floating_base, include_feet=include_feet
        )
        adapter = PinocchioForceAdapter.from_spatial_oracle(
            floating_base=floating_base, include_feet=include_feet
        )
        assert (
            adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) == CapabilityLevel.FULL
        )

        snap = snapshot_from_adapter(
            adapter,
            expected.q,
            expected.v,
            expected.a,
            input_provenance=expected.input_provenance,
            model_provenance=expected.model_provenance,
            grf_observed=expected.grf_observed,
        )
        assert snap.nv() == expected.nv()
        assert snap.nc() == 6
        np.testing.assert_allclose(snap.M, expected.M, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(snap.h, expected.h, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(snap.J_c, expected.J_c, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(snap.Jdot_v, expected.Jdot_v, rtol=1e-12, atol=1e-12)
        assert snap.bare_club.mass_kg == pytest.approx(expected.bare_club.mass_kg)
        assert len(snap.foot_contacts) == len(expected.foot_contacts)

        report = admit(snap)
        assert report.verdict == AdmissionVerdict.ADMITTED
        assert report.admitted_trajectory is not None


def test_closure_convention_invariance_in_snapshot_and_torque_allocator() -> None:
    """snapshot_from_adapter and MultiEngineTorqueAllocator canonicalize any native closure convention."""
    expected, _ = spatial_two_arm_club_snapshot(floating_base=True, include_feet=True)
    conventions = (
        PINOCCHIO_CLOSURE_CONVENTION,
        DRAKE_CLOSURE_CONVENTION,
        MUJOCO_CLOSURE_CONVENTION,
        OPENSIM_CLOSURE_CONVENTION,
    )

    baseline_adapter = _NativeConventionFixtureAdapter(PINOCCHIO_CLOSURE_CONVENTION)
    baseline_alloc = MultiEngineTorqueAllocator(baseline_adapter).allocate_frame(
        expected.q, expected.v, expected.a
    )

    for conv in conventions:
        adapter = _NativeConventionFixtureAdapter(conv)
        snap = snapshot_from_adapter(adapter, expected.q, expected.v, expected.a)
        np.testing.assert_allclose(snap.J_c, expected.J_c, rtol=1e-12, atol=1e-12)
        np.testing.assert_allclose(snap.Jdot_v, expected.Jdot_v, rtol=1e-12, atol=1e-12)

        alloc = MultiEngineTorqueAllocator(adapter).allocate_frame(
            expected.q, expected.v, expected.a
        )
        np.testing.assert_allclose(
            alloc.tau_actuated,
            baseline_alloc.tau_actuated,
            rtol=1e-10,
            atol=1e-10,
        )
        np.testing.assert_allclose(
            alloc.lambda_grip,
            baseline_alloc.lambda_grip,
            rtol=1e-10,
            atol=1e-10,
        )


def test_pinocchio_verify_acceleration_parity_uses_constrained_kkt_dynamics() -> None:
    """verify_acceleration_parity uses constrained KKT dynamics and stays invariant under J_c^T lambda shifts."""
    adapter = PinocchioForceAdapter.from_spatial_oracle(
        floating_base=False, include_feet=False
    )
    snap, _ = spatial_two_arm_club_snapshot(floating_base=False, include_feet=False)

    # Project snap.a onto the exact acceleration manifold J_c a + Jdot_v = 0
    j_c = snap.J_c
    m_inv_jt = np.linalg.solve(snap.M, j_c.T)
    g_mat = j_c @ m_inv_jt
    delta_lam = np.linalg.solve(g_mat, j_c @ snap.a + snap.Jdot_v)
    a_constrained = snap.a - m_inv_jt @ delta_lam

    # Nominal generalized torque reproducing a_constrained on the manifold
    tau_nominal = snap.M @ a_constrained + snap.h
    # Add a large internal grip force shift J_c^T * delta_lambda:
    # Unconstrained ABA would diverge by M^{-1} J_c^T delta_lambda, whereas
    # constrained KKT dynamics absorbs J_c^T delta_lambda into the constraint impulse.
    extra_lambda = np.array([45.0, -30.0, 25.0, 4.0, -3.5, 2.0], dtype=np.float64)
    tau_shifted = tau_nominal + j_c.T @ extra_lambda

    parity_err = adapter.verify_acceleration_parity(
        snap.q, snap.v, tau_shifted, a_constrained
    )
    assert parity_err < 1e-9


def test_live_pinocchio_anthro_driver_snapshot_and_kkt_parity_when_available() -> None:
    """When C++ pinocchio is installed, PinocchioForceAdapter on full_body_spec_anthro_driver passes snapshot & KKT parity."""
    try:
        import pinocchio
    except ImportError:
        return
    if getattr(pinocchio, "__file__", None) is None:
        return

    spec_path = (
        Path(__file__).resolve().parents[3]
        / "docs"
        / "development"
        / "full_body_models"
        / "full_body_spec_anthro_driver.json"
    )
    adapter = PinocchioForceAdapter(spec_path.read_bytes())
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) == CapabilityLevel.FULL

    nv = int(adapter.nv)
    q0 = np.asarray(pinocchio.neutral(adapter._model), dtype=np.float64)
    v0 = np.zeros(nv, dtype=np.float64)
    a0 = np.zeros(nv, dtype=np.float64)

    raw_snap = snapshot_from_adapter(
        adapter,
        q0,
        v0,
        a0,
        input_provenance=InputProvenance(
            kinematic_source="pinocchio_neutral",
            derivative_method="analytical",
            time_sync_status="SYNCHRONIZED",
        ),
        model_provenance=ModelProvenance(
            spec_sha256="anthro-driver-v1",
            inertial_source="anthropometric_table",
            bare_club_bench_id="cad",
            engine_name="pinocchio",
        ),
    )
    assert raw_snap.nv() == nv
    assert raw_snap.nc() == 6

    # Repair initial neutral configuration onto the bilateral grip closure manifold
    report = admit(raw_snap, repair="project_to_closure", plant=adapter)
    assert report.verdict == AdmissionVerdict.REPAIRED
    assert report.admitted_trajectory is not None
    repaired_snap = report.admitted_trajectory.snapshots[0]

    # Re-extract a fresh snapshot at the repaired configuration: must be ADMITTED directly
    fresh_repaired = snapshot_from_adapter(
        adapter, repaired_snap.q, repaired_snap.v, repaired_snap.a
    )
    recheck = admit(fresh_repaired)
    assert recheck.verdict == AdmissionVerdict.ADMITTED

    # Verify live Pinocchio constraintDynamics acceleration parity under internal wrench shift
    tau_base = fresh_repaired.M @ fresh_repaired.a + fresh_repaired.h
    extra_lambda = np.array([15.0, -10.0, 8.0, 1.2, -0.8, 0.5], dtype=np.float64)
    tau_shifted = tau_base + fresh_repaired.J_c.T @ extra_lambda
    parity_err = adapter.verify_acceleration_parity(
        fresh_repaired.q, fresh_repaired.v, tau_shifted, fresh_repaired.a
    )
    assert parity_err < 1e-9

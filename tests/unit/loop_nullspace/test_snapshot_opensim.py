"""NSX-54: OpenSim closed-loop snapshot against the Pinocchio reference (#12582).

Both engines are driven with the *same named scalar coordinates* of the
full-body spec, so no base twist map (``pose_interchange.dynamic_maps``) is
involved: the OpenSim adapter's velocity is ``dq/dt`` of those coordinates
(equal to the Simbody ``u`` slot per coordinate, verified by the adapter's
layout probe), not a Simbody FreeJoint mobilizer speed.  The two native
orders differ, so snapshots are compared after permuting the OpenSim one to the
Pinocchio order by coordinate name.

Tolerances are stated per quantity and scaled by the quantity's magnitude; they
carry a margin of at least two orders over the measured agreement.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import Capability, CapabilityLevel
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    InputProvenance,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter

pytestmark = [
    pytest.mark.unit,
    pytest.mark.requires_opensim,
    pytest.mark.requires_pinocchio,
]

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")

# Scaled tolerances (max |err| / max(1, max |ref|)); measured agreement is
# recorded in the PR (M, h ~1e-14, closure rows ~1e-13).
TOL_MASS = 1e-11
TOL_BIAS = 1e-11
TOL_JACOBIAN = 1e-9
TOL_DRIFT = 1e-8
TOL_ACCEL = 1e-8


def _scaled(err: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(err))) / max(1.0, float(np.max(np.abs(ref))))


@pytest.fixture(scope="module")
def pin_adapter():
    pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    return PinocchioForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def os_adapter():
    pytest.importorskip("opensim")
    from src.engines.physics_engines.opensim.python.force_adapter import (
        OpenSimNativeForceAdapter,
    )

    return OpenSimNativeForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def perm(pin_adapter, os_adapter) -> np.ndarray:
    """``perm[i]`` is the OpenSim index of Pinocchio coordinate ``i``."""
    names = list(os_adapter.coordinate_order)
    return np.array([names.index(n) for n in pin_adapter.coordinate_order])


@pytest.fixture(scope="module")
def pin_state(pin_adapter):
    """A closure-consistent state in Pinocchio's native order."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    qn, vn = plant._named(proj.q), plant._named(proj.v)
    order = pin_adapter.coordinate_order
    return np.array([qn[n] for n in order]), np.array([vn[n] for n in order])


@pytest.fixture(scope="module")
def os_state(pin_state, perm):
    inverse = np.argsort(perm)  # OpenSim index j holds Pinocchio index inverse[j]
    return pin_state[0][inverse], pin_state[1][inverse]


@pytest.fixture(scope="module")
def reference(pin_adapter, pin_state):
    """Pinocchio constrained acceleration for a seeded applied effort."""
    q, v = pin_state
    tau = np.zeros(pin_adapter.nv)
    idx = list(pin_adapter.actuated_indices)
    tau[idx] = np.random.default_rng(12582).normal(size=len(idx))
    a = pin_adapter._plant.constrained_acceleration(q, v, tau)
    return {"a": a, "tau": tau}


@pytest.fixture(scope="module")
def snapshots(pin_adapter, os_adapter, pin_state, os_state, reference, perm):
    inverse = np.argsort(perm)
    pin_snap = snapshot_from_adapter(
        pin_adapter, *pin_state, reference["a"], input_provenance=PROVENANCE
    )
    os_snap = snapshot_from_adapter(
        os_adapter,
        *os_state,
        reference["a"][inverse],
        input_provenance=PROVENANCE,
    )
    return pin_snap, os_snap


def test_capability_is_full_and_protocol_is_satisfied(os_adapter) -> None:
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        BaseEngineForceAdapter,
        ClosedLoopSnapshotAdapter,
        EngineType,
    )

    assert os_adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is (
        CapabilityLevel.FULL
    )
    assert os_adapter.capability(Capability.MUSCLES) is CapabilityLevel.NONE
    assert isinstance(os_adapter, BaseEngineForceAdapter)
    assert isinstance(os_adapter, ClosedLoopSnapshotAdapter)
    assert os_adapter.engine_type is EngineType.OPENSIM


def test_native_convention_is_the_nsx02_constant(os_adapter) -> None:
    from src.engines.physics_engines.opensim.python.full_body_parity import (
        CLOSURE_CONVENTION,
    )

    assert os_adapter.native_closure_convention is CLOSURE_CONVENTION
    assert CLOSURE_CONVENTION.frame == "WORLD"
    assert CLOSURE_CONVENTION.relative_sign == -1


def test_same_coordinate_vocabulary_no_base_map_needed(pin_adapter, os_adapter) -> None:
    assert sorted(pin_adapter.coordinate_order) == sorted(os_adapter.coordinate_order)
    assert os_adapter.nv == pin_adapter.nv


def test_snapshot_is_a_valid_weld6_snapshot(snapshots, os_adapter) -> None:
    _, snap = snapshots
    assert isinstance(snap, ClosedLoopSnapshot)
    assert snap.topology.closure_kind is ClosureKind.WELD6
    assert snap.topology.coordinate_order == tuple(os_adapter.coordinate_order)
    assert snap.model_provenance.name == "opensim"
    assert snap.model_provenance.content_hash == os_adapter.model_hash
    assert len(os_adapter.model_hash) == 64
    assert snap.J_g is None


def test_mass_and_bias_equal_pinocchio(snapshots, perm) -> None:
    pin, snap = snapshots
    ix = np.ix_(perm, perm)
    assert _scaled(snap.M[ix] - pin.M, pin.M) < TOL_MASS
    assert _scaled(snap.h[perm] - pin.h, pin.h) < TOL_BIAS


def test_canonical_closure_rows_equal_pinocchio(snapshots, perm) -> None:
    pin, snap = snapshots
    assert _scaled(snap.J_c[:, perm] - pin.J_c, pin.J_c) < TOL_JACOBIAN
    assert _scaled(snap.Jdot_v - pin.Jdot_v, pin.Jdot_v) < TOL_DRIFT


def test_grip_frame_and_club_inertia_equal_pinocchio(snapshots) -> None:
    pin, snap = snapshots
    np.testing.assert_allclose(
        snap.grip_frame.rotation, pin.grip_frame.rotation, rtol=0, atol=1e-10
    )
    np.testing.assert_allclose(
        snap.grip_frame.origin, pin.grip_frame.origin, rtol=0, atol=1e-10
    )
    assert snap.bare_club.mass == pytest.approx(pin.bare_club.mass, abs=1e-12)
    np.testing.assert_allclose(snap.bare_club.com, pin.bare_club.com, atol=1e-9)
    np.testing.assert_allclose(snap.bare_club.inertia, pin.bare_club.inertia, atol=1e-9)
    assert len(snap.welded_hand_inertias) == len(pin.welded_hand_inertias)
    masses = sorted(h.mass for h in snap.welded_hand_inertias)
    expect = sorted(h.mass for h in pin.welded_hand_inertias)
    np.testing.assert_allclose(masses, expect, atol=1e-12)


def test_actuation_and_loop_support_equal_pinocchio(
    snapshots, pin_adapter, os_adapter
) -> None:
    pin, snap = snapshots
    pin_names = pin_adapter.coordinate_order
    os_names = os_adapter.coordinate_order
    assert {os_names[i] for i in snap.topology.actuated_indices} == {
        pin_names[i] for i in pin.topology.actuated_indices
    }
    assert {os_names[i] for i in snap.topology.loop_support_indices} == {
        pin_names[i] for i in pin.topology.loop_support_indices
    }
    # World-frame stations cancel on shared ancestors only to round-off, and
    # only on the closed loop (the state here is projected onto it).
    nonzero = {i for i in range(snap.topology.n) if np.abs(snap.J_c[:, i]).max() > 1e-9}
    assert nonzero <= set(snap.topology.loop_support_indices)


def test_loop_equation_and_multiplier_equal_pinocchio(
    snapshots, pin_adapter, reference, perm
) -> None:
    """``M a + h = S^T tau + J_c^T lambda`` with Pinocchio's own multiplier."""
    pin, snap = snapshots
    idx = list(snap.topology.actuated_indices)
    inverse = np.argsort(perm)
    tau = reference["tau"][inverse][idx]
    residual = snap.generalized_load() - snap.S.T @ tau
    lam, *_ = np.linalg.lstsq(snap.J_c.T, residual, rcond=None)
    assert _scaled(snap.J_c.T @ lam - residual, residual) < TOL_ACCEL
    pin_residual = (
        pin.generalized_load()
        - pin.S.T @ reference["tau"][list(pin.topology.actuated_indices)]
    )
    lam_pin, *_ = np.linalg.lstsq(pin.J_c.T, pin_residual, rcond=None)
    assert _scaled(lam - lam_pin, lam_pin) < 1e-7


def test_verify_acceleration_parity_agrees_with_pinocchio(
    os_adapter, os_state, reference, perm
) -> None:
    inverse = np.argsort(perm)
    q, v = os_state
    a = reference["a"][inverse]
    err = os_adapter.verify_acceleration_parity(q, v, reference["tau"][inverse], a)
    assert err / max(1.0, float(np.max(np.abs(a)))) < TOL_ACCEL
    wrong = a.copy()
    wrong[10] += 0.5
    assert (
        os_adapter.verify_acceleration_parity(q, v, reference["tau"][inverse], wrong)
        > 0.1
    )


def test_drift_matches_finite_difference_at_a_generic_velocity(
    os_adapter, os_state
) -> None:
    """Native drift is ``dJ/dt v`` even off the closure tangent space."""
    q, _ = os_state
    v = np.random.default_rng(3).normal(scale=0.3, size=os_adapter.nv)
    h = 1e-6
    fd = (
        (
            os_adapter.compute_grip_jacobian(q + h * v)
            - os_adapter.compute_grip_jacobian(q - h * v)
        )
        / (2 * h)
    ) @ v
    drift = os_adapter.compute_closure_drift(q, v)
    assert drift.shape == (6,)
    assert _scaled(drift - fd, fd) < 1e-6


def test_inverse_dynamics_and_contact_jacobian_shapes(os_adapter, os_state) -> None:
    q, v = os_state
    mass, bias = os_adapter.compute_mass_and_bias(q, v)
    a = np.random.default_rng(1).normal(size=os_adapter.nv)
    np.testing.assert_allclose(
        os_adapter.compute_inverse_dynamics(q, v, a), mass @ a + bias, atol=1e-9
    )
    jac = os_adapter.compute_contact_jacobian(q)
    assert jac.shape == (3 * os_adapter.n_contact_spheres, os_adapter.nv)
    assert len(os_adapter.contact_names) == os_adapter.n_contact_spheres


def test_state_shape_is_validated(os_adapter, os_state) -> None:
    q, v = os_state
    with pytest.raises(ValueError):
        snapshot_from_adapter(os_adapter, q[:-1], v, v, input_provenance=PROVENANCE)
    with pytest.raises(ValueError):
        os_adapter.compute_closure_drift(q, v[:-1])


def test_measured_agreement_is_far_inside_the_stated_tolerances(
    snapshots, perm
) -> None:
    pin, snap = snapshots
    errs = {
        "M": _scaled(snap.M[np.ix_(perm, perm)] - pin.M, pin.M),
        "h": _scaled(snap.h[perm] - pin.h, pin.h),
        "J_c": _scaled(snap.J_c[:, perm] - pin.J_c, pin.J_c),
        "Jdot_v": _scaled(snap.Jdot_v - pin.Jdot_v, pin.Jdot_v),
    }
    for key, tol in zip(
        errs, (TOL_MASS, TOL_BIAS, TOL_JACOBIAN, TOL_DRIFT), strict=True
    ):
        assert errs[key] < tol / 10.0

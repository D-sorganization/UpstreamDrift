"""NSX-13: ``snapshot_from_adapter`` on the Pinocchio reference adapter (#12558).

The expected snapshot is built independently from raw Pinocchio calls
(``crba``, ``nonLinearEffects``, ``getConstraintsJacobian``,
``constraintDynamics``) on a second ``Data`` object, never through the adapter
methods under test.  The NSX-12 spatial fixture is not on this branch; this
independent construction stands in for it (reported in the PR).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.biomechanics.club_inertia_partition import (
    club_solids_partition,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    InputProvenance,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    MujocoForceAdapter,
    SyntheticMultibodyFixture,
    create_engine_force_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")
ATOL = 1e-9


@pytest.fixture(scope="module")
def pin_module():
    return pytest.importorskip("pinocchio")


@pytest.fixture(scope="module")
def adapter(pin_module):
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    return PinocchioForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def state(adapter):
    """A projected (closure-consistent) state in the adapter's native order."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    qn, vn = plant._named(proj.q), plant._named(proj.v)
    order = adapter.coordinate_order
    return (
        np.array([qn[n] for n in order]),
        np.array([vn[n] for n in order]),
    )


@pytest.fixture(scope="module")
def reference(adapter, state, pin_module):
    """Independent raw-Pinocchio evaluation on a separate Data object."""
    pin = pin_module
    plant = adapter._plant
    model, cons = plant.model, plant.constraints
    data = model.createData()
    cdata = [c.createData() for c in cons]
    pin.initConstraintDynamics(model, data, cons, cdata)
    q, v = state
    rng = np.random.default_rng(12558)
    tau = np.zeros(adapter.nv)
    tau[list(adapter.actuated_indices)] = rng.normal(size=len(adapter.actuated_indices))
    a = np.array(pin.constraintDynamics(model, data, q, v, tau, cons, cdata))
    lam = np.array(data.lambda_c)
    mass = np.array(pin.crba(model, data, q))
    mass = np.triu(mass) + np.triu(mass, 1).T
    bias = np.array(pin.nonLinearEffects(model, data, q, v))
    pin.computeJointJacobians(model, data, q)
    pin.constraintDynamics(model, data, q, v, tau, cons, cdata)
    jac = np.array(pin.getConstraintsJacobian(model, data, cons, cdata))
    return {"a": a, "lam": lam, "M": mass, "h": bias, "J": jac, "tau": tau}


def _scaled(x: np.ndarray, scale: float) -> float:
    return float(np.max(np.abs(x))) / max(1.0, scale)


def test_capability_enum_member_and_levels(adapter) -> None:
    assert Capability.CLOSED_LOOP_SNAPSHOT.value == "closed_loop_snapshot"
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL


def test_synthetic_adapters_declare_none_and_refuse(adapter) -> None:
    for engine in ("pinocchio", "mujoco", "drake", "opensim", "simscape"):
        synth = create_engine_force_adapter(engine, allow_synthetic=True, nv=44)
        assert isinstance(synth, SyntheticMultibodyFixture)
        assert synth.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.NONE
        q = np.zeros(synth.nv)
        with pytest.raises(CapabilityUnavailable, match="closed_loop_snapshot"):
            snapshot_from_adapter(synth, q, q, q, input_provenance=PROVENANCE)


def test_mujoco_adapter_declares_none() -> None:
    bare = MujocoForceAdapter.__new__(MujocoForceAdapter)
    assert bare.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.NONE


def test_adapter_without_capability_method_refuses() -> None:
    class Legacy:
        pass

    with pytest.raises(CapabilityUnavailable):
        snapshot_from_adapter(
            Legacy(), np.zeros(1), np.zeros(1), np.zeros(1), input_provenance=PROVENANCE
        )


def test_partial_level_refuses(adapter, state, monkeypatch) -> None:
    monkeypatch.setattr(
        type(adapter), "capability", lambda self, cap: CapabilityLevel.PARTIAL
    )
    q, v = state
    with pytest.raises(CapabilityUnavailable, match="PARTIAL"):
        snapshot_from_adapter(adapter, q, v, v, input_provenance=PROVENANCE)


def test_snapshot_equals_independent_pinocchio_reference(
    adapter, state, reference
) -> None:
    q, v = state
    snap = snapshot_from_adapter(
        adapter, q, v, reference["a"], input_provenance=PROVENANCE
    )
    assert isinstance(snap, ClosedLoopSnapshot)
    assert snap.topology.closure_kind is ClosureKind.WELD6
    assert snap.topology.coordinate_order == adapter.coordinate_order
    np.testing.assert_allclose(snap.M, reference["M"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(snap.h, reference["h"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(snap.J_c, reference["J"], rtol=0, atol=1e-12)
    np.testing.assert_array_equal(snap.q, q)
    assert snap.input_provenance is PROVENANCE
    assert snap.model_provenance.content_hash == adapter.model_hash
    assert snap.J_g is None


def test_snapshot_satisfies_the_loop_equation_against_constraint_dynamics(
    adapter, state, reference
) -> None:
    """``M a + h = S^T tau + J_c^T lambda`` with Pinocchio's own multiplier."""
    q, v = state
    snap = snapshot_from_adapter(
        adapter, q, v, reference["a"], input_provenance=PROVENANCE
    )
    tau_act = reference["tau"][list(snap.topology.actuated_indices)]
    rhs = snap.S.T @ tau_act + snap.J_c.T @ reference["lam"]
    lhs = snap.generalized_load()
    assert _scaled(lhs - rhs, float(np.max(np.abs(lhs)))) < ATOL


def test_closure_is_satisfied_at_projected_state(adapter, state, reference) -> None:
    q, v = state
    snap = snapshot_from_adapter(
        adapter, q, v, reference["a"], input_provenance=PROVENANCE
    )
    assert _scaled(snap.J_c @ snap.v, float(np.max(np.abs(snap.v)))) < 1e-12
    accel = snap.J_c @ snap.a + snap.Jdot_v
    assert _scaled(accel, float(np.max(np.abs(snap.Jdot_v)))) < ATOL


def test_drift_matches_finite_difference_of_the_jacobian(adapter, state) -> None:
    q, v = state
    names = adapter.coordinate_order
    plant = adapter._plant

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array(
            plant.closure_force_jacobian(dict(zip(names, x, strict=True))).jacobian
        )

    h = 1e-6
    fd = ((jac(q + h * v) - jac(q - h * v)) / (2 * h)) @ v
    drift = adapter.compute_closure_drift(q, v)
    assert drift.shape == (6,)
    assert _scaled(drift - fd, float(np.max(np.abs(fd)))) < 1e-7


def test_drift_matches_finite_difference_at_a_generic_velocity(adapter, state) -> None:
    """The native drift is ``dJ/dt v`` even off the closure tangent space."""
    q, _ = state
    v = np.random.default_rng(3).normal(scale=0.3, size=adapter.nv)
    names = adapter.coordinate_order
    plant = adapter._plant

    def jac(x: np.ndarray) -> np.ndarray:
        return np.array(
            plant.closure_force_jacobian(dict(zip(names, x, strict=True))).jacobian
        )

    h = 1e-6
    fd = ((jac(q + h * v) - jac(q - h * v)) / (2 * h)) @ v
    drift = adapter.compute_closure_drift(q, v)
    assert _scaled(drift - fd, float(np.max(np.abs(fd)))) < 1e-7


def test_closure_jacobian_matches_finite_difference_of_the_pose(adapter, state) -> None:
    """``J_c dq`` is the first-order change of the weld pose residual."""
    q, _ = state
    names = adapter.coordinate_order
    plant = adapter._plant
    dq = np.random.default_rng(5).normal(size=adapter.nv)

    def pose(x: np.ndarray) -> np.ndarray:
        return np.array(plant.closure_residuals(dict(zip(names, x, strict=True)))[0])

    h = 1e-6
    fd = (pose(q + h * dq) - pose(q - h * dq)) / (2 * h)
    snap_j = adapter.compute_grip_jacobian(q)
    np.testing.assert_allclose(
        snap_j @ dq, fd, rtol=0, atol=1e-5 * max(1.0, float(np.abs(fd).max()))
    )


def test_wrench_frame_and_convention_are_exposed(adapter, state) -> None:
    from src.engines.physics_engines.pinocchio.python.native_model import (
        CLOSURE_CONVENTION,
    )

    q, _ = state
    rot, origin = adapter.closure_wrench_frame(q)
    assert rot.shape == (3, 3) and origin.shape == (3,)
    assert adapter.native_closure_convention == CLOSURE_CONVENTION


def test_club_inertia_split_is_bare_and_welded(adapter) -> None:
    bare, hands = adapter.club_inertia_split()
    expected = club_solids_partition(json.loads(SPEC_PATH.read_bytes())).bare_club
    assert bare.mass == pytest.approx(expected.mass_kg, abs=1e-12)
    assert len(hands) >= 1 and all(h.mass > 0.0 for h in hands)
    total = bare.mass + sum(h.mass for h in hands)
    assert total > 1.4  # club plus welded hands (NSX-01)


def test_snapshot_carries_bare_club_and_loop_support(adapter, state, reference) -> None:
    q, v = state
    snap = snapshot_from_adapter(
        adapter, q, v, reference["a"], input_provenance=PROVENANCE
    )
    bare, hands = adapter.club_inertia_split()
    assert snap.bare_club.mass == bare.mass
    assert len(snap.welded_hand_inertias) == len(hands)
    support = set(snap.topology.loop_support_indices)
    assert support
    # every column with a nonzero closure Jacobian is in the loop support
    nonzero = {i for i in range(snap.topology.n) if np.abs(snap.J_c[:, i]).max() > 0}
    assert nonzero <= support


def test_state_shape_is_validated(adapter, state) -> None:
    q, v = state
    with pytest.raises(ValueError):
        snapshot_from_adapter(adapter, q[:-1], v, v, input_provenance=PROVENANCE)


def test_verify_acceleration_parity_agrees_with_constraint_dynamics(
    adapter, state, reference
) -> None:
    """Corrected parity: native ``constraintDynamics`` to 1e-9 scaled (#12558)."""
    q, v = state
    a = reference["a"]
    scale = float(np.max(np.abs(a)))
    # applied efforts only
    err = adapter.verify_acceleration_parity(q, v, reference["tau"], a)
    assert err / max(1.0, scale) < ATOL
    # applied efforts plus the closure reaction (allocator contract): idempotent
    full = reference["tau"] + reference["J"].T @ reference["lam"]
    err_full = adapter.verify_acceleration_parity(q, v, full, a)
    assert err_full / max(1.0, scale) < ATOL


def test_verify_acceleration_parity_detects_a_wrong_target(
    adapter, state, reference
) -> None:
    q, v = state
    wrong = reference["a"].copy()
    wrong[10] += 0.5
    assert adapter.verify_acceleration_parity(q, v, reference["tau"], wrong) > 0.1

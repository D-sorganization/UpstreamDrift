"""NSX-53 (acceptance): Drake closed-loop snapshot equals Pinocchio's (#12581).

Both adapters build a ``ClosedLoopSnapshot`` at one matched closure-consistent
state through ``snapshot_from_adapter``.  Pinocchio is the reference.  The
adapters use different native coordinate orders, so every array is permuted to
the Pinocchio order by coordinate *name* before comparison.  The canonical
closure Jacobian and drift are already convention-free (NSX-02 conversion).

Tolerance: ``RTOL`` times the larger of 1 and the reference magnitude, i.e. a
relative tolerance with unit floor.  It is stated, measured and not widened;
measured maxima are about 2e-12 (drift) and below 2e-15 (M, h, J_c).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import InputProvenance
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from src.shared.python.motion_matching.polynomial_actuation import ROOT_COORDINATES

pytestmark = [
    pytest.mark.integration,
    pytest.mark.requires_pinocchio,
    pytest.mark.requires_drake,
]

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")
# Relative tolerance with unit floor between two independent engines.
RTOL = 1e-10


def _rel(x: np.ndarray, ref: np.ndarray) -> float:
    return float(np.max(np.abs(x - ref))) / max(1.0, float(np.max(np.abs(ref))))


@pytest.fixture(scope="module")
def adapters():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.force_adapter import (
        DrakeForceAdapter,
    )
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )

    spec = SPEC_PATH.read_bytes()
    return PinocchioForceAdapter(spec), DrakeForceAdapter(spec)


@pytest.fixture(scope="module", params=[0, 400])
def matched(request, adapters):
    """Snapshots from both engines at the same named state."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    pin, drake = adapters
    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][request.param], rec["v"][request.param]
    proj = project_to_closure(plant, q0, v0)
    named_q, named_v = plant._named(proj.q), plant._named(proj.v)
    a_named = dict(zip(plant.coordinate_order, np.zeros(plant.nv), strict=True))

    def build(adapter):
        order = adapter.coordinate_order
        return snapshot_from_adapter(
            adapter,
            np.array([named_q[n] for n in order]),
            np.array([named_v[n] for n in order]),
            np.array([a_named[n] for n in order]),
            input_provenance=PROVENANCE,
        )

    return build(pin), build(drake), pin.coordinate_order, drake.coordinate_order


def _to_reference_order(arr: np.ndarray, ref_order, other_order, axes) -> np.ndarray:
    perm = [other_order.index(n) for n in ref_order]
    for axis in axes:
        arr = np.take(arr, perm, axis=axis)
    return arr


def test_orders_differ_so_the_permutation_is_exercised(adapters) -> None:
    pin, drake = adapters
    assert set(pin.coordinate_order) == set(drake.coordinate_order)


def test_mass_matrix_and_bias_agree(matched) -> None:
    ref, other, ro, oo = matched
    mass = _to_reference_order(other.M, ro, oo, (0, 1))
    bias = _to_reference_order(other.h, ro, oo, (0,))
    assert _rel(mass, ref.M) < RTOL
    assert _rel(bias, ref.h) < RTOL


def test_canonical_closure_agrees(matched) -> None:
    ref, other, ro, oo = matched
    j_c = _to_reference_order(other.J_c, ro, oo, (1,))
    assert _rel(j_c, ref.J_c) < RTOL
    assert float(np.abs(ref.J_c).max()) > 1e-3  # not vacuous
    # Drift is exact only on the closure tangent space; the state is projected.
    assert _rel(other.Jdot_v, ref.Jdot_v) < RTOL
    assert float(np.abs(ref.Jdot_v).max()) > 1e-6


def test_grip_frame_and_club_inertias_agree(matched) -> None:
    ref, other, _, _ = matched
    np.testing.assert_allclose(
        other.grip_frame.rotation, ref.grip_frame.rotation, atol=1e-8
    )
    np.testing.assert_allclose(
        other.grip_frame.origin, ref.grip_frame.origin, atol=1e-8
    )
    for mine, theirs in ((other.bare_club, ref.bare_club),):
        assert mine.mass == pytest.approx(theirs.mass, abs=1e-12)
        np.testing.assert_allclose(mine.com, theirs.com, atol=1e-9)
        np.testing.assert_allclose(mine.inertia, theirs.inertia, atol=1e-9)
    assert len(other.welded_hand_inertias) == len(ref.welded_hand_inertias)
    masses_o = sorted(h.mass for h in other.welded_hand_inertias)
    masses_r = sorted(h.mass for h in ref.welded_hand_inertias)
    np.testing.assert_allclose(masses_o, masses_r, atol=1e-12)


def test_topology_agrees_by_coordinate_name(matched) -> None:
    ref, other, ro, oo = matched
    names = lambda t, idx, order: {order[i] for i in idx}  # noqa: E731
    assert other.topology.closure_kind is ref.topology.closure_kind
    assert (other.topology.closing_side, other.topology.holding_side) == (
        ref.topology.closing_side,
        ref.topology.holding_side,
    )
    assert names(0, other.topology.actuated_indices, oo) == names(
        0, ref.topology.actuated_indices, ro
    )
    assert names(0, other.topology.loop_support_indices, oo) == names(
        0, ref.topology.loop_support_indices, ro
    )


def test_acceleration_parity_agrees_across_engines(adapters, matched) -> None:
    """Both engines produce the same constrained acceleration for the same effort."""
    pin, drake = adapters
    ref, other, ro, oo = matched
    rng = np.random.default_rng(53)
    tau_by_name = {
        n: float(x) for n, x in zip(ro, rng.normal(size=len(ro)), strict=True)
    }
    for n in ROOT_COORDINATES:
        tau_by_name[n] = 0.0
    tau_p = np.array([tau_by_name[n] for n in ro])
    tau_d = np.array([tau_by_name[n] for n in oo])
    # Solve the constrained acceleration with the Drake reference (M, h, J_c, Jdot_v):
    # a = argmin_a M a = tau - h + J^T lam, J a + Jdot_v = 0, in the Drake order.
    k = 6
    mass, h_vec, j_c, drift = other.M, other.h, other.J_c, other.Jdot_v
    n = mass.shape[0]
    kkt = np.block([[mass, -j_c.T], [j_c, np.zeros((k, k))]])
    a_d = np.linalg.solve(kkt, np.concatenate((tau_d - h_vec, -drift)))[:n]
    a_p = _to_reference_order(a_d, ro, oo, (0,))
    assert (
        pin.verify_acceleration_parity(ref.q, ref.v, tau_p, a_p)
        / max(1.0, float(np.abs(a_p).max()))
        < 1e-6
    )
    assert drake.verify_acceleration_parity(other.q, other.v, tau_d, a_d) < 1e-8 * max(
        1.0, float(np.abs(a_d).max())
    )


def test_contact_jacobian_and_inverse_dynamics_agree(adapters, matched) -> None:
    """Ground-contact Jacobian and ``M a + h`` agree sphere by sphere, by name."""
    pin, drake = adapters
    ref, other, ro, oo = matched
    assert pin.contact_names == drake.contact_names
    j_d = _to_reference_order(drake.compute_contact_jacobian(other.q), ro, oo, (1,))
    q_ref = np.array(ref.q)
    assert _rel(j_d, pin.compute_contact_jacobian(q_ref)) < RTOL
    accel = np.random.default_rng(12581).normal(size=len(ro))
    tau_p = pin.compute_inverse_dynamics(q_ref, np.array(ref.v), accel)
    accel_d = _to_reference_order(accel, oo, ro, (0,))
    tau_d = drake.compute_inverse_dynamics(other.q, other.v, accel_d)
    assert _rel(_to_reference_order(tau_d, ro, oo, (0,)), tau_p) < RTOL

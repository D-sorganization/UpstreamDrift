"""NSX-51: MuJoCo closed-loop snapshot by hard-KKT evaluation (#12579).

The snapshot is assembled from ``M``, ``h`` and the weld Jacobian of the
compiled tree; the closure is enforced as a rigid constraint, not through the
solref/solimp compliance of stock ``mj_step``.  References are built from raw
MuJoCo calls on a second ``MjData`` and from a numpy KKT solve, never through
the adapter methods under test.  Cross-engine equality with Pinocchio is in
``test_cross_engine_snapshot.py``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.engine_core.capabilities import (
    Capability,
    CapabilityLevel,
    CapabilityUnavailable,
)
from src.shared.python.loop_nullspace.contracts import (
    ClosedLoopSnapshot,
    ClosureKind,
    InputProvenance,
)
from src.shared.python.loop_nullspace.snapshot import (
    has_closed_loop_protocol,
    parity_effort,
    snapshot_from_adapter,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_mujoco]

REPO = Path(__file__).resolve().parents[3]
EVIDENCE = REPO / "docs/development/full_body_models/evidence/ground_support"
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"
PROVENANCE = InputProvenance(kind="test", source="dynamics_record frame 400")
ATOL = 1e-9


def _scaled(x: np.ndarray, scale: float) -> float:
    return float(np.max(np.abs(x))) / max(1.0, scale)


@pytest.fixture(scope="module")
def mj():
    return pytest.importorskip("mujoco")


@pytest.fixture(scope="module")
def adapter(mj):
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        MujocoForceAdapter,
    )

    return MujocoForceAdapter(SPEC_PATH.read_bytes())


@pytest.fixture(scope="module")
def state(adapter):
    """A closure-consistent state, in the adapter's native (dof) order."""
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    plant = VectorPlant("mujoco", SPEC_PATH.read_bytes())
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
def raw(adapter, state, mj):
    """Independent raw MuJoCo evaluation and an exact numpy KKT solve."""
    model = adapter._model.model
    data = mj.MjData(model)
    q, v = state
    data.qpos[:] = q
    data.qvel[:] = v
    mj.mj_forward(model, data)
    mass = np.zeros((model.nv, model.nv))
    mj.mj_fullM(model, mass, data.qM)
    sites = [model.site(f"native_closure_{s}").id for s in ("a", "b")]
    jacs = []
    for site in sites:
        jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
        mj.mj_jacSite(model, data, jp, jr, site)
        jacs.append(np.vstack([jp, jr]))
    jac = jacs[0] - jacs[1]
    rng = np.random.default_rng(12579)
    tau = np.zeros(model.nv)
    tau[list(adapter.actuated_indices)] = rng.normal(size=len(adapter.actuated_indices))
    # exact rigid-closure solve: [M -J^T; J 0] [a; lam] = [tau - h; -drift]
    h = data.qfrc_bias.copy()
    step = 1e-6
    jdots = []
    for sign in (+1, -1):
        data.qpos[:] = q + sign * step * v
        mj.mj_forward(model, data)
        pair = []
        for site in sites:
            jp, jr = np.zeros((3, model.nv)), np.zeros((3, model.nv))
            mj.mj_jacSite(model, data, jp, jr, site)
            pair.append(np.vstack([jp, jr]))
        jdots.append(pair[0] - pair[1])
    drift = ((jdots[0] - jdots[1]) / (2 * step)) @ v
    data.qpos[:] = q
    mj.mj_forward(model, data)
    nv = model.nv
    kkt = np.block([[mass, -jac.T], [jac, np.zeros((6, 6))]])
    sol = np.linalg.solve(kkt, np.concatenate([tau - h, -drift]))
    return {
        "M": mass,
        "h": h,
        "J": jac,
        "drift": drift,
        "tau": tau,
        "a": sol[:nv],
        "lam": sol[nv:],
    }


def test_capability_is_full_and_protocol_is_complete(adapter) -> None:
    assert adapter.capability(Capability.CLOSED_LOOP_SNAPSHOT) is CapabilityLevel.FULL
    assert has_closed_loop_protocol(adapter)
    from src.engines.physics_engines.mujoco.python.native_model import (
        CLOSURE_CONVENTION,
    )

    assert adapter.native_closure_convention == CLOSURE_CONVENTION


def test_other_capabilities_stay_none(adapter) -> None:
    others = [c for c in Capability if c is not Capability.CLOSED_LOOP_SNAPSHOT]
    assert others
    assert all(adapter.capability(c) is CapabilityLevel.NONE for c in others)


def test_snapshot_does_not_depend_on_the_stock_kkt_regularisation(
    adapter, state, raw
) -> None:
    """The snapshot never calls the Tikhonov solve; the parity audit is exact."""
    q, v = state
    model = adapter._model
    saved = model.kkt_regularization
    try:
        model.kkt_regularization = 1e-3
        snap = snapshot_from_adapter(
            adapter, q, v, raw["a"], input_provenance=PROVENANCE
        )
        err = adapter.verify_acceleration_parity(q, v, raw["tau"], raw["a"])
    finally:
        model.kkt_regularization = saved
    np.testing.assert_allclose(snap.M, raw["M"], rtol=0, atol=1e-10)
    assert err / max(1.0, float(np.max(np.abs(raw["a"])))) < 1e-7


def test_coordinate_order_matches_state_vector_order(adapter) -> None:
    """The adapter order is the MuJoCo dof order its vectors use (not spec order)."""
    model = adapter._model
    by_dof = sorted(model._indices, key=model._indices.get)
    assert list(adapter.coordinate_order) == by_dof
    assert sorted(adapter.coordinate_order) == sorted(model.coordinate_order)


def test_synthetic_mujoco_still_refuses() -> None:
    from src.shared.python.motion_matching.multi_engine_torque_allocator import (
        create_engine_force_adapter,
    )

    synth = create_engine_force_adapter("mujoco", allow_synthetic=True, nv=44)
    z = np.zeros(synth.nv)
    with pytest.raises(CapabilityUnavailable, match="closed_loop_snapshot"):
        snapshot_from_adapter(synth, z, z, z, input_provenance=PROVENANCE)


def test_snapshot_equals_independent_raw_mujoco(adapter, state, raw) -> None:
    from src.engines.physics_engines.mujoco.python.native_model import (
        CLOSURE_CONVENTION,
    )
    from src.shared.python.loop_nullspace.closure_convention import (
        to_canonical_closure,
    )

    q, v = state
    snap = snapshot_from_adapter(adapter, q, v, raw["a"], input_provenance=PROVENANCE)
    assert isinstance(snap, ClosedLoopSnapshot)
    assert snap.topology.closure_kind is ClosureKind.WELD6
    assert snap.topology.coordinate_order == tuple(adapter.coordinate_order)
    np.testing.assert_allclose(snap.M, raw["M"], rtol=0, atol=1e-10)
    np.testing.assert_allclose(snap.h, raw["h"], rtol=0, atol=1e-10)
    rot, origin = adapter.closure_wrench_frame(q)
    j_c, jdot_v = to_canonical_closure(
        raw["J"],
        raw["drift"],
        native=CLOSURE_CONVENTION,
        native_frame_rotation=np.eye(3),
        grip_frame_rotation=rot,
        native_reference_point=origin,
        grip_reference_point=origin,
    )
    np.testing.assert_allclose(snap.J_c, j_c, rtol=0, atol=1e-10)
    scale = float(np.max(np.abs(jdot_v)))
    assert _scaled(snap.Jdot_v - jdot_v, scale) < 1e-7
    assert snap.J_g is None
    assert snap.model_provenance.content_hash == adapter.model_hash


def test_drift_method_is_recorded_in_provenance(adapter, state, raw) -> None:
    q, v = state
    snap = snapshot_from_adapter(adapter, q, v, raw["a"], input_provenance=PROVENANCE)
    assert adapter.closure_drift_method == "central_finite_difference"
    assert "drift=central_finite_difference" in snap.model_provenance.version
    assert snap.model_provenance.name == "mujoco"


def test_loop_equation_holds_with_the_exact_rigid_multiplier(
    adapter, state, raw
) -> None:
    """``M a + h = S^T tau + J_c^T lambda`` with the hard-KKT ``a`` and ``lambda``."""
    from src.shared.python.loop_nullspace.closure_convention import (
        closure_conversion_matrix,
    )
    from src.engines.physics_engines.mujoco.python.native_model import (
        CLOSURE_CONVENTION,
    )

    q, v = state
    snap = snapshot_from_adapter(adapter, q, v, raw["a"], input_provenance=PROVENANCE)
    rot, origin = adapter.closure_wrench_frame(q)
    c_mat = closure_conversion_matrix(
        native=CLOSURE_CONVENTION,
        native_frame_rotation=np.eye(3),
        grip_frame_rotation=rot,
        native_reference_point=origin,
        grip_reference_point=origin,
    )
    lam_c = np.linalg.solve(c_mat.T, raw["lam"])  # lambda_native = C^T lambda_c
    tau_act = raw["tau"][list(snap.topology.actuated_indices)]
    lhs = snap.generalized_load()
    rhs = snap.S.T @ tau_act + snap.J_c.T @ lam_c
    assert _scaled(lhs - rhs, float(np.max(np.abs(lhs)))) < ATOL


def test_closure_acceleration_constraint_is_exact(adapter, state, raw) -> None:
    q, v = state
    snap = snapshot_from_adapter(adapter, q, v, raw["a"], input_provenance=PROVENANCE)
    assert _scaled(snap.J_c @ snap.v, float(np.max(np.abs(snap.v)))) < 1e-10
    accel = snap.J_c @ snap.a + snap.Jdot_v
    assert _scaled(accel, float(np.max(np.abs(snap.Jdot_v)))) < ATOL


def test_central_difference_drift_is_second_order_accurate(adapter, state) -> None:
    """Halving the step changes the drift far less than the engine's own jacDot."""
    q, v = state
    model = adapter._model
    fine = model.closure_drift_central_difference(q, v, step=1e-6)
    coarse = model.closure_drift_central_difference(q, v, step=2e-6)
    stock = adapter.compute_closure_drift(q, v)
    scale = float(np.max(np.abs(fine)))
    assert _scaled(fine - coarse, scale) < 1e-8
    np.testing.assert_array_equal(stock, fine)


def test_stock_jacdot_agrees_with_central_difference_to_first_order(
    adapter, state
) -> None:
    """``mj_jacDot`` (engine-internal forward difference) is the declared fallback."""
    q, v = state
    model = adapter._model
    adapter._prepare_state(q, v)
    _, stock = model.evaluate_weld_closure()
    central = model.closure_drift_central_difference(q, v)
    assert _scaled(stock - central, float(np.max(np.abs(central)))) < 1e-4


def test_closure_jacobian_matches_finite_difference_of_the_weld_pose(
    adapter, state, mj
) -> None:
    q, _ = state
    model = adapter._model.model
    data = mj.MjData(model)
    sites = [model.site(f"native_closure_{s}").id for s in ("a", "b")]
    dq = np.random.default_rng(5).normal(size=model.nv)

    def pose(x: np.ndarray) -> np.ndarray:
        from scipy.spatial.transform import Rotation

        data.qpos[:] = x
        mj.mj_kinematics(model, data)
        a, b = sites
        ra, rb = (data.site_xmat[i].reshape(3, 3) for i in (a, b))
        return np.r_[
            data.site_xpos[a] - data.site_xpos[b],
            Rotation.from_matrix(ra @ rb.T).as_rotvec(),
        ]

    h = 1e-6
    fd = (pose(q + h * dq) - pose(q - h * dq)) / (2 * h)
    np.testing.assert_allclose(
        adapter.compute_grip_jacobian(q) @ dq,
        fd,
        rtol=0,
        atol=1e-5 * max(1.0, float(np.abs(fd).max())),
    )


def test_club_inertia_split_is_bare_and_welded(adapter) -> None:
    import json

    from src.shared.python.biomechanics.club_inertia_partition import (
        club_solids_partition,
    )

    bare, hands = adapter.club_inertia_split()
    expected = club_solids_partition(json.loads(SPEC_PATH.read_bytes()))
    assert bare.mass == pytest.approx(expected.bare_club.mass_kg, abs=1e-12)
    assert len(hands) == len(expected.welded_hands) >= 1
    assert sum(h.mass for h in hands) == pytest.approx(
        sum(h.mass_kg for h in expected.welded_hands.values()), abs=1e-12
    )


def test_club_inertia_com_agrees_with_the_mujoco_body_in_world_axes(
    adapter, state, mj
) -> None:
    """The closure-frame club COM equals the MuJoCo body-frame COM, in the world."""
    q, _ = state
    bare, _ = adapter.club_inertia_split()
    rot, origin = adapter.closure_wrench_frame(q)
    com_world = rot @ np.asarray(bare.com) + origin
    solid = adapter._model._solid_group("club")
    model, data = adapter._model.model, adapter._model.data
    adapter._prepare_state(q)
    body = int(model.site_bodyid[adapter._model._closure[1]])
    expected = (
        np.array(data.xmat[body]).reshape(3, 3) @ np.asarray(solid.com_m)
        + data.xpos[body]
    )
    np.testing.assert_allclose(com_world, expected, atol=1e-9)


def test_topology_loop_support_covers_every_nonzero_jacobian_column(
    adapter, state
) -> None:
    q, v = state
    topo = adapter.closure_topology()
    jac = adapter.compute_grip_jacobian(q)
    # Common-ancestor columns cancel only up to the site separation (MuJoCo takes
    # each site's linear rows at its own point), i.e. round-off on a closed loop.
    nonzero = {i for i in range(topo.n) if np.abs(jac[:, i]).max() > 1e-9}
    assert nonzero <= set(topo.loop_support_indices)
    assert set(topo.actuated_indices) == set(adapter.actuated_indices)
    assert set(topo.unactuated_indices) == set(range(6))


def test_state_shape_is_validated(adapter, state) -> None:
    q, v = state
    with pytest.raises(ValueError):
        snapshot_from_adapter(adapter, q[:-1], v, v, input_provenance=PROVENANCE)


def test_parity_is_the_hard_kkt_acceleration_with_applied_effort_only(
    adapter, state, raw
) -> None:
    q, v = state
    err = adapter.verify_acceleration_parity(q, v, raw["tau"], raw["a"])
    assert err / max(1.0, float(np.max(np.abs(raw["a"])))) < 1e-7


def test_parity_detects_a_wrong_target_and_an_unconstrained_target(
    adapter, state, raw
) -> None:
    q, v = state
    wrong = raw["a"].copy()
    wrong[10] += 0.5
    assert adapter.verify_acceleration_parity(q, v, raw["tau"], wrong) > 0.1
    free = np.linalg.solve(raw["M"], raw["tau"] - raw["h"])
    assert adapter.verify_acceleration_parity(q, v, raw["tau"], free) > 1e-3


def test_parity_effort_passes_applied_effort_only(adapter) -> None:
    tau, jac, lam = np.ones(3), np.eye(6)[:, :3], np.arange(6.0)
    np.testing.assert_array_equal(parity_effort(adapter, tau, jac, lam), tau)


REGISTRY = REPO / "tests/integration/cross_engine/divergence_registry.yaml"
STOCK_FRAMES = (100, 250, 400, 500)


def _stock_closure_ratios(adapter, mj, frame: int) -> list[float]:
    """``|J qacc_stock + Jdot v| / |J a_free + Jdot v|`` for two effort levels.

    Stock ``mj_forward`` (contacts and limits off so only the weld acts) against
    the closure acceleration the exact rigid solve drives to zero.
    """
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    model = adapter._model.model
    flags = model.opt.disableflags
    model.opt.disableflags |= (
        mj.mjtDisableBit.mjDSBL_CONTACT | mj.mjtDisableBit.mjDSBL_LIMIT
    )
    try:
        plant = VectorPlant("mujoco", SPEC_PATH.read_bytes())
        with np.load(EVIDENCE / "dynamics_record.npz") as rec:
            proj = project_to_closure(plant, rec["q"][frame], rec["v"][frame])
        qn, vn = plant._named(proj.q), plant._named(proj.v)
        q = np.array([qn[n] for n in adapter.coordinate_order])
        v = np.array([vn[n] for n in adapter.coordinate_order])
        mass, bias = adapter.compute_mass_and_bias(q, v)
        jac = adapter.compute_grip_jacobian(q)
        drift = adapter.compute_closure_drift(q, v)
        ratios = []
        for scale in (0.0, 1.0):
            tau = np.zeros(model.nv)
            tau[6:] = scale * np.random.default_rng(frame).normal(size=model.nv - 6)
            data = mj.MjData(model)
            data.qpos[:], data.qvel[:], data.qfrc_applied[:] = q, v, tau
            mj.mj_forward(model, data)
            free = np.linalg.solve(mass, tau - bias)
            stock = np.max(np.abs(jac @ data.qacc + drift))
            ratios.append(float(stock / np.max(np.abs(jac @ free + drift))))
        return ratios
    finally:
        model.opt.disableflags = flags


def test_stock_soft_stepping_is_a_separate_registered_divergence(adapter, mj) -> None:
    """Stock ``mj_forward`` keeps the weld compliant (solref/solimp, review C).

    The exact rigid solve closes the loop to round-off; stock stepping leaves a
    closure-acceleration residual that is a finite fraction of the free one.  It
    has its own divergence-registry entry with the measured envelope as its
    tolerance, and is never the source of a snapshot.
    """
    from src.shared.python.engine_core.cross_engine_validator import (
        DivergenceRegistry,
    )

    entry = DivergenceRegistry.from_yaml(REGISTRY).find(
        check_name="closed_loop_snapshot_conformance",
        metric_name="closure_acceleration_residual_ratio",
        engine1="mujoco-stock-step",
        engine2="mujoco-hard-kkt",
    )
    assert entry is not None and entry.id == "mujoco-stock-soft-weld-vs-hard-kkt"
    assert adapter._model.model.neq >= 1
    ratios = [r for f in STOCK_FRAMES for r in _stock_closure_ratios(adapter, mj, f)]
    assert min(ratios) > 1e-3, "stock stepping unexpectedly closes the loop exactly"
    assert max(ratios) <= entry.tolerance, (
        f"stock closure residual ratio {max(ratios):.3g} exceeds the registered "
        f"envelope {entry.tolerance}"
    )

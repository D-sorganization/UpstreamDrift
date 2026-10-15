"""NSX-15: closed-loop decomposition on ForceNullSpace (#12560)."""

from __future__ import annotations

import dataclasses
import json

import numpy as np
import pytest

from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    BalanceInconsistentError,
    Decomposition,
    ZeroSideCase,
    decompose,
    decompose_trajectory,
    dimension_accounting,
    min_metric_torque,
    min_metric_torque_trajectory,
    zero_side_solve,
    zero_side_trajectory,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)

pytestmark = pytest.mark.unit

ACTUATIONS = ("full", "club_unactuated")
TOL = 1e-9


def _planar(kind: str, actuation: str):
    if kind == "straight_arm":
        return straight_arm_snapshot(actuation=actuation)
    return planar_golfer_snapshot(generic_planar_config(), actuation=actuation)


CASES = [(k, a) for k in ("generic", "straight_arm") for a in ACTUATIONS]


def _golden(kind, actuation):
    case = _planar(kind, actuation)
    assert case.golden is not None
    return case.snapshot, case.golden


def _metric(snap, which: str) -> AllocationMetric:
    if which == "identity":
        return AllocationMetric.identity(snap.topology.n_actuated)
    return AllocationMetric.inverse_mass(snap)


# ------------------------------------------------------------ dimensions --


@pytest.mark.parametrize(("kind", "actuation"), CASES)
def test_dimension_counts_equal_goldens(kind, actuation):
    snap, golden = _golden(kind, actuation)
    want = golden.get("dimensions")
    d = decompose(snap, weights=AllocationMetric.identity(snap.topology.n_actuated))
    assert d.rank_A == want["rank_A"]
    assert d.dim_x == want["dim_x"]
    assert d.nullity == want["equality_nullity"]
    assert d.multiplier_gauge == want["multiplier_gauge"]
    assert d.attainable_torque_variation == want["attainable_torque_variation"]
    assert d.rank_J_c == want["rank_J_c"]
    assert d.rank_J_cu_T == want["rank_J_cu_T"]
    assert d.nullity == d.multiplier_gauge + d.attainable_torque_variation
    assert d.N.shape == (want["dim_x"], want["equality_nullity"])


def test_straight_arm_rank_drop():
    full = decompose(
        _planar("generic", "full").snapshot, weights=AllocationMetric.identity(5)
    )
    sing = decompose(
        _planar("straight_arm", "full").snapshot, weights=AllocationMetric.identity(5)
    )
    assert sing.rank_J_c == full.rank_J_c - 1
    assert sing.multiplier_gauge == 1
    assert sing.attainable_torque_variation == full.attainable_torque_variation - 1
    assert sing.nullity == full.nullity


@pytest.mark.parametrize(
    ("variant", "kwargs"),
    [
        ("fixed_base_arms_only", {}),
        ("floating_root_no_feet", {"floating_root": True}),
        ("floating_root_two_point_feet", {"floating_root": True, "feet": True}),
    ],
)
def test_spatial_dimension_accounting_equals_goldens(variant, kwargs):
    case = spatial_two_arm_club_snapshot(**kwargs)
    want = case.golden.payload
    acc = dimension_accounting(case.snapshot)
    assert acc.dim_x == want["dim_x"]
    assert acc.rank_A == want["rank_A"]
    assert acc.nullity == want["equality_nullity"]
    assert acc.n_g == want["n_g"]
    assert acc.rank_J_c == want["rank_J_c"]
    assert acc.rank_J_cu_T == want["rank_J_cu_T"]
    assert acc.multiplier_gauge == want["multiplier_gauge"]
    assert (
        acc.attainable_torque_variation == want["attainable_torque_variation_fg_fixed"]
    )
    assert acc.multiplier_gauge_fg_free == want["gauge_total_fg_free"]
    assert (
        acc.attainable_torque_variation_fg_free
        == want["attainable_torque_variation_fg_free"]
    )
    assert acc.rank_stacked == want["rank_stacked_J_c_J_g"]


def test_spatial_fixed_base_decomposes_with_golden_counts():
    case = spatial_two_arm_club_snapshot()
    d = decompose(case.snapshot, weights=AllocationMetric.identity(14))
    assert (d.rank_A, d.nullity, d.multiplier_gauge) == (14, 6, 0)
    assert d.attainable_torque_variation == 6


def test_floating_root_unbalanced_reports_inconsistency():
    case = spatial_two_arm_club_snapshot(floating_root=True)
    with pytest.raises(BalanceInconsistentError, match="unactuated") as exc:
        decompose(case.snapshot, weights=AllocationMetric.identity(14))
    assert exc.value.residual_scaled > 1e-6
    assert isinstance(exc.value, ValueError)


# ------------------------------------------------------------- zero side --


@pytest.mark.parametrize(("kind", "actuation"), CASES)
def test_zero_side_classification_is_exact(kind, actuation):
    snap, golden = _golden(kind, actuation)
    for side, want in golden.get("zero_side").items():
        res = zero_side_solve(snap, side)
        assert res.case is ZeroSideCase[want["case"]], (side, res)
        assert list(res.indices) == want["indices"]
        assert res.n_equations == want["n_equations"]
        assert res.n_unknowns == want["n_unknowns"]
        assert res.rank == want["rank"]
        assert res.nullity == want["nullity"]
        if res.case is ZeroSideCase.INCONSISTENT:
            assert res.residual == pytest.approx(want["residual"], rel=1e-9)
            assert res.residual_scaled > 1e-6
            assert res.label == "minimum-residual side effort"
        else:
            assert res.residual == pytest.approx(0.0, abs=1e-12)
            assert res.residual_scaled < 1e-9
            assert res.label == ""
        np.testing.assert_allclose(
            res.multiplier, want["lambda_min_residual_min_norm"], rtol=1e-9, atol=1e-10
        )


def test_zero_side_records_damping_and_never_relaxes():
    snap, _ = _golden("generic", "full")
    res = zero_side_solve(snap, "lead")
    assert res.damping == 0.0
    assert res.case is ZeroSideCase.INCONSISTENT
    damped = zero_side_solve(snap, "trail", damping=1e-3)
    assert damped.damping == 1e-3
    assert damped.case is ZeroSideCase.CONSISTENT_UNIQUE
    # damping never turns an inconsistent system into a consistent one
    assert zero_side_solve(snap, "lead", damping=1e-3).case is ZeroSideCase.INCONSISTENT


def test_zero_side_explicit_indices_override_and_validation():
    snap, _ = _golden("generic", "full")
    res = zero_side_solve(snap, "trail", indices=(3, 4))
    assert res.indices == (3, 4)
    with pytest.raises(ValueError, match="side"):
        zero_side_solve(snap, "left_knee")
    with pytest.raises(ValueError, match="indices"):
        zero_side_solve(snap, "trail", indices=(9,))
    with pytest.raises(ValueError, match="damping"):
        zero_side_solve(snap, "trail", damping=-1.0)


def test_zero_side_spatial_chain_is_overdetermined_but_rank_six():
    case = spatial_two_arm_club_snapshot()
    res = zero_side_solve(case.snapshot, "lead")
    assert res.n_equations == 7
    assert res.n_unknowns == 6
    assert res.rank == 6


# --------------------------------------------------------------- metrics --


def test_metric_validation_and_labels():
    snap, _ = _golden("generic", "full")
    assert AllocationMetric.inverse_mass(snap).label == "C-METRIC-MINV"
    assert AllocationMetric.identity(5).label == "C-METRIC-IDENTITY"
    with pytest.raises(ValueError, match="positive definite"):
        AllocationMetric(np.diag([1.0, 1.0, 0.0, 1.0, 1.0]), "bad")
    with pytest.raises(ValueError, match="symmetric"):
        AllocationMetric(np.triu(np.ones((3, 3))) + 5 * np.eye(3), "asym")
    with pytest.raises(ValueError, match="label"):
        AllocationMetric(np.eye(2), "")
    with pytest.raises(ValueError, match="shape"):
        min_metric_torque(snap, np.eye(3))
    with pytest.raises(TypeError):
        decompose(snap, weights="M_inv")  # type: ignore[arg-type]


def test_raw_matrix_metric_is_labelled_user_declared():
    snap, _ = _golden("generic", "full")
    d = decompose(snap, weights=np.eye(5))  # type: ignore[arg-type]
    assert d.metric_label == "USER_DECLARED"


@pytest.mark.parametrize(("kind", "actuation"), CASES)
@pytest.mark.parametrize("which", ["identity", "minv"])
def test_min_metric_torque_equals_golden(kind, actuation, which):
    snap, golden = _golden(kind, actuation)
    want = golden.get("min_metric_torque", which)
    metric = _metric(snap, which)
    tau = min_metric_torque(snap, metric)
    np.testing.assert_allclose(tau, want["tau_actuated"], rtol=1e-9, atol=1e-10)
    assert float(tau @ metric.W @ tau) == pytest.approx(want["objective"], rel=1e-9)
    d = decompose(snap, weights=metric)
    np.testing.assert_allclose(d.torque_particular, tau, atol=1e-12)
    np.testing.assert_allclose(
        d.x_p[d.n_actuated : d.n_actuated + d.m],
        want["lambda_min_norm"],
        rtol=1e-9,
        atol=1e-10,
    )


def test_min_metric_torque_minv_closed_form():
    snap, golden = _golden("generic", "full")
    w = AllocationMetric.inverse_mass(snap).W
    b = snap.generalized_load()
    jc = snap.J_c
    lam = np.linalg.solve(jc @ w @ jc.T, jc @ w @ b)
    closed = b - jc.T @ lam
    np.testing.assert_allclose(min_metric_torque(snap, w), closed, atol=1e-10)
    np.testing.assert_allclose(
        closed, golden.get("min_metric_torque", "minv", "closed_form_tau"), atol=1e-10
    )


def test_min_metric_is_minimal_among_family():
    snap, _ = _golden("generic", "full")
    metric = AllocationMetric.inverse_mass(snap)
    d = decompose(snap, weights=metric)
    base = d.torque_particular
    obj = float(base @ metric.W @ base)
    rng = np.random.default_rng(3)
    for _ in range(20):
        t = d.torque(rng.normal(size=d.nullity))
        assert float(t @ metric.W @ t) >= obj - 1e-12


# ------------------------------------------------------------ equivalence --


def _kkt_acceleration(snap, tau_act, lam_free=None):
    """Independent constrained forward dynamics: ``a`` for given actuated torque."""
    n, m = snap.M.shape[0], snap.J_c.shape[0]
    kkt = np.block([[snap.M, -snap.J_c.T], [snap.J_c, np.zeros((m, m))]])
    rhs = np.concatenate([snap.S.T @ tau_act - snap.h, -snap.Jdot_v])
    sol = np.linalg.solve(kkt, rhs)
    return sol[:n], sol[n:]


@pytest.mark.parametrize(("kind", "actuation"), CASES)
def test_planar_family_reproduces_measured_acceleration(kind, actuation):
    snap, _ = _golden(kind, actuation)
    if np.linalg.matrix_rank(snap.J_c) < snap.J_c.shape[0]:
        pytest.skip("KKT matrix is singular at the straight-arm rank drop")
    d = decompose(snap, weights=AllocationMetric.inverse_mass(snap))
    rng = np.random.default_rng(11)
    for _ in range(10):
        z = rng.normal(size=d.nullity) * 10.0
        a, _lam = _kkt_acceleration(snap, d.torque(z))
        assert np.max(np.abs(a - snap.a)) / max(1.0, np.max(np.abs(snap.a))) < TOL


@pytest.mark.parametrize("kind", ["generic"])
def test_family_satisfies_balance_exactly(kind):
    for actuation in ACTUATIONS:
        snap, _ = _golden(kind, actuation)
        d = decompose(snap, weights=AllocationMetric.identity(snap.topology.n_actuated))
        a_mat = np.hstack([snap.S.T, snap.J_c.T])
        z = np.random.default_rng(0).normal(size=d.nullity)
        res = a_mat @ d.x(z) - snap.generalized_load()
        assert np.max(np.abs(res)) < 1e-10
        assert np.max(np.abs(a_mat @ d.N)) < 1e-9


def _constraint_dynamics(case_kwargs):
    """Pinocchio ``constraintDynamics`` with the spatial fixture's weld."""
    pin = pytest.importorskip("pinocchio")
    from src.shared.python.loop_nullspace import spatial_fixture

    sp = spatial_fixture._build_closed(case_kwargs.get("floating_root", False), False)
    frames = sp.model.frames
    trail, lead = frames[sp.trail_grip], frames[sp.lead_grip]
    constraint = pin.RigidConstraintModel(
        pin.ContactType.CONTACT_6D,
        sp.model,
        trail.parentJoint,
        trail.placement,
        lead.parentJoint,
        lead.placement,
        pin.ReferenceFrame.LOCAL,
    )
    data = sp.model.createData()
    cdata = constraint.createData()
    pin.initConstraintDynamics(sp.model, data, [constraint], [cdata])

    def accel(q, v, tau):
        return np.array(
            pin.constraintDynamics(sp.model, data, q, v, tau, [constraint], [cdata])
        )

    return accel


@pytest.mark.requires_pinocchio
@pytest.mark.parametrize("floating_root", [False, True])
def test_constraint_dynamics_reproduces_measured_acceleration(floating_root):
    kwargs = {"floating_root": floating_root}
    base = spatial_two_arm_club_snapshot(**kwargs).snapshot
    accel = _constraint_dynamics(kwargs)
    n_a = base.topology.n_actuated
    rng = np.random.default_rng(5)
    tau0 = np.zeros(base.topology.n)
    tau0[list(base.topology.actuated_indices)] = rng.normal(size=n_a)
    a_meas = accel(base.q, base.v, tau0)
    snap = dataclasses.replace(base, a=a_meas)
    d = decompose(snap, weights=AllocationMetric.inverse_mass(snap))
    assert d.nullity == 6
    for _ in range(8):
        z = rng.normal(size=d.nullity) * 5.0
        tau = np.zeros(snap.topology.n)
        tau[list(snap.topology.actuated_indices)] = d.torque(z)
        a = accel(snap.q, snap.v, tau)
        scale = max(1.0, float(np.max(np.abs(a_meas))))
        assert np.max(np.abs(a - a_meas)) / scale < TOL


# ------------------------------------------------------------ evidence ----


def test_diagnostics_are_serializable_and_carry_evidence():
    snap, _ = _golden("generic", "club_unactuated")
    d = decompose(snap, weights=AllocationMetric.inverse_mass(snap))
    payload = d.to_dict()
    text = json.dumps(payload)
    back = json.loads(text)
    assert back["rank_A"] == d.rank_A
    assert back["nullity"] == d.nullity
    assert back["metric_label"] == "C-METRIC-MINV"
    assert back["scales"] == {"force": 100.0, "moment": 5.0}
    assert len(back["singular_values_scaled"]) == d.rank_A
    assert back["condition_number"] == pytest.approx(d.condition_number)
    assert back["x_p"] == pytest.approx(list(d.x_p))
    rec = d.records["attainable_torque_variation"]
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.model_provenance == snap.model_provenance
    assert rec.input_provenance == snap.input_provenance
    assert any(c.startswith("C-SCALE-FORCE-MOMENT") for c in rec.conventions)
    tau_rec = d.records["torque_particular"]
    assert tau_rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert "C-METRIC-MINV" in " ".join(tau_rec.hypotheses)
    assert isinstance(d, Decomposition)
    assert (
        json.loads(json.dumps(zero_side_solve(snap, "lead").to_dict()))["case"]
        == "INCONSISTENT"
    )


def test_decomposition_arrays_are_read_only():
    snap, _ = _golden("generic", "full")
    d = decompose(snap, weights=AllocationMetric.identity(5))
    with pytest.raises(ValueError, match="read-only"):
        d.x_p[0] = 1.0
    with pytest.raises(ValueError, match="read-only"):
        d.N[0, 0] = 1.0


def test_wrong_metric_size_rejected():
    snap, _ = _golden("generic", "club_unactuated")
    with pytest.raises(ValueError, match="shape"):
        decompose(snap, weights=AllocationMetric.identity(5))


# ---------------------------------------------------------- admission -----


@pytest.fixture(scope="module")
def admitted() -> AdmittedTrajectory:
    snap = planar_golfer_snapshot(generic_planar_config()).snapshot
    report = admit(
        [snap] * 4,
        gates=AdmissionGates(require_position=False),
        times=np.array([0.0, 0.01, 0.02, 0.03]),
    )
    return AdmittedTrajectory.from_report(report)


def test_trajectory_entry_points_accept_admitted_trajectory(admitted):
    metric = AllocationMetric.identity(5)
    out = decompose_trajectory(admitted, weights=metric)
    assert len(out) == 4
    assert all(isinstance(o, Decomposition) for o in out)
    zs = zero_side_trajectory(admitted, "trail")
    assert [z.case for z in zs] == [ZeroSideCase.CONSISTENT_UNIQUE] * 4
    taus = min_metric_torque_trajectory(admitted, metric)
    assert len(taus) == 4
    np.testing.assert_allclose(taus[0], out[0].torque_particular)


def test_trajectory_entry_points_accept_metric_factory(admitted):
    out = decompose_trajectory(admitted, weights=AllocationMetric.inverse_mass)
    assert out[0].metric_label == "C-METRIC-MINV"


@pytest.mark.parametrize(
    "call",
    [
        lambda t: decompose_trajectory(t, weights=AllocationMetric.identity(5)),
        lambda t: zero_side_trajectory(t, "trail"),
        lambda t: min_metric_torque_trajectory(t, AllocationMetric.identity(5)),
    ],
)
def test_trajectory_entry_points_reject_unadmitted_input(call, admitted):
    snap = planar_golfer_snapshot(generic_planar_config()).snapshot
    with pytest.raises(TypeError, match="AdmittedTrajectory"):
        call([snap, snap, snap])
    with pytest.raises(TypeError, match="AdmittedTrajectory"):
        call(tuple(admitted.snapshots))
    with pytest.raises(TypeError, match="AdmittedTrajectory"):
        call(admitted.report)

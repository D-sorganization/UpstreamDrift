"""NSX-17: invariant quantities with validity gates (#12562)."""

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
    Decomposition,
    decompose,
    decompose_trajectory,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
    spatial_two_arm_club_snapshot,
    straight_arm_snapshot,
)
from src.shared.python.loop_nullspace.invariants import (
    INVARIANT_NAMES,
    InvariantReport,
    compute_invariants,
    min_metric_equivalent,
    tangent_basis,
)

pytestmark = pytest.mark.unit

TOL = 1e-9
INV = AllocationDependence.INVARIANT
UNAV = AllocationDependence.UNAVAILABLE
TIMES = np.array([0.0, 0.01, 0.02, 0.03])
LOOSE = AdmissionGates(
    require_position=False,
    phi_dot_tol=1e9,
    phi_ddot_tol=1e9,
    stance_velocity_tol=1e9,
)
STRICT = AdmissionGates(require_position=False)


def _planar(kind: str, actuation: str):
    if kind == "straight_arm":
        return straight_arm_snapshot(actuation=actuation).snapshot
    return planar_golfer_snapshot(generic_planar_config(), actuation=actuation).snapshot


def _admit(snap, gates=STRICT):
    report = admit([snap] * 4, gates=gates, times=TIMES)
    return AdmittedTrajectory.from_report(report)


def _report(snap, metric=None, gates=STRICT, **options):
    adm = _admit(snap, gates)
    metric = metric or AllocationMetric.identity(snap.topology.n_actuated)
    decs = decompose_trajectory(adm, weights=metric)
    return adm, decs, compute_invariants(adm, decs, **options)


CASES = [
    (k, a) for k in ("generic", "straight_arm") for a in ("full", "club_unactuated")
]


def _scaled_err(a, b):
    return float(np.max(np.abs(np.asarray(a) - np.asarray(b)))) / max(
        1.0, float(np.max(np.abs(b)))
    )


# ------------------------------------------------------------- structure --


@pytest.mark.parametrize(("kind", "actuation"), CASES)
def test_planar_report_records_and_gates(kind, actuation):
    snap = _planar(kind, actuation)
    _, _, rep = _report(snap)
    assert isinstance(rep, InvariantReport)
    assert len(rep.frames) == 4
    for frame in rep.frames:
        assert set(frame.records) == set(INVARIANT_NAMES)
        for name in ("loop_power", "tangent_pairing", "min_metric_torque"):
            rec = frame.records[name]
            assert rec.allocation_dependence is INV, name
            assert np.all(np.isfinite(rec.values))
        for name in ("net_club_wrench", "base_centroidal_rows"):
            rec = frame.records[name]
            assert rec.allocation_dependence is UNAV, name
            assert rec.reason and np.all(np.isnan(rec.values))
    assert "club_ne" in rep.frames[0].records["net_club_wrench"].reason
    assert "floating base" in rep.frames[0].records["base_centroidal_rows"].reason


# ------------------------------------------------------------ invariance --


@pytest.mark.parametrize(("kind", "actuation"), CASES)
def test_invariants_do_not_move_under_null_space_perturbation(kind, actuation):
    snap = _planar(kind, actuation)
    _, decs, rep = _report(snap)
    dec = decs[0]
    rec = rep.frames[0].records
    n_a = snap.topology.n_actuated
    support = [
        i
        for i in snap.topology.loop_support_indices
        if i in snap.topology.actuated_indices
    ]
    pos = [snap.topology.actuated_indices.index(i) for i in support]
    basis = tangent_basis(snap, dec)
    rng = np.random.default_rng(12562)
    for _ in range(8):
        z = 3.0 * rng.normal(size=dec.nullity)
        tau = dec.torque(z)
        assert tau.shape == (n_a,)
        power = float(np.sum(tau[pos] * snap.v[support]))
        assert _scaled_err(power, rec["loop_power"].values[0]) < TOL
        pairing = basis.T @ (snap.S.T @ tau)
        assert _scaled_err(pairing, rec["tangent_pairing"].values) < TOL
        p_w = min_metric_equivalent(tau, dec, AllocationMetric.identity(n_a))
        assert _scaled_err(p_w, rec["min_metric_torque"].values) < TOL


def test_tangent_basis_spans_null_space_of_closure():
    snap = _planar("generic", "full")
    dec = decompose(snap, weights=AllocationMetric.identity(5))
    basis = tangent_basis(snap, dec)
    assert basis.shape == (5, 5 - dec.rank_J_c)
    assert np.max(np.abs(snap.J_c @ basis)) < 1e-12
    assert np.max(np.abs(basis.T @ basis - np.eye(basis.shape[1]))) < 1e-12


def test_min_metric_torque_equals_decomposition_particular():
    snap = _planar("generic", "full")
    _, decs, rep = _report(snap)
    got = rep.frames[0].records["min_metric_torque"].values
    assert _scaled_err(got, decs[0].torque_particular) < 1e-8


def test_min_metric_label_and_values_change_with_w():
    snap = _planar("generic", "full")
    _, _, a = _report(snap, AllocationMetric.identity(5))
    _, _, b = _report(snap, AllocationMetric.inverse_mass(snap))
    ra, rb = (
        a.frames[0].records["min_metric_torque"],
        b.frames[0].records["min_metric_torque"],
    )
    assert "C-METRIC-IDENTITY" in ra.conventions
    assert "C-METRIC-MINV" in rb.conventions
    assert ra.conventions != rb.conventions
    assert np.max(np.abs(ra.values - rb.values)) > 1e-6
    # the metric-free quantities do not care
    for name in ("loop_power", "tangent_pairing"):
        assert (
            _scaled_err(
                a.frames[0].records[name].values, b.frames[0].records[name].values
            )
            < TOL
        )


def test_explicit_metric_must_match_decomposition_label():
    snap = _planar("generic", "full")
    adm = _admit(snap)
    decs = decompose_trajectory(adm, weights=AllocationMetric.identity(5))
    with pytest.raises(ValueError, match="metric"):
        compute_invariants(adm, decs, metric=AllocationMetric.inverse_mass(snap))
    ok = compute_invariants(adm, decs, metric=AllocationMetric.identity(5))
    base = compute_invariants(adm, decs)
    assert (
        _scaled_err(
            ok.frames[0].records["min_metric_torque"].values,
            base.frames[0].records["min_metric_torque"].values,
        )
        < 1e-8
    )


# ----------------------------------------------------------------- gates --


def _velocity_violating(snap):
    jc = snap.J_c
    bump = np.linalg.pinv(jc) @ np.ones(jc.shape[0])
    return dataclasses.replace(snap, v=snap.v + 1e-2 * bump)


def test_phi_dot_violation_makes_velocity_gated_items_unavailable_both_ways():
    snap = _planar("generic", "full")
    bad = _velocity_violating(snap)
    _, _, rep = _report(bad, gates=LOOSE)
    for name in ("loop_power", "tangent_pairing"):
        rec = rep.frames[0].records[name]
        assert rec.allocation_dependence is UNAV, name
        assert "phi_dot" in rec.reason and np.all(np.isnan(rec.values))
    # the metric-lifted torque has no velocity gate
    assert rep.frames[0].records["min_metric_torque"].allocation_dependence is INV
    # the clean state passes the same gate
    _, _, good = _report(snap, gates=LOOSE)
    assert good.frames[0].records["loop_power"].allocation_dependence is INV


def test_custom_gate_tolerance_flips_the_item():
    snap = _planar("generic", "full")
    bad = _velocity_violating(snap)
    adm = _admit(bad, LOOSE)
    decs = decompose_trajectory(adm, weights=AllocationMetric.identity(5))
    relaxed = AdmissionGates(require_position=False, phi_dot_tol=1e9)
    rep = compute_invariants(adm, decs, gates=relaxed)
    assert rep.frames[0].records["loop_power"].allocation_dependence is INV


def test_gate_tolerance_is_per_frame():
    snap = _planar("generic", "full")
    bad = _velocity_violating(snap)
    report = admit([snap, bad, snap, snap], gates=LOOSE, times=TIMES)
    adm = AdmittedTrajectory.from_report(report)
    decs = decompose_trajectory(adm, weights=AllocationMetric.identity(5))
    rep = compute_invariants(adm, decs)
    kinds = [f.records["loop_power"].allocation_dependence for f in rep.frames]
    assert kinds == [INV, UNAV, INV, INV]
    assert [r.allocation_dependence for r in rep.records("loop_power")] == kinds


def _with_ground_load(column: int, speed: float):
    """Planar snapshot with one ground-load row on ``column``.

    ``speed`` is the contact velocity ``J_g v`` the row is made to have.
    """
    snap = _planar("generic", "full")
    other = next(i for i in range(5) if i != column and abs(snap.v[i]) > 1e-3)
    row = np.zeros((1, 5))
    row[0, column] = 1.0
    row[0, other] = (speed - snap.v[column]) / snap.v[other]
    return dataclasses.replace(snap, J_g=row)


def test_loop_power_ground_load_gates_both_ways():
    snap = _planar("generic", "full")
    inside = snap.topology.loop_support_indices[0]
    # stationary contact, load inside the support: invariant
    ok = _with_ground_load(inside, 0.0)
    ok_rec = _report(ok)[2].frames[0].records["loop_power"]
    assert ok_rec.allocation_dependence is INV
    # moving contact: contact power (J_g v)^T f_g is free, so unavailable
    moving = _with_ground_load(inside, 1.0)
    rec = _report(moving)[2].frames[0].records["loop_power"]
    assert rec.allocation_dependence is UNAV
    assert "contact power" in rec.reason
    # stationary contact but the load acts on an actuated coordinate outside the
    # declared loop support: the support power would depend on f_g
    topo = dataclasses.replace(
        ok.topology,
        loop_support_indices=tuple(
            i for i in ok.topology.loop_support_indices if i != inside
        ),
    )
    outside = dataclasses.replace(ok, topology=topo)
    rec = _report(outside)[2].frames[0].records["loop_power"]
    assert rec.allocation_dependence is UNAV
    assert "outside the loop support" in rec.reason


def test_loop_power_is_invariant_with_free_stationary_ground_load():
    snap = _with_ground_load(
        _planar("generic", "full").topology.loop_support_indices[0], 0.0
    )
    _, decs, rep = _report(snap)
    dec = decs[0]
    assert dec.nullity > 0
    inside, _ = (list(snap.topology.loop_support_indices), None)
    rng = np.random.default_rng(3)
    want = rep.frames[0].records["loop_power"].values[0]
    for _ in range(5):
        tau = dec.torque(rng.normal(size=dec.nullity))
        got = float(tau[inside] @ snap.v[inside])
        assert abs(got - want) <= TOL * max(1.0, abs(want))


def test_tangent_pairing_with_ground_load_gate_both_ways():
    snap = _planar("generic", "full")
    basis = tangent_basis(snap, decompose(snap, weights=AllocationMetric.identity(5)))
    loaded = _with_ground_load(0, 0.0)
    assert np.max(np.abs(loaded.J_g @ basis)) > 1e-3
    rec = _report(loaded)[2].frames[0].records["tangent_pairing"]
    assert rec.allocation_dependence is UNAV
    assert "J_g N_c" in rec.reason
    # a ground row in the row space of J_c has J_g N_c = 0: the gate passes
    ortho = dataclasses.replace(snap, J_g=snap.J_c[:1])
    rec = _report(ortho)[2].frames[0].records["tangent_pairing"]
    assert rec.allocation_dependence is INV


def test_precondition_errors():
    snap = _planar("generic", "full")
    adm = _admit(snap)
    decs = decompose_trajectory(adm, weights=AllocationMetric.identity(5))
    with pytest.raises(TypeError, match="AdmittedTrajectory"):
        compute_invariants(list(adm.snapshots), decs)
    with pytest.raises(TypeError, match="Decomposition"):
        compute_invariants(adm, decs[0])
    with pytest.raises(ValueError, match="one Decomposition per frame"):
        compute_invariants(adm, decs[:2])
    with pytest.raises(ValueError, match="club_ne"):
        compute_invariants(adm, decs, club_ne=[{}])
    assert isinstance(decs[0], Decomposition)


def test_report_is_json_serializable_and_round_trips_reasons():
    snap = _planar("generic", "club_unactuated")
    _, _, rep = _report(snap)
    text = json.dumps(rep.to_dict(), allow_nan=False)
    data = json.loads(text)
    assert len(data["frames"]) == 4
    rec = data["frames"][0]["records"]
    assert rec["loop_power"]["allocation_dependence"] == "INVARIANT"
    assert rec["net_club_wrench"]["allocation_dependence"] == "UNAVAILABLE"
    assert rec["net_club_wrench"]["values"] is None
    assert rec["net_club_wrench"]["reason"]


# ------------------------------------------------------ floating base / GRF --


def _spatial(floating: bool, feet: bool):
    """Spatial snapshot with the base rows made dynamically balanceable.

    The fixture's rates are kinematically admissible but not dynamically
    consistent for a floating base, so ``h`` on the base rows is set so that
    ``b_u = J_cu^T lambda_0 + J_gu^T f_0`` for a random ``lambda_0, f_0``.
    """
    snap = spatial_two_arm_club_snapshot(floating_root=floating, feet=feet).snapshot
    unact = list(snap.topology.unactuated_indices)
    if not unact:
        return snap
    rng = np.random.default_rng(7)
    target = snap.J_c[:, unact].T @ rng.normal(size=snap.J_c.shape[0])
    if snap.J_g is not None:
        target = target + snap.J_g[:, unact].T @ rng.normal(size=snap.J_g.shape[0])
    h = np.array(snap.h)
    h[unact] = target - (snap.M @ snap.a)[unact]
    return dataclasses.replace(snap, h=h)


def _with_consistent_grf(snap):
    dec = decompose(snap, weights=AllocationMetric.identity(snap.topology.n_actuated))
    n_a = snap.topology.n_actuated
    m = snap.topology.closure_kind.rows
    return dataclasses.replace(snap, grf_observed=np.array(dec.x_p[n_a + m :])), dec


@pytest.mark.requires_pinocchio
def test_base_rows_unavailable_without_floating_base_or_contact_or_grf():
    for snap, word in (
        (_spatial(False, False), "floating base"),
        (_spatial(True, False), "ground contact"),
        (_spatial(True, True), "no observed ground reaction"),
    ):
        _, _, rep = _report(snap, gates=LOOSE)
        rec = rep.frames[0].records["base_centroidal_rows"]
        assert rec.allocation_dependence is UNAV
        assert word in rec.reason


@pytest.mark.requires_pinocchio
def test_base_rows_with_observed_grf_are_invariant_for_fixed_ground_load():
    snap, dec = _with_consistent_grf(_spatial(True, True))
    _, decs, rep = _report(snap, gates=LOOSE)
    rec = rep.frames[0].records["base_centroidal_rows"]
    assert rec.allocation_dependence is INV
    unact = list(snap.topology.unactuated_indices)
    expected = snap.J_g[:, unact].T @ snap.grf_observed
    assert _scaled_err(rec.values, expected) < TOL
    # unactuated-row balance b_u - J_cu^T lambda = J_gu^T f_g, over the part of
    # the null space that leaves the ground load fixed
    n_a, m = snap.topology.n_actuated, snap.topology.closure_kind.rows
    d = decs[0]
    fg_rows = d.N[n_a + m :]
    _, sv, vt = np.linalg.svd(fg_rows)
    free = vt[int(np.count_nonzero(sv > 1e-10)) :].T
    assert free.shape[1] > 0
    rng = np.random.default_rng(1)
    b_u = snap.generalized_load()[unact]
    for _ in range(5):
        x = d.x_p + d.N @ (free @ rng.normal(size=free.shape[1]))
        lam = x[n_a : n_a + m]
        lhs = b_u - snap.J_c[:, unact].T @ lam
        assert _scaled_err(lhs, rec.values) < TOL
    assert dec.nullity > 0


@pytest.mark.requires_pinocchio
def test_base_rows_unavailable_when_observed_grf_is_inconsistent():
    snap, _ = _with_consistent_grf(_spatial(True, True))
    unact = list(snap.topology.unactuated_indices)
    # decouple the closure from the base: the base rows are then pure ground load
    j_c = np.array(snap.J_c)
    j_c[:, unact] = 0.0
    iso = dataclasses.replace(snap, J_c=j_c)
    dec = decompose(iso, weights=AllocationMetric.identity(iso.topology.n_actuated))
    n_a, m = iso.topology.n_actuated, iso.topology.closure_kind.rows
    f_ok = np.array(dec.x_p[n_a + m :])
    good = dataclasses.replace(iso, grf_observed=f_ok)
    _, _, rep = _report(good, gates=LOOSE)
    assert rep.frames[0].records["base_centroidal_rows"].allocation_dependence is INV
    bad = dataclasses.replace(iso, grf_observed=f_ok + 1e3)
    _, _, rep = _report(bad, gates=LOOSE)
    rec = rep.frames[0].records["base_centroidal_rows"]
    assert rec.allocation_dependence is UNAV
    assert "inconsistent" in rec.reason

"""NSX-30: constrained ZTCF/ZVCF on the closed loop (#12570).

Independent references: an augmented KKT block solve on the planar oracle and
raw Pinocchio ``constraintDynamics`` (on its own ``Data``) on the anthro model.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.decomposition import AllocationMetric, decompose
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.fixtures import (
    load_golden,
    planar_golfer_snapshot,
)
from src.shared.python.loop_nullspace.planar_model import (
    PlanarGolferParams,
    PlanarGolferProvider,
)
from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    constrained_counterfactual_report,
)
from src.shared.python.simulation_backends.protocol import (
    ConstrainedEquationsOfMotionProvider,
    EquationsOfMotionProvider,
)
from src.shared.python.simulation_backends.ztcf_zvcf import (
    CanonicalDynamicsTrajectory,
    drift_and_control_split,
    evaluate_ztcf_zvcf_on_canonical_trajectory,
    zero_velocity_control_preserved_acceleration,
    ztcf_acceleration,
    zvcf_acceleration,
)

pytestmark = pytest.mark.unit
TOL = 1e-9


def _scaled(x: np.ndarray, scale: float) -> float:
    return float(np.max(np.abs(x))) / max(1.0, scale)


def _kkt(M, load, J, Jdot_v):
    n, m = M.shape[0], J.shape[0]
    kkt = np.block([[M, -J.T], [J, np.zeros((m, m))]])
    sol = np.linalg.solve(kkt, np.concatenate([load, -Jdot_v]))
    return sol[:n], sol[n:]


@pytest.fixture(scope="module")
def planar():
    case = planar_golfer_snapshot()
    params = PlanarGolferParams.from_golden(load_golden("planar_two_arm_full"))
    return case.snapshot, PlanarGolferProvider(params)


def test_protocol_membership(planar) -> None:
    _, provider = planar
    assert isinstance(provider, ConstrainedEquationsOfMotionProvider)
    assert isinstance(provider, EquationsOfMotionProvider)

    class Plain:
        def mass_matrix(self, q):
            return np.eye(2)

        def bias_forces(self, q, v):
            return np.zeros(2)

    assert isinstance(Plain(), EquationsOfMotionProvider)
    assert not isinstance(Plain(), ConstrainedEquationsOfMotionProvider)


def test_ztcf_re_solves_the_reaction_against_the_kkt_oracle(planar) -> None:
    s, provider = planar
    a = ztcf_acceleration(provider, s.q, s.v)
    ref, _ = _kkt(s.M, -s.h, s.J_c, s.Jdot_v)
    assert _scaled(a - ref, np.max(np.abs(ref))) < TOL
    assert _scaled(s.J_c @ a + s.Jdot_v, np.max(np.abs(s.Jdot_v))) < TOL
    free = np.linalg.solve(s.M, -s.h)
    assert np.max(np.abs(a - free)) > 1e-3  # not the unconstrained value


def test_zvcf_sets_velocity_and_control_to_zero_on_the_loop(planar) -> None:
    s, provider = planar
    a = zvcf_acceleration(provider, s.q)
    h0 = provider.bias_forces(s.q, np.zeros(5))
    ref, _ = _kkt(s.M, -h0, s.J_c, provider.closure_drift(s.q, np.zeros(5)))
    assert _scaled(a - ref, np.max(np.abs(ref))) < TOL
    assert np.max(np.abs(s.J_c @ a)) < 1e-9


def test_control_split_and_control_preserved_variant(planar) -> None:
    s, provider = planar
    tau = s.S.T @ np.array([1.0, -2.0, 0.5, 0.3, 1.7])
    drift, control = drift_and_control_split(provider, s.q, s.v, tau)
    total, _ = _kkt(s.M, tau - s.h, s.J_c, s.Jdot_v)
    assert _scaled(drift + control - total, np.max(np.abs(total))) < TOL
    kept = zero_velocity_control_preserved_acceleration(provider, s.q, tau)
    h0 = provider.bias_forces(s.q, np.zeros(5))
    ref, _ = _kkt(s.M, tau - h0, s.J_c, np.zeros(2))
    assert _scaled(kept - ref, np.max(np.abs(ref))) < TOL


def test_measured_acceleration_is_ztcf_plus_control_total(planar) -> None:
    case = planar_golfer_snapshot()
    s, golden = case.snapshot, case.golden
    tau_a = golden.array("min_metric_torque", "minv", "tau_actuated")
    report = constrained_counterfactual_report(
        PlanarGolferProvider(PlanarGolferParams.from_golden(golden)),
        s.q,
        s.v,
        s.S.T @ tau_a,
    )
    assert _scaled(report.ztcf + report.control_total - s.a, np.max(np.abs(s.a))) < 1e-8


def test_dependence_labels_follow_nsx00(planar) -> None:
    s, provider = planar
    report = constrained_counterfactual_report(provider, s.q, s.v, s.S.T @ np.ones(5))
    dep = report.dependence
    assert dep["ztcf"] is AllocationDependence.INVARIANT
    assert dep["zvcf"] is AllocationDependence.INVARIANT
    assert dep["control_total"] is AllocationDependence.INVARIANT
    assert dep["per_actuator"] is AllocationDependence.FAMILY_DEPENDENT
    assert report.hypotheses["per_actuator"]


def test_total_is_invariant_and_per_actuator_is_family_dependent(planar) -> None:
    """Across NSX-15 family members: total P_c S^T tau is fixed, shares move."""
    s, provider = planar
    dec = decompose(s, weights=AllocationMetric.identity(5))
    assert dec.nullity > 0
    reports = []
    for z in (
        np.zeros(dec.nullity),
        np.full(dec.nullity, 2.0),
        np.linspace(-1, 1, dec.nullity),
    ):
        reports.append(
            constrained_counterfactual_report(provider, s.q, s.v, s.S.T @ dec.torque(z))
        )
    base = reports[0]
    scale = np.max(np.abs(base.control_total))
    for other in reports[1:]:
        assert _scaled(other.control_total - base.control_total, scale) < TOL
        assert _scaled(other.ztcf - base.ztcf, 1.0) < TOL
    assert np.max(np.abs(reports[1].per_actuator - base.per_actuator)) > 1e-3
    for r in reports:
        assert _scaled(r.per_actuator.sum(axis=1) - r.control_total, scale) < TOL


def test_canonical_trajectory_dispatches_to_the_constrained_path(planar) -> None:
    s, provider = planar
    traj = CanonicalDynamicsTrajectory(
        t=np.array([0.0]),
        q=s.q[None, :],
        v=s.v[None, :],
        tau=(s.S.T @ np.ones(5))[None, :],
    )
    res = evaluate_ztcf_zvcf_on_canonical_trajectory(provider, traj)
    assert (
        np.max(np.abs(res.ztcf_acceleration[0] - ztcf_acceleration(provider, s.q, s.v)))
        == 0
    )
    assert np.max(np.abs(s.J_c @ res.ztcf_acceleration[0] + s.Jdot_v)) < 1e-9


def test_unconstrained_provider_path_is_unchanged() -> None:
    class Plain:
        def mass_matrix(self, q):
            return np.diag([2.0, 4.0])

        def bias_forces(self, q, v):
            return np.array([2.0, 8.0])

    a = ztcf_acceleration(Plain(), np.zeros(2), np.zeros(2))
    np.testing.assert_allclose(a, [-1.0, -2.0])


def test_report_rejects_bad_inputs(planar) -> None:
    s, provider = planar
    with pytest.raises(ValueError, match="tau"):
        constrained_counterfactual_report(provider, s.q, s.v, np.zeros(4))
    with pytest.raises(TypeError, match="ConstrainedEquationsOfMotionProvider"):
        constrained_counterfactual_report(object(), s.q, s.v, np.zeros(5))


# ------------------------------------------------------------- Pinocchio ----

EVIDENCE = (
    Path(__file__).resolve().parents[3]
    / "docs/development/full_body_models/evidence/ground_support"
)
SPEC_PATH = EVIDENCE / "full_body_spec_hipcal_scaled.json"


@pytest.fixture(scope="module")
def pin_case():
    pin = pytest.importorskip("pinocchio")
    from src.engines.physics_engines.pinocchio.python.force_adapter import (
        PinocchioForceAdapter,
    )
    from src.shared.python.motion_matching.same_input import (
        VectorPlant,
        project_to_closure,
    )

    adapter = PinocchioForceAdapter(SPEC_PATH.read_bytes())
    plant = VectorPlant("pinocchio", SPEC_PATH.read_bytes())
    with np.load(EVIDENCE / "dynamics_record.npz") as rec:
        q0, v0 = rec["q"][400], rec["v"][400]
    proj = project_to_closure(plant, q0, v0)
    qn, vn = plant._named(proj.q), plant._named(proj.v)
    order = adapter.coordinate_order
    q = np.array([qn[n] for n in order])
    v = np.array([vn[n] for n in order])
    return pin, adapter, q, v


def _pin_constraint_dynamics(pin, adapter, q, v, tau):
    model, cons = adapter._plant.model, adapter._plant.constraints
    data = model.createData()
    cdata = [c.createData() for c in cons]
    pin.initConstraintDynamics(model, data, cons, cdata)
    a = np.array(pin.constraintDynamics(model, data, q, v, tau, cons, cdata))
    return a, np.array(data.lambda_c)


@pytest.mark.requires_pinocchio
def test_pinocchio_ztcf_zvcf_match_constraint_dynamics(pin_case) -> None:
    from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider

    pin, adapter, q, v = pin_case
    provider = AdapterConstrainedProvider(adapter)
    assert isinstance(provider, ConstrainedEquationsOfMotionProvider)
    zeros = np.zeros(adapter.nv)

    ztcf_ref, _ = _pin_constraint_dynamics(pin, adapter, q, v, zeros)
    ztcf = ztcf_acceleration(provider, q, v)
    assert _scaled(ztcf - ztcf_ref, np.max(np.abs(ztcf_ref))) < TOL

    zvcf_ref, _ = _pin_constraint_dynamics(pin, adapter, q, zeros, zeros)
    zvcf = zvcf_acceleration(provider, q)
    assert _scaled(zvcf - zvcf_ref, np.max(np.abs(zvcf_ref))) < TOL

    jc = provider.closure_jacobian(q)
    assert _scaled(jc @ ztcf + provider.closure_drift(q, v), 1.0) < 1e-8
    assert (
        np.max(
            np.abs(
                ztcf
                - np.linalg.solve(provider.mass_matrix(q), -provider.bias_forces(q, v))
            )
        )
        > 1e-6
    )


@pytest.mark.requires_pinocchio
def test_pinocchio_report_splits_invariant_from_family_dependent(pin_case) -> None:
    from src.shared.python.loop_nullspace.provider import AdapterConstrainedProvider

    pin, adapter, q, v = pin_case
    provider = AdapterConstrainedProvider(adapter)
    rng = np.random.default_rng(12570)
    tau = np.zeros(adapter.nv)
    act = list(adapter.actuated_indices)
    tau[act] = rng.normal(size=len(act))
    report = constrained_counterfactual_report(provider, q, v, tau)
    full_ref, _ = _pin_constraint_dynamics(pin, adapter, q, v, tau)
    assert (
        _scaled(report.ztcf + report.control_total - full_ref, np.max(np.abs(full_ref)))
        < TOL
    )
    assert report.dependence["control_total"] is AllocationDependence.INVARIANT
    assert report.dependence["per_actuator"] is AllocationDependence.FAMILY_DEPENDENT

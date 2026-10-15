"""NSX-61: kinematic ensemble and envelope union/intersection (#12586).

Planar oracles only.  The sampling distribution is a declared hypothesis; the
union must contain every member envelope and the intersection must be
contained in every member.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from src.shared.python.loop_nullspace import registry
from src.shared.python.loop_nullspace.admission import (
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.ensemble import (
    ENSEMBLE_HYPOTHESIS_ID,
    EnsembleSource,
    SamplingHypothesis,
    ensemble_envelope,
    envelope_union_intersection,
    kinematic_ensemble,
)
from src.shared.python.loop_nullspace.envelopes import BoundStatus, EnvelopeSeries
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
)
from tests.unit.loop_nullspace.test_admission import TIMES, _PlanarPlant

pytestmark = pytest.mark.unit

B, U, I = BoundStatus.BOUNDED, BoundStatus.UNBOUNDED, BoundStatus.INFEASIBLE  # noqa: E741


@pytest.fixture(scope="module")
def plant() -> _PlanarPlant:
    return _PlanarPlant()


@pytest.fixture(scope="module")
def nominal(plant) -> AdmittedTrajectory:
    snap = planar_golfer_snapshot(generic_planar_config()).snapshot
    report = admit([snap] * 4, gates=AdmissionGates(), times=TIMES, plant=plant)
    return AdmittedTrajectory.from_report(report)


def _source(nominal, plant, sigma=0.05, gates=None) -> EnsembleSource:
    return EnsembleSource(
        nominal=nominal,
        plant=plant,
        rebuild=plant.rebuild,
        sampling=SamplingHypothesis(sigma_q=sigma),
        gates=gates,
    )


# ---------------------------------------------------------------- sampling --
def test_hypothesis_is_declared_in_the_registry():
    assert registry.hypothesis(ENSEMBLE_HYPOTHESIS_ID).id == ENSEMBLE_HYPOTHESIS_ID
    assert SamplingHypothesis(sigma_q=0.1).hypothesis_id == ENSEMBLE_HYPOTHESIS_ID


def test_seeded_determinism(nominal, plant):
    a = kinematic_ensemble(_source(nominal, plant), 6, 7)
    b = kinematic_ensemble(_source(nominal, plant), 6, 7)
    c = kinematic_ensemble(_source(nominal, plant), 6, 8)
    qa = [t.snapshots[0].q for t in a.accepted]
    qb = [t.snapshots[0].q for t in b.accepted]
    qc = [t.snapshots[0].q for t in c.accepted]
    assert len(qa) == 6
    assert all(np.array_equal(x, y) for x, y in zip(qa, qb, strict=True))
    assert not np.allclose(qa[0], qc[0])
    assert a.sample_indices == b.sample_indices


def test_samples_are_projected_distinct_and_close_to_nominal(nominal, plant):
    res = kinematic_ensemble(_source(nominal, plant, sigma=0.05), 5, 1)
    q0 = nominal.snapshots[0].q
    assert res.n_accepted == 5 and res.n_repaired >= 1
    for traj in res.accepted:
        assert isinstance(traj, AdmittedTrajectory)
        for snap in traj.snapshots:
            assert np.max(np.abs(plant.closure_pose_residual(snap.q))) < 1e-6
        assert 0.0 < np.linalg.norm(traj.snapshots[0].q - q0) < 1.0
    assert res.hypothesis_id == ENSEMBLE_HYPOTHESIS_ID
    assert res.seed == 1 and res.sigma_q == 0.05


def test_zero_samples_refused(nominal, plant):
    with pytest.raises(ValueError, match="n must be"):
        kinematic_ensemble(_source(nominal, plant), 0, 1)
    with pytest.raises(ValueError, match="sigma_q"):
        SamplingHypothesis(sigma_q=0.0)
    with pytest.raises(TypeError, match="AdmittedTrajectory"):
        EnsembleSource(
            nominal=object(),
            plant=plant,
            rebuild=plant.rebuild,
            sampling=SamplingHypothesis(sigma_q=0.1),
        )


# --------------------------------------------------------------- rejection --
def test_rejection_accounting_all_rejected(nominal, plant):
    # Analytic velocities disagree with the (constant-q) central difference,
    # so the derivative gate rejects every sample; each is counted once.
    gates = dataclasses.replace(
        nominal.report.gates, derivative_method="central_difference"
    )
    res = kinematic_ensemble(_source(nominal, plant, gates=gates), 5, 3)
    assert res.n_accepted == 0 and res.n_rejected == 5
    assert res.rejected_by_reason == {"derivative": 5}
    assert res.n_requested == 5
    assert res.rejection_rate == 1.0


def test_rejection_accounting_projection_failure_is_counted(nominal, plant):
    res = kinematic_ensemble(_source(nominal, plant, sigma=40.0), 30, 11)
    assert res.n_accepted + res.n_rejected == 30
    assert sum(res.rejected_by_reason.values()) == res.n_rejected
    assert len(res.accepted) == res.n_accepted == len(res.sample_indices)
    assert set(res.rejected_by_reason) <= {
        "projection_failed",
        "position",
        "velocity",
        "acceleration",
    }
    assert res.n_rejected > 0 and res.n_accepted > 0


# ---------------------------------------------------------------- envelope --
def _series(lower, upper, sl=None, su=None) -> EnvelopeSeries:
    lower, upper = np.asarray(lower, float), np.asarray(upper, float)
    f, q = lower.shape
    sl = sl or [[B] * q] * f
    su = su or [[B] * q] * f
    e = tuple(tuple(() for _ in range(q)) for _ in range(f))
    flag = tuple(tuple(False for _ in range(q)) for _ in range(f))
    return EnvelopeSeries(
        names=tuple(f"x{j}" for j in range(q)),
        units=("N m",) * q,
        lower=lower,
        upper=upper,
        status_lower=tuple(tuple(r) for r in sl),
        status_upper=tuple(tuple(r) for r in su),
        sign_class=tuple(tuple(None for _ in range(q)) for _ in range(f)),
        binding_lower=e,
        binding_upper=e,
        binding_hypotheses_lower=e,
        binding_hypotheses_upper=e,
        lower_rests_on_unverified=flag,
        upper_rests_on_unverified=flag,
        frame_rests_on_unverified=(False,) * f,
        frame_hypothesis_ids=(("H-TORQUE-CAPACITY",),) * f,
        chebyshev_margin=np.zeros(f),
        chebyshev_center=(np.zeros(1),) * f,
        chebyshev_status=(B,) * f,
        chebyshev_metric="m",
        cone_error_bound=np.zeros(f),
        simplex_iterations=0,
        warm_started=False,
    )


MEMBERS = [
    _series([[0.0, -1.0], [1.0, 0.0]], [[4.0, 2.0], [5.0, 3.0]]),
    _series([[1.0, -2.0], [0.5, 1.0]], [[5.0, 1.0], [3.0, 4.0]]),
    _series([[-1.0, 0.0], [2.0, -1.0]], [[3.0, 3.0], [6.0, 2.0]]),
]


def test_union_contains_and_intersection_is_contained_in_every_member():
    env = envelope_union_intersection(MEMBERS, labels=["a", "b", "c"])
    assert env.member_labels == ("a", "b", "c") and env.n_members == 3
    for m in MEMBERS:
        assert np.all(env.union_lower <= m.lower) and np.all(env.union_upper >= m.upper)
        ok = ~env.intersection_empty
        assert np.all(env.intersection_lower[ok] >= m.lower[ok])
        assert np.all(env.intersection_upper[ok] <= m.upper[ok])
    np.testing.assert_allclose(env.union_lower, [[-1, -2], [0.5, -1]])
    np.testing.assert_allclose(env.union_upper, [[5, 3], [6, 4]])
    np.testing.assert_allclose(env.intersection_lower, [[1, 0], [2, 1]])
    np.testing.assert_allclose(env.intersection_upper, [[3, 1], [3, 2]])
    assert not env.intersection_empty.any()


def test_disjoint_members_give_an_empty_intersection_not_an_error():
    a = _series([[0.0]], [[1.0]])
    b = _series([[2.0]], [[3.0]])
    env = envelope_union_intersection([a, b])
    assert env.intersection_empty[0, 0]
    assert np.isnan(env.intersection_lower[0, 0])
    assert env.intersection_status[0][0] is BoundStatus.INFEASIBLE
    assert (env.union_lower[0, 0], env.union_upper[0, 0]) == (0.0, 3.0)
    assert env.member_labels == ("member-0", "member-1")


def test_unbounded_and_infeasible_members():
    a = _series([[0.0]], [[1.0]])
    unb = _series([[np.nan]], [[2.0]], sl=[[U]])
    inf = _series([[np.nan]], [[np.nan]], sl=[[I]], su=[[I]])
    env = envelope_union_intersection([a, unb])
    assert np.isnan(env.union_lower[0, 0]) and env.union_upper[0, 0] == 2.0
    assert env.union_status_lower[0][0] is U
    assert env.intersection_lower[0, 0] == 0.0
    assert env.intersection_upper[0, 0] == 1.0
    env = envelope_union_intersection([a, inf])  # empty member: skipped in union
    assert (env.union_lower[0, 0], env.union_upper[0, 0]) == (0.0, 1.0)
    assert env.intersection_empty[0, 0]


def test_envelope_input_validation():
    with pytest.raises(ValueError, match="at least one"):
        envelope_union_intersection([])
    with pytest.raises(ValueError, match="labels"):
        envelope_union_intersection(MEMBERS, labels=["a"])
    with pytest.raises(ValueError, match="labels"):
        envelope_union_intersection(MEMBERS, labels=["a", "a", "b"])
    with pytest.raises(ValueError, match="shape"):
        envelope_union_intersection([MEMBERS[0], _series([[0.0]], [[1.0]])])
    with pytest.raises(TypeError, match="EnvelopeSeries"):
        envelope_union_intersection([object()])


def test_records_carry_the_ensemble_hypothesis():
    env = envelope_union_intersection(MEMBERS)
    model = ModelProvenance(name="planar", version="1")
    src = InputProvenance(kind="test", source="unit")
    recs = env.records(model, src)
    assert set(recs) == {"x0:union", "x0:intersection", "x1:union", "x1:intersection"}
    rec = recs["x0:union"]
    assert rec.allocation_dependence is AllocationDependence.FAMILY_DEPENDENT
    assert ENSEMBLE_HYPOTHESIS_ID in rec.hypotheses
    assert "H-TORQUE-CAPACITY" in rec.hypotheses
    assert rec.values.shape == (2, 2)


def test_ensemble_envelope_runs_series_factory_over_accepted(nominal, plant):
    res = kinematic_ensemble(_source(nominal, plant), 3, 5)
    seen = []

    def series_of(traj):
        seen.append(traj)
        return MEMBERS[len(seen) - 1]

    env = ensemble_envelope(res, series_of)
    assert len(seen) == 3 and env.n_members == 3
    assert env.member_labels == tuple(f"sample-{i}" for i in res.sample_indices)
    empty = dataclasses.replace(res, accepted=(), sample_indices=())
    with pytest.raises(ValueError, match="no accepted"):
        ensemble_envelope(empty, series_of)

"""NSX-61 (#12586): ensemble envelope hulls from real LPs (planar oracle).

Each accepted sample is decomposed, its per-frame torque-box feasible set is
solved by NSX-22 ``static_envelopes`` (two LPs per quantity), and the hull
containment properties are checked on the resulting real bounds.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

pytest.importorskip("highspy")
pytest.importorskip("clarabel")

from src.shared.python.loop_nullspace.admission import (  # noqa: E402
    AdmissionGates,
    AdmittedTrajectory,
    admit,
)
from src.shared.python.loop_nullspace.decomposition import decompose  # noqa: E402
from src.shared.python.loop_nullspace.ensemble import (  # noqa: E402
    EnsembleSource,
    SamplingHypothesis,
    ensemble_envelope,
    kinematic_ensemble,
)
from src.shared.python.loop_nullspace.envelopes import (  # noqa: E402
    BoundStatus,
    joint_torque_quantities,
    static_envelopes,
)
from src.shared.python.loop_nullspace.feasible_rows import FeasibleRow  # noqa: E402
from src.shared.python.loop_nullspace.feasible_set import FeasibleSet  # noqa: E402
from src.shared.python.loop_nullspace.fixtures import (  # noqa: E402
    generic_planar_config,
    planar_golfer_snapshot,
)
from tests.unit.loop_nullspace.test_admission import TIMES, _PlanarPlant  # noqa: E402

pytestmark = pytest.mark.unit

CAP = 1.0e3
TOL = 1e-9


def _feasible_set(snap) -> FeasibleSet:
    dec = decompose(snap, weights=np.eye(snap.topology.n_actuated))
    rows = []
    for i in range(dec.n_actuated):
        rows.append(FeasibleRow(dec.N[i], CAP - dec.x_p[i], f"T:{i}:up", "T", "torque"))
        rows.append(
            FeasibleRow(-dec.N[i], CAP + dec.x_p[i], f"T:{i}:lo", "T", "torque")
        )
    return dataclasses.replace(
        FeasibleSet.from_rows(dec.nullity, rows), decomposition=dec
    )


def _series_of(traj: AdmittedTrajectory):
    frames = [_feasible_set(s) for s in traj.snapshots]
    return static_envelopes(frames, joint_torque_quantities())


def test_hull_of_real_lp_envelopes_contains_and_is_contained():
    plant = _PlanarPlant()
    snap = planar_golfer_snapshot(generic_planar_config()).snapshot
    report = admit([snap] * 4, gates=AdmissionGates(), times=TIMES, plant=plant)
    source = EnsembleSource(
        nominal=AdmittedTrajectory.from_report(report),
        plant=plant,
        rebuild=plant.rebuild,
        sampling=SamplingHypothesis(sigma_q=0.1),
    )
    result = kinematic_ensemble(source, 4, 21)
    assert result.n_accepted >= 3
    members = [_series_of(t) for t in result.accepted]
    assert all(
        s is BoundStatus.BOUNDED for m in members for r in m.status_lower for s in r
    )
    env = ensemble_envelope(result, _series_of)
    assert env.n_members == result.n_accepted
    # Members differ, so the hull is a real test (not all identical).
    assert not np.allclose(members[0].lower, members[1].lower)
    for m in members:
        assert np.all(env.union_lower <= m.lower + TOL)
        assert np.all(env.union_upper >= m.upper - TOL)
        ok = ~env.intersection_empty
        assert np.all(env.intersection_lower[ok] >= m.lower[ok] - TOL)
        assert np.all(env.intersection_upper[ok] <= m.upper[ok] + TOL)
    # Every quantity's union is attained by some member.
    lows = np.min([m.lower for m in members], axis=0)
    highs = np.max([m.upper for m in members], axis=0)
    np.testing.assert_allclose(env.union_lower, lows, atol=TOL)
    np.testing.assert_allclose(env.union_upper, highs, atol=TOL)

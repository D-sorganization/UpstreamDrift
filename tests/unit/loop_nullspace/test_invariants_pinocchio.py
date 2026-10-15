"""NSX-17: invariant report on the Pinocchio anthro fixture (#12562)."""

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
    decompose_trajectory,
)
from src.shared.python.loop_nullspace.evidence import AllocationDependence
from src.shared.python.loop_nullspace.grip_split import net_club_wrench
from src.shared.python.loop_nullspace.invariants import (
    compute_invariants,
    tangent_basis,
)
from src.shared.python.loop_nullspace.snapshot import snapshot_from_adapter
from tests.unit.loop_nullspace.test_snapshot_from_adapter import (  # noqa: F401
    PROVENANCE,
    adapter,
    pin_module,  # noqa: F401
    reference,
    state,
)

pytestmark = [pytest.mark.unit, pytest.mark.requires_pinocchio]

INV = AllocationDependence.INVARIANT
UNAV = AllocationDependence.UNAVAILABLE
TIMES = np.array([0.0, 0.01, 0.02])


def _run(adapter, state, a, gates, drift=0.0):  # noqa: F811
    q, v = state
    snap = snapshot_from_adapter(adapter, q, v, a, input_provenance=PROVENANCE)
    if drift:
        snap = dataclasses.replace(snap, Jdot_v=snap.Jdot_v + drift)
    adm = AdmittedTrajectory.from_report(admit([snap] * 3, gates=gates, times=TIMES))
    metric = AllocationMetric.identity(snap.topology.n_actuated)
    decs = decompose_trajectory(adm, weights=metric)
    ne = adapter.club_newton_euler(q, v, a)
    return snap, decs, compute_invariants(adm, decs, club_ne=[ne] * 3)


def test_anthro_report_is_invariant_gated_and_json_serializable(
    adapter,  # noqa: F811
    state,  # noqa: F811
    reference,  # noqa: F811
) -> None:
    gates = AdmissionGates(require_position=False)
    snap, decs, rep = _run(adapter, state, reference["a"], gates)
    rec = rep.frames[0].records
    for name in ("net_club_wrench", "loop_power", "tangent_pairing"):
        assert rec[name].allocation_dependence is INV, rec[name].reason
    assert rec["min_metric_torque"].allocation_dependence is INV
    assert rec["base_centroidal_rows"].allocation_dependence is UNAV
    assert "ground contact" in rec["base_centroidal_rows"].reason
    ne = adapter.club_newton_euler(*state, reference["a"])
    want = net_club_wrench(snap, ne).values
    assert np.max(np.abs(rec["net_club_wrench"].values - want)) <= 1e-10 * max(
        1.0, float(np.max(np.abs(want)))
    )

    dec = decs[0]
    basis = tangent_basis(snap, dec)
    inside = [
        i
        for i in snap.topology.actuated_indices
        if i in set(snap.topology.loop_support_indices)
    ]
    pos = [snap.topology.actuated_indices.index(i) for i in inside]
    rng = np.random.default_rng(12562)
    p_ref = rec["loop_power"].values[0]
    t_ref = rec["tangent_pairing"].values
    for _ in range(4):
        tau = dec.torque(rng.normal(size=dec.nullity))
        power = float(tau[pos] @ snap.v[inside])
        assert abs(power - p_ref) <= 1e-9 * max(1.0, abs(p_ref))
        pair = basis.T @ (snap.S.T @ tau)
        assert np.max(np.abs(pair - t_ref)) <= 1e-9 * max(1.0, np.max(np.abs(t_ref)))

    data = json.loads(json.dumps(rep.to_dict(), allow_nan=False))
    assert len(data["frames"]) == 3
    assert data["frames"][0]["records"]["net_club_wrench"]["values"] is not None


def test_phi_ddot_violation_makes_net_wrench_unavailable(
    adapter,  # noqa: F811
    state,  # noqa: F811
    reference,  # noqa: F811
) -> None:
    loose = AdmissionGates(require_position=False, phi_ddot_tol=1e9, phi_dot_tol=1e9)
    _, _, rep = _run(adapter, state, reference["a"], loose, drift=1.0)
    rec = rep.frames[0].records["net_club_wrench"]
    assert rec.allocation_dependence is UNAV
    assert "phi_ddot" in rec.reason and np.all(np.isnan(rec.values))

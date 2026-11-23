"""Rank-history diagnostic and near-rank-loss negative controls (Refs #15494)."""

from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.research.nullspace_rank_audit import audit_rank_history, main
from src.shared.python.loop_nullspace.decomposition import (
    AllocationMetric,
    BalanceInconsistentError,
    CharacteristicScales,
    decompose,
    dimension_accounting,
)
from src.shared.python.loop_nullspace.fixtures import (
    generic_planar_config,
    planar_golfer_snapshot,
)

pytestmark = pytest.mark.unit


def _snapshot(epsilon=1.0):
    snapshot = planar_golfer_snapshot(generic_planar_config()).snapshot
    jacobian = np.zeros_like(snapshot.J_c)
    jacobian[0, 0], jacobian[1, 1] = 1.0, epsilon
    return replace(snapshot, J_c=jacobian)


def test_near_rank_loss_is_tolerance_sensitive_without_balance_rank_loss():
    snapshots = [_snapshot(), _snapshot(1e-7), _snapshot(0.0)]
    report = audit_rank_history(snapshots, [0.0, 0.1, 0.2], [1e-9, 1e-5])
    assert report["tolerance_sensitive_frames"] == [1]
    assert report["sweeps"][0]["changed_frames"] == [2]
    assert report["sweeps"][1]["changed_frames"] == [1]
    assert [c["rank_J_c"] for c in report["sweeps"][0]["counts"]] == [2, 2, 1]
    assert [c["rank_J_c"] for c in report["sweeps"][1]["counts"]] == [2, 1, 1]
    # Full actuation preserves balance rank even when closure loses rank.
    assert {c["rank_A"] for c in report["sweeps"][0]["counts"]} == {5}
    assert report["time_units"] == "s"
    assert report["count_units"] == "1"
    assert report["classification"] == "diagnostic_only"
    json.dumps(report, allow_nan=False)


def test_counts_match_existing_accounting_with_declared_scales():
    snapshot = _snapshot(1e-6)
    scales = CharacteristicScales(force=80.0, moment=4.0)
    report = audit_rank_history([snapshot], [0.0], [1e-9, 1e-4], scales=scales)
    assert report["scales"] == scales.to_dict()
    for sweep in report["sweeps"]:
        expected = dimension_accounting(
            snapshot, scales=scales, rank_rtol=sweep["rtol"]
        )
        assert sweep["counts"] == [expected.to_dict()]
        assert sweep["changed_frames"] == []


def test_unactuated_balance_rank_is_reported_independently():
    snapshot = planar_golfer_snapshot(
        generic_planar_config(), actuation="club_unactuated"
    ).snapshot
    jacobian = np.zeros_like(snapshot.J_c)
    jacobian[0, 0], jacobian[1, 2] = 1.0, 1e-7
    report = audit_rank_history([replace(snapshot, J_c=jacobian)], [0.0], [1e-9, 1e-5])
    assert [s["counts"][0]["rank_A"] for s in report["sweeps"]] == [5, 4]
    assert [s["counts"][0]["nullity"] for s in report["sweeps"]] == [1, 2]


@pytest.mark.parametrize("times", [[], [0, 0], [1, 0], [0, np.nan], [[0, 1]]])
def test_invalid_timebase_is_rejected(times):
    with pytest.raises(ValueError, match="times"):
        audit_rank_history([_snapshot(), _snapshot()], times, [1e-9, 1e-5])


@pytest.mark.parametrize(
    "rtols", [[], [1e-9], [1e-9, 1e-9], [0, 0.1], [0.1, 1], [np.nan, 0.1], [[0.1, 0.2]]]
)
def test_invalid_tolerances_are_rejected(rtols):
    with pytest.raises(ValueError, match="rtols"):
        audit_rank_history([_snapshot()], [0.0], rtols)


def test_empty_and_non_snapshot_inputs_are_rejected():
    with pytest.raises(ValueError, match="snapshots"):
        audit_rank_history([], [], [1e-9, 1e-5])
    with pytest.raises(TypeError, match="ClosedLoopSnapshot"):
        audit_rank_history([object()], [0.0], [1e-9, 1e-5])


def test_changed_topology_is_rejected():
    snapshot = _snapshot()
    topology = replace(snapshot.topology, closing_side="other")
    with pytest.raises(ValueError, match="topology"):
        audit_rank_history(
            [snapshot, replace(snapshot, topology=topology)], [0, 1], [1e-9, 1e-5]
        )


def test_infeasible_load_still_produces_only_a_rank_diagnostic():
    snapshot = planar_golfer_snapshot(
        generic_planar_config(), actuation="club_unactuated"
    ).snapshot
    load = np.zeros_like(snapshot.h)
    load[2] = 1.0
    snapshot = replace(
        snapshot,
        J_c=np.zeros_like(snapshot.J_c),
        a=np.zeros_like(snapshot.a),
        h=load,
    )
    report = audit_rank_history([snapshot], [0], [1e-9, 1e-5])
    assert report["classification"] == "diagnostic_only"
    assert report["tolerance_sensitive_frames"] == []
    with pytest.raises(BalanceInconsistentError):
        decompose(snapshot, weights=AllocationMetric.identity(4))


def test_ground_dimension_and_scale_type_mismatch_are_rejected():
    snapshot = _snapshot()
    grounded = replace(snapshot, J_g=np.zeros((1, 5)))
    with pytest.raises(ValueError, match="ground-load"):
        audit_rank_history([snapshot, grounded], [0, 1], [1e-9, 1e-5])
    with pytest.raises(TypeError, match="scales"):
        audit_rank_history([snapshot], [0], [1e-9, 1e-5], scales="invalid")


def test_synthetic_cli_emits_strict_json(capsys):
    main()
    report = json.loads(capsys.readouterr().out)
    assert report["fixture"] == "synthetic_diagonal_closure"
    assert report["tolerance_sensitive_frames"] == [1]
    assert "not a replay" in " ".join(report["non_claims"])

"""Recorded solution comparison contracts for optimization Slice 5."""

from dataclasses import replace
import json

import numpy as np
import pytest

from scripts.compare_optimization_bundles import compare_bundles, main
from src.shared.python.optimization.solution_bundle import (
    SolutionBundle,
    SolutionBundleError,
    save_solution_bundle,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def bundles(tmp_path):
    grid = np.zeros((2, 1))
    arrays = dict.fromkeys(
        (
            "q_ref",
            "qd_ref",
            "tau_ff_ref",
            "q_rollout",
            "qd_rollout",
            "tau_cmd",
            "tau_applied",
            "delta_tau_coeffs",
        ),
        grid,
    )
    arrays.update(dict.fromkeys(("q0", "qd0", "kp", "kd"), np.zeros(1)))
    arrays.update(
        time=np.array([0.0, 1.0]),
        knots=np.array([0.0, 1.0]),
        controller_state0=np.zeros(0),
    )
    baseline = SolutionBundle(
        solution_id="baseline",
        parent_id=None,
        git_sha="a" * 40,
        model_hash="m" * 64,
        asset_hashes={"club": "c" * 64},
        frame="world_z_up",
        joint_order=("hinge",),
        control={"hold": "zoh"},
        integrator={"dt": 0.01},
        environment={"gravity": 9.81},
        optimizer_seed=1,
        impact_event={"t": 0.5, "q": [0.0], "qd": [0.0]},
        flight_metrics={"carry_m": 200.0, "ball_speed_ms": 60.0, "rollout_m": 20.0},
        feasibility_margins={"torque_nm": 2.0},
        assumptions=("synthetic",),
        arrays=arrays,
    )
    candidate = replace(
        baseline,
        solution_id="candidate",
        parent_id="baseline",
        flight_metrics={"carry_m": 210.0, "ball_speed_ms": 62.0},
        feasibility_margins={"torque_nm": -1.0, "tracking_rad": 0.1},
        assumptions=("synthetic", "rigid shaft"),
        integrator={"dt": 0.005},
    )
    paths = (tmp_path / "baseline", tmp_path / "candidate")
    save_solution_bundle(baseline, paths[0])
    save_solution_bundle(candidate, paths[1])
    return baseline, candidate, paths


def test_recorded_deltas_and_missing_evidence_are_explicit(bundles):
    _, _, paths = bundles
    report = compare_bundles(*paths)
    assert report["flight_metrics"]["carry_m"] == {
        "baseline": 200.0,
        "candidate": 210.0,
        "delta": 10.0,
    }
    assert report["flight_metrics"]["clubhead_speed_ms"] == {
        "baseline": None,
        "candidate": None,
        "delta": None,
    }
    assert report["rollout_metrics"]["rollout_m"]["delta"] is None
    assert "rollout_m" not in report["flight_metrics"]
    assert report["feasibility_margins"]["torque_nm"]["candidate"] == -1.0
    assert report["feasibility_margins"]["tracking_rad"]["baseline"] is None
    assert report["forward_replay"] == {"status": "unverified", "drift": None}
    assert report["candidate"]["assumptions"] == ["synthetic", "rigid shaft"]
    assert report["baseline"]["integrator"] != report["candidate"]["integrator"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_hash", "other"),
        ("asset_hashes", {"club": "other"}),
        ("frame", "y_up"),
        ("joint_order", ("other",)),
        ("environment", {"gravity": 0.0}),
    ],
)
def test_incompatible_inputs_rejected(bundles, tmp_path, field, value):
    _, candidate, paths = bundles
    other = save_solution_bundle(replace(candidate, **{field: value}), tmp_path / field)
    with pytest.raises(ValueError, match=field):
        compare_bundles(paths[0], other)


@pytest.mark.parametrize("value", [True, "210", None, [210.0]])
def test_metric_types_not_silently_coerced(bundles, tmp_path, value):
    _, candidate, paths = bundles
    other = save_solution_bundle(
        replace(candidate, flight_metrics={"carry_m": value}), tmp_path / "bad"
    )
    with pytest.raises(ValueError, match="carry_m"):
        compare_bundles(paths[0], other)


def test_cli_uses_codec_and_emits_deterministic_json(bundles, capsys):
    _, _, paths = bundles
    args = ["--baseline", str(paths[0]), "--candidate", str(paths[1])]
    assert main(args) == 0
    first = capsys.readouterr().out
    assert json.loads(first) == compare_bundles(*paths)
    assert main(args) == 0
    assert capsys.readouterr().out == first
    arrays = paths[1] / "arrays.npz"
    arrays.write_bytes(arrays.read_bytes() + b"tampered")
    with pytest.raises(SolutionBundleError, match="arrays_sha256"):
        compare_bundles(*paths)


def test_overflowed_delta_rejected(bundles, tmp_path):
    baseline, candidate, _ = bundles
    a = save_solution_bundle(
        replace(baseline, flight_metrics={"carry_m": -1e308}), tmp_path / "a"
    )
    b = save_solution_bundle(
        replace(candidate, flight_metrics={"carry_m": 1e308}), tmp_path / "b"
    )
    with pytest.raises(ValueError, match="finite"):
        compare_bundles(a, b)

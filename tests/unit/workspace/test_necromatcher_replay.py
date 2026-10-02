"""Authored forward replay executes dynamics without source-clock inference."""

import json
from dataclasses import replace
import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def replay_case(native_fit_case):
    library, source, fit = native_fit_case
    # Lift the synthetic initial pose clear of ground penetration; this is not player calibration.
    for q in fit["q"]:
        q[2] = 2.0
    source.write_text(json.dumps(fit))
    saved = library.add_fit("replay-fit", "practice", source)
    coefficients = np.zeros((44, 2))
    coefficients[6, 1] = 1.0
    payload = {
        "schema_version": "necromatcher/effort-profile/2",
        "model_id": fit["model_id"],
        "model_hash": fit["model_hash"],
        "fit_id": saved.dataset_id,
        "fit_hash": saved.metadata["hash"],
        "dofs": fit["coordinate_order"],
        "coordinate_units": fit["coordinate_units"],
        "effort_units": ["N"] * 3 + ["N*m"] * 41,
        "timebase": "physical_seconds",
        "provenance": {
            "kind": "authored",
            "description": "Diagnostic commands; no measured forces",
        },
        "segments": [
            {
                "start_s": 0.5,
                "end_s": 0.504,
                "is_bernstein": True,
                "coefficients": coefficients.tolist(),
            }
        ],
    }
    path = source.parent / "replay-profile.json"
    path.write_text(json.dumps(payload))
    library.add_profile("replay-profile", "practice", path)
    return library, path, payload


def test_replay_runs_independent_dynamics_and_preserves_authored_clock(replay_case):
    from src.shared.python.workspace.necromatcher_replay import (
        ReplayOptions,
        replay_authored_profile,
    )
    from src.shared.python.simulation_backends import Trace

    library, _, _ = replay_case
    result = replay_authored_profile(
        library, "replay-profile", ReplayOptions(0, (0.0,) * 44, 0.004, 0.001)
    )
    assert isinstance(result, Trace)
    np.testing.assert_allclose(result.t, [0.5, 0.501, 0.502, 0.503, 0.504])
    assert result.q.shape == (5, 44)
    assert result.v.shape == (5, 44)
    assert np.any(result.v[1:])
    np.testing.assert_allclose(result.u[:, 6], [0, 0.25, 0.5, 0.75, 1])
    assert result.torques is None
    assert result.meta["physical_source_time_qualified"] is False
    assert result.meta["scientific_qualified"] is False
    assert result.meta["independent_replay_executed"] is True
    assert result.meta["verification_refinement"] == 4
    assert json.loads(result.meta["coordinate_units_json"]) == ["m"] * 3 + ["rad"] * 41
    assert json.loads(result.meta["effort_units_json"]) == ["N"] * 3 + ["N*m"] * 41
    assert result.meta["initial_grip_gap_m"] > 0
    np.testing.assert_array_equal(result.q[0], library.load_fit("replay-fit")["q"][0])
    np.testing.assert_array_equal(result.v[0], np.zeros(44))
    assert result.meta["root_policy"] == "unactuated"
    from src.shared.python.simulation_backends.trace_io import read_trace, write_trace

    path = library.root / "replay.h5"
    write_trace(result, path)
    recalled = read_trace(path)
    np.testing.assert_array_equal(recalled.q, result.q)
    assert (
        recalled.meta["coordinate_units_json"] == result.meta["coordinate_units_json"]
    )
    assert recalled.meta["source_frame_json"] == result.meta["source_frame_json"]
    assert not recalled.meta["scientific_qualified"]


@pytest.mark.parametrize(
    "field,value",
    [
        ("dt_s", 0),
        ("duration_s", float("inf")),
        ("record_every", True),
        ("refinement", 1),
        ("initial_rates", (True,) * 44),
        ("source_frame_index", True),
        ("duration_s", 0.0045),
    ],
)
def test_invalid_replay_parameters_rejected(field, value):
    from src.shared.python.workspace.necromatcher_replay import ReplayOptions

    options = {
        "source_frame_index": 0,
        "initial_rates": (0.0,) * 44,
        "duration_s": 0.004,
        "dt_s": 0.001,
    }
    options[field] = value
    with pytest.raises(ValueError):
        ReplayOptions(**options)


def test_root_commands_are_rejected_before_simulator_can_silently_zero_them(
    replay_case,
):
    from src.shared.python.workspace.necromatcher_replay import (
        ReplayOptions,
        replay_authored_profile,
    )

    library, path, payload = replay_case
    # Nonzero interior coefficients must be rejected even when endpoint commands are zero.
    payload["segments"][0]["coefficients"] = np.zeros((44, 3)).tolist()
    payload["segments"][0]["coefficients"][0][1] = 1.0
    path.write_text(json.dumps(payload))
    library.add_profile("root-profile", "practice", path)
    with pytest.raises(ValueError, match="root"):
        replay_authored_profile(
            library, "root-profile", ReplayOptions(0, (0.0,) * 44, 0.004, 0.001)
        )


def test_replay_rejects_unknown_frame_rates_and_horizon(replay_case):
    from src.shared.python.workspace.necromatcher_replay import (
        ReplayOptions,
        replay_authored_profile,
    )

    library, _, _ = replay_case
    options = ReplayOptions(0, (0.0,) * 44, 0.004, 0.001)
    for changed in (
        replace(options, source_frame_index=1),
        replace(options, initial_rates=(0.0,)),
        replace(options, duration_s=0.006),
    ):
        with pytest.raises((IndexError, ValueError)):
            replay_authored_profile(library, "replay-profile", changed)

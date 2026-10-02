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


@pytest.fixture
def replay_trace(replay_case):
    from src.shared.python.workspace import ReplayOptions, replay_authored_profile

    library, _, _ = replay_case
    trace = replay_authored_profile(
        library, "replay-profile", ReplayOptions(0, (0.0,) * 44, 0.004, 0.001)
    )
    return library, trace


def test_replay_versions_survive_recall_and_portable_export(replay_trace):
    from zipfile import ZipFile
    from src.shared.python.simulation_backends.trace_io import write_trace
    from src.shared.python.core.contracts.exceptions import StateError

    library, trace = replay_trace
    source = library.root / "run.h5"
    write_trace(trace, source)
    saved = library.add_replay("saved-replay", "practice", source)
    assert saved.kind == "authored_replay"
    assert saved.metadata["qualification"] == "unqualified_authored_replay"
    recalled = library.load_replay(saved.dataset_id)
    np.testing.assert_array_equal(recalled.q, trace.q)
    assert recalled.meta["profile_hash"] == trace.meta["profile_hash"]
    with pytest.raises(StateError):
        library.add_replay("saved-replay", "practice", source)
    package = library.root / "swing.zip"
    library.export_swing("practice", package)
    with ZipFile(package) as archive:
        manifest = json.loads(archive.read("manifest.json"))
        entry = next(
            x for x in manifest["assets"] if x["dataset_id"] == saved.dataset_id
        )
        assert entry["metadata"]["schema"] == "simulation_backend.trace/2.1.0"
        assert archive.read(entry["path"]) == source.read_bytes()


@pytest.mark.parametrize(
    "mutation",
    [
        "profile_hash",
        "model_hash",
        "source_frame",
        "units",
        "qualified",
        "initial_pose",
        "rates",
        "controls",
        "clock",
        "diagnostic",
        "missing_option",
    ],
)
def test_replay_import_rejects_relabelled_or_malformed_evidence(replay_trace, mutation):
    from src.shared.python.simulation_backends.trace_io import write_trace

    library, trace = replay_trace
    meta = dict(trace.meta)
    if mutation in {"profile_hash", "model_hash"}:
        meta[mutation] = "sha256:" + "0" * 64
    elif mutation == "source_frame":
        meta["source_frame_json"] = "{}"
    elif mutation == "units":
        meta["coordinate_units_json"] = json.dumps(["rad"] * 44)
    elif mutation == "qualified":
        meta["scientific_qualified"] = True
    elif mutation == "initial_pose":
        trace.q[0, 0] += 1
    elif mutation == "rates":
        trace.v[0, 0] += 1
    elif mutation == "controls":
        trace.u[1, 6] += 1
    elif mutation == "clock":
        trace.t[-1] += 0.001
    elif mutation == "diagnostic":
        meta["max_grip_gap_m"] = float("nan")
    elif mutation == "missing_option":
        meta.pop("duration_s")
    trace.meta = meta
    source = library.root / "invalid.h5"
    write_trace(trace, source)
    with pytest.raises(ValueError):
        library.add_replay("invalid-replay", "practice", source)
    assert not any(x.dataset_id == "invalid-replay" for x in library.assets("practice"))


def test_replay_recall_and_export_recheck_parent_bytes(replay_trace):
    from pathlib import Path
    from src.shared.python.simulation_backends.trace_io import write_trace

    library, trace = replay_trace
    source = library.root / "run.h5"
    write_trace(trace, source)
    library.add_replay("saved-replay", "practice", source)
    parent = library.load_asset("replay-profile")
    Path(parent.path).write_text("{}")
    for action in (
        lambda: library.load_replay("saved-replay"),
        lambda: library.export_swing("practice", library.root / "invalid.zip"),
    ):
        with pytest.raises(ValueError, match="hash mismatch"):
            action()


def test_shared_api_imports_and_recalls_replay_versions(replay_trace):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes import necromatcher as routes
    from src.shared.python.simulation_backends.trace_io import write_trace

    library, trace = replay_trace
    source = library.root / "run.h5"
    write_trace(trace, source)
    app = FastAPI()
    app.include_router(routes.router)
    app.dependency_overrides[routes.get_library] = lambda: library
    app.dependency_overrides[routes.require_local_client] = lambda: None
    with TestClient(app) as client:
        response = client.post(
            "/necromatcher/swings/practice/replays",
            json={"id": "api-replay", "source_path": str(source)},
        )
        assert response.status_code == 201, response.text
        assert response.json()["kind"] == "authored_replay"
        summary = client.get("/necromatcher/replays/api-replay")
        assert summary.status_code == 200, summary.text
        assert summary.json()["sample_count"] == 5
        assert summary.json()["metadata"]["scientific_qualified"] is False
        download = client.get("/necromatcher/replays/api-replay/data")
        assert download.status_code == 200
        assert download.content == source.read_bytes()
        assert client.get("/necromatcher/replays/replay-profile").status_code == 422


def test_malformed_trace_file_has_an_admission_error(replay_case):
    library, _, _ = replay_case
    source = library.root / "broken.h5"
    source.write_bytes(b"not an HDF5 trace")
    with pytest.raises(ValueError, match="canonical trace"):
        library.add_replay("broken-replay", "practice", source)


def test_replay_storage_preserves_an_uneven_final_recording_stride(replay_case):
    from src.shared.python.workspace import ReplayOptions, replay_authored_profile
    from src.shared.python.simulation_backends.trace_io import write_trace

    library, _, _ = replay_case
    trace = replay_authored_profile(
        library,
        "replay-profile",
        ReplayOptions(0, (0.0,) * 44, 0.003, 0.001, record_every=2),
    )
    np.testing.assert_allclose(trace.t, [0.5, 0.502, 0.503])
    source = library.root / "uneven.h5"
    write_trace(trace, source)
    library.add_replay("uneven-replay", "practice", source)
    np.testing.assert_array_equal(library.load_replay("uneven-replay").t, trace.t)


def test_replay_storage_requires_the_native_root_coordinate_order(replay_trace):
    from src.shared.python.workspace.necromatcher_replay_storage import _check_samples
    from src.shared.python.workspace import ReplayOptions

    library, trace = replay_trace
    controls = library.load_effort_profile("replay-profile", trace.meta["model_id"])
    swapped = replace(
        controls, dofs=(controls.dofs[1], controls.dofs[0]) + controls.dofs[2:]
    )
    with pytest.raises(ValueError, match="root coordinate order"):
        _check_samples(
            trace,
            swapped,
            library.load_fit("replay-fit"),
            ReplayOptions(0, (0.0,) * 44, 0.004, 0.001),
        )

"""Opt-in native replay→owned clean worker→retained trajectory evidence."""

from __future__ import annotations

import json
from pathlib import Path
import time
from zipfile import ZipFile

import numpy as np
import pytest
from tests.unit.workspace import test_necromatcher_replay as replay_fixtures

replay_case = replay_fixtures.replay_case

pytestmark = [pytest.mark.live_simulation, pytest.mark.integration]


def test_native_saved_replay_reaches_owned_impact_bundle(
    replay_case, tmp_path: Path
) -> None:
    """No native, impact, flight, stamp or worker substitutions are permitted."""
    from src.shared.python import workspace as owner
    from src.shared.python.simulation_backends.trace_io import write_trace
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding

    pytest.importorskip("upstream_physics")
    assert hasattr(owner, "NativeImpactSession"), (
        "Impact session must use the public workspace facade"
    )
    library, _, _ = replay_case
    rates = (20.0,) + (0.0,) * 43
    trace = owner.replay_authored_profile(
        library, "replay-profile", owner.ReplayOptions(0, rates, 0.004, 0.001)
    )
    trace_file = tmp_path / "native-replay.h5"
    write_trace(trace, trace_file)
    library.add_replay("native-replay", "practice", trace_file)
    binding = load_native_fit_binding(library, "replay-fit")
    body = "solid_reference:GolfSwing3D_Kinetic/Club/Clubface Vector"
    rotation, _ = binding.plant.frame_poses({"head": (body, (0, 0, 0))}, trace.q[0])[
        body
    ]
    geometry = owner.ReplayImpactGeometry(
        body,
        (0, 0, 0),
        rotation.T @ np.array([1, 0, 0]),
        rotation.T @ np.array([0, 0, 1]),
        0.2,
        0.005,
        "Synthetic declared face, body origin and effective inertia; no historical anatomy",
    )
    selection = owner.ReplayImpactSelection(
        0,
        np.eye(3),
        (0, 0, 0),
        "Authored initial recorded sample, not detected contact",
    )
    expected = owner.extract_replay_impact_state(
        library, "native-replay", geometry, selection
    )
    session = owner.NativeImpactSession(library)
    try:
        run = session.submit("native-replay", geometry, selection, 60.0)["run_id"]
        deadline = time.monotonic() + 65
        while True:
            view = session.view("native-replay", run)
            if view["status"] not in {"pending", "running"}:
                break
            if time.monotonic() > deadline:
                pytest.fail(
                    "Owned native impact run did not close inside the test deadline"
                )
            time.sleep(0.2)
        assert view["status"] == "succeeded", view
        assert view["acceptance"] == "rejected"
        assert view["execution_verified"] and view["download_available"]
        with ZipFile(session.download("native-replay", run)) as archive:
            assert set(archive.namelist()) == {
                "trajectory.json",
                "impact-receipt.json",
                "result.json",
                "request.json",
            }
            extracted = tmp_path / "downloaded"
            archive.extractall(extracted)
        recalled = owner.load_replay_impact_receipt(
            extracted / "impact-receipt.json", extracted / "trajectory.json"
        )
        np.testing.assert_array_equal(
            recalled.clubhead_velocity, expected.clubhead_velocity
        )
        np.testing.assert_array_equal(
            recalled.clubhead_angular_velocity, expected.clubhead_angular_velocity
        )
        np.testing.assert_array_equal(
            recalled.clubhead_orientation, expected.clubhead_orientation
        )
        result = json.loads((extracted / "result.json").read_bytes())
        assert result["impact_state"]["ball_velocity"][0] > 0
        assert result["scientific_qualified"] is False
        record = json.loads((extracted / "trajectory.json").read_bytes())
        coordinator = owner.ShotTrajectoryHandoffCoordinator(repo_root=tmp_path)
        web = coordinator.load_into_ball_flight_web(extracted / "trajectory.json")
        qt = coordinator.load_into_shot_tracer(extracted / "trajectory.json")
        assert len(web.samples) == len(qt.positions) == len(record["samples"])
        assert (
            coordinator.load_into_impact_explorer(extracted / "trajectory.json")[
                "samples"
            ]
            == record["samples"]
        )
    finally:
        session.close()
    reopened = owner.NativeImpactSession(library)
    try:
        assert reopened.view("native-replay", run)["download_available"]
        path = library.root / "impact-runs" / run / "output" / "trajectory.json"
        path.write_bytes(path.read_bytes() + b"\n")
        with pytest.raises((ValueError, RuntimeError)):
            reopened.download("native-replay", run)
    finally:
        reopened.close()

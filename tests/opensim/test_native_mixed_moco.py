"""Mixed Moco export preserves assistance semantics and fresh native replay."""

from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from tests.opensim.test_native_mixed_actuation import _profile
from tests.opensim.test_native_mixed_actuation import mixed_model as mixed_model
from tests.opensim.test_native_moco_runner import _write_request
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_runner import (
    NativeMocoRequest,
    prepare_native_moco,
    solve_native_moco,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
    load_native_moco_request,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_moco_replay import (
    export_native_moco_bundle,
    replay_native_moco_bundle,
    score_native_moco_replay,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_initial_bindings import (
    MocoInitialBindings,
)
from src.engines.physics_engines.opensim.python.tour_matching.moco_tracking import (
    MocoTrackingConfig,
)
from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
    MusclePassiveLimits,
    PassiveReadinessPolicy,
)
from src.engines.physics_engines.opensim.python.tour_matching.registration import (
    CaptureRegistration,
)
from src.engines.physics_engines.opensim.python.tour_matching.trc import write_trc
from src.shared.python.motion_matching.tour_capture_contract import TourCapture

pytestmark = pytest.mark.unit


@pytest.fixture
def mixed_request(
    mixed_model: tuple[Any, Any, Path], tmp_path: Path
) -> NativeMocoRequest:
    import opensim as osim

    model, state, path = mixed_model
    names = model.getStateVariableNames()
    initial = {
        names.get(i): model.getStateVariableValue(state, names.get(i))
        for i in range(names.getSize())
    }
    model.addMarker(
        osim.Marker("load_marker", model.getBodySet().get("load"), osim.Vec3(0))
    )
    model.addMarker(
        osim.Marker("arm_marker", model.getBodySet().get("arm"), osim.Vec3(0.1, 0, 0))
    )
    model.finalizeConnections()
    model.initSystem()
    model.printToXML(str(path))
    times = np.linspace(0, 0.04, 9)
    displacement = 0.0002 * (times / times[-1]) ** 2
    points = np.zeros((len(times), 2, 3))
    points[:, :, 0] = 0.31 + displacement[:, None]
    points[:, 1, 0] += 0.1 * np.cos(displacement)
    points[:, 1, 1] = 0.1 * np.sin(displacement)
    trc = write_trc(
        TourCapture(
            times, ("load_marker", "arm_marker"), points, np.ones((len(times), 2), bool)
        ),
        tmp_path / "mixed.trc",
        rate_hz=200,
    )
    table = osim.TimeSeriesTable()
    labels = osim.StdVectorString()
    for name in initial:
        labels.append(name)
    table.setColumnLabels(labels)
    for t in times:
        row = osim.RowVector(len(initial))
        for i, value in enumerate(initial.values()):
            row[i] = value
        table.appendRow(float(t), row)
    guess = tmp_path / "mixed_guess.sto"
    osim.STOFileAdapter.write(table, str(guess))
    bounds = {
        name: (
            (value - 0.1, value + 0.1)
            if name.endswith("/value")
            else (-1.0, 1.0)
            if name.endswith("/speed")
            else (0.01, 1.0)
            if name.endswith("/activation")
            else (0.05, 0.2)
        )
        for name, value in initial.items()
    }
    profile = _profile()
    policy = PassiveReadinessPolicy(
        hashlib.sha256(model.dump().encode()).hexdigest(),
        "predeclared synthetic envelope only",
        tuple(
            MusclePassiveLimits(
                c.path,
                (0.1, 10.0),
                1000.0,
                1000.0,
                False,
                False,
                "synthetic fixture only",
            )
            for c in profile.channels[:2]
        ),
    )

    def sha(p: Path) -> str:
        return hashlib.sha256(p.read_bytes()).hexdigest()

    return NativeMocoRequest(
        path,
        trc,
        guess,
        sha(path),
        sha(trc),
        sha(guess),
        MocoInitialBindings(
            bounds, initial, {c.path: c.control_bounds for c in profile.channels}
        ),
        MocoTrackingConfig(
            horizon_s=0.04,
            mesh_interval_s=0.002,
            optim_max_iterations=300,
            optim_constraint_tolerance=1e-7,
            optim_convergence_tolerance=1e-6,
            allow_unused_references=False,
            effort_weight=1e-8,
            marker_weight=1e6,
        ),
        {"load_marker": 1.0, "arm_marker": 1.0},
        {
            "load_marker": ("/bodyset/load", (0.0, 0.0, 0.0)),
            "arm_marker": ("/bodyset/arm", (0.1, 0.0, 0.0)),
        },
        CaptureRegistration(np.eye(3), np.zeros(3)),
        "/ground",
        policy,
        {},
        mixed_actuation=profile,
    )


def test_mixed_request_identity_bounds_and_json(
    mixed_request: NativeMocoRequest, tmp_path: Path
) -> None:
    profile = mixed_request.mixed_actuation
    changed = replace(
        profile,
        channels=(
            *profile.channels[:-1],
            replace(profile.channels[-1], role=profile.channels[-2].role),
        ),
    )
    assert (
        replace(mixed_request, mixed_actuation=changed).identity_sha256
        != mixed_request.identity_sha256
    )
    with pytest.raises(ValueError, match="bounds"):
        replace(
            mixed_request,
            bindings=replace(
                mixed_request.bindings,
                control_bounds={
                    **mixed_request.bindings.control_bounds,
                    "/forceset/upper_torque": (-0.2, 0.2),
                },
            ),
        )
    path = tmp_path / "request.json"
    _write_request(mixed_request, path)
    payload = json.loads(path.read_text())
    payload["mixed_actuation"] = asdict(profile)
    path.write_text(json.dumps(payload))
    assert (
        load_native_moco_request(path).identity_sha256 == mixed_request.identity_sha256
    )
    payload["mixed_actuation"]["channels"][0]["inferred"] = True
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError, match="missing or unknown"):
        load_native_moco_request(path)


def test_native_mixed_moco_solve_export_and_independent_replay(
    mixed_request: NativeMocoRequest, tmp_path: Path
) -> None:
    directory = tmp_path / "run"
    prepared = prepare_native_moco(mixed_request, directory)
    assert prepared.ready_for_software_solve, prepared.blockers
    strict = prepare_native_moco(
        replace(mixed_request, mixed_actuation=None), tmp_path / "strict"
    )
    assert "nonmuscle-assistance-unqualified" in strict.blockers
    solved = solve_native_moco(mixed_request, prepared, directory)
    assert solved.success, solved.status
    exported = export_native_moco_bundle(mixed_request, prepared, solved, directory)
    profile = mixed_request.mixed_actuation
    changed = replace(
        profile,
        channels=(
            *profile.channels[:-1],
            replace(profile.channels[-1], role=profile.channels[-2].role),
        ),
    )
    with pytest.raises(ValueError):
        replay_native_moco_bundle(
            exported, mixed_request.model_path, directory, mixed_actuation=changed
        )
    with pytest.raises(ValueError):
        replay_native_moco_bundle(exported, mixed_request.model_path, directory)
    native = replay_native_moco_bundle(
        exported,
        mixed_request.model_path,
        directory,
        mixed_actuation=mixed_request.mixed_actuation,
    )
    assert native.channel_paths == tuple(
        c.path for c in mixed_request.mixed_actuation.channels
    )
    for name in ("/jointset/slider/pelvis_tx/value", "/jointset/pin/arm_flex_r/value"):
        column = native.state_names.index(name)
        assert abs(native.states[-1, column] - native.states[0, column]) > 1e-4
    assert np.any(np.abs(native.actuations[:, -2:]) > 1e-6)
    evidence = np.load(directory / "native_replay.npz", allow_pickle=False)
    assert "muscle_forces_n" not in evidence and "applied_excitations" not in evidence
    assert evidence["output_units"].tolist() == ["N", "N", "N", "N*m"]
    np.testing.assert_array_equal(evidence["applied_controls"], native.applied_controls)
    score = score_native_moco_replay(
        mixed_request, prepared, solved, exported, native, directory
    )
    assert score.marker_rmse_m < 2e-4
    import opensim as osim

    trajectory = osim.MocoTrajectory(str(directory / "solution.sto"))
    knots = np.asarray(trajectory.getTimeMat()).reshape(-1)
    indices = np.searchsorted(native.times, knots)
    errors = {
        name: float(
            np.max(
                np.abs(
                    np.asarray(trajectory.getStateMat(name)).reshape(-1)
                    - native.states[indices, i]
                )
            )
        )
        for i, name in enumerate(native.state_names)
    }
    (directory / "per_state_error.json").write_text(json.dumps(errors, indent=2))
    # Synthetic numerical tolerances, separated by state kind and physical unit.
    limits = {"value": 1e-7, "speed": 1e-5, "activation": 1e-3, "fiber_length": 1e-5}
    assert all(
        error < limits[name.rsplit("/", 1)[-1]] for name, error in errors.items()
    ), errors
    assert np.isfinite(native.powers_w).all() and np.isfinite(native.work_j).all()

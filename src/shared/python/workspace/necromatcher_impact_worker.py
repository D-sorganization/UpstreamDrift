"""SDK-first owned replay impact/flight worker; no reference or flight fallback.

Authored replay seconds remain research inputs, never historical clock calibration.
Only a complete, reauthenticated three-artifact bundle is published.
"""

from __future__ import annotations

import hashlib
import json
import os
import re
from dataclasses import asdict
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from typing import Any


def _api() -> SimpleNamespace:
    from .necromatcher import NecromatcherLibrary
    from .necromatcher_impact import extract_replay_impact_state
    from .necromatcher_impact_execution import impact_execution_stamp
    from .necromatcher_impact_jobs import impact_job_parents
    from .necromatcher_impact_receipt import export_replay_impact_receipt
    from .trajectory_handoff import ShotTrajectoryHandoffCoordinator

    return SimpleNamespace(
        library=NecromatcherLibrary,
        stamp=impact_execution_stamp,
        parents=impact_job_parents,
        extract=extract_replay_impact_state,
        coordinator=ShotTrajectoryHandoffCoordinator,
        receipt=export_replay_impact_receipt,
    )


def _json_value(value: Any) -> Any:
    """Detach actual dataclass arrays, retaining finite JSON values and units."""
    if isinstance(value, dict):
        return {key: _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if hasattr(value, "tolist"):
        return _json_value(value.tolist())
    return value


def _equal(actual: Any, expected: Any, name: str) -> None:
    if json.dumps(actual, sort_keys=True, allow_nan=False) != json.dumps(
        expected, sort_keys=True, allow_nan=False
    ):
        raise ValueError(f"Impact {name} changed or differs")


def _check_stamp(api: SimpleNamespace, expected: dict[str, Any]) -> None:
    """Observation timestamps are not implementation/runtime identity."""
    keys = ("source_commit", "source_sha256", "runtime_sha256")
    actual = api.stamp()
    if not isinstance(expected, dict) or any(
        not isinstance(expected.get(key), str) or not expected[key] for key in keys
    ):
        raise ValueError("Impact execution stamp requires stable identity pins")
    _equal(
        {key: actual.get(key) for key in keys},
        {key: expected[key] for key in keys},
        "execution stamp",
    )


def _summary(result: Any) -> dict[str, Any]:
    """Persist actual effective parameters; no claim that defaults were calibrated."""
    from src.shared.python.physics.ball_properties import BallProperties

    return _json_value(
        {
            "environment": asdict(result.environment),
            "impact_params": asdict(result.impact_params),
            "impact_method": "RIGID_BODY",
            "ball_assumptions": asdict(BallProperties()),
            "launch_conditions": asdict(result.launch_conditions),
            "impact_state": asdict(result.impact_state),
            "summary": {
                name: getattr(result, name)
                for name in (
                    "carry_m",
                    "max_height_m",
                    "flight_time_s",
                    "landing_angle_deg",
                )
            },
            "pipeline_metadata": result.metadata,
            "flight_settings": {"max_time_s": 10.0, "dt_s": 0.01},
        }
    )


def compute_native_impact(request: dict[str, Any]) -> dict[str, Any]:
    """Extract recorded state and publish actual pipeline artifacts exclusively.

    Preconditions: caller is the SDK-first clean worker; output_root is a
    server-owned absent path. Parents are reauthenticated around execution.
    Postconditions: exact hashes/sizes bind a complete bundle; qualification
    remains false. Exceptions publish no successful output directory.
    """
    from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_json

    from .necromatcher_impact_contracts import (
        ReplayImpactGeometry,
        ReplayImpactSelection,
    )

    keys = {
        "library_root",
        "replay_id",
        "geometry",
        "selection",
        "execution_stamp",
        "output_root",
    }
    if not isinstance(request, dict) or set(request) != keys:
        raise ValueError("Impact compute request requires exact fields")
    for name in ("library_root", "replay_id", "output_root"):
        if not isinstance(request[name], str) or not request[name].strip():
            raise ValueError(f"Impact {name} requires a nonempty string")
    geometry = ReplayImpactGeometry.from_record(request["geometry"])
    selection = ReplayImpactSelection.from_record(request["selection"])
    output = Path(request["output_root"])
    if output.exists() or output.is_symlink():
        raise FileExistsError("Impact output must be absent")
    api = _api()
    _check_stamp(api, request["execution_stamp"])
    library = api.library(request["library_root"])
    parents = api.parents(library, request["replay_id"])
    state = api.extract(library, request["replay_id"], geometry, selection)
    for key, value in parents.items():
        _equal(state.metadata.get(key), value, "extracted parent")
    output.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(prefix=".impact-", dir=output.parent) as temporary:
        staged = Path(temporary) / "output"
        staged.mkdir()
        coordinator = api.coordinator(repo_root=staged)
        result, trajectory = coordinator.simulate_and_export_trajectory(
            state, "trajectory"
        )
        api.receipt(state, trajectory, staged / "impact-receipt.json")
        record = {
            "schema": "necromatcher/impact-result/1",
            "parents": parents,
            "geometry": geometry.to_record(),
            "selection": selection.to_record(),
            "execution_stamp": request["execution_stamp"],
            "physical_source_time_qualified": False,
            "scientific_qualified": False,
            "replay_clock_policy": "authored_simulation_seconds",
            "extraction_metadata": state.metadata,
            **_summary(result),
        }
        atomic_write_json(staged / "result.json", record)
        _equal(api.parents(library, request["replay_id"]), parents, "parents")
        _check_stamp(api, request["execution_stamp"])
        hashes, sizes = {}, {}
        for path in staged.iterdir():
            raw = path.read_bytes()
            hashes[path.name] = "sha256:" + hashlib.sha256(raw).hexdigest()
            sizes[path.name] = len(raw)
        if output.exists() or output.is_symlink():
            raise FileExistsError("Impact output appeared during execution")
        os.rename(staged, output)
    return {
        "artifact_hashes": hashes,
        "artifact_sizes": sizes,
        "summary": record["summary"],
    }


def main() -> None:
    """Load SDK before workspace imports; consume the server-owned job envelope."""
    import sys

    path = Path(sys.argv[1])
    if any(
        candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)()
        for candidate in (path, *path.parents)
    ):
        raise ValueError("Impact job path ancestors must be link-free")
    request = json.loads(path.read_bytes())
    keys = {
        "kind",
        "run_id",
        "library_root",
        "replay_id",
        "parents",
        "geometry",
        "selection",
        "budget_wall_s",
        "execution_stamp",
    }
    if not isinstance(request, dict) or set(request) != keys:
        raise ValueError("Impact job envelope requires exact fields")
    if (
        request["kind"] != "necromatcher/impact-job/1"
        or not isinstance(request["run_id"], str)
        or re.fullmatch(r"[a-f0-9]{32}", request["run_id"]) is None
        or request["run_id"] != path.parent.name
        or path.name != "request.json"
        or path.parent.parent.name != "impact-runs"
        or path.parent.parent.parent.resolve()
        != Path(request["library_root"]).resolve()
    ):
        raise ValueError("Impact job path and identity differ")
    library_root = Path(request["library_root"])
    if any(
        candidate.is_symlink() or getattr(candidate, "is_junction", lambda: False)()
        for candidate in (library_root, *library_root.parents)
    ):
        raise ValueError("Impact library path ancestors must be link-free")
    import mujoco  # noqa: F401 -- SDK-first native execution is deliberate.

    from .necromatcher import NecromatcherLibrary
    from .necromatcher_impact_jobs import impact_job_parents

    _check_stamp(_api(), request["execution_stamp"])
    library = NecromatcherLibrary(request["library_root"])
    _equal(
        impact_job_parents(library, request["replay_id"]),
        request["parents"],
        "queued parents",
    )
    compute = {
        key: request[key]
        for key in (
            "library_root",
            "replay_id",
            "geometry",
            "selection",
            "execution_stamp",
        )
    }
    compute["output_root"] = str(path.parent / "output")
    sys.stdout.write(json.dumps(compute_native_impact(compute), allow_nan=False))


if __name__ == "__main__":
    main()

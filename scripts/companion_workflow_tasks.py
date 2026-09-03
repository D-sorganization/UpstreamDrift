"""Deterministic, headless tasks exercised by the companion workflow authority.

These tasks are small adapters over existing UpstreamDrift public surfaces. They
produce reviewable evidence artifacts; they do not create a second scientific
calculation authority or claim that optional GUI/engine runtimes are qualified.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import math
import subprocess
import sys
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path
from typing import Any


def _repo_root() -> Path:
    root = Path.cwd().resolve()
    if not (root / "pyproject.toml").is_file():
        raise ValueError("companion workflow tasks must run from the repository root")
    return root


def _output_path(root: Path, raw: str) -> Path:
    from scripts.companion_catalog import validate_repo_relative

    relative = validate_repo_relative(Path(raw))
    destination = root.joinpath(*relative.parts).resolve()
    try:
        destination.relative_to(root)
    except ValueError as exc:  # pragma: no cover - defense after lexical validation
        raise ValueError(f"output escapes repository root: {raw}") from exc
    destination.parent.mkdir(parents=True, exist_ok=True)
    return destination


def _source_commit(root: Path) -> str:
    completed = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        check=False,
        capture_output=True,
        encoding="utf-8",
        errors="replace",
        shell=False,
    )
    commit = completed.stdout.strip().lower()
    if completed.returncode != 0 or len(commit) != 40:
        raise RuntimeError("workflow task could not resolve an exact source commit")
    return commit


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
        newline="\n",
    )


def _rounded(values: Sequence[float]) -> list[float]:
    return [round(float(value), 12) for value in values]


def _reference_simulation() -> Any:
    import numpy as np

    from src.shared.python.pendulum_simulator.physics import PendulumParams
    from src.shared.python.pendulum_simulator.simulation import run_simulation

    params = PendulumParams(
        m1=5.0,
        m2=0.3,
        L1=0.65,
        L2=1.1,
        mClub=0.2,
        b1=0.05,
        b2=0.02,
    )
    state = np.array([math.radians(-35.0), math.radians(70.0), 0.2, -0.1])
    return run_simulation(
        params,
        state,
        0.05,
        lambda _time: (12.0, -3.0),
        dt=0.01,
    )


def _installation_verification(root: Path, output: Path) -> int:
    from scripts.ci.verify_installation import check_python_version
    from src.shared.python.config.model_registry import ModelRegistry

    python_ok, _message = check_python_version()
    registry = ModelRegistry(
        config_path=root / "src/config/models.yaml",
        strict=True,
        discovery_mode="local-only",
    )
    _write_json(
        output,
        {
            "check_scope": "provider workflow core imports and local registry",
            "model_records": len(registry.get_all_models()),
            "python_major_minor": f"{sys.version_info.major}.{sys.version_info.minor}",
            "python_supported": python_ok,
            "source_commit": _source_commit(root),
            "status": "passed" if python_ok else "failed",
        },
    )
    return 0 if python_ok else 10


def _model_launch_resolution(root: Path, output: Path) -> int:
    from src.shared.python.config.model_registry import ModelRegistry

    registry = ModelRegistry(
        config_path=root / "src/config/models.yaml",
        strict=True,
        discovery_mode="local-only",
    )
    model = registry.get_model("model_explorer")
    if model is None or model.path is None:
        raise RuntimeError("model_explorer launch record is missing")
    entry_point = root / model.path
    if not entry_point.is_file():
        raise RuntimeError(f"model_explorer entry point is missing: {model.path}")
    _write_json(
        output,
        {
            "entry_point": model.path,
            "launch_argv": ["python3", model.path],
            "launch_mode": "headless-resolution-only",
            "program_id": model.id,
            "source_commit": _source_commit(root),
            "status": "resolved",
        },
    )
    return 0


def _reference_simulation_task(root: Path, output: Path) -> int:
    result = _reference_simulation()
    _write_json(
        output,
        {
            "coordinate_order": ["theta1", "phi", "dtheta1", "dphi"],
            "final_state": _rounded(result.states[-1]),
            "frame": "planar pendulum model frame",
            "n_steps": result.n_steps,
            "source_commit": _source_commit(root),
            "time_end_s": round(float(result.t[-1]), 12),
            "units": ["rad", "rad", "rad/s", "rad/s"],
        },
    )
    return 0


def _reference_analysis(root: Path, output: Path) -> int:
    result = _reference_simulation()
    tip_speeds = [
        result.joint_velocities_at(i)["tip_speed"] for i in range(result.n_steps)
    ]
    initial_energy = result.energy_at(0)
    final_energy = result.energy_at(result.n_steps - 1)
    _write_json(
        output,
        {
            "energy_change_j": round(
                float(final_energy["total"] - initial_energy["total"]), 12
            ),
            "estimands": ["maximum tip speed", "total mechanical energy change"],
            "max_tip_speed_m_s": round(float(max(tip_speeds)), 12),
            "n_steps": result.n_steps,
            "source_commit": _source_commit(root),
        },
    )
    return 0


def _launch_monitor_roundtrip(root: Path, output: Path, csv_output: Path) -> int:
    from src.tools.launch_monitor_model import import_session

    fixture = root / "tests/fixtures/launch_monitor/trackman.csv"
    session = import_session(fixture)
    canonical_columns = [
        column
        for column in (
            "shot_id",
            "club_speed",
            "ball_speed",
            "launch_angle",
            "spin_rate",
            "carry_distance",
        )
        if column in session.shots.columns
    ]
    if len(canonical_columns) < 5:
        raise RuntimeError("TrackMan fixture did not map the canonical metrics")
    session.shots.loc[:, canonical_columns].to_csv(
        csv_output,
        index=False,
        lineterminator="\n",
    )
    _write_json(
        output,
        {
            "canonical_columns": canonical_columns,
            "export_path": csv_output.relative_to(root).as_posix(),
            "profile_id": session.manifest.profile_id,
            "row_count": int(len(session.shots)),
            "source_commit": _source_commit(root),
            "source_fixture": "tests/fixtures/launch_monitor/trackman.csv",
            "source_sha256": session.manifest.file_sha256,
        },
    )
    return 0


def _program_catalog_export(root: Path, output: Path) -> int:
    from scripts import companion_catalog

    catalog = companion_catalog.build_catalog(root, require_clean=False)
    selected_ids = {"model_explorer", "pendulum_simulator", "pose_studio"}
    records = [
        {
            "availability": program["availability"],
            "entry_point": program["entry_point"],
            "id": program["id"],
            "name": program["name"],
        }
        for program in catalog["programs"]
        if program["id"] in selected_ids
    ]
    if len(records) != len(selected_ids):
        raise RuntimeError("program catalog slice is incomplete")
    _write_json(
        output,
        {
            "programs": records,
            "source_commit": catalog["source"]["commit"],
        },
    )
    return 0


def _zero_torque_counterfactual(root: Path, output: Path) -> int:
    import numpy as np

    from src.shared.python.pendulum_simulator.counterfactual import (
        zero_torque_joint_forces_double,
    )
    from src.shared.python.pendulum_simulator.physics import PendulumParams

    state = np.array([0.3, -0.2, 0.5, -0.3])
    params = PendulumParams(m1=5.0, m2=0.3, L1=0.65, L2=1.1, mClub=0.2)
    forces = zero_torque_joint_forces_double(state, params)
    _write_json(
        output,
        {
            "counterfactual": "instantaneous zero applied driving torque",
            "forces_n": {
                name: _rounded(value) for name, value in sorted(forces.items())
            },
            "source_commit": _source_commit(root),
            "state_order": ["theta1", "phi", "dtheta1", "dphi"],
        },
    )
    return 0


def _reference_report(root: Path, output: Path) -> int:
    result = _reference_simulation()
    tip_speeds = [
        result.joint_velocities_at(i)["tip_speed"] for i in range(result.n_steps)
    ]
    commit = _source_commit(root)
    output.write_text(
        "\n".join(
            [
                "# Governed Reference Simulation Report",
                "",
                f"- Source commit: `{commit}`",
                f"- Samples: {result.n_steps}",
                f"- Maximum tip speed: {max(tip_speeds):.9f} m/s",
                "- Scope: deterministic software workflow evidence only",
                "- Limitation: this short synthetic trace is not a golfer validation dataset",
                "",
            ]
        ),
        encoding="utf-8",
        newline="\n",
    )
    return 0


def _reference_plot(root: Path, output: Path) -> int:
    import matplotlib

    matplotlib.use("Agg")
    from matplotlib.figure import Figure

    result = _reference_simulation()
    tip_speeds = [
        result.joint_velocities_at(i)["tip_speed"] for i in range(result.n_steps)
    ]
    figure = Figure(figsize=(6.4, 3.6), dpi=100, layout="constrained")
    axes = figure.subplots()
    axes.plot(result.t, tip_speeds, color="#4c78a8", linewidth=2)
    axes.set_title("Reference Double-Pendulum Tip Speed")
    axes.set_xlabel("Time (s)")
    axes.set_ylabel("Tip Speed (m/s)")
    axes.grid(alpha=0.25)
    figure.savefig(
        output,
        format="png",
        metadata={"Software": "UpstreamDrift", "SourceCommit": _source_commit(root)},
    )
    return 0


def _failure_evidence(
    root: Path,
    output: Path,
    *,
    failure_class: str,
    error_type: str,
    detail: str,
) -> None:
    _write_json(
        output,
        {
            "detail": detail,
            "error_type": error_type,
            "failure_class": failure_class,
            "source_commit": _source_commit(root),
            "status": "expected-failure",
        },
    )


def _unsupported_dependency(root: Path, output: Path) -> int:
    dependency = "upstreamdrift_companion_missing_dependency_fixture_v1"
    if importlib.util.find_spec(dependency) is not None:  # pragma: no cover
        raise RuntimeError("reserved missing-dependency fixture unexpectedly exists")
    _failure_evidence(
        root,
        output,
        failure_class="unsupported-dependency",
        error_type="ModuleNotFoundError",
        detail=f"Required fixture dependency is unavailable: {dependency}",
    )
    return 20


def _bad_input(root: Path, output: Path) -> int:
    from src.shared.python.pendulum_simulator.physics import PendulumParams

    try:
        PendulumParams(m1=0.0, m2=0.3, L1=0.65, L2=1.1)
    except ValueError as exc:
        _failure_evidence(
            root,
            output,
            failure_class="bad-input",
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        return 21
    raise RuntimeError("bad-input fixture did not fail closed")  # pragma: no cover


def _unavailable_engine(root: Path, output: Path) -> int:
    engine_id = "companion-missing-engine-fixture"
    _failure_evidence(
        root,
        output,
        failure_class="unavailable-engine",
        error_type="EngineUnavailableError",
        detail=f"Deterministic fixture engine is unavailable: {engine_id}",
    )
    return 22


def _stale_version(root: Path, output: Path) -> int:
    from scripts import companion_publication

    policy = companion_publication.load_compatibility_policy(root)
    stale = dict(policy)
    stale["current"] = "0.0.0"
    try:
        companion_publication.validate_compatibility_policy(root, stale)
    except companion_publication.PublicationContractError as exc:
        _failure_evidence(
            root,
            output,
            failure_class="stale-version",
            error_type=type(exc).__name__,
            detail=str(exc),
        )
        return 23
    raise RuntimeError("stale-version fixture did not fail closed")  # pragma: no cover


_TASKS: dict[str, Callable[[Path, Path], int]] = {
    "installation-verification": _installation_verification,
    "model-launch-resolution": _model_launch_resolution,
    "reference-simulation": _reference_simulation_task,
    "reference-analysis": _reference_analysis,
    "program-catalog-export": _program_catalog_export,
    "zero-torque-counterfactual": _zero_torque_counterfactual,
    "reference-report": _reference_report,
    "reference-plot": _reference_plot,
    "unsupported-dependency": _unsupported_dependency,
    "bad-input": _bad_input,
    "unavailable-engine": _unavailable_engine,
    "stale-version": _stale_version,
}


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("task", choices=sorted((*_TASKS, "launch-monitor-roundtrip")))
    parser.add_argument("--output", required=True)
    parser.add_argument("--csv-output")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run one deterministic task and return its governed exit code."""
    args = _parser().parse_args(argv)
    root = _repo_root()
    output = _output_path(root, args.output)
    if args.task == "launch-monitor-roundtrip":
        if not args.csv_output:
            raise ValueError("launch-monitor-roundtrip requires --csv-output")
        csv_output = _output_path(root, args.csv_output)
        return _launch_monitor_roundtrip(root, output, csv_output)
    if args.csv_output:
        raise ValueError("--csv-output is only valid for launch-monitor-roundtrip")
    return _TASKS[args.task](root, output)


if __name__ == "__main__":
    raise SystemExit(main())

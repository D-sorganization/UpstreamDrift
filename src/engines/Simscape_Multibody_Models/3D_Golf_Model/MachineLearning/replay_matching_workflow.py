"""Closed-loop MATLAB replay diagnostics harness (issue #3970).

Drives the full replay pipeline:

1. Validates inputs (polynomial coefficients MAT, target club CSV, optional
   start-state MAT).
2. If ``matlab.engine`` is importable, invokes the MATLAB driver
   ``run_replay.m`` to load ``GolfSwing3D_Kinetic``, apply polynomial inputs,
   simulate, and export the simulated club CSV plus optional joint-velocity
   CSV.
3. If MATLAB is unavailable, the simulation step is skipped. Diagnostics still
   run if a previously simulated CSV is supplied (or if one is found inside
   the run directory).
4. Calls :mod:`evaluate_matching_workflow` to compute canonical Metrics
   (target/sim trajectory comparison, residuals, fitness JSON, plots).
5. Writes a run manifest documenting inputs, outputs, scenario, git commit
   (best-effort), and timestamp under
   ``data/processed/matching_reports/<scenario>/<timestamp>/``.

The harness is intentionally tolerant of missing MATLAB so it remains usable
on CI and developer machines without a Simulink install. It raises a clear
``ReplayError`` when the MATLAB simulation step itself fails.
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import shutil
import subprocess
import sys
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

_HERE = Path(__file__).resolve().parent
DEFAULT_REPORTS_DIR = _HERE / "data" / "processed" / "matching_reports"
DEFAULT_MATLAB_DRIVER = _HERE.parent / "matlab" / "MachineLearning" / "run_replay.m"
SUPPORTED_SCENARIOS = ("full-swing", "downswing")
SIM_CSV_NAME = "simulated_club_motion.csv"
JOINT_VELOCITY_CSV_NAME = "simulated_joint_velocity.csv"


class ReplayError(RuntimeError):
    """Raised when the closed-loop replay cannot complete."""


@dataclass
class ReplayInputs:
    """Validated input bundle for a replay run."""

    polynomial_mat: Path
    target_csv: Path
    scenario: str
    start_state_mat: Path | None = None
    torque_csv: Path | None = None
    existing_sim_csv: Path | None = None
    joint_velocity_csv: Path | None = None


@dataclass
class ReplayResult:
    """Outputs from a completed replay run."""

    run_dir: Path
    manifest_path: Path
    metrics_path: Path | None
    sim_csv: Path | None
    matlab_used: bool
    skipped_reason: str | None = None
    report: dict[str, Any] = field(default_factory=dict)


def _validate_scenario(scenario: str) -> str:
    if scenario not in SUPPORTED_SCENARIOS:
        raise ValueError(
            f"scenario must be one of {SUPPORTED_SCENARIOS}, got {scenario!r}"
        )
    return scenario


def _validate_existing_file(path: Path | None, label: str) -> Path | None:
    if path is None:
        return None
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(f"{label} does not exist: {path}")
    return path


def build_inputs(
    polynomial_mat: Path,
    target_csv: Path,
    scenario: str,
    start_state_mat: Path | None = None,
    torque_csv: Path | None = None,
    existing_sim_csv: Path | None = None,
    joint_velocity_csv: Path | None = None,
) -> ReplayInputs:
    """Validate input artefacts before launching MATLAB or diagnostics."""
    polynomial_mat = Path(polynomial_mat)
    target_csv = Path(target_csv)
    if polynomial_mat.suffix.lower() != ".mat":
        raise ValueError(f"polynomial_mat must be a .mat file: {polynomial_mat}")
    _validate_existing_file(polynomial_mat, "polynomial coefficients MAT")
    _validate_existing_file(target_csv, "target club CSV")
    return ReplayInputs(
        polynomial_mat=polynomial_mat,
        target_csv=target_csv,
        scenario=_validate_scenario(scenario),
        start_state_mat=_validate_existing_file(
            Path(start_state_mat) if start_state_mat else None, "start-state MAT"
        ),
        torque_csv=_validate_existing_file(
            Path(torque_csv) if torque_csv else None, "torque CSV"
        ),
        existing_sim_csv=_validate_existing_file(
            Path(existing_sim_csv) if existing_sim_csv else None,
            "existing simulated CSV",
        ),
        joint_velocity_csv=_validate_existing_file(
            Path(joint_velocity_csv) if joint_velocity_csv else None,
            "joint velocity CSV",
        ),
    )


def _matlab_engine_available() -> bool:
    """Return ``True`` when ``matlab.engine`` can be imported."""
    try:
        spec = importlib.util.find_spec("matlab.engine")
    except (ImportError, ValueError):
        return False
    return spec is not None


def _git_commit() -> str | None:
    git = shutil.which("git")
    if git is None:
        return None
    try:
        out = subprocess.run(
            [git, "rev-parse", "HEAD"],
            cwd=str(_HERE),
            capture_output=True,
            text=True,
            timeout=5,
            check=False,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    if out.returncode != 0:
        return None
    commit = out.stdout.strip()
    return commit or None


def _make_run_dir(
    output_root: Path, scenario: str, timestamp: datetime | None = None
) -> Path:
    when = timestamp or datetime.now(UTC)  # noqa: UP017 - mypy stub lacks datetime.UTC
    label = when.strftime("%Y%m%dT%H%M%SZ")
    run_dir = output_root / scenario / label
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _run_matlab_driver(
    inputs: ReplayInputs,
    run_dir: Path,
    matlab_driver: Path,
    sim_csv: Path,
    joint_velocity_csv: Path,
    engine_module: Any,
) -> None:
    """Invoke the MATLAB driver via ``matlab.engine``.

    Raises ``ReplayError`` if the MATLAB call fails.
    """
    if not matlab_driver.exists():
        raise ReplayError(f"MATLAB driver not found: {matlab_driver}")
    eng = engine_module.start_matlab()
    try:
        eng.addpath(str(matlab_driver.parent), nargout=0)
        eng.addpath(str(_HERE / "matlab"), nargout=0)
        try:
            eng.run_replay(
                str(inputs.polynomial_mat),
                inputs.scenario,
                str(inputs.start_state_mat) if inputs.start_state_mat else "",
                str(sim_csv),
                str(joint_velocity_csv),
                nargout=0,
            )
        except Exception as exc:  # noqa: BLE001 - any MATLAB error must surface
            raise ReplayError(
                f"MATLAB replay failed for scenario {inputs.scenario!r}: {exc}"
            ) from exc
    finally:
        try:
            eng.quit()
        except Exception:  # noqa: BLE001 - best effort shutdown
            LOGGER.debug("matlab.engine quit raised; ignoring", exc_info=True)


def _load_diagnostics_module() -> Any:
    """Import ``evaluate_matching_workflow`` co-located with this file."""
    script = _HERE / "evaluate_matching_workflow.py"
    spec = importlib.util.spec_from_file_location("evaluate_matching_workflow", script)
    if spec is None or spec.loader is None:
        raise ReplayError(f"Cannot load diagnostics module at {script}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def _write_manifest(
    manifest_path: Path,
    inputs: ReplayInputs,
    run_dir: Path,
    sim_csv: Path | None,
    metrics_path: Path | None,
    matlab_used: bool,
    skipped_reason: str | None,
    timestamp: datetime,
) -> None:
    payload = {
        "scenario": inputs.scenario,
        "timestamp_utc": timestamp.strftime("%Y-%m-%dT%H:%M:%SZ"),
        "git_commit": _git_commit(),
        "matlab_used": matlab_used,
        "skipped_reason": skipped_reason,
        "inputs": {
            "polynomial_mat": str(inputs.polynomial_mat),
            "target_csv": str(inputs.target_csv),
            "start_state_mat": (
                str(inputs.start_state_mat) if inputs.start_state_mat else None
            ),
            "torque_csv": str(inputs.torque_csv) if inputs.torque_csv else None,
            "existing_sim_csv": (
                str(inputs.existing_sim_csv) if inputs.existing_sim_csv else None
            ),
            "joint_velocity_csv": (
                str(inputs.joint_velocity_csv) if inputs.joint_velocity_csv else None
            ),
        },
        "outputs": {
            "run_dir": str(run_dir),
            "sim_csv": str(sim_csv) if sim_csv else None,
            "metrics_json": str(metrics_path) if metrics_path else None,
        },
    }
    manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def replay(
    polynomial_mat: Path,
    target_csv: Path,
    scenario: str,
    output_root: Path = DEFAULT_REPORTS_DIR,
    start_state_mat: Path | None = None,
    torque_csv: Path | None = None,
    existing_sim_csv: Path | None = None,
    joint_velocity_csv: Path | None = None,
    impact_time: float | None = None,
    impact_window_s: float = 0.02,
    effort_weight: float = 1.0e-8,
    smoothness_weight: float = 1.0e-10,
    run_label: str | None = None,
    matlab_driver: Path = DEFAULT_MATLAB_DRIVER,
    matlab_engine_module: Any | None = None,
    timestamp: datetime | None = None,
) -> ReplayResult:
    """Run the closed-loop replay end-to-end and return the report bundle."""
    inputs = build_inputs(
        polynomial_mat=polynomial_mat,
        target_csv=target_csv,
        scenario=scenario,
        start_state_mat=start_state_mat,
        torque_csv=torque_csv,
        existing_sim_csv=existing_sim_csv,
        joint_velocity_csv=joint_velocity_csv,
    )
    when = timestamp or datetime.now(UTC)  # noqa: UP017 - mypy stub lacks datetime.UTC
    run_dir = _make_run_dir(Path(output_root), inputs.scenario, when)
    sim_csv = run_dir / SIM_CSV_NAME
    joint_csv = run_dir / JOINT_VELOCITY_CSV_NAME

    matlab_used = False
    skipped_reason: str | None = None

    engine_module = matlab_engine_module
    if engine_module is None and _matlab_engine_available():
        try:
            engine_module = importlib.import_module("matlab.engine")
        except ImportError as exc:
            LOGGER.info("matlab.engine import failed: %s", exc)
            engine_module = None

    if engine_module is not None:
        _run_matlab_driver(
            inputs=inputs,
            run_dir=run_dir,
            matlab_driver=Path(matlab_driver),
            sim_csv=sim_csv,
            joint_velocity_csv=joint_csv,
            engine_module=engine_module,
        )
        matlab_used = True
    else:
        skipped_reason = "matlab.engine unavailable; using existing simulated CSV"
        if inputs.existing_sim_csv is not None:
            shutil.copy2(inputs.existing_sim_csv, sim_csv)
        else:
            sim_csv = sim_csv if sim_csv.exists() else None  # type: ignore[assignment]
            LOGGER.warning(
                "MATLAB unavailable and no existing simulated CSV provided; "
                "diagnostics will skip the matching step."
            )

    if inputs.joint_velocity_csv is not None and not joint_csv.exists():
        shutil.copy2(inputs.joint_velocity_csv, joint_csv)

    sim_csv_resolved: Path | None = sim_csv if (sim_csv and sim_csv.exists()) else None
    joint_csv_resolved: Path | None = joint_csv if joint_csv.exists() else None

    diagnostics = _load_diagnostics_module()
    label = run_label or run_dir.name
    report = diagnostics.evaluate(
        target_csv=inputs.target_csv,
        sim_csv=sim_csv_resolved,
        torque_csv=inputs.torque_csv,
        output_dir=run_dir,
        scenario=inputs.scenario,
        run_label=label,
        impact_time=impact_time,
        impact_window_s=impact_window_s,
        effort_weight=effort_weight,
        smoothness_weight=smoothness_weight,
        joint_velocity_csv=joint_csv_resolved,
    )
    metrics_path = run_dir / f"{label}_matching_metrics.json"
    if not metrics_path.exists():
        metrics_path = None  # type: ignore[assignment]

    manifest_path = run_dir / "run_manifest.json"
    _write_manifest(
        manifest_path=manifest_path,
        inputs=inputs,
        run_dir=run_dir,
        sim_csv=sim_csv_resolved,
        metrics_path=metrics_path,
        matlab_used=matlab_used,
        skipped_reason=skipped_reason,
        timestamp=when,
    )

    return ReplayResult(
        run_dir=run_dir,
        manifest_path=manifest_path,
        metrics_path=metrics_path,
        sim_csv=sim_csv_resolved,
        matlab_used=matlab_used,
        skipped_reason=skipped_reason,
        report=report,
    )


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--polynomial-mat", type=Path, required=True)
    parser.add_argument("--target-csv", type=Path, required=True)
    parser.add_argument("--scenario", choices=SUPPORTED_SCENARIOS, default="full-swing")
    parser.add_argument("--start-state-mat", type=Path)
    parser.add_argument("--torque-csv", type=Path)
    parser.add_argument("--existing-sim-csv", type=Path)
    parser.add_argument("--joint-velocity-csv", type=Path)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_REPORTS_DIR)
    parser.add_argument("--impact-time", type=float)
    parser.add_argument("--impact-window-s", type=float, default=0.02)
    parser.add_argument("--effort-weight", type=float, default=1.0e-8)
    parser.add_argument("--smoothness-weight", type=float, default=1.0e-10)
    parser.add_argument("--run-label")
    parser.add_argument("--matlab-driver", type=Path, default=DEFAULT_MATLAB_DRIVER)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(
        level=os.environ.get("REPLAY_LOG_LEVEL", "INFO"),
        format="%(message)s",
    )
    args = parse_args(argv)
    result = replay(
        polynomial_mat=args.polynomial_mat,
        target_csv=args.target_csv,
        scenario=args.scenario,
        output_root=args.output_root,
        start_state_mat=args.start_state_mat,
        torque_csv=args.torque_csv,
        existing_sim_csv=args.existing_sim_csv,
        joint_velocity_csv=args.joint_velocity_csv,
        impact_time=args.impact_time,
        impact_window_s=args.impact_window_s,
        effort_weight=args.effort_weight,
        smoothness_weight=args.smoothness_weight,
        run_label=args.run_label,
        matlab_driver=args.matlab_driver,
    )
    LOGGER.info("Replay run directory: %s", result.run_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

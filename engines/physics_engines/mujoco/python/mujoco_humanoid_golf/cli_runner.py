"""Headless CLI runner for batch golf swing simulations."""

from __future__ import annotations

import argparse
import csv
import json
from collections.abc import Iterable, Mapping, MutableMapping, Sequence
from pathlib import Path
from typing import Any

import mujoco
import numpy as np

from .biomechanics import BiomechanicalAnalyzer, SwingRecorder
from .control_system import ControlSystem, ControlType
from .models import (
    ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    CHAOTIC_PENDULUM_XML,
    DOUBLE_PENDULUM_XML,
    FULL_BODY_GOLF_SWING_XML,
    MYOARM_SIMPLE_PATH,
    MYOBODY_PATH,
    MYOUPPERBODY_PATH,
    TRIPLE_PENDULUM_XML,
    UPPER_BODY_GOLF_SWING_XML,
)

MODEL_SPECS: Mapping[str, Mapping[str, str]] = {
    "chaotic_pendulum": {"mode": "xml_string", "value": CHAOTIC_PENDULUM_XML},
    "double_pendulum": {"mode": "xml_string", "value": DOUBLE_PENDULUM_XML},
    "triple_pendulum": {"mode": "xml_string", "value": TRIPLE_PENDULUM_XML},
    "upper_body": {"mode": "xml_string", "value": UPPER_BODY_GOLF_SWING_XML},
    "full_body": {"mode": "xml_string", "value": FULL_BODY_GOLF_SWING_XML},
    "advanced_biomech": {
        "mode": "xml_string",
        "value": ADVANCED_BIOMECHANICAL_GOLF_SWING_XML,
    },
    "myoupperbody": {"mode": "xml_path", "value": MYOUPPERBODY_PATH},
    "myobody": {"mode": "xml_path", "value": MYOBODY_PATH},
    "myoarm_simple": {"mode": "xml_path", "value": MYOARM_SIMPLE_PATH},
}


def _resolve_model_path(xml_path: str) -> Path:
    """Resolve repository-relative MJCF paths."""
    path = Path(xml_path)
    if path.is_absolute():
        return path
    return Path(__file__).parent.parent.parent / path


def load_model(model_key: str) -> tuple[mujoco.MjModel, mujoco.MjData]:
    """Load a MuJoCo model from the catalog or filesystem."""
    spec = MODEL_SPECS.get(model_key)
    if spec is None:
        candidate = _resolve_model_path(model_key)
        if not candidate.exists():
            raise ValueError(
                f"Unknown model '{model_key}'. Available: "
                f"{', '.join(sorted(MODEL_SPECS))}",
            )
        model = mujoco.MjModel.from_xml_path(candidate.as_posix())
    elif spec["mode"] == "xml_string":
        model = mujoco.MjModel.from_xml_string(spec["value"])
    else:
        model = mujoco.MjModel.from_xml_path(
            _resolve_model_path(spec["value"]).as_posix(),
        )

    data = mujoco.MjData(model)
    return model, data


def apply_control_preset(
    control_system: ControlSystem,
    preset: Mapping[str, Any],
) -> None:
    """Apply control parameters loaded from a JSON structure."""
    for entry in preset.get("actuators", []):
        idx = int(entry["index"])
        ctrl_type = entry.get("type", "constant").lower()
        if ctrl_type == "constant":
            control_system.set_control_type(idx, ControlType.CONSTANT)
            control_system.set_constant_value(idx, float(entry.get("value", 0.0)))
        elif ctrl_type == "polynomial":
            coeffs = np.array(entry.get("coefficients", [0.0] * 7), dtype=np.float64)
            if coeffs.shape[0] != 7:
                raise ValueError(
                    "Polynomial coefficients must contain exactly 7 values",
                )
            control_system.set_polynomial_coeffs(idx, coeffs)
            control_system.set_control_type(idx, ControlType.POLYNOMIAL)
        elif ctrl_type == "sine":
            control_system.set_sine_wave_params(
                idx,
                amplitude=float(entry.get("amplitude", 0.0)),
                frequency=float(entry.get("frequency", 1.0)),
                phase=float(entry.get("phase", 0.0)),
            )
            control_system.set_control_type(idx, ControlType.SINE_WAVE)
        elif ctrl_type == "step":
            control_system.set_step_params(
                idx,
                step_time=float(entry.get("step_time", 0.0)),
                step_value=float(entry.get("step_value", 0.0)),
            )
            control_system.set_control_type(idx, ControlType.STEP)
        else:
            raise ValueError(f"Unsupported control type '{ctrl_type}' in preset file")


def run_simulation(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    duration_s: float,
    control_system: ControlSystem,
) -> SwingRecorder:
    """Simulate the provided model for the requested duration."""
    analyzer = BiomechanicalAnalyzer(model, data)
    recorder = SwingRecorder()
    recorder.start_recording()

    steps = max(1, int(duration_s / model.opt.timestep))
    for _ in range(steps):
        control_system.update_time(data.time)
        velocities = data.qvel[: model.nu] if model.nu <= len(data.qvel) else None
        data.ctrl[:] = control_system.compute_control_vector(velocities)
        mujoco.mj_step(model, data)
        recorder.record_frame(analyzer.extract_full_state())

    recorder.stop_recording()
    return recorder


def summarize_run(recorder: SwingRecorder) -> MutableMapping[str, float]:
    """Return high-level metrics required by optimization workflows."""
    summary: MutableMapping[str, float] = {}
    times, speeds = recorder.get_time_series("club_head_speed")
    if len(speeds) > 0:
        idx = int(np.argmax(speeds))
        summary["max_club_speed_mps"] = float(speeds[idx])
        summary["max_club_speed_mph"] = float(speeds[idx] * 2.23694)
        summary["time_of_max_speed_s"] = float(times[idx])
        summary["samples"] = float(len(speeds))
        summary["recorded_duration_s"] = float(times[-1] - times[0])

    _, total_energy = recorder.get_time_series("total_energy")
    if len(total_energy) > 0:
        summary["peak_total_energy_j"] = float(np.max(total_energy))

    return summary


def export_json(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist telemetry to a JSON file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def export_csv(path: Path, payload: Mapping[str, Any]) -> None:
    """Persist telemetry to a CSV file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    keys = [key for key, value in payload.items() if isinstance(value, Sequence)]
    max_len = max((len(payload[key]) for key in keys), default=0)

    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(keys)
        for row_idx in range(max_len):
            row = []
            for key in keys:
                values = payload[key]
                row.append(values[row_idx] if row_idx < len(values) else "")
            writer.writerow(row)


def execute_run(
    *,
    model: str,
    duration: float,
    timestep: float | None,
    control_config: Path | None,
    output_json: Path | None,
    output_csv: Path | None,
    show_summary: bool,
) -> MutableMapping[str, float] | None:
    """Execute a single run and optionally emit telemetry."""
    model_obj, data = load_model(model)
    if timestep is not None:
        model_obj.opt.timestep = timestep

    control_system = ControlSystem(model_obj.nu)
    if control_config:
        preset_payload = json.loads(control_config.read_text(encoding="utf-8"))
        apply_control_preset(control_system, preset_payload)

    recorder = run_simulation(
        model_obj,
        data,
        duration_s=duration,
        control_system=control_system,
    )

    export_payload = recorder.export_to_dict()
    if output_json:
        export_json(output_json, export_payload)
    if output_csv:
        export_csv(output_csv, export_payload)

    if show_summary:
        summary = summarize_run(recorder)
        return summary

    return None


def run_batch(batch_path: Path, base_args: argparse.Namespace) -> None:
    """Execute every entry described in a batch configuration file."""
    spec = json.loads(batch_path.read_text(encoding="utf-8"))
    runs: Iterable[Mapping[str, Any]]
    if isinstance(spec, Mapping) and "runs" in spec:
        runs = spec["runs"]
    elif isinstance(spec, list):
        runs = spec
    else:
        raise ValueError("Batch file must be a list or contain a 'runs' array")

    for entry in runs:
        name = entry.get("name", entry.get("model", "unnamed_run"))
        summary = execute_run(
            model=entry["model"],
            duration=float(entry.get("duration", base_args.duration)),
            timestep=entry.get("timestep", base_args.timestep),
            control_config=(
                Path(entry["control_config"]) if "control_config" in entry else None
            ),
            output_json=Path(entry["output_json"]) if "output_json" in entry else None,
            output_csv=Path(entry["output_csv"]) if "output_csv" in entry else None,
            show_summary=entry.get("summary", base_args.summary),
        )
        if summary is not None:
            # We use direct print here as this is a CLI tool explicitly asked for
            # summary output
            print(f"[{name}] Summary:")  # noqa: T201
            for key, value in summary.items():
                print(f"  {key}: {value:.6g}")  # noqa: T201


def build_parser() -> argparse.ArgumentParser:
    """Create the CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="python -m python.mujoco_humanoid_golf.cli_runner",
        description="Run MuJoCo golf swing simulations without the GUI.",
    )
    parser.add_argument("--model", help="Model key or MJCF path", required=False)
    parser.add_argument(
        "--duration",
        type=float,
        default=2.0,
        help="Simulation duration in seconds",
    )
    parser.add_argument(
        "--timestep",
        type=float,
        help="Override integrator timestep [s]",
    )
    parser.add_argument(
        "--control-config",
        type=Path,
        help="JSON file describing actuator controls",
    )
    parser.add_argument("--output-json", type=Path, help="Path to save telemetry JSON")
    parser.add_argument("--output-csv", type=Path, help="Path to save telemetry CSV")
    parser.add_argument(
        "--batch-config",
        type=Path,
        help="JSON file describing multiple runs",
    )
    parser.add_argument(
        "--summary",
        action="store_true",
        help="Print summary metrics (max club speed, energy, duration)",
    )
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point for the CLI module."""
    parser = build_parser()
    args = parser.parse_args(argv)

    if args.batch_config:
        run_batch(args.batch_config, args)
        return 0

    if not args.model:
        parser.error("--model is required when --batch-config is not provided")

    summary = execute_run(
        model=args.model,
        duration=args.duration,
        timestep=args.timestep,
        control_config=args.control_config,
        output_json=args.output_json,
        output_csv=args.output_csv,
        show_summary=args.summary,
    )

    if summary is not None:
        # We use direct print here as this is a CLI tool explicitly asked for
        # summary output
        print("Summary:")  # noqa: T201
        for key, value in summary.items():
            print(f"  {key}: {value:.6g}")  # noqa: T201

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

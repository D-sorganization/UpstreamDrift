"""Command-line entry point for cross-engine comparison reports."""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

from .comparison import (
    ComparisonInput,
    compare,
    render_markdown_report,
    write_report,
)
from .factory import make_backend
from .model_params import GolfModelParams
from .protocol import SimState

_DEFAULT_Q = (1.2, -0.6)


def build_parser() -> argparse.ArgumentParser:
    """Build the CC-27 comparison CLI parser."""
    parser = argparse.ArgumentParser(
        prog="python -m src.shared.python.simulation_backends.compare_cli",
        description="Run selected simulation backends and write a comparison report.",
    )
    parser.add_argument(
        "--engines",
        required=True,
        help="Comma-separated backend names, for example: ode,mujoco.",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=100,
        help="Number of integration steps to run.",
    )
    parser.add_argument(
        "--dt",
        type=float,
        default=0.005,
        help="Integration step size in seconds.",
    )
    parser.add_argument(
        "--controls",
        type=Path,
        help="Optional JSON or CSV control history. A single column is repeated.",
    )
    parser.add_argument(
        "--control-dim",
        type=int,
        default=2,
        help="Control dimension used when repeating one-column input.",
    )
    parser.add_argument(
        "--initial-q",
        default=",".join(str(value) for value in _DEFAULT_Q),
        help="Comma-separated initial generalized positions.",
    )
    parser.add_argument(
        "--initial-v",
        default=None,
        help="Comma-separated initial generalized velocities. Defaults to zeros.",
    )
    parser.add_argument(
        "--format",
        choices=("markdown", "json"),
        default="markdown",
        help="Output report format.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        help="Report path. Omit to write the report to stdout.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    """Run the comparison CLI."""
    parser = build_parser()
    args = parser.parse_args(argv)
    engine_names = _parse_engine_names(args.engines)
    params = GolfModelParams.default()
    engines = [make_backend(name, params) for name in engine_names]
    controls = _load_controls(args.controls, args.horizon, args.control_dim)
    initial_state = _initial_state(args.initial_q, args.initial_v)
    report = compare(
        engines,
        ComparisonInput(
            horizon=args.horizon,
            dt=args.dt,
            controls=controls,
            initial_state=initial_state,
        ),
        labels=engine_names,
    )
    if args.output is not None:
        write_report(report, args.output, format=args.format)
        sys.stdout.write(f"report written: {args.output}\n")
        return 0
    if args.format == "json":
        sys.stdout.write(json.dumps(report.to_dict(), indent=2, sort_keys=True))
        sys.stdout.write("\n")
    else:
        sys.stdout.write(render_markdown_report(report))
    return 0


def _parse_engine_names(raw: str) -> tuple[str, ...]:
    names = tuple(name.strip() for name in raw.split(",") if name.strip())
    if len(names) < 2:
        raise ValueError("--engines must name at least two backends")
    return names


def _initial_state(raw_q: str, raw_v: str | None) -> SimState:
    q = np.array(_parse_float_list(raw_q), dtype=float)
    if raw_v is None:
        v = np.zeros_like(q)
    else:
        v = np.array(_parse_float_list(raw_v), dtype=float)
    return SimState(q=q, v=v)


def _parse_float_list(raw: str) -> list[float]:
    values = [item.strip() for item in raw.split(",") if item.strip()]
    if not values:
        raise ValueError("numeric list must be non-empty")
    return [float(value) for value in values]


def _load_controls(
    path: Path | None,
    horizon: int,
    control_dim: int,
) -> np.ndarray | None:
    if path is None:
        return None
    if control_dim <= 0:
        raise ValueError("--control-dim must be positive")
    if path.suffix.lower() == ".json":
        with path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        data = np.asarray(payload, dtype=float)
    else:
        data = _read_csv_controls(path)
    controls = _coerce_controls(data, control_dim)
    if controls.shape[0] != horizon:
        raise ValueError(
            f"controls have {controls.shape[0]} rows, expected horizon {horizon}"
        )
    return controls


def _read_csv_controls(path: Path) -> np.ndarray:
    rows: list[list[float]] = []
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.reader(handle)
        for row in reader:
            if not row:
                continue
            rows.append([float(item) for item in row])
    return np.asarray(rows, dtype=float)


def _coerce_controls(data: np.ndarray, control_dim: int) -> np.ndarray:
    arr = np.asarray(data, dtype=float)
    if arr.ndim == 1:
        arr = arr.reshape(-1, 1)
    if arr.ndim != 2:
        raise ValueError(f"controls must be 1-D or 2-D, got {arr.ndim}-D")
    if arr.shape[1] == 1 and control_dim > 1:
        arr = np.repeat(arr, control_dim, axis=1)
    return arr


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

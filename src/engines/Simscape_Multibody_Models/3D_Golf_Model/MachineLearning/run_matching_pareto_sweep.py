"""Run a Pareto sweep for Golf ML club-matching torque optimization."""

import argparse
import csv
import json
import logging
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_OUTPUT_DIR = SCRIPT_DIR / "data" / "processed" / "pareto_sweep"


@dataclass(frozen=True)
class SweepRun:
    label: str
    effort_weight: float
    smoothness_weight: float
    torque_csv: Path
    summary_json: Path
    best_loss: float | None
    tracking_loss: float | None
    optimizer_effort_loss: float | None
    optimizer_smoothness_loss: float | None
    l2_torque_effort: float | None
    smoothness_l2: float | None
    objective: float | None


def parse_positive_float_grid(value: str, label: str) -> list[float]:
    """Parse a non-empty comma-separated grid of positive finite floats."""
    values: list[float] = []
    for raw_item in value.split(","):
        item = raw_item.strip()
        if not item:
            continue
        try:
            parsed = float(item)
        except ValueError as exc:
            raise ValueError(f"{label} contains a non-float value: {item!r}") from exc
        if not math.isfinite(parsed) or parsed <= 0.0:
            raise ValueError(f"{label} values must be positive finite floats")
        values.append(parsed)
    if not values:
        raise ValueError(f"{label} must contain at least one positive float")
    return values


def _default_device() -> str:
    try:
        import torch
    except ImportError:
        return "cpu"
    return "cuda" if torch.cuda.is_available() else "cpu"


def _load_optimizer() -> Any:
    if str(SCRIPT_DIR) not in sys.path:
        sys.path.insert(0, str(SCRIPT_DIR))
    from optimize_torque_sequence_for_club import optimize_sequence as optimizer

    return optimizer


def optimize_sequence(
    *,
    checkpoint_path: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    output_csv: Path,
    steps: int,
    learning_rate: float,
    effort_weight: float,
    smoothness_weight: float,
    device_name: str,
) -> Any:
    """Monkeypatchable adapter around the existing torque-sequence optimizer."""
    optimizer = _load_optimizer()
    return optimizer(
        checkpoint_path=checkpoint_path,
        desired_club_csv=desired_club_csv,
        reference_body_csv=reference_body_csv,
        output_csv=output_csv,
        steps=steps,
        learning_rate=learning_rate,
        effort_weight=effort_weight,
        smoothness_weight=smoothness_weight,
        device_name=device_name,
    )


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _latest_history(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return {}
    latest = history[-1]
    return latest if isinstance(latest, dict) else {}


def _best_history(summary: dict[str, Any]) -> dict[str, Any]:
    history = summary.get("history")
    if not isinstance(history, list) or not history:
        return {}
    candidates = [item for item in history if isinstance(item, dict)]
    if not candidates:
        return {}
    return min(candidates, key=lambda item: float(item.get("loss", math.inf)))


def _float_or_none(value: Any) -> float | None:
    if value is None:
        return None
    try:
        parsed = float(value)
    except (TypeError, ValueError):
        return None
    return parsed if math.isfinite(parsed) else None


def _time_values(rows: list[dict[str, str]]) -> list[float]:
    if not rows:
        return []
    values: list[float] = []
    for index, row in enumerate(rows):
        value = _float_or_none(row.get("time"))
        values.append(float(index) if value is None else value)
    return values


def _time_steps(time: list[float]) -> list[float]:
    if len(time) <= 1:
        return [1.0] * max(len(time), 1)
    diffs = [abs(time[index + 1] - time[index]) for index in range(len(time) - 1)]
    finite = [value for value in diffs if math.isfinite(value) and value > 1.0e-12]
    fallback = sorted(finite)[len(finite) // 2] if finite else 1.0
    return [*diffs, fallback]


def torque_effort_metrics(torque_csv: Path) -> dict[str, Any]:
    with torque_csv.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        return {"available": False, "reason": "torque CSV is empty"}

    numeric_columns: list[str] = []
    for column in rows[0]:
        if column == "time":
            continue
        if all(_float_or_none(row.get(column)) is not None for row in rows):
            numeric_columns.append(column)
    if not numeric_columns:
        return {"available": False, "reason": "no numeric torque columns found"}

    time = _time_values(rows)
    dt = _time_steps(time)
    values = [
        [
            value if (value := _float_or_none(row[column])) is not None else 0.0
            for column in numeric_columns
        ]
        for row in rows
    ]
    l2_effort = sum(
        sum(value * value for value in row) * dt[index]
        for index, row in enumerate(values)
    )
    l1_impulse = sum(
        sum(abs(value) for value in row) * dt[index] for index, row in enumerate(values)
    )
    peak_abs = max(abs(value) for row in values for value in row)
    smoothness = 0.0
    if len(values) > 1:
        step = max(sorted(dt)[len(dt) // 2], 1.0e-12)
        for previous, current in zip(values, values[1:], strict=False):
            smoothness += (
                sum(
                    ((right - left) / step) ** 2
                    for left, right in zip(previous, current, strict=True)
                )
                * step
            )

    return {
        "available": True,
        "columns": numeric_columns,
        "rows": len(rows),
        "l2_torque_effort": l2_effort,
        "l1_torque_impulse": l1_impulse,
        "peak_abs_torque": peak_abs,
        "mean_abs_torque": l1_impulse / max(sum(dt) * len(numeric_columns), 1.0e-12),
        "smoothness_l2": smoothness,
    }


def _run_label(effort_weight: float, smoothness_weight: float) -> str:
    return f"effort_{effort_weight:.6g}_smooth_{smoothness_weight:.6g}".replace("+", "")


def run_single(
    *,
    checkpoint: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    output_dir: Path,
    effort_weight: float,
    smoothness_weight: float,
    steps: int,
    learning_rate: float,
    scenario: str,
    device: str,
) -> SweepRun:
    label = _run_label(effort_weight, smoothness_weight)
    torque_csv = output_dir / f"{label}_torques.csv"
    summary_json = torque_csv.with_suffix(".summary.json")
    optimize_result = optimize_sequence(
        checkpoint_path=checkpoint,
        desired_club_csv=desired_club_csv,
        reference_body_csv=reference_body_csv,
        output_csv=torque_csv,
        steps=steps,
        learning_rate=learning_rate,
        effort_weight=effort_weight,
        smoothness_weight=smoothness_weight,
        device_name=device,
    )
    summary = _read_json(summary_json)
    if isinstance(optimize_result, dict):
        summary.update(optimize_result)
    effort = torque_effort_metrics(torque_csv)
    best = _best_history(summary) or _latest_history(summary)
    tracking_loss = _float_or_none(best.get("tracking_loss"))
    optimizer_effort = _float_or_none(best.get("effort_loss"))
    optimizer_smoothness = _float_or_none(best.get("smoothness_loss"))
    l2_effort = _float_or_none(effort.get("l2_torque_effort"))
    smoothness_l2 = _float_or_none(effort.get("smoothness_l2"))
    objective = _objective(
        tracking_loss=tracking_loss,
        l2_effort=l2_effort,
        smoothness_l2=smoothness_l2,
        effort_weight=effort_weight,
        smoothness_weight=smoothness_weight,
        fallback=_float_or_none(summary.get("best_loss")),
    )
    summary.update(
        {
            "scenario": scenario,
            "effort_weight": effort_weight,
            "smoothness_weight": smoothness_weight,
            "run_label": label,
            "torque_effort": effort,
            "pareto_objective": objective,
        }
    )
    _write_json(summary_json, summary)
    return SweepRun(
        label=label,
        effort_weight=effort_weight,
        smoothness_weight=smoothness_weight,
        torque_csv=torque_csv,
        summary_json=summary_json,
        best_loss=_float_or_none(summary.get("best_loss")),
        tracking_loss=tracking_loss,
        optimizer_effort_loss=optimizer_effort,
        optimizer_smoothness_loss=optimizer_smoothness,
        l2_torque_effort=l2_effort,
        smoothness_l2=smoothness_l2,
        objective=objective,
    )


def _objective(
    *,
    tracking_loss: float | None,
    l2_effort: float | None,
    smoothness_l2: float | None,
    effort_weight: float,
    smoothness_weight: float,
    fallback: float | None,
) -> float | None:
    if tracking_loss is None:
        return fallback
    objective = tracking_loss
    if l2_effort is not None:
        objective += effort_weight * l2_effort
    if smoothness_l2 is not None:
        objective += smoothness_weight * smoothness_l2
    return objective


def _metric(run: SweepRun, name: str) -> float | None:
    if name == "error":
        return run.tracking_loss if run.tracking_loss is not None else run.best_loss
    if name == "effort":
        if run.l2_torque_effort is not None:
            return run.l2_torque_effort
        return run.optimizer_effort_loss
    if name == "smoothness":
        if run.smoothness_l2 is not None:
            return run.smoothness_l2
        return run.optimizer_smoothness_loss
    return None


def select_candidates(runs: list[SweepRun]) -> dict[str, SweepRun]:
    if not runs:
        raise ValueError("Cannot select candidates from an empty sweep")
    best_error = min(runs, key=lambda run: _metric(run, "error") or math.inf)
    best_effort = min(runs, key=lambda run: _metric(run, "effort") or math.inf)
    return {
        "best_low_error": best_error,
        "best_low_effort": best_effort,
        "knee_point": _knee_point(runs),
    }


def _knee_point(runs: list[SweepRun]) -> SweepRun:
    error_values = [_metric(run, "error") for run in runs]
    effort_values = [_metric(run, "effort") for run in runs]
    finite_error = [value for value in error_values if value is not None]
    finite_effort = [value for value in effort_values if value is not None]
    if not finite_error or not finite_effort:
        return min(runs, key=lambda run: run.objective or math.inf)
    min_error, max_error = min(finite_error), max(finite_error)
    min_effort, max_effort = min(finite_effort), max(finite_effort)
    error_span = max(max_error - min_error, 1.0e-12)
    effort_span = max(max_effort - min_effort, 1.0e-12)

    def score(run: SweepRun) -> float:
        error = _metric(run, "error")
        effort = _metric(run, "effort")
        if error is None or effort is None:
            return math.inf
        return math.hypot(
            (error - min_error) / error_span,
            (effort - min_effort) / effort_span,
        )

    return min(runs, key=score)


def write_pareto_summary(output_dir: Path, runs: list[SweepRun]) -> dict[str, SweepRun]:
    candidates = select_candidates(runs)
    csv_path = output_dir / "pareto_summary.csv"
    fieldnames = [
        "run_label",
        "effort_weight",
        "smoothness_weight",
        "best_loss",
        "tracking_loss",
        "optimizer_effort_loss",
        "optimizer_smoothness_loss",
        "l2_torque_effort",
        "smoothness_l2",
        "objective",
        "torque_csv",
        "summary_json",
        "candidate_roles",
    ]
    role_by_label: dict[str, list[str]] = {}
    for role, run in candidates.items():
        role_by_label.setdefault(run.label, []).append(role)
    with csv_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for run in runs:
            writer.writerow(
                {
                    "run_label": run.label,
                    "effort_weight": run.effort_weight,
                    "smoothness_weight": run.smoothness_weight,
                    "best_loss": run.best_loss,
                    "tracking_loss": run.tracking_loss,
                    "optimizer_effort_loss": run.optimizer_effort_loss,
                    "optimizer_smoothness_loss": run.optimizer_smoothness_loss,
                    "l2_torque_effort": run.l2_torque_effort,
                    "smoothness_l2": run.smoothness_l2,
                    "objective": run.objective,
                    "torque_csv": str(run.torque_csv),
                    "summary_json": str(run.summary_json),
                    "candidate_roles": ";".join(role_by_label.get(run.label, [])),
                }
            )
    _write_markdown(output_dir / "pareto_summary.md", runs, candidates)
    _write_plot(output_dir / "pareto_front.png", runs, candidates)
    return candidates


def _format_metric(value: float | None) -> str:
    return "n/a" if value is None else f"{value:.8g}"


def _write_markdown(
    path: Path, runs: list[SweepRun], candidates: dict[str, SweepRun]
) -> None:
    lines = [
        "# Golf ML Pareto Sweep",
        "",
        "## Selected Candidates",
    ]
    for role, run in candidates.items():
        lines.append(
            "- "
            f"{role}: `{run.label}` "
            f"(tracking `{_format_metric(_metric(run, 'error'))}`, "
            f"effort `{_format_metric(_metric(run, 'effort'))}`, "
            f"smoothness `{_format_metric(_metric(run, 'smoothness'))}`)"
        )
    lines.extend(
        [
            "",
            "## Runs",
            "",
            (
                "| run | effort weight | smoothness weight | tracking | effort | "
                "smoothness | objective |"
            ),
            "| --- | ---: | ---: | ---: | ---: | ---: | ---: |",
        ]
    )
    for run in runs:
        lines.append(
            f"| `{run.label}` | {run.effort_weight:.8g} | "
            f"{run.smoothness_weight:.8g} | "
            f"{_format_metric(_metric(run, 'error'))} | "
            f"{_format_metric(_metric(run, 'effort'))} | "
            f"{_format_metric(_metric(run, 'smoothness'))} | "
            f"{_format_metric(run.objective)} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_plot(
    path: Path, runs: list[SweepRun], candidates: dict[str, SweepRun]
) -> None:
    points: list[tuple[SweepRun, float, float]] = []
    for run in runs:
        effort = _metric(run, "effort")
        error = _metric(run, "error")
        if effort is not None and error is not None:
            points.append((run, effort, error))
    if not points:
        return
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    fig, axis = plt.subplots(figsize=(8, 5))
    axis.scatter([effort for _, effort, _ in points], [error for _, _, error in points])
    for role, run in candidates.items():
        effort = _metric(run, "effort")
        error = _metric(run, "error")
        if effort is not None and error is not None:
            axis.annotate(role, (effort, error))
    axis.set_xlabel("L2 torque effort")
    axis.set_ylabel("tracking loss")
    axis.set_title("Golf ML matching Pareto sweep")
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def run_sweep(
    *,
    checkpoint: Path,
    desired_club_csv: Path,
    reference_body_csv: Path,
    output_dir: Path,
    effort_weights: list[float],
    smoothness_weights: list[float],
    steps: int,
    learning_rate: float,
    scenario: str,
    device: str,
) -> list[SweepRun]:
    if steps <= 0:
        raise ValueError("steps must be positive")
    if learning_rate <= 0.0 or not math.isfinite(learning_rate):
        raise ValueError("learning-rate must be a positive finite float")
    output_dir.mkdir(parents=True, exist_ok=True)
    runs: list[SweepRun] = []
    for effort_weight in effort_weights:
        for smoothness_weight in smoothness_weights:
            LOGGER.info(
                "Running Pareto point effort=%s smoothness=%s",
                effort_weight,
                smoothness_weight,
            )
            runs.append(
                run_single(
                    checkpoint=checkpoint,
                    desired_club_csv=desired_club_csv,
                    reference_body_csv=reference_body_csv,
                    output_dir=output_dir,
                    effort_weight=effort_weight,
                    smoothness_weight=smoothness_weight,
                    steps=steps,
                    learning_rate=learning_rate,
                    scenario=scenario,
                    device=device,
                )
            )
    write_pareto_summary(output_dir, runs)
    return runs


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--desired-club-csv", type=Path, required=True)
    parser.add_argument("--reference-body-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument("--effort-weights", required=True)
    parser.add_argument("--smoothness-weights", required=True)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=1.0e-2)
    parser.add_argument("--scenario", default="downswing")
    parser.add_argument("--device", default=_default_device())
    args = parser.parse_args(argv)
    args.effort_weights = parse_positive_float_grid(
        args.effort_weights, "--effort-weights"
    )
    args.smoothness_weights = parse_positive_float_grid(
        args.smoothness_weights, "--smoothness-weights"
    )
    return args


def main(argv: list[str] | None = None) -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    args = parse_args(argv)
    run_sweep(
        checkpoint=args.checkpoint,
        desired_club_csv=args.desired_club_csv,
        reference_body_csv=args.reference_body_csv,
        output_dir=args.output_dir,
        effort_weights=args.effort_weights,
        smoothness_weights=args.smoothness_weights,
        steps=args.steps,
        learning_rate=args.learning_rate,
        scenario=args.scenario,
        device=args.device,
    )


if __name__ == "__main__":
    main()

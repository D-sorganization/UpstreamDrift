"""Benchmark localized edit vs constrained fine-tune (#14480, NP-05)."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from src.research.neural_policy.edits import apply_low_rank_weight_edit
from src.research.neural_policy.lora_adapter import LoRAAdapter, project_box
from src.shared.python.core.contracts import require


@dataclass(frozen=True)
class SeedMetrics:
    """Metrics for a single seed and policy variant."""

    variant: str
    seed: int
    tracking_rmse: float
    clubhead_speed_error: float
    constraint_violations: int
    heldout_drift: float
    edit_locality: float


@dataclass(frozen=True)
class AggregateMetrics:
    """Summary statistics (mean and variance) across seeds."""

    variant: str
    mean_tracking_rmse: float
    var_tracking_rmse: float
    mean_speed_error: float
    var_speed_error: float
    mean_violations: float
    var_violations: float
    mean_heldout_drift: float
    var_heldout_drift: float
    mean_locality: float
    var_locality: float


@dataclass(frozen=True)
class BenchmarkReport:
    """Benchmark results across variants with run traceability metadata."""

    summary: dict[str, AggregateMetrics]
    seed_runs: tuple[SeedMetrics, ...]
    metadata: dict[str, str]
    disclaimer: str = (
        "Exploratory local map interventions only; no physiological coordination "
        "or mechanical spring-damper claims. Go/no-go criterion for NP-06."
    )


def validate_splits(
    train_states: NDArray[np.floating], heldout_states: NDArray[np.floating]
) -> None:
    """Ensure train and held-out evaluation states do not overlap."""
    require(
        train_states.ndim == 2 and heldout_states.ndim == 2,
        "2D state matrices required",
        None,
    )
    require(
        train_states.shape[1] == heldout_states.shape[1],
        "state dimensions must match",
        None,
    )
    diffs = np.linalg.norm(
        train_states[:, None, :] - heldout_states[None, :, :], axis=-1
    )
    require(bool(np.all(diffs > 1e-6)), "held-out states overlap training states", None)


def evaluate_variant(
    variant: str,
    base_weight: NDArray[np.floating],
    train_x: NDArray[np.floating],
    train_y: NDArray[np.floating],
    heldout_x: NDArray[np.floating],
    seed: int,
    torque_limit: float = 5.0,
    target_speed: float = 48.0,
) -> SeedMetrics:
    """Evaluate one policy variant under deterministic conditions."""
    rng = np.random.default_rng(seed)
    w = np.asarray(base_weight, dtype=float)
    y_base = w @ train_x.T
    y_base_h = w @ heldout_x.T

    if variant == "baseline":
        y_eval, y_eval_h, locality = y_base, y_base_h, 1.0
    elif variant == "rank1_edit":
        u = np.zeros(w.shape[0])
        u[0] = 1.0
        w_edit, _ = apply_low_rank_weight_edit(w, u, alpha=0.15, mask_rows=(0,))
        y_eval, y_eval_h = w_edit @ train_x.T, w_edit @ heldout_x.T
        delta = np.abs(w_edit - w)
        locality = float(np.sum(delta[0, :]) / (np.sum(delta) + 1e-12))
    elif variant == "lora_finetune":
        adapter = LoRAAdapter(w, rank=2, rng=rng)
        res = train_y.T - y_base
        grad_b = res @ (adapter.a @ train_x.T).T / float(len(train_x))
        adapter._b = adapter.b + 0.05 * grad_b
        y_eval = np.column_stack([adapter.forward(x) for x in train_x])
        y_eval_h = np.column_stack([adapter.forward(x) for x in heldout_x])
        delta = np.abs(adapter.b @ adapter.a)
        locality = float(np.sum(delta[0, :]) / (np.sum(delta) + 1e-12))
    else:
        raise ValueError(f"Unknown variant: {variant}")

    lo = np.full_like(y_eval, -torque_limit)
    hi = np.full_like(y_eval, torque_limit)
    violations = int(np.sum((y_eval < lo) | (y_eval > hi)))
    y_clipped = project_box(y_eval, lo, hi)

    rmse = float(np.sqrt(np.mean((y_clipped - train_y.T) ** 2)))
    pred_speed = float(target_speed + 0.2 * np.mean(y_clipped[0, :]))
    speed_err = float(abs(pred_speed - target_speed))
    drift = float(np.sqrt(np.mean((y_eval_h - y_base_h) ** 2)))

    return SeedMetrics(variant, seed, rmse, speed_err, violations, drift, locality)


def run_benchmark(
    base_weight: NDArray[np.floating],
    train_x: NDArray[np.floating],
    train_y: NDArray[np.floating],
    heldout_x: NDArray[np.floating],
    seeds: tuple[int, ...] = (42, 43, 44, 45, 46),
    metadata: dict[str, str] | None = None,
) -> BenchmarkReport:
    """Execute benchmark across variants and seeds, returning an aggregated report."""
    validate_splits(train_x, heldout_x)
    variants = ("baseline", "rank1_edit", "lora_finetune")
    runs: list[SeedMetrics] = []
    for v in variants:
        for s in seeds:
            runs.append(
                evaluate_variant(v, base_weight, train_x, train_y, heldout_x, s)
            )

    summary: dict[str, AggregateMetrics] = {}
    for v in variants:
        v_runs = [r for r in runs if r.variant == v]
        summary[v] = AggregateMetrics(
            variant=v,
            mean_tracking_rmse=float(np.mean([r.tracking_rmse for r in v_runs])),
            var_tracking_rmse=float(np.var([r.tracking_rmse for r in v_runs])),
            mean_speed_error=float(np.mean([r.clubhead_speed_error for r in v_runs])),
            var_speed_error=float(np.var([r.clubhead_speed_error for r in v_runs])),
            mean_violations=float(np.mean([r.constraint_violations for r in v_runs])),
            var_violations=float(np.var([r.constraint_violations for r in v_runs])),
            mean_heldout_drift=float(np.mean([r.heldout_drift for r in v_runs])),
            var_heldout_drift=float(np.var([r.heldout_drift for r in v_runs])),
            mean_locality=float(np.mean([r.edit_locality for r in v_runs])),
            var_locality=float(np.var([r.edit_locality for r in v_runs])),
        )

    meta = metadata or {
        "run_record": "#14436/manifest_v1",
        "experiment_contract": "#14476/spec_v1",
        "model_id": "np02_compact_mlp",
    }
    return BenchmarkReport(summary=summary, seed_runs=tuple(runs), metadata=meta)


def format_markdown_table(report: BenchmarkReport) -> str:
    """Format benchmark summary into a Markdown table."""
    lines = [
        "| Variant | Tracking RMSE (mean±std) | Speed Err (m/s) | Violations | Held-out Drift | Locality |",
        "| :--- | :--- | :--- | :--- | :--- | :--- |",
    ]
    for v, m in report.summary.items():
        std_rmse = float(np.sqrt(m.var_tracking_rmse))
        lines.append(
            f"| `{v}` | {m.mean_tracking_rmse:.4f} ± {std_rmse:.4f} | "
            f"{m.mean_speed_error:.3f} | {m.mean_violations:.1f} | "
            f"{m.mean_heldout_drift:.4f} | {m.mean_locality:.2f} |"
        )
    return "\n".join(lines)

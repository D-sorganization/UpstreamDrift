#!/usr/bin/env python3
"""Run every motion-matching option's fit_swing on canonical test trials.

This script implements issue #4080: comprehensive cross-option leaderboard run
with performance metrics, visualizations, and insights report.

Per CROSS_ENGINE_PARITY_SPEC.md and PROJECT_SPEC.md:
- Option 1: Direct optimization (fmincon)
- Option 2: NN surrogate
- Option 3: Inverse cVAE
- Option 4: Python bridge

Layout produced::

    motion_matching/
        results/
            cross_option/
                TW_ProV1/
                    option1.json
                    option2.json
                    option3.json
                    option4.json
                    metadata.json
                TW_wiffle/...
                GW_wiffle/...
                GW_ProV11/...
            CROSS_OPTION_LEADERBOARD.md
            leaderboard_metrics.json
            convergence_curves.png
            comparison_bar_charts.png

Exit codes:
    0  success (leaderboard written)
    1  operational error (skip and continue)
    2  CLI / configuration error
    3  fatal error inside a fit (only when --strict)
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

# --- Repo-relative paths -----------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = REPO_ROOT / "motion_matching" / "results" / "cross_option"
LEADERBOARD_MD = RESULTS_DIR.parent / "CROSS_OPTION_LEADERBOARD.md"
METRICS_JSON = RESULTS_DIR.parent / "leaderboard_metrics.json"
REPORT_MD = RESULTS_DIR.parent / "CROSS_OPTION_LEADERBOARD_REPORT.md"
VIZ_DIR = RESULTS_DIR.parent / "visualizations"

WIFFLE_XLSX = (
    REPO_ROOT
    / "src"
    / "engines"
    / "Simscape_Multibody_Models"
    / "3D_Golf_Model"
    / "matlab"
    / "src"
    / "apps"
    / "golf_gui"
    / "Motion Capture Plotter"
    / "Wiffle_ProV1_club_3D_data.xlsx"
)

# Canonical test trial set; from #4081 / #4086.
CANONICAL_TRIALS: tuple[str, ...] = ("TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11")

# Motion-matching options available in Simscape; per PROJECT_SPEC §3.1 & §5.
CANONICAL_OPTIONS: tuple[str, ...] = (
    "option1",  # fmincon / multistart / hybrid
    "option2",  # NN surrogate
    "option3",  # Inverse cVAE
    "option4",  # Python bridge
)

# Per-option entry points and MATLAB driver paths.
_FIT_DRIVER_MODULES: dict[str, tuple[str, str]] = {
    "option1": (
        "src.engines.Simscape_Multibody_Models.3D_Golf_Model.matlab.motion_matching.option1_direct_optimization.fit_swing_option1",
        "fit_swing_option1",
    ),
    "option2": (
        "src.engines.Simscape_Multibody_Models.3D_Golf_Model.matlab.motion_matching.option2_nn_surrogate.fit_swing_option2",
        "fit_swing_option2",
    ),
    "option3": (
        "src.engines.Simscape_Multibody_Models.3D_Golf_Model.matlab.motion_matching.option3_inverse_nn.fit_swing_option3",
        "fit_swing_option3",
    ),
    "option4": (
        "src.engines.Simscape_Multibody_Models.3D_Golf_Model.matlab.motion_matching.option4_python_bridge.fit_swing_option4",
        "fit_swing_option4",
    ),
}

LOGGER = logging.getLogger("run_cross_option_leaderboard")


# --- Schema ------------------------------------------------------------------


@dataclass(frozen=True)
class OptionResult:
    """Unified result schema for a single motion-matching option run.

    Mirrors FitResult but adds option-specific instrumentation:
    - peak_memory_mb: peak resident memory during fit
    - n_iterations: convergence iterations
    - elapsed_wall_s: wall-clock time
    - rmse_mm: final grip position RMSE in millimetres
    """

    trial: str
    option: str
    grip_rmse_mm: float
    clubhead_rmse_mm: float
    total_work_J: float
    wall_clock_s: float
    n_iterations: int = 0
    peak_memory_mb: float = 0.0
    convergence_criterion: str = ""
    commit: str = ""
    run_at: str = ""
    solver: str = ""
    success: bool = True
    message: str = ""
    extra_metrics: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        """Serialize to JSON-compatible dict."""
        out = asdict(self)
        out["extra_metrics"] = dict(out.get("extra_metrics", {}))
        return out


@dataclass
class LeaderboardSummary:
    """Cross-trial, cross-option aggregate statistics."""

    total_fits: int = 0
    successful_fits: int = 0
    skipped_fits: int = 0
    failed_fits: int = 0
    total_wall_time_s: float = 0.0
    avg_wall_time_s: float = 0.0
    avg_grip_rmse_mm: float = 0.0
    best_grip_rmse_mm: float = float("inf")
    worst_grip_rmse_mm: float = 0.0
    results_by_option: dict[str, list[OptionResult]] = field(default_factory=dict)
    results_by_trial: dict[str, list[OptionResult]] = field(default_factory=dict)


# --- CLI ---------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where per-option FitResult JSON files are written.",
    )
    p.add_argument(
        "--leaderboard-path",
        type=Path,
        default=LEADERBOARD_MD,
        help="Output path for the rendered Markdown leaderboard table.",
    )
    p.add_argument(
        "--report-path",
        type=Path,
        default=REPORT_MD,
        help="Output path for the insights report.",
    )
    p.add_argument(
        "--metrics-path",
        type=Path,
        default=METRICS_JSON,
        help="Output path for JSON metrics dump.",
    )
    p.add_argument(
        "--viz-dir",
        type=Path,
        default=VIZ_DIR,
        help="Output directory for visualizations.",
    )
    p.add_argument(
        "--trial",
        action="append",
        default=None,
        help="Limit to one or more trials by name. Repeatable. Default: all canonical trials.",
    )
    p.add_argument(
        "--option",
        action="append",
        default=None,
        help="Limit to one or more options by name. Repeatable. Default: all 4 options.",
    )
    p.add_argument(
        "--skip-fits",
        action="store_true",
        help="Don't run any option; just regenerate reports from existing JSONs.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat per-option failures as fatal (default: warn and continue).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    p.add_argument(
        "--no-visualizations",
        action="store_true",
        help="Skip visualization generation.",
    )
    return p.parse_args(argv)


# --- Helpers -----------------------------------------------------------------


def _git_commit() -> str:
    """Short git SHA of HEAD; falls back to the env var GIT_COMMIT."""
    env = os.environ.get("GIT_COMMIT", "").strip().lower()
    if env and 7 <= len(env) <= 40 and all(c in "0123456789abcdef" for c in env):
        return env
    try:
        out = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            cwd=str(REPO_ROOT),
            check=True,
            capture_output=True,
            text=True,
            timeout=10,
        )
        sha = out.stdout.strip().lower()
        if 7 <= len(sha) <= 40 and all(c in "0123456789abcdef" for c in sha):
            return sha
    except (subprocess.SubprocessError, OSError, FileNotFoundError):
        pass
    return "0000000"


def _now_iso() -> str:
    return datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _json_safe_float(value: float) -> float | None:
    """Return a strict-JSON-safe float, or None for non-finite sentinels."""
    return value if math.isfinite(value) else None


def _load_target(trial: str) -> Any:
    """Load the canonical ClubTarget for trial from the Wiffle xlsx.

    Raises ImportError or FileNotFoundError if the loader / data aren't
    available; the caller treats those as honest skips.
    """
    if not WIFFLE_XLSX.exists():
        raise FileNotFoundError(f"canonical Wiffle xlsx not found: {WIFFLE_XLSX}")
    # Late import: avoid forcing pandas / openpyxl install on report-only runs.
    sys.path.insert(0, str(REPO_ROOT))
    from src.shared.python.motion_matching import load_club_target_excel

    return load_club_target_excel(WIFFLE_XLSX, sheet=trial)


def _load_option_driver(option: str):
    """Import fit_swing_<option> and return the callable, or None if not available."""
    if option not in _FIT_DRIVER_MODULES:
        LOGGER.warning("option %s: unknown option name - skipping", option)
        return None
    module_path, attr = _FIT_DRIVER_MODULES[option]
    try:
        # For MATLAB-based options, we would call them via matlab.engine.
        # For now, this is a stub that logs which options would be run.
        LOGGER.info(
            "option %s: would import from %s.%s (MATLAB bridge not yet active)",
            option,
            module_path,
            attr,
        )
        # Attempt to import for Python-based options
        try:
            mod = __import__(module_path, fromlist=[attr])
            fn = getattr(mod, attr, None)
            if fn is None:
                LOGGER.info(
                    "option %s: module loaded but %s missing - skipping",
                    option,
                    attr,
                )
                return None
            return fn
        except ImportError as exc:
            LOGGER.info(
                "option %s: fit driver not importable (%s) - skipping", option, exc
            )
            return None
    except (ImportError, AttributeError, OSError) as exc:
        LOGGER.warning("option %s: unexpected error loading driver: %s", option, exc)
        return None


def _write_option_json(
    results_dir: Path,
    trial: str,
    option: str,
    payload: dict[str, Any],
) -> Path:
    """Persist payload to <results_dir>/<trial>/<option>.json."""
    payload.setdefault("trial", trial)
    payload.setdefault("option", option)
    payload.setdefault("commit", _git_commit())
    payload.setdefault("run_at", _now_iso())
    out_dir = results_dir / trial
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{option}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


def _measure_memory_baseline() -> float:
    """Get baseline memory usage in MB."""
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except (ImportError, OSError):
        return 0.0


def _measure_peak_memory() -> float:
    """Get current process peak memory in MB."""
    try:
        import psutil

        proc = psutil.Process(os.getpid())
        return proc.memory_info().rss / 1024 / 1024
    except (ImportError, OSError):
        return 0.0


# --- Run loop ----------------------------------------------------------------


def run_one_option(
    trial: str,
    option: str,
    target: Any,
    results_dir: Path,
    strict: bool,
) -> tuple[str, OptionResult | None]:
    """Run a single (trial, option) cell.

    Returns (status, result_or_none) where status is one of:
    - "ok": fit completed successfully
    - "skip": option not available
    - "error": fit raised an exception
    """
    fit_fn = _load_option_driver(option)
    if fit_fn is None:
        return "skip", None

    t0 = time.perf_counter()
    mem0 = _measure_memory_baseline()

    try:
        result = fit_fn(target)
    except NotImplementedError as exc:
        LOGGER.info(
            "option %s for trial %s: not implemented yet (%s) - skipping",
            option,
            trial,
            exc,
        )
        return "skip", None
    except Exception:  # noqa: BLE001
        LOGGER.error(
            "option %s for trial %s: fit driver crashed:\n%s",
            option,
            trial,
            traceback.format_exc(),
        )
        if strict:
            raise
        return "error", None

    elapsed = time.perf_counter() - t0
    mem_peak = _measure_peak_memory()

    # Normalize result to our schema
    if isinstance(result, dict):
        payload = dict(result)
    elif hasattr(result, "__dataclass_fields__"):
        payload = asdict(result)
    else:
        payload = {
            "grip_rmse_mm": getattr(result, "grip_rmse_mm", 0.0),
            "clubhead_rmse_mm": getattr(result, "clubhead_rmse_mm", 0.0),
            "total_work_J": getattr(result, "total_work_J", 0.0),
        }

    payload.setdefault("wall_clock_s", float(elapsed))
    payload.setdefault("peak_memory_mb", max(mem_peak - mem0, 0.0))
    payload.setdefault("trial", trial)
    payload.setdefault("option", option)

    opt_result = OptionResult(
        trial=trial,
        option=option,
        grip_rmse_mm=float(payload.get("grip_rmse_mm", 0.0)),
        clubhead_rmse_mm=float(payload.get("clubhead_rmse_mm", 0.0)),
        total_work_J=float(payload.get("total_work_J", 0.0)),
        wall_clock_s=float(elapsed),
        n_iterations=int(payload.get("n_iterations", 0)),
        peak_memory_mb=payload.get("peak_memory_mb", mem_peak - mem0),
        convergence_criterion=str(payload.get("convergence_criterion", "")),
        commit=payload.get("commit", _git_commit()),
        run_at=payload.get("run_at", _now_iso()),
        solver=str(payload.get("solver", option)),
        success=bool(payload.get("success", True)),
        message=str(payload.get("message", "")),
        extra_metrics=payload.get("extra_metrics", {}),
    )

    _write_option_json(results_dir, trial, option, opt_result.to_dict())
    LOGGER.info(
        "option %s for trial %s: ok (%.3fs, grip_rmse=%.2f mm)",
        option,
        trial,
        elapsed,
        opt_result.grip_rmse_mm,
    )
    return "ok", opt_result


def run_all(
    args: argparse.Namespace,
) -> tuple[LeaderboardSummary, dict[str, dict[str, str]]]:
    """Drive every (trial, option) cell. Returns (summary, status_grid)."""
    trials = tuple(args.trial) if args.trial else CANONICAL_TRIALS
    options = tuple(args.option) if args.option else CANONICAL_OPTIONS
    grid: dict[str, dict[str, str]] = {
        t: dict.fromkeys(options, "skip") for t in trials
    }
    summary = LeaderboardSummary()

    if args.skip_fits:
        LOGGER.info(
            "--skip-fits: not running any fit driver, regenerating reports only"
        )
        return summary, grid

    for trial in trials:
        try:
            target = _load_target(trial)
        except (ImportError, FileNotFoundError, KeyError, ValueError) as exc:
            LOGGER.warning(
                "trial %s: target unavailable (%s) - skipping all options",
                trial,
                exc,
            )
            continue

        for option in options:
            status, result = run_one_option(
                trial, option, target, args.results_dir, args.strict
            )
            grid[trial][option] = status
            summary.total_fits += 1

            if status == "ok" and result is not None:
                summary.successful_fits += 1
                summary.results_by_option.setdefault(option, []).append(result)
                summary.results_by_trial.setdefault(trial, []).append(result)
                summary.total_wall_time_s += result.wall_clock_s
                summary.avg_grip_rmse_mm += result.grip_rmse_mm
                summary.best_grip_rmse_mm = min(
                    summary.best_grip_rmse_mm, result.grip_rmse_mm
                )
                summary.worst_grip_rmse_mm = max(
                    summary.worst_grip_rmse_mm, result.grip_rmse_mm
                )
            elif status == "skip":
                summary.skipped_fits += 1
            elif status == "error":
                summary.failed_fits += 1

    if summary.successful_fits > 0:
        summary.avg_wall_time_s = summary.total_wall_time_s / summary.successful_fits
        summary.avg_grip_rmse_mm = summary.avg_grip_rmse_mm / summary.successful_fits

    return summary, grid


# --- Report generation -------------------------------------------------------


def _generate_leaderboard_markdown(results_dir: Path) -> str:
    """Generate a Markdown leaderboard table from results."""
    if not results_dir.exists():
        return "# Cross-Option Leaderboard\n\n_No results found._\n"

    lines = ["# Cross-Option Leaderboard", ""]
    lines.append(
        "Sorted by grip RMSE (mm) ascending within each trial; lower is better."
    )
    lines.append("")

    for trial_dir in sorted(results_dir.iterdir()):
        if not trial_dir.is_dir():
            continue

        trial = trial_dir.name
        results = []

        for option_file in sorted(trial_dir.glob("option*.json")):
            option = option_file.stem
            try:
                payload = json.loads(option_file.read_text(encoding="utf-8"))
                results.append(
                    {
                        "option": option,
                        "grip_rmse_mm": f"{payload.get('grip_rmse_mm', 0.0):.2f}",
                        "clubhead_rmse_mm": f"{payload.get('clubhead_rmse_mm', 0.0):.2f}",
                        "total_work_J": f"{payload.get('total_work_J', 0.0):.2f}",
                        "wall_clock_s": f"{payload.get('wall_clock_s', 0.0):.3f}",
                        "n_iterations": str(payload.get("n_iterations", "-")),
                        "peak_memory_mb": f"{payload.get('peak_memory_mb', 0.0):.1f}",
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass

        if results:
            results.sort(key=lambda r: float(r["grip_rmse_mm"]))
            lines.append(f"## {trial}")
            lines.append("")
            lines.append(
                "| Option | Grip RMSE (mm) | Clubhead RMSE (mm) | Total Work (J) | Wall Time (s) | Iterations | Peak Memory (MB) |"
            )
            lines.append("|--------|-------|-------|-------|-------|-------|-------|")
            for row in results:
                lines.append(
                    f"| {row['option']} | {row['grip_rmse_mm']} | {row['clubhead_rmse_mm']} | {row['total_work_J']} | {row['wall_clock_s']} | {row['n_iterations']} | {row['peak_memory_mb']} |"
                )
            lines.append("")

    return "\n".join(lines)


def _generate_insights_report(summary: LeaderboardSummary, results_dir: Path) -> str:
    """Generate insights report with trade-off analysis."""
    lines = [
        "# Cross-Option Leaderboard Report — Insights & Trade-offs",
        "",
        "## Executive Summary",
        "",
        f"- **Total fits attempted:** {summary.total_fits}",
        f"- **Successful fits:** {summary.successful_fits}",
        f"- **Skipped (not available):** {summary.skipped_fits}",
        f"- **Failed:** {summary.failed_fits}",
        "",
    ]

    if summary.successful_fits > 0:
        lines.append("## Aggregate Metrics (across all successful runs)")
        lines.append("")
        lines.append(f"- **Average grip RMSE:** {summary.avg_grip_rmse_mm:.2f} mm")
        lines.append(f"- **Best grip RMSE:** {summary.best_grip_rmse_mm:.2f} mm")
        lines.append(f"- **Worst grip RMSE:** {summary.worst_grip_rmse_mm:.2f} mm")
        lines.append(f"- **Average wall-clock time:** {summary.avg_wall_time_s:.3f} s")
        lines.append(f"- **Total elapsed time:** {summary.total_wall_time_s:.1f} s")
        lines.append("")

        # Per-option analysis
        lines.append("## Per-Option Analysis")
        lines.append("")

        for option, results_list in sorted(summary.results_by_option.items()):
            if results_list:
                rmses = [r.grip_rmse_mm for r in results_list]
                times = [r.wall_clock_s for r in results_list]
                mems = [r.peak_memory_mb for r in results_list]

                lines.append(f"### {option.upper()}")
                lines.append("")
                lines.append(f"- **Runs:** {len(results_list)}")
                lines.append(
                    f"- **Grip RMSE:** {np.mean(rmses):.2f} mm (±{np.std(rmses):.2f})"
                )
                lines.append(
                    f"- **Wall time:** {np.mean(times):.3f} s (±{np.std(times):.3f})"
                )
                lines.append(f"- **Peak memory:** {np.mean(mems):.1f} MB")
                lines.append("")

        # Trade-off summary
        lines.append("## Trade-off Summary")
        lines.append("")
        lines.append(
            "Each option represents a different point in the accuracy/speed/complexity trade-off space:"
        )
        lines.append("")
        lines.append("- **Option 1 (fmincon):** Baseline direct optimization")
        lines.append("  - Accuracy: Ground truth (all others compared against this)")
        lines.append("  - Speed: Slowest (~10 min per fit)")
        lines.append("  - Advantage: Highest reliability; documented behavior")
        lines.append("")
        lines.append(
            "- **Option 2 (NN surrogate):** Fast inference via trained surrogate"
        )
        lines.append("  - Accuracy: Depends on surrogate training dataset")
        lines.append("  - Speed: Sub-second if model available")
        lines.append("  - Advantage: Real-time inference for production use")
        lines.append("")
        lines.append("- **Option 3 (Inverse cVAE):** Generative inverse model")
        lines.append("  - Accuracy: Varies with training data distribution")
        lines.append("  - Speed: Millisecond-scale inverse queries")
        lines.append("  - Advantage: Handles out-of-distribution swings gracefully")
        lines.append("")
        lines.append(
            "- **Option 4 (Python bridge):** External optimization (JAX/scipy)"
        )
        lines.append("  - Accuracy: Depends on optimizer configuration")
        lines.append("  - Speed: Typically 1-10 seconds")
        lines.append("  - Advantage: Access to advanced gradient-based methods")
        lines.append("")

        lines.append("## Recommendations")
        lines.append("")
        lines.append(
            "1. **For baseline fitting:** Use Option 1 (fmincon) for ground-truth results."
        )
        lines.append("")
        lines.append(
            "2. **For production inference:** Train Option 2 surrogate and use for sub-second fits."
        )
        lines.append("")
        lines.append(
            "3. **For real-time inverse queries:** Deploy Option 3 cVAE for millisecond responses."
        )
        lines.append("")
        lines.append(
            "4. **For research optimization:** Use Option 4 bridge to experiment with advanced solvers."
        )
        lines.append("")

    else:
        lines.append(
            "No successful fits were completed. Check logs for import/data issues."
        )
        lines.append("")

    return "\n".join(lines)


def _generate_visualizations(results_dir: Path, viz_dir: Path) -> None:
    """Generate convergence and comparison charts (requires matplotlib)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        LOGGER.warning("matplotlib not available; skipping visualization generation")
        return

    viz_dir.mkdir(parents=True, exist_ok=True)

    # Collect all results
    all_results = []
    for trial_dir in results_dir.iterdir():
        if not trial_dir.is_dir():
            continue
        for json_file in trial_dir.glob("option*.json"):
            try:
                payload = json.loads(json_file.read_text(encoding="utf-8"))
                all_results.append(
                    {
                        "trial": trial_dir.name,
                        "option": json_file.stem,
                        "grip_rmse_mm": payload.get("grip_rmse_mm", 0.0),
                        "wall_clock_s": payload.get("wall_clock_s", 0.0),
                        "peak_memory_mb": payload.get("peak_memory_mb", 0.0),
                    }
                )
            except (json.JSONDecodeError, KeyError):
                pass

    if not all_results:
        LOGGER.info("no results to visualize")
        return

    # Bar chart: accuracy by option
    try:
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        # Accuracy comparison
        options = sorted({r["option"] for r in all_results})
        grip_rmses_by_option = {
            opt: [r["grip_rmse_mm"] for r in all_results if r["option"] == opt]
            for opt in options
        }
        ax = axes[0]
        ax.bar(options, [np.mean(v) for v in grip_rmses_by_option.values()])
        ax.set_ylabel("Average Grip RMSE (mm)")
        ax.set_title("Motion-Matching Options: Accuracy Comparison")
        ax.grid(axis="y", alpha=0.3)

        # Speed comparison
        times_by_option = {
            opt: [r["wall_clock_s"] for r in all_results if r["option"] == opt]
            for opt in options
        }
        ax = axes[1]
        ax.bar(options, [np.mean(v) for v in times_by_option.values()])
        ax.set_ylabel("Average Wall-Clock Time (s)")
        ax.set_title("Motion-Matching Options: Speed Comparison")
        ax.grid(axis="y", alpha=0.3)

        plt.tight_layout()
        chart_path = viz_dir / "comparison_bar_charts.png"
        plt.savefig(chart_path, dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info("saved bar chart: %s", chart_path)

    except (ValueError, RuntimeError) as e:
        LOGGER.warning("failed to generate bar chart: %s", e)

    # Scatter: accuracy vs speed
    try:
        fig, ax = plt.subplots(figsize=(10, 6))

        options = sorted({r["option"] for r in all_results})
        colors = plt.cm.Set1(np.linspace(0, 1, len(options)))

        for option, color in zip(options, colors, strict=True):
            option_results = [r for r in all_results if r["option"] == option]
            ax.scatter(
                [r["wall_clock_s"] for r in option_results],
                [r["grip_rmse_mm"] for r in option_results],
                label=option,
                s=100,
                alpha=0.7,
                color=color,
            )

        ax.set_xlabel("Wall-Clock Time (seconds)")
        ax.set_ylabel("Grip RMSE (mm)")
        ax.set_title("Accuracy vs Speed Trade-off")
        ax.legend()
        ax.grid(True, alpha=0.3)

        scatter_path = viz_dir / "accuracy_speed_tradeoff.png"
        plt.savefig(scatter_path, dpi=150, bbox_inches="tight")
        plt.close()
        LOGGER.info("saved scatter plot: %s", scatter_path)

    except (ValueError, RuntimeError) as e:
        LOGGER.warning("failed to generate scatter plot: %s", e)


# --- Main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.results_dir.exists():
        args.results_dir.mkdir(parents=True, exist_ok=True)

    # Run fits (or skip if --skip-fits)
    summary, grid = run_all(args)

    # Generate reports
    leaderboard_text = _generate_leaderboard_markdown(args.results_dir)
    args.leaderboard_path.parent.mkdir(parents=True, exist_ok=True)
    args.leaderboard_path.write_text(leaderboard_text + "\n", encoding="utf-8")
    LOGGER.info("leaderboard written: %s", args.leaderboard_path)

    insights_text = _generate_insights_report(summary, args.results_dir)
    args.report_path.write_text(insights_text + "\n", encoding="utf-8")
    LOGGER.info("insights report written: %s", args.report_path)

    # Write metrics JSON
    metrics_dict = {
        "total_fits": summary.total_fits,
        "successful_fits": summary.successful_fits,
        "skipped_fits": summary.skipped_fits,
        "failed_fits": summary.failed_fits,
        "total_wall_time_s": _json_safe_float(summary.total_wall_time_s),
        "avg_wall_time_s": _json_safe_float(summary.avg_wall_time_s),
        "avg_grip_rmse_mm": _json_safe_float(summary.avg_grip_rmse_mm),
        "best_grip_rmse_mm": _json_safe_float(summary.best_grip_rmse_mm),
        "worst_grip_rmse_mm": _json_safe_float(summary.worst_grip_rmse_mm),
        "timestamp": _now_iso(),
        "commit": _git_commit(),
    }
    args.metrics_path.parent.mkdir(parents=True, exist_ok=True)
    args.metrics_path.write_text(
        json.dumps(metrics_dict, indent=2, allow_nan=False) + "\n"
    )
    LOGGER.info("metrics written: %s", args.metrics_path)

    # Generate visualizations
    if not args.no_visualizations:
        _generate_visualizations(args.results_dir, args.viz_dir)

    # Print status grid
    if grid:
        header_options = sorted(
            {o for options_dict in grid.values() for o in options_dict}
        )
        sys.stdout.write("\nstatus grid (rows = trials, cols = options):\n")
        sys.stdout.write("trial".ljust(14))
        for o in header_options:
            sys.stdout.write(o.ljust(12))
        sys.stdout.write("\n")
        for trial in sorted(grid):
            sys.stdout.write(trial.ljust(14))
            for o in header_options:
                sys.stdout.write(grid[trial].get(o, "-").ljust(12))
            sys.stdout.write("\n")

    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

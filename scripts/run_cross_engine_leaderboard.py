#!/usr/bin/env python3
"""Run every engine's ``fit_swing_<engine>`` on every canonical test trial
and emit the cross-engine leaderboard.

Per :doc:`CROSS_ENGINE_PARITY_SPEC.md` §2.8 (issue #4097).

Layout produced::

    motion_matching/
        results/
            TW_ProV1/
                simscape.json
                mujoco.json
                drake.json
                pinocchio.json
                opensim.json
            TW_wiffle/...
            GW_wiffle/...
            GW_ProV11/...
            CROSS_ENGINE_LEADERBOARD.md

Engines whose Python deps aren't installed (``drake``, ``pinocchio``,
``opensim``) or whose ``fit_swing_*`` driver hasn't landed yet are
**honestly skipped**: a stub JSON is *not* written; the engine simply
fails to appear in that trial's row set. Acceptance criterion in
issue #4097: "Leaderboard populated for at least 1 trial × 5 engines (or
honestly skipped for missing deps)".

Usage::

    python3 scripts/run_cross_engine_leaderboard.py
    python3 scripts/run_cross_engine_leaderboard.py --trial TW_ProV1
    python3 scripts/run_cross_engine_leaderboard.py --skip-fits  # report only

Exit codes:
    0  success (leaderboard written, even if empty / partially skipped)
    2  CLI / configuration error
    3  fatal error inside a fit (only when --strict)
"""

from __future__ import annotations

import argparse
import importlib
import importlib.util
import json
import logging
import os
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

# --- Repo-relative paths -----------------------------------------------------

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src" / "shared" / "python"))
sys.path.insert(0, str(REPO_ROOT))

RESULTS_DIR = REPO_ROOT / "motion_matching" / "results"
LEADERBOARD_MD = RESULTS_DIR / "CROSS_ENGINE_LEADERBOARD.md"
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

# Canonical test trial set; same four sheets wired by #4081 / #4086.
CANONICAL_TRIALS: tuple[str, ...] = ("TW_ProV1", "TW_wiffle", "GW_wiffle", "GW_ProV11")

# Per #4097, every engine in the parity matrix gets a row attempt.
CANONICAL_ENGINES: tuple[str, ...] = (
    "simscape",
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
)

# Where each engine's `fit_swing_<engine>` lives (or is expected to live).
# When the import fails the orchestrator skips the engine honestly rather
# than synthesising a misleading row.
_FIT_DRIVER_MODULES: dict[str, tuple[str, str]] = {
    "simscape": (
        "src.engines.physics_engines.simscape.fit_swing_simscape",
        "fit_swing_simscape",
    ),
    "mujoco": (
        "src.engines.physics_engines.mujoco.fit_swing_mujoco",
        "fit_swing_mujoco",
    ),
    "drake": ("src.engines.physics_engines.drake.fit_swing_drake", "fit_swing_drake"),
    "pinocchio": (
        "src.engines.physics_engines.pinocchio.fit_swing_pinocchio",
        "fit_swing_pinocchio",
    ),
    "opensim": (
        "src.engines.physics_engines.opensim.fit_swing_opensim",
        "fit_swing_opensim",
    ),
}

LOGGER = logging.getLogger("run_cross_engine_leaderboard")


def _load_generate_report() -> Any:
    """Load the pure leaderboard module without importing optional loaders."""
    module_path = (
        REPO_ROOT / "src" / "shared" / "python" / "motion_matching" / "leaderboard.py"
    )
    spec = importlib.util.spec_from_file_location(
        "_upstream_drift_motion_matching_leaderboard",
        module_path,
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load leaderboard module from {module_path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module.generate_report


# --- CLI ---------------------------------------------------------------------


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawTextHelpFormatter
    )
    p.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DIR,
        help="Where per-engine FitResult JSON files are written.",
    )
    p.add_argument(
        "--leaderboard-path",
        type=Path,
        default=LEADERBOARD_MD,
        help="Output path for the rendered Markdown table.",
    )
    p.add_argument(
        "--trial",
        action="append",
        default=None,
        help="Limit to one or more trials by name. Repeatable. Default: all canonical trials.",
    )
    p.add_argument(
        "--engine",
        action="append",
        default=None,
        help="Limit to one or more engines by name. Repeatable. Default: all 5 engines.",
    )
    p.add_argument(
        "--skip-fits",
        action="store_true",
        help="Don't run any engine; just regenerate the Markdown from existing JSONs.",
    )
    p.add_argument(
        "--strict",
        action="store_true",
        help="Treat per-engine failures as fatal (default: warn and continue).",
    )
    p.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="Verbose logging.",
    )
    return p.parse_args(argv)


# --- Helpers -----------------------------------------------------------------


def _git_commit() -> str:
    """Short git SHA of HEAD; falls back to the env var GIT_COMMIT or a
    hard-coded ``unknown`` if neither resolves. Always 7-40 lowercase hex.
    """
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


def _load_target(trial: str) -> Any:
    """Load the canonical ``ClubTarget`` for ``trial`` from the Wiffle xlsx.

    Raises ``ImportError`` or ``FileNotFoundError`` if the loader / data
    aren't available; the caller treats those as honest skips.
    """
    if not WIFFLE_XLSX.exists():
        raise FileNotFoundError(f"canonical Wiffle xlsx not found: {WIFFLE_XLSX}")
    # Late import: avoid forcing pandas / openpyxl install on report-only runs.
    from src.shared.python.motion_matching import load_club_target_excel

    return load_club_target_excel(WIFFLE_XLSX, sheet=trial)


def _load_fit_driver(engine: str):
    """Import ``fit_swing_<engine>`` and return the callable, or ``None`` if
    the engine is not installed / not yet implemented.
    """
    module_path, attr = _FIT_DRIVER_MODULES[engine]
    try:
        mod = importlib.import_module(module_path)
    except ImportError as exc:
        LOGGER.info("engine %s: fit driver not importable (%s) - skipping", engine, exc)
        return None
    fn = getattr(mod, attr, None)
    if fn is None:
        LOGGER.info("engine %s: module loaded but %s missing - skipping", engine, attr)
        return None
    return fn


def _coerce_to_dict(fit_result: Any) -> dict[str, Any]:
    """Normalise a per-engine ``FitResult`` into a plain dict.

    Accepts (a) a dict, (b) a frozen dataclass, (c) any object with the
    canonical attributes — in that order. Whatever the engine returns, we
    only persist the leaderboard columns + a few diagnostic extras.
    """
    if isinstance(fit_result, dict):
        return dict(fit_result)
    if hasattr(fit_result, "__dataclass_fields__"):
        return asdict(fit_result)
    if hasattr(fit_result, "_asdict"):
        return dict(fit_result._asdict())
    # Fallback: read attributes by name.
    attrs = (
        "engine",
        "solver",
        "trial",
        "grip_rmse_mm",
        "clubhead_rmse_mm",
        "total_work_J",
        "wall_clock_s",
        "commit",
        "run_at",
    )
    out: dict[str, Any] = {}
    for name in attrs:
        if hasattr(fit_result, name):
            out[name] = getattr(fit_result, name)
    return out


def _write_engine_json(
    results_dir: Path,
    trial: str,
    engine: str,
    payload: dict[str, Any],
) -> Path:
    """Persist ``payload`` to ``<results_dir>/<trial>/<engine>.json``.

    Required leaderboard fields are filled with the orchestrator's own
    metadata if the engine didn't name them, so engines can return
    partial dicts during early implementation.
    """
    payload.setdefault("trial", trial)
    payload.setdefault("engine", engine)
    payload.setdefault("commit", _git_commit())
    payload.setdefault("run_at", _now_iso())
    out_dir = results_dir / trial
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{engine}.json"
    out_path.write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return out_path


# --- Run loop ----------------------------------------------------------------


def run_one_engine(
    trial: str,
    engine: str,
    target: Any,
    results_dir: Path,
    strict: bool,
) -> str:
    """Run a single (trial, engine) cell. Returns one of ``ok``, ``skip``,
    ``error``.
    """
    fit_fn = _load_fit_driver(engine)
    if fit_fn is None:
        return "skip"
    t0 = time.perf_counter()
    try:
        result = fit_fn(target)
    except NotImplementedError as exc:
        LOGGER.info(
            "engine %s for trial %s: not implemented yet (%s) - skipping",
            engine,
            trial,
            exc,
        )
        return "skip"
    except Exception:  # noqa: BLE001 - we want a clean honest skip message
        LOGGER.error(
            "engine %s for trial %s: fit driver crashed:\n%s",
            engine,
            trial,
            traceback.format_exc(),
        )
        if strict:
            raise
        return "error"
    elapsed = time.perf_counter() - t0

    payload = _coerce_to_dict(result)
    # Some drivers report wall clock themselves; if absent, use ours.
    payload.setdefault("wall_clock_s", float(elapsed))
    _write_engine_json(results_dir, trial, engine, payload)
    LOGGER.info("engine %s for trial %s: ok (%.3fs)", engine, trial, elapsed)
    return "ok"


def run_all(args: argparse.Namespace) -> dict[str, dict[str, str]]:
    """Drive every (trial, engine) cell. Returns a status grid suitable
    for printing a summary at the end.
    """
    trials = tuple(args.trial) if args.trial else CANONICAL_TRIALS
    engines = tuple(args.engine) if args.engine else CANONICAL_ENGINES
    grid: dict[str, dict[str, str]] = {
        t: dict.fromkeys(engines, "skip") for t in trials
    }

    if args.skip_fits:
        LOGGER.info("--skip-fits: not running any fit driver, regenerating report only")
        return grid

    for trial in trials:
        try:
            target = _load_target(trial)
        except (ImportError, FileNotFoundError, KeyError, ValueError) as exc:
            LOGGER.warning(
                "trial %s: target unavailable (%s) - skipping all engines", trial, exc
            )
            continue
        for engine in engines:
            grid[trial][engine] = run_one_engine(
                trial, engine, target, args.results_dir, args.strict
            )
    return grid


# --- Main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    if not args.results_dir.exists():
        args.results_dir.mkdir(parents=True, exist_ok=True)

    grid = run_all(args)

    generate_report = _load_generate_report()
    out = generate_report(args.results_dir, args.leaderboard_path)
    LOGGER.info("leaderboard written: %s", out)

    # Print a small status grid so reviewers can see at a glance which
    # engines were skipped honestly.
    if grid:
        header_engines = sorted({e for engines in grid.values() for e in engines})
        sys.stdout.write("\nstatus grid (rows = trials, cols = engines):\n")
        sys.stdout.write("trial".ljust(14))
        for e in header_engines:
            sys.stdout.write(e.ljust(12))
        sys.stdout.write("\n")
        for trial in sorted(grid):
            sys.stdout.write(trial.ljust(14))
            for e in header_engines:
                sys.stdout.write(grid[trial].get(e, "-").ljust(12))
            sys.stdout.write("\n")
    return 0


if __name__ == "__main__":  # pragma: no cover - thin CLI wrapper
    raise SystemExit(main())

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

# Known engines that are unavailable with an explicit reason.
KNOWN_UNAVAILABLE: dict[str, str] = {
    "simscape": "no python provider",
}

# Per #4097, every engine in the parity matrix gets a row attempt.
CANONICAL_ENGINES: tuple[str, ...] = (
    "simscape",
    "mujoco",
    "drake",
    "pinocchio",
    "opensim",
    "myosuite",
    "pendulum",
)

# Where each engine's driver lives.
# When the import fails the orchestrator skips the engine honestly rather
# than synthesising a misleading row.
_FIT_DRIVER_MODULES: dict[str, tuple[str, str]] = {
    "simscape": (
        "src.engines.physics_engines.simscape.fit_swing_simscape",
        "fit_swing_simscape",
    ),
    "mujoco": (
        "src.engines.physics_engines.mujoco.python.motion_matching.fit_swing",
        "fit_swing_mujoco",
    ),
    "drake": (
        "src.engines.physics_engines.drake.python.motion_matching.fit_swing",
        "fit_swing_drake",
    ),
    "pinocchio": (
        "src.engines.physics_engines.pinocchio.python.motion_matching.fit_swing",
        "fit_swing_pinocchio",
    ),
    "opensim": (
        "src.engines.physics_engines.opensim.python.motion_matching.fit_swing",
        "fit_swing_opensim",
    ),
    "myosuite": (
        "src.engines.physics_engines.myosuite.python.motion_matching.provider",
        "MyoSuiteFitSwingProvider",
    ),
    "pendulum": (
        "src.engines.physics_engines.pendulum.python.motion_matching.provider",
        "PendulumFitSwingProvider",
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
        "--trials",
        type=str,
        default=None,
        help="Path to trials directory or trial fixture file.",
    )
    p.add_argument(
        "--out",
        type=Path,
        default=None,
        help="Base path for output leaderboard files (.md and .json).",
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


def _load_target(trial: str, trials_dir: Path | None = None) -> Any:
    """Load the canonical ``ClubTarget`` for ``trial`` from the Wiffle xlsx or a fixture.

    Raises ``ImportError`` or ``FileNotFoundError`` if the loader / data
    aren't available; the caller treats those as honest skips.
    """
    if trials_dir is not None and trials_dir.is_dir():
        fixture_json = trials_dir / f"{trial}.json"
        if fixture_json.exists():
            import numpy as np
            from src.shared.python.motion_matching.club_target import (
                ClubTarget,
                SourceProvenance,
            )

            data = json.loads(fixture_json.read_text(encoding="utf-8"))
            t = np.array(data["time"], dtype=np.float64)
            t_adj = t - t[0]
            n = len(t_adj)
            impact = min(n, max(1, int(data.get("impact_idx", n))))
            return ClubTarget(
                time=t_adj,
                butt=np.array(data["butt"], dtype=np.float64),
                clubhead=np.array(data["clubhead"], dtype=np.float64),
                club_quat=np.array(data["club_quat"], dtype=np.float64),
                impact_idx=impact,
                source=SourceProvenance(
                    filename=data.get("source_filename", fixture_json.name),
                    format="synthetic",
                    subject_id="fixture",
                    trial_id=trial,
                    sha256="fixture",
                ),
            )

    if not WIFFLE_XLSX.exists():
        raise FileNotFoundError(f"canonical Wiffle xlsx not found: {WIFFLE_XLSX}")
    # Late import: avoid forcing pandas / openpyxl install on report-only runs.
    from src.shared.python.motion_matching import AlignOptions, load_club_target_excel

    return load_club_target_excel(WIFFLE_XLSX, sheet=trial, opts=AlignOptions())


def _load_fit_driver(engine: str) -> Any:
    """Obtain a callable or provider for ``engine``, or None if unavailable."""
    if engine in KNOWN_UNAVAILABLE:
        return None

    # First try the canonical provider registry
    try:
        from src.shared.python.motion_matching.provider import get_provider

        return get_provider(engine)
    except (KeyError, ImportError):
        pass
    except Exception as exc:  # noqa: BLE001
        LOGGER.info("engine %s: provider lookup failed (%s)", engine, exc)

    # If not registered yet, try importing the module from _FIT_DRIVER_MODULES
    if engine in _FIT_DRIVER_MODULES:
        module_path, attr = _FIT_DRIVER_MODULES[engine]
        try:
            mod = importlib.import_module(module_path)
            try:
                from src.shared.python.motion_matching.provider import get_provider

                return get_provider(engine)
            except (KeyError, ImportError):
                pass
            return getattr(mod, attr, None)
        except ImportError as exc:
            LOGGER.info(
                "engine %s: fit driver not importable (%s) - skipping", engine, exc
            )
            return None
    return None


def _coerce_to_dict(fit_result: Any) -> dict[str, Any]:
    """Normalise a per-engine ``FitResult`` into a plain dict."""
    out: dict[str, Any] = {}

    def _get(name: str, default: Any = None) -> Any:
        if isinstance(fit_result, dict):
            return fit_result.get(name, default)
        return getattr(fit_result, name, default)

    attrs = (
        "engine",
        "trial",
        "grip_rmse_mm",
        "clubhead_rmse_mm",
        "body_marker_rmse_mm",
        "total_work_J",
        "wall_clock_s",
        "commit",
        "run_at",
    )
    for name in attrs:
        val = _get(name)
        if val is not None and not hasattr(val, "shape"):
            out[name] = val

    # Method / solver name
    method = _get("method") or _get("solver")
    if method is not None:
        out["solver"] = str(method)

    # Bridge CanonicalFitResult attributes to LeaderboardRow columns
    if "grip_rmse_mm" not in out or out["grip_rmse_mm"] is None:
        final_rmse = _get("final_rmse_m")
        if final_rmse is not None:
            out["grip_rmse_mm"] = float(final_rmse) * 1000.0
            if "clubhead_rmse_mm" not in out or out["clubhead_rmse_mm"] is None:
                out["clubhead_rmse_mm"] = float(final_rmse) * 1000.0

    if "total_work_J" not in out or out["total_work_J"] is None:
        work = _get("final_total_work_J")
        out["total_work_J"] = float(work) if work is not None else 0.0

    if "body_marker_rmse_mm" not in out or out["body_marker_rmse_mm"] is None:
        out["body_marker_rmse_mm"] = 0.0

    if "commit" not in out or not out["commit"]:
        commit = _get("git_commit")
        if commit:
            out["commit"] = str(commit)

    if "run_at" not in out or not out["run_at"]:
        run_at = _get("timestamp_utc")
        if run_at:
            out["run_at"] = str(run_at)

    if "run_at" in out and isinstance(out["run_at"], str):
        if out["run_at"].endswith("+00:00"):
            out["run_at"] = out["run_at"][:-6] + "Z"

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


def _handle_unavailable_engine(
    trial: str,
    engine: str,
    results_dir: Path,
) -> str:
    """Write an explicit unavailable payload for an engine with known missing runtime."""
    reason = KNOWN_UNAVAILABLE[engine]
    payload = {
        "trial": trial,
        "engine": engine,
        "solver": f"unavailable: {reason}",
        "status": f"unavailable: {reason}",
        "grip_rmse_mm": None,
        "clubhead_rmse_mm": None,
        "body_marker_rmse_mm": None,
        "total_work_J": None,
        "wall_clock_s": 0.0,
        "commit": _git_commit(),
        "run_at": _now_iso(),
    }
    _write_engine_json(results_dir, trial, engine, payload)
    return "unavailable"


def run_one_engine(
    trial: str,
    engine: str,
    target: Any,
    results_dir: Path,
    strict: bool,
    json_path: Path | None = None,
) -> str:
    """Run a single (trial, engine) cell. Returns one of ``ok``, ``skip``,
    ``error``, ``unavailable``.
    """
    if engine in KNOWN_UNAVAILABLE:
        return _handle_unavailable_engine(trial, engine, results_dir)

    driver = _load_fit_driver(engine)
    if driver is None:
        return "skip"
    t0 = time.perf_counter()
    try:
        if hasattr(driver, "fit_swing"):
            from src.shared.python.motion_matching.provider import FitOptions

            if (
                engine == "mujoco"
                and hasattr(target, "time")
                and target.time.shape[0] != 301
            ):
                from src.engines.physics_engines.mujoco.python.motion_matching.fit_swing import (
                    FitOptions as MujocoFitOptions,
                    SimOptions,
                )

                dt = (
                    float(target.time[1] - target.time[0])
                    if len(target.time) > 1
                    else 0.001
                )
                rate = round(1.0 / dt) if dt > 0 else 1000
                m_opts = MujocoFitOptions(
                    sim=SimOptions(T_s=float(target.time[-1]), output_rate_hz=rate)
                )
                opts = FitOptions(maxiter=200, engine_options=m_opts)
            else:
                opts = FitOptions(maxiter=200)
            result = driver.fit_swing(target, opts)
        else:
            result = driver(target)
    except (NotImplementedError, ModuleNotFoundError, ImportError) as exc:
        LOGGER.info(
            "engine %s for trial %s: not available or not implemented (%s) - skipping",
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

    if json_path is not None:
        try:
            from src.shared.python.motion_matching.leaderboard import append_row

            ver = (
                driver.engine_version()
                if hasattr(driver, "engine_version")
                else "unknown"
            )
            append_row(engine, result, ver, json_path=json_path, target_id=trial)
        except Exception as exc:  # noqa: BLE001
            LOGGER.warning("Could not append to JSON leaderboard: %s", exc)

    LOGGER.info("engine %s for trial %s: ok (%.3fs)", engine, trial, elapsed)
    return "ok"


def run_all(
    args: argparse.Namespace,
    json_path: Path | None = None,
) -> dict[str, dict[str, str]]:
    """Drive every (trial, engine) cell. Returns a status grid suitable
    for printing a summary at the end.
    """
    trials_dir: Path | None = None
    if getattr(args, "trials", None):
        p = Path(args.trials)
        if p.is_dir():
            trials_dir = p
            if not args.trial:
                json_files = sorted(p.glob("*.json"))
                if json_files:
                    args.trial = [f.stem for f in json_files]

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
            target = _load_target(trial, trials_dir=trials_dir)
        except (ImportError, FileNotFoundError, KeyError, ValueError) as exc:
            LOGGER.warning(
                "trial %s: target unavailable (%s) - skipping all engines", trial, exc
            )
            continue
        for engine in engines:
            grid[trial][engine] = run_one_engine(
                trial,
                engine,
                target,
                args.results_dir,
                args.strict,
                json_path=json_path,
            )
    return grid


# --- Main --------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )

    json_path: Path | None = None
    if getattr(args, "out", None):
        out_base = Path(args.out)
        if out_base.suffix == ".md":
            args.leaderboard_path = out_base
            json_path = out_base.with_suffix(".json")
        elif out_base.suffix == ".json":
            json_path = out_base
            args.leaderboard_path = out_base.with_suffix(".md")
        else:
            args.leaderboard_path = out_base.with_suffix(".md")
            json_path = out_base.with_suffix(".json")
    else:
        env_json = os.environ.get("UD_LEADERBOARD_JSON_PATH", "").strip()
        if env_json:
            json_path = Path(env_json)

    if not args.results_dir.exists():
        args.results_dir.mkdir(parents=True, exist_ok=True)

    grid = run_all(args, json_path=json_path)

    generate_report = _load_generate_report()
    out = generate_report(args.results_dir, args.leaderboard_path)
    LOGGER.info("leaderboard written: %s", out)

    if json_path is not None and not json_path.exists():
        json_path.parent.mkdir(parents=True, exist_ok=True)
        json_path.write_text("[]\n", encoding="utf-8")

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

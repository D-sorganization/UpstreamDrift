"""Cross-engine motion-matching leaderboard.

Aggregates per-engine ``FitResult`` JSON files emitted by
``fit_swing_<engine>(target)`` drivers into a single Markdown comparison
table per :doc:`VISUALIZATION_SPEC.md` "Comparison across options".

The on-disk layout consumed by :func:`generate_report` is:

    <results_dir>/
        <trial>/
            <engine>.json       -- one FitResult per (trial, engine)

A ``FitResult`` JSON document is the canonical schema described in
``CROSS_ENGINE_PARITY_SPEC.md`` §2.4 / §2.8 with at minimum these fields:

    {
        "engine":            str,    # "simscape" | "mujoco" | "drake"
                                     # | "pinocchio" | "opensim"
        "solver":            str,    # e.g. "fmincon-sqp+ms8", "ipopt", "lm"
        "trial":             str,    # e.g. "TW_ProV1"
        "grip_rmse_mm":      float,  # >= 0
        "clubhead_rmse_mm":  float,  # >= 0
        "total_work_J":      float,  # >= 0 (regularised effort, summed)
        "wall_clock_s":      float,  # >= 0
        "commit":            str,    # 7-40 hex chars; short or full git SHA
        "run_at":            str,    # ISO-8601 UTC, "...Z"
    }

Additional fields are tolerated and ignored — engines emit richer payloads
(``coefficients``, ``solver_options``, ``n_iterations`` ...) but only the
columns above show up in the leaderboard.

Public API
----------
    LeaderboardRow      -- frozen dataclass; the leaderboard row schema.
    LeaderboardError    -- raised on schema violations or unreadable input.
    load_results        -- read every ``<trial>/<engine>.json`` under a dir.
    render_markdown     -- format a list of FitResults as a Markdown table.
    generate_report     -- end-to-end: read directory -> write ``.md`` file.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

__all__ = [
    "FitResult",
    "LeaderboardRow",
    "LeaderboardError",
    "load_results",
    "render_markdown",
    "generate_report",
    "append_row",
    "maybe_append_row",
    "sync_leaderboard_from_ledger",
    "rows_from_parity_report",
    "sync_leaderboard_from_parity_report",
    "JSON_LEADERBOARD_COLUMNS",
    "default_json_path",
    "valid_engines",
]

# --- Schema ------------------------------------------------------------------

_VALID_ENGINES: frozenset[str] = frozenset(
    {
        "simscape",
        "mujoco",
        "drake",
        "pinocchio",
        "opensim",
        "myosuite",
        "pendulum",
    }
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")


def valid_engines() -> frozenset[str]:
    """Return the set of recognized engine identifiers for the leaderboard."""
    return _VALID_ENGINES


# Canonical column order for the Markdown table (matches issue #4097 spec).
_COLUMNS: tuple[str, ...] = (
    "engine",
    "solver",
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "body_marker_rmse_mm",
    "total_work_J",
    "wall_clock_s",
    "commit",
    "run_at",
)
_REQUIRED_FIELDS: tuple[str, ...] = ("trial", *_COLUMNS)
_NONNEG_FIELDS: tuple[str, ...] = (
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "body_marker_rmse_mm",
    "total_work_J",
    "wall_clock_s",
)


class LeaderboardError(ValueError):
    """Raised when a ``FitResult`` JSON file is malformed or the directory
    layout does not match ``<trial>/<engine>.json``.
    """


def _validate_strings(record: LeaderboardRow) -> None:
    """Trial / engine / solver string fields must be present and recognised."""
    if not isinstance(record.trial, str) or not record.trial:
        raise LeaderboardError(
            f"trial must be a non-empty string, got {record.trial!r}"
        )
    if record.engine not in _VALID_ENGINES:
        raise LeaderboardError(
            f"engine must be one of {sorted(_VALID_ENGINES)}, got {record.engine!r}"
        )
    if not isinstance(record.solver, str) or not record.solver:
        raise LeaderboardError(
            f"solver must be a non-empty string, got {record.solver!r}"
        )


def _validate_numbers(record: LeaderboardRow) -> None:
    """Numeric fields must be finite and non-negative (or None if unavailable)."""
    if record.solver.startswith("unavailable"):
        return
    for name in _NONNEG_FIELDS:
        value = getattr(record, name)
        if value is None:
            continue
        if not isinstance(value, (int, float)) or isinstance(value, bool):
            raise LeaderboardError(f"{name} must be a number, got {value!r}")
        if value < 0:
            raise LeaderboardError(f"{name} must be >= 0, got {value!r}")


def _validate_commit(commit: str) -> None:
    if not _COMMIT_RE.match(commit):
        raise LeaderboardError(
            f"commit must be 7-40 lowercase hex chars, got {commit!r}"
        )


def _validate_timestamp(run_at: str) -> None:
    if not _ISO8601_RE.match(run_at):
        raise LeaderboardError(
            f"run_at must be ISO-8601 UTC ending in 'Z', got {run_at!r}"
        )
    # Cheap final sanity check: parse the timestamp.
    try:
        datetime.strptime(run_at, "%Y-%m-%dT%H:%M:%SZ")
        return
    except ValueError:
        pass
    try:
        datetime.strptime(run_at, "%Y-%m-%dT%H:%M:%S.%fZ")
    except ValueError as exc:
        raise LeaderboardError(f"run_at unparseable as ISO-8601: {run_at!r}") from exc


@dataclass(frozen=True)
class LeaderboardRow:
    """Cross-engine leaderboard row.

    Mirrors the per-engine ``fit_swing_<engine>`` JSON contract from
    CROSS_ENGINE_PARITY_SPEC.md §2.8. All fields are required and validated
    in :meth:`__post_init__`.
    """

    trial: str
    engine: str
    solver: str
    grip_rmse_mm: float | None
    clubhead_rmse_mm: float | None
    body_marker_rmse_mm: float | None
    total_work_J: float | None
    wall_clock_s: float | None
    commit: str
    run_at: str

    def __post_init__(self) -> None:
        _validate_strings(self)
        _validate_numbers(self)
        _validate_commit(self.commit)
        _validate_timestamp(self.run_at)

    @classmethod
    def from_dict(cls, data: dict, trial: str) -> LeaderboardRow:
        """Build a leaderboard row from a parsed JSON dict.

        ``trial`` is supplied separately because most fit drivers know
        which trial they ran without re-stating it; if the JSON does name
        ``trial`` and it disagrees with the directory name, that is a
        :class:`LeaderboardError`.
        """
        if not isinstance(data, dict):
            raise LeaderboardError(f"expected JSON object, got {type(data).__name__}")
        if "trial" in data and data["trial"] != trial:
            raise LeaderboardError(
                f"trial mismatch: directory says {trial!r}, payload says {data['trial']!r}"
            )
        kwargs = {name: data.get(name) for name in _REQUIRED_FIELDS}
        kwargs["trial"] = trial
        is_unavail = str(data.get("solver", "")).startswith("unavailable") or str(
            data.get("status", "")
        ).startswith("unavailable")
        missing = [
            name
            for name, v in kwargs.items()
            if v is None and (not is_unavail or name not in _NONNEG_FIELDS)
        ]
        if missing:
            raise LeaderboardError(
                f"missing required field(s) {sorted(missing)} in leaderboard JSON"
            )
        return cls(**kwargs)  # type: ignore[arg-type]

    def as_row(self) -> dict[str, str]:
        """Return a stringly-typed mapping for the Markdown row."""
        out: dict[str, str] = {}
        for name in _COLUMNS:
            value = getattr(self, name)
            if value is None:
                out[name] = "-"
            elif isinstance(value, float):
                out[name] = f"{value:.3f}"
            else:
                out[name] = str(value)
        return out


FitResult = LeaderboardRow


# --- I/O ---------------------------------------------------------------------


def load_results(results_dir: Path) -> dict[str, list[FitResult]]:
    """Read every ``<trial>/<engine>.json`` under ``results_dir``.

    Returns a mapping ``trial -> [FitResult, ...]``. Files whose basename
    is not a recognised engine are skipped silently — the leaderboard does
    not police that subdirectory for unrelated artefacts. JSON parse errors
    or schema violations raise :class:`LeaderboardError` so callers see the
    failure rather than a silently-incomplete table.
    """
    if not isinstance(results_dir, Path):
        raise TypeError(f"results_dir must be a Path, got {type(results_dir).__name__}")
    out: dict[str, list[FitResult]] = {}
    if not results_dir.exists():
        return out
    if not results_dir.is_dir():
        raise LeaderboardError(f"{results_dir} is not a directory")

    for trial_dir in sorted(results_dir.iterdir()):
        if not trial_dir.is_dir():
            continue
        rows = _load_trial_dir(trial_dir)
        if rows:
            out[trial_dir.name] = rows
    return out


def _load_trial_dir(trial_dir: Path) -> list[FitResult]:
    """Read every recognised ``<engine>.json`` under one trial directory."""
    trial = trial_dir.name
    rows: list[FitResult] = []
    for engine_file in sorted(trial_dir.glob("*.json")):
        engine = engine_file.stem
        if engine not in _VALID_ENGINES:
            # Tolerate sibling artefacts; the layout is conventional, not
            # exclusive.
            continue
        try:
            payload = json.loads(engine_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LeaderboardError(f"could not parse {engine_file}: {exc}") from exc
        # Inject the engine if the JSON does not name itself; this lets
        # writers be lazy.
        if isinstance(payload, dict):
            payload.setdefault("engine", engine)
        rows.append(FitResult.from_dict(payload, trial=trial))
    return rows


# --- Rendering ---------------------------------------------------------------


def _format_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[str]:
    """Format a Markdown table with column-width alignment."""
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(row[col]))
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    body = [
        "| " + " | ".join(row[col].ljust(widths[col]) for col in columns) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def render_markdown(results: dict[str, list[FitResult]]) -> str:
    """Render the full leaderboard Markdown.

    Sections are sorted by trial name (alphabetical). Within a trial, rows
    are sorted by ``grip_rmse_mm`` ascending so the most accurate engine
    appears first. The output is deterministic — same input, same bytes —
    so the file is safe to commit and diff across PRs.
    """
    lines: list[str] = ["# Cross-engine leaderboard", ""]
    if not results:
        lines.append(
            "_No FitResult JSON files found. Engines that have not yet "
            "implemented `fit_swing_<engine>` are honestly skipped._"
        )
        lines.append("")
        return "\n".join(lines)

    lines.append(
        "Sorted by `grip_rmse_mm` ascending within each trial; lower is better."
    )
    lines.append("")

    for trial in sorted(results.keys()):
        rows = sorted(
            results[trial],
            key=lambda r: (
                float("inf")
                if r.grip_rmse_mm is None
                or (isinstance(r.grip_rmse_mm, float) and math.isnan(r.grip_rmse_mm))
                else r.grip_rmse_mm
            ),
        )
        lines.append(f"## {trial}")
        lines.append("")
        lines.extend(_format_table([r.as_row() for r in rows], _COLUMNS))
        lines.append("")
    return "\n".join(lines)


def generate_report(results_dir: Path, output_path: Path) -> Path:
    """Read every FitResult under ``results_dir`` and write a Markdown
    leaderboard to ``output_path``.

    Returns the absolute path of the written file. The parent directory of
    ``output_path`` is created if it does not exist. The output is
    deterministic — same inputs always produce the same bytes — so the
    leaderboard file is safe to commit.
    """
    if not isinstance(results_dir, Path):
        raise TypeError(f"results_dir must be a Path, got {type(results_dir).__name__}")
    if not isinstance(output_path, Path):
        raise TypeError(f"output_path must be a Path, got {type(output_path).__name__}")
    results = load_results(results_dir)
    text = render_markdown(results)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(
        text + ("" if text.endswith("\n") else "\n"), encoding="utf-8"
    )
    return output_path.resolve()


# --- JSON leaderboard writer (issue #4713) -----------------------------------

JSON_LEADERBOARD_COLUMNS: tuple[str, ...] = (
    "engine",
    "engine_version",
    "target_id",
    "theta",
    "residual_rms",
    "body_marker_rms",
    "wallclock",
    "commit_sha",
)


def default_json_path() -> Path:
    """Return the canonical path for ``cross_engine_leaderboard.json``."""
    env = os.environ.get("UD_LEADERBOARD_JSON_PATH", "").strip()
    if env:
        return Path(env)
    here = Path(__file__).resolve()
    for parent in here.parents:
        if (parent / ".git").exists() or (
            (parent / "docs").is_dir()
            and (parent / "src").is_dir()
            and (parent / "pyproject.toml").is_file()
        ):
            return parent / "reports" / "cross_engine_leaderboard.json"
    return Path("reports") / "cross_engine_leaderboard.json"


def _short_commit(commit: str | None) -> str:
    if not commit:
        return "0000000"
    s = str(commit).strip().lower()
    if not re.match(r"^[0-9a-f]{7,40}$", s):
        return hashlib.sha256(s.encode("utf-8"), usedforsecurity=False).hexdigest()[:12]
    return s


def _coerce_theta(value: Any) -> list[float]:
    """Normalise a theta vector into a JSON-serialisable list of floats."""
    if value is None:
        return []
    if hasattr(value, "tolist"):
        value = value.tolist()
    if not isinstance(value, (list, tuple)):
        raise LeaderboardError(f"theta must be a vector, got {type(value).__name__}")
    out: list[float] = []
    for i, x in enumerate(value):
        try:
            f = float(x)
        except (TypeError, ValueError) as exc:
            raise LeaderboardError(f"theta[{i}] not numeric: {x!r}") from exc
        if f != f:
            raise LeaderboardError(f"theta[{i}] is NaN")
        out.append(f)
    return out


def _row_from_fit_result(
    engine: str,
    fit_result: Any,
    engine_version: str,
    target_id: str | None,
) -> dict[str, Any]:
    if not isinstance(engine, str) or not engine:
        raise LeaderboardError(f"engine must be a non-empty string, got {engine!r}")
    if engine not in _VALID_ENGINES:
        raise LeaderboardError(
            f"engine must be one of {sorted(_VALID_ENGINES)}, got {engine!r}"
        )
    if not isinstance(engine_version, str) or not engine_version:
        engine_version = "unknown"

    def _attr(*names: str, default: Any = None) -> Any:
        if isinstance(fit_result, dict):
            for n in names:
                if n in fit_result:
                    return fit_result[n]
            return default
        for n in names:
            if hasattr(fit_result, n):
                return getattr(fit_result, n)
        return default

    theta = _coerce_theta(_attr("theta_optimal", "theta", "coefficients"))
    residual_rms = _attr("final_rmse_m", "residual_rms", "grip_rmse_mm")
    if residual_rms is None:
        raise LeaderboardError(
            "fit_result must expose final_rmse_m / residual_rms / grip_rmse_mm"
        )
    residual_rms = float(residual_rms)
    if residual_rms < 0 or residual_rms != residual_rms:
        raise LeaderboardError(
            f"residual_rms must be a finite non-negative float, got {residual_rms!r}"
        )

    wallclock = _attr("wall_clock_s", "wallclock", default=0.0)
    wallclock = float(wallclock)
    if wallclock < 0 or wallclock != wallclock:
        raise LeaderboardError(
            f"wallclock must be a finite non-negative float, got {wallclock!r}"
        )

    commit = _attr("git_commit", "commit_sha", "commit", default=None)
    target = (
        target_id
        if target_id is not None
        else _attr("target_id", "target_hash", "trial", "trial_id", default="unknown")
    )
    target = str(target).strip() or "unknown"

    run_at = _attr(
        "timestamp_utc",
        "run_at",
        default=datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
    )
    run_at_str = str(run_at)
    if run_at_str.endswith("+00:00"):
        run_at_str = run_at_str[:-6] + "Z"
    solver = _attr("method", "solver", default="unknown")
    iterations = _attr("iterations", "n_iterations", default=0)

    return {
        "engine": engine,
        "engine_version": str(engine_version),
        "target_id": target,
        "theta": theta,
        "residual_rms": residual_rms,
        "body_marker_rms": float(
            _attr("body_marker_rms", "body_marker_rmse_mm", default=0.0)
        ),
        "wallclock": wallclock,
        "commit_sha": _short_commit(commit),
        "run_at": run_at_str,
        "solver": str(solver),
        "iterations": int(iterations) if iterations is not None else 0,
    }


def append_row(
    engine: str,
    fit_result: Any,
    engine_version: str,
    *,
    json_path: Path | None = None,
    target_id: str | None = None,
) -> Path:
    """Append one fit-result row to ``cross_engine_leaderboard.json``.

    Required schema (issue #4713 acceptance criteria):
        engine, engine_version, target_id, theta, residual_rms,
        wallclock, commit_sha

    Optional diagnostic columns ``run_at``, ``solver``, ``iterations``
    are also stamped when available.
    """
    path = json_path if json_path is not None else default_json_path()
    if not isinstance(path, Path):
        raise TypeError(f"json_path must be a Path, got {type(path).__name__}")

    row = _row_from_fit_result(engine, fit_result, engine_version, target_id)

    if path.exists():
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LeaderboardError(
                f"existing leaderboard JSON at {path} is not valid JSON: {exc}"
            ) from exc
        if not isinstance(data, list):
            raise LeaderboardError(
                f"existing leaderboard JSON at {path} must be a list, "
                f"got {type(data).__name__}"
            )
        existing = data
    else:
        existing = []

    existing.append(row)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(existing, indent=2, sort_keys=False) + "\n",
        encoding="utf-8",
    )
    return path.resolve()


def maybe_append_row(
    engine: str,
    fit_result: Any,
    engine_version: str,
    *,
    json_path: Path | None = None,
    target_id: str | None = None,
) -> Path | None:
    """Call :func:`append_row` only when ``UD_LEADERBOARD_PUBLISH=1``.

    Providers wire this into canonical ``fit_swing`` paths so CI runs
    accumulate ``reports/cross_engine_leaderboard.json`` for free, while
    every other consumer stays unaffected. Failures are demoted to
    warnings so a leaderboard hiccup never breaks a fit.
    """
    if os.environ.get("UD_LEADERBOARD_PUBLISH", "").strip() != "1":
        return None
    try:
        return append_row(
            engine,
            fit_result,
            engine_version,
            json_path=json_path,
            target_id=target_id,
        )
    except Exception as exc:  # noqa: BLE001 - never break the fit
        import logging as _logging

        _logging.getLogger(__name__).warning(
            "leaderboard.append_row failed (engine=%s): %s", engine, exc
        )
        return None


def sync_leaderboard_from_ledger(
    ledger_rows: Any,
    *,
    json_path: Path | None = None,
) -> Path:
    """Populate cross_engine_leaderboard.json from ledger rows unconditionally."""
    path = json_path if json_path is not None else default_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_out: list[dict[str, Any]] = []
    for row in ledger_rows:
        eng = getattr(row, "engine", None)
        if eng not in _VALID_ENGINES:
            continue
        metrics = getattr(row, "metrics", None)
        whole_rmse = getattr(metrics, "whole_marker_rmse_m", None) if metrics else None
        if whole_rmse is None:
            continue
        cand_sha = getattr(row, "candidate_sha", None) or "unknown"
        capture = getattr(row, "capture", None) or "unknown"
        run_dict = {
            "engine": eng,
            "engine_version": "unknown",
            "target_id": capture,
            "theta": [],
            "residual_rms": float(whole_rmse),
            "body_marker_rms": float(whole_rmse),
            "wallclock": float(getattr(row, "horizon_s", None) or 0.0),
            "commit_sha": _short_commit(cand_sha),
            "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "solver": "unknown",
            "iterations": 0,
        }
        rows_out.append(run_dict)
    path.write_text(
        json.dumps(rows_out, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return path.resolve()


def rows_from_parity_report(report: Any) -> list[dict[str, Any]]:
    """Convert UnifiedParityReport engine rows into leaderboard JSON row dictionaries."""
    rows_out: list[dict[str, Any]] = []
    cand_id = getattr(report, "candidate_id", "unknown")
    cand_sha = getattr(report, "candidate_sha256", "unknown")
    engine_rows = getattr(report, "engine_rows", {})
    for eng, row in engine_rows.items():
        if eng not in _VALID_ENGINES:
            continue
        status = getattr(row, "status", "")
        if status == "unavailable":
            continue
        metrics = getattr(row, "shared_metrics", None) or {}
        whole_rmse = metrics.get("whole_marker_rmse_m", 0.0)
        pt_diffs = getattr(row, "pointwise_differences", {})
        if not whole_rmse and "marker_diff_m" in pt_diffs:
            whole_rmse = getattr(pt_diffs["marker_diff_m"], "rms_diff", 0.0)
        run_dict = {
            "engine": eng,
            "engine_version": getattr(row, "model_sha256", "unknown") or "unknown",
            "target_id": cand_id,
            "theta": [],
            "residual_rms": float(whole_rmse),
            "body_marker_rms": float(whole_rmse),
            "total_work_J": float(getattr(row, "total_work_J", 0.0) or 0.0),
            "wallclock": float(getattr(row, "wall_clock_s", 0.0) or 0.0),
            "commit_sha": _short_commit(cand_sha),
            "run_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
            "solver": "matching_plant",
            "iterations": 0,
        }
        rows_out.append(run_dict)
    return rows_out


def sync_leaderboard_from_parity_report(
    report: Any,
    *,
    json_path: Path | None = None,
) -> Path:
    """Populate cross_engine_leaderboard.json from a UnifiedParityReport."""
    path = json_path if json_path is not None else default_json_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    rows_out = rows_from_parity_report(report)
    path.write_text(
        json.dumps(rows_out, indent=2, sort_keys=False) + "\n", encoding="utf-8"
    )
    return path.resolve()


# --- Module-level metadata ---------------------------------------------------

# Re-export for documentation / introspection.
COLUMNS: tuple[str, ...] = _COLUMNS
SUPPORTED_ENGINES: frozenset[str] = _VALID_ENGINES
SCHEMA_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FitResult))

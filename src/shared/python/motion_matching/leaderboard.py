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
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, fields
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.shared.python.contracts import require

__all__ = [
    "FitResult",
    "Leaderboard",
    "LeaderboardRow",
    "LeaderboardError",
    "group_by_capture",
    "load_results",
    "render_markdown",
    "render_per_capture_markdown",
    "render_side_by_side_table",
    "generate_report",
    "append_row",
    "maybe_append_row",
    "sync_leaderboard_from_ledger",
    "rows_from_parity_report",
    "sync_leaderboard_from_parity_report",
    "JSON_LEADERBOARD_COLUMNS",
    "DEFAULT_SIDE_BY_SIDE_METRICS",
    "PER_CAPTURE_COLUMNS",
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

# Canonical columns for per-capture Markdown table
PER_CAPTURE_COLUMNS: tuple[str, ...] = (
    "trial",
    "engine",
    "solver",
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "body_marker_rmse_mm",
    "total_work_J",
    "wall_clock_s",
    "verdict",
)

# Default metric columns for side-by-side comparison table
DEFAULT_SIDE_BY_SIDE_METRICS: tuple[str, ...] = (
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "body_marker_rmse_mm",
    "total_work_J",
    "verdict",
)


class LeaderboardError(ValueError):
    """Raised when a ``FitResult`` JSON file is malformed or the directory
    layout does not match ``<trial>/<engine>.json``.
    """


def _validate_strings(record: LeaderboardRow) -> None:
    """Trial / engine / solver / capture string fields must be present and recognised."""
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
    require(
        isinstance(record.capture, str) and bool(record.capture.strip()),
        "capture must be a non-empty string",
        record.capture,
    )
    if record.verdict is not None:
        require(
            isinstance(record.verdict, str) and bool(record.verdict.strip()),
            "verdict must be a non-empty string",
            record.verdict,
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
    capture: str = "driver"
    verdict: str | None = None

    def __post_init__(self) -> None:
        _validate_strings(self)
        _validate_numbers(self)
        _validate_commit(self.commit)
        _validate_timestamp(self.run_at)

    @classmethod
    def from_dict(
        cls,
        data: dict,
        trial: str,
        capture: str | None = None,
    ) -> LeaderboardRow:
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

        # Capture resolution: explicit param > data["capture"] > default "driver"
        if capture is not None:
            kwargs["capture"] = capture
        elif "capture" in data:
            kwargs["capture"] = data["capture"]
        else:
            kwargs["capture"] = "driver"

        if "verdict" in data:
            kwargs["verdict"] = data["verdict"]

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

    @property
    def display_verdict(self) -> str:
        """Verdict string for leaderboard presentation.

        Only an explicitly supplied ``verdict`` is ever shown here — this
        never invents a PASS/FAIL. An unavailable solver or a row with no
        grip RMSE reads as "not run"; everything else without a verdict
        reads as "-".
        """
        if self.verdict is not None and self.verdict.strip():
            return self.verdict.strip()
        if self.solver.startswith("unavailable") or self.grip_rmse_mm is None:
            return "not run"
        return "-"

    def as_row(self, columns: tuple[str, ...] | None = None) -> dict[str, str]:
        """Return a stringly-typed mapping for the Markdown row."""
        cols = columns if columns is not None else _COLUMNS
        out: dict[str, str] = {}
        for name in cols:
            if name == "verdict":
                out[name] = self.display_verdict
                continue
            if name == "capture":
                out[name] = self.capture
                continue
            value = getattr(self, name, None)
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


def _load_engine_files(
    dir_path: Path, trial: str, *, capture_default: str | None
) -> list[FitResult]:
    """Read every recognised ``<engine>.json`` file directly inside ``dir_path``.

    ``capture_default`` names the capture to assume when a JSON payload
    does not state its own ``capture`` field (``None`` for the flat
    ``<trial>/<engine>.json`` layout, where :class:`FitResult` falls back
    to "driver"; the capture subdirectory name for the nested layout).
    """
    rows: list[FitResult] = []
    for engine_file in sorted(dir_path.glob("*.json")):
        engine = engine_file.stem
        if engine not in _VALID_ENGINES:
            # Tolerate sibling artefacts; the layout is conventional, not
            # exclusive.
            continue
        try:
            payload = json.loads(engine_file.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            raise LeaderboardError(f"could not parse {engine_file}: {exc}") from exc
        if isinstance(payload, dict):
            # Inject the engine if the JSON does not name itself; this lets
            # writers be lazy.
            payload.setdefault("engine", engine)
            if capture_default is not None:
                payload.setdefault("capture", capture_default)
            cap = payload.get("capture")
        else:
            cap = capture_default
        rows.append(FitResult.from_dict(payload, trial=trial, capture=cap))
    return rows


def _load_trial_dir(trial_dir: Path) -> list[FitResult]:
    """Read every recognised ``<engine>.json`` under one trial directory.

    Supports two on-disk layouts:
        <trial>/<engine>.json            -- capture defaults to "driver"
        <trial>/<capture>/<engine>.json  -- capture named by the subdirectory
    """
    trial = trial_dir.name
    rows = _load_engine_files(trial_dir, trial, capture_default=None)
    for sub_dir in sorted(trial_dir.iterdir()):
        if sub_dir.is_dir():
            rows.extend(
                _load_engine_files(sub_dir, trial, capture_default=sub_dir.name)
            )
    return rows


# --- Grouping and Leaderboard Data Model --------------------------------------


def group_by_capture(
    results: Sequence[LeaderboardRow] | Mapping[str, Sequence[LeaderboardRow]],
) -> dict[str, list[LeaderboardRow]]:
    """Group leaderboard rows by capture name.

    Preconditions:
        - Every row must have a non-empty capture string.
        - No duplicate (trial, engine, capture) rows within the row set.
          Two trials of the same engine and capture are legitimate and are
          both kept.

    Args:
        results: A sequence of LeaderboardRow objects or a mapping of
            trial/key -> sequence of LeaderboardRow objects.

    Returns:
        Mapping of capture name -> list of LeaderboardRow objects.
    """
    flat_rows: list[LeaderboardRow] = []
    if isinstance(results, Mapping):
        for trial_rows in results.values():
            flat_rows.extend(trial_rows)
    else:
        flat_rows.extend(results)

    seen_keys: set[tuple[str, str, str]] = set()
    grouped: dict[str, list[LeaderboardRow]] = {}

    for row in flat_rows:
        require(
            isinstance(row.capture, str) and bool(row.capture.strip()),
            "capture name must be a non-empty string",
            row.capture,
        )
        key = (row.trial, row.engine, row.capture)
        require(
            key not in seen_keys,
            f"duplicate (trial, engine, capture) row: {key}",
            key,
        )
        seen_keys.add(key)
        grouped.setdefault(row.capture, []).append(row)

    return grouped


class Leaderboard:
    """Leaderboard data model grouped by capture."""

    def __init__(
        self,
        rows: Sequence[LeaderboardRow]
        | Mapping[str, Sequence[LeaderboardRow]]
        | None = None,
    ) -> None:
        self._rows: list[LeaderboardRow] = []
        self._by_capture: dict[str, list[LeaderboardRow]] = {}
        if rows:
            self._by_capture = group_by_capture(rows)
            for cap_rows in self._by_capture.values():
                self._rows.extend(cap_rows)

    @classmethod
    def from_dir(cls, results_dir: Path) -> Leaderboard:
        """Load all results from a directory into a Leaderboard instance."""
        results = load_results(results_dir)
        return cls(results)

    @property
    def rows(self) -> list[LeaderboardRow]:
        return list(self._rows)

    @property
    def captures(self) -> list[str]:
        return sorted(self._by_capture.keys())

    @property
    def by_capture(self) -> dict[str, list[LeaderboardRow]]:
        return {k: list(v) for k, v in self._by_capture.items()}

    def get_capture_rows(self, capture: str) -> list[LeaderboardRow]:
        require(
            isinstance(capture, str) and bool(capture.strip()),
            "capture name must be a non-empty string",
            capture,
        )
        return list(self._by_capture.get(capture, []))

    def filter_by_capture(self, capture: str | None) -> Leaderboard:
        if capture is None or capture.strip().lower() == "all":
            return Leaderboard(self._rows)
        return Leaderboard(self.get_capture_rows(capture.strip()))

    def render_per_capture_markdown(self, capture: str | None = None) -> str:
        lb = self.filter_by_capture(capture)
        return render_per_capture_markdown(lb)

    def render_side_by_side_table(
        self,
        metrics: tuple[str, ...] | None = None,
    ) -> str:
        return render_side_by_side_table(self, metrics=metrics)

    def render_markdown(self, capture: str | None = None) -> str:
        lb = self.filter_by_capture(capture)
        return render_markdown(lb)


# --- Rendering ---------------------------------------------------------------


def _format_table(rows: list[dict[str, str]], columns: tuple[str, ...]) -> list[str]:
    """Format a Markdown table with column-width alignment."""
    widths = {col: len(col) for col in columns}
    for row in rows:
        for col in columns:
            widths[col] = max(widths[col], len(row.get(col, "")))
    header = "| " + " | ".join(col.ljust(widths[col]) for col in columns) + " |"
    sep = "| " + " | ".join("-" * widths[col] for col in columns) + " |"
    body = [
        "| " + " | ".join(row.get(col, "").ljust(widths[col]) for col in columns) + " |"
        for row in rows
    ]
    return [header, sep, *body]


def render_side_by_side_table(
    results: Sequence[LeaderboardRow]
    | Mapping[str, Sequence[LeaderboardRow]]
    | Leaderboard,
    *,
    metrics: tuple[str, ...] | None = None,
) -> str:
    """Render a side-by-side comparison table across captures.

    One row per (trial, engine) pair, one column group per capture. Keying
    on trial as well as engine keeps two trials of the same engine from
    collapsing into a single row. Missing results show as 'not run', never
    as zero.

    Args:
        results: Leaderboard instance, row sequence, or mapping.
        metrics: Column metric names to include for each capture.

    Returns:
        Formatted Markdown table string.
    """
    if isinstance(results, Leaderboard):
        by_capture = results.by_capture
        all_rows = results.rows
    else:
        by_capture = group_by_capture(results)
        all_rows = [r for rows in by_capture.values() for r in rows]

    if not all_rows:
        return "_No results available for side-by-side comparison._"

    metric_cols = metrics if metrics is not None else DEFAULT_SIDE_BY_SIDE_METRICS
    captures = sorted(by_capture.keys())
    trial_engine_pairs = sorted({(r.trial, r.engine) for r in all_rows})

    row_lookup: dict[tuple[str, str, str], LeaderboardRow] = {
        (r.trial, r.engine, r.capture): r for r in all_rows
    }

    columns: list[str] = ["trial", "engine"]
    for cap in captures:
        for m in metric_cols:
            columns.append(f"{cap} {m}")

    table_rows: list[dict[str, str]] = []
    for trial, eng in trial_engine_pairs:
        row_dict: dict[str, str] = {"trial": trial, "engine": eng}
        for cap in captures:
            entry = row_lookup.get((trial, eng, cap))
            for m in metric_cols:
                col_name = f"{cap} {m}"
                if entry is None:
                    row_dict[col_name] = "not run"
                elif m == "verdict":
                    row_dict[col_name] = entry.display_verdict
                else:
                    val = getattr(entry, m, None)
                    if val is None:
                        row_dict[col_name] = "not run"
                    elif isinstance(val, float):
                        row_dict[col_name] = f"{val:.3f}"
                    else:
                        row_dict[col_name] = str(val)
        table_rows.append(row_dict)

    lines = _format_table(table_rows, tuple(columns))
    return "\n".join(lines)


def render_per_capture_markdown(
    results: Sequence[LeaderboardRow]
    | Mapping[str, Sequence[LeaderboardRow]]
    | Leaderboard,
    *,
    capture: str | None = None,
) -> str:
    """Render per-capture Markdown tables and side-by-side comparison table.

    For each capture, renders one Markdown table (trial, engine, match
    metrics, verdict) with one row per (trial, engine) pair seen anywhere
    in the leaderboard — keying on trial as well as engine keeps two
    trials of the same engine from collapsing into a single row, and a
    (trial, engine) pair absent from a given capture renders as "not run"
    rather than being silently dropped. Then renders a side-by-side table
    with one row per (trial, engine) pair and one column group per
    capture. Missing results show as 'not run', never as zero.

    Args:
        results: Leaderboard, row list, or mapping of trial -> rows.
        capture: Optional capture filter.

    Returns:
        Title-cased formatted Markdown document string.
    """
    lb = results if isinstance(results, Leaderboard) else Leaderboard(results)

    if capture is not None and capture.strip().lower() != "all":
        lb = lb.filter_by_capture(capture.strip())

    captures = lb.captures
    if not captures:
        return (
            "# Cross-Engine Leaderboard per Capture\n\n"
            "_No FitResult JSON files found. Engines that have not yet "
            "implemented `fit_swing_<engine>` are honestly skipped._\n"
        )

    lines: list[str] = [
        "# Cross-Engine Leaderboard per Capture",
        "",
        "Sorted by `grip_rmse_mm` ascending within each capture; lower is better.",
        "",
    ]

    trial_engine_pairs = sorted({(r.trial, r.engine) for r in lb.rows})
    rows_by_key: dict[tuple[str, str, str], LeaderboardRow] = {
        (r.trial, r.engine, r.capture): r for r in lb.rows
    }

    def _sort_key(d: dict[str, str]) -> tuple[str, float]:
        v = d.get("grip_rmse_mm", "not run")
        try:
            grip = float("inf") if v in ("not run", "-") else float(v)
        except ValueError:
            grip = float("inf")
        return (d.get("trial", ""), grip)

    # 1. Per-capture tables
    for cap in captures:
        table_rows: list[dict[str, str]] = []
        for trial, eng in trial_engine_pairs:
            row = rows_by_key.get((trial, eng, cap))
            if row is not None:
                table_rows.append(row.as_row(PER_CAPTURE_COLUMNS))
            else:
                table_rows.append(
                    {
                        "trial": trial,
                        "engine": eng,
                        "solver": "not run",
                        "grip_rmse_mm": "not run",
                        "clubhead_rmse_mm": "not run",
                        "body_marker_rmse_mm": "not run",
                        "total_work_J": "not run",
                        "wall_clock_s": "not run",
                        "verdict": "not run",
                    }
                )

        sorted_rows = sorted(table_rows, key=_sort_key)

        lines.append(f"## Capture: {cap}")
        lines.append("")
        lines.extend(_format_table(sorted_rows, PER_CAPTURE_COLUMNS))
        lines.append("")

    # 2. Side-by-side table
    lines.append("## Side-by-Side Comparison")
    lines.append("")
    sbs_table = render_side_by_side_table(lb)
    lines.append(sbs_table)
    lines.append("")

    return "\n".join(lines)


def render_markdown(
    results: dict[str, list[FitResult]] | Sequence[FitResult] | Leaderboard,
    *,
    capture: str | None = None,
) -> str:
    """Render the full leaderboard Markdown.

    Renders the legacy trial-by-trial table when every row belongs to the
    single default capture ("driver"). Once more than one capture is
    present, delegates to :func:`render_per_capture_markdown` for the
    per-capture tables plus the side-by-side comparison table.
    """
    lb = results if isinstance(results, Leaderboard) else Leaderboard(results)
    if capture is not None and capture.strip().lower() != "all":
        lb = lb.filter_by_capture(capture.strip())

    if not lb.rows:
        lines: list[str] = ["# Cross-engine leaderboard", ""]
        lines.append(
            "_No FitResult JSON files found. Engines that have not yet "
            "implemented `fit_swing_<engine>` are honestly skipped._"
        )
        lines.append("")
        return "\n".join(lines)

    if lb.captures != ["driver"]:
        return render_per_capture_markdown(lb)

    results_dict: dict[str, list[FitResult]] = {}
    for r in lb.rows:
        results_dict.setdefault(r.trial, []).append(r)

    lines = ["# Cross-engine leaderboard", ""]
    lines.append(
        "Sorted by `grip_rmse_mm` ascending within each trial; lower is better."
    )
    lines.append("")

    for trial in sorted(results_dict.keys()):
        rows = sorted(
            results_dict[trial],
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


def generate_report(
    results_dir: Path,
    output_path: Path,
    *,
    capture: str | None = None,
) -> Path:
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
    text = render_markdown(results, capture=capture)
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

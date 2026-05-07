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
    FitResult           -- frozen dataclass; the leaderboard row schema.
    LeaderboardError    -- raised on schema violations or unreadable input.
    load_results        -- read every ``<trial>/<engine>.json`` under a dir.
    render_markdown     -- format a list of FitResults as a Markdown table.
    generate_report     -- end-to-end: read directory -> write ``.md`` file.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass, fields
from datetime import datetime
from pathlib import Path

__all__ = [
    "FitResult",
    "LeaderboardError",
    "load_results",
    "render_markdown",
    "generate_report",
]

# --- Schema ------------------------------------------------------------------

_VALID_ENGINES: frozenset[str] = frozenset(
    {"simscape", "mujoco", "drake", "pinocchio", "opensim"}
)
_COMMIT_RE = re.compile(r"^[0-9a-f]{7,40}$")
_ISO8601_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?Z$")

# Canonical column order for the Markdown table (matches issue #4097 spec).
_COLUMNS: tuple[str, ...] = (
    "engine",
    "solver",
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "total_work_J",
    "wall_clock_s",
    "commit",
    "run_at",
)
_REQUIRED_FIELDS: tuple[str, ...] = ("trial", *_COLUMNS)
_NONNEG_FIELDS: tuple[str, ...] = (
    "grip_rmse_mm",
    "clubhead_rmse_mm",
    "total_work_J",
    "wall_clock_s",
)


class LeaderboardError(ValueError):
    """Raised when a ``FitResult`` JSON file is malformed or the directory
    layout does not match ``<trial>/<engine>.json``.
    """


def _validate_strings(record: FitResult) -> None:
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


def _validate_numbers(record: FitResult) -> None:
    """Numeric fields must be finite and non-negative."""
    for name in _NONNEG_FIELDS:
        value = getattr(record, name)
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
class FitResult:
    """Cross-engine leaderboard row.

    Mirrors the per-engine ``fit_swing_<engine>`` JSON contract from
    CROSS_ENGINE_PARITY_SPEC.md §2.8. All fields are required and validated
    in :meth:`__post_init__`.
    """

    trial: str
    engine: str
    solver: str
    grip_rmse_mm: float
    clubhead_rmse_mm: float
    total_work_J: float
    wall_clock_s: float
    commit: str
    run_at: str

    def __post_init__(self) -> None:
        _validate_strings(self)
        _validate_numbers(self)
        _validate_commit(self.commit)
        _validate_timestamp(self.run_at)

    @classmethod
    def from_dict(cls, data: dict, trial: str) -> FitResult:
        """Build a FitResult from a parsed JSON dict.

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
        missing = [name for name, v in kwargs.items() if v is None]
        if missing:
            raise LeaderboardError(
                f"missing required field(s) {sorted(missing)} in FitResult JSON"
            )
        return cls(**kwargs)  # type: ignore[arg-type]

    def as_row(self) -> dict[str, str]:
        """Return a stringly-typed mapping for the Markdown row."""
        out: dict[str, str] = {}
        for name in _COLUMNS:
            value = getattr(self, name)
            if isinstance(value, float):
                out[name] = f"{value:.3f}"
            else:
                out[name] = str(value)
        return out


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
        rows = sorted(results[trial], key=lambda r: r.grip_rmse_mm)
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


# --- Module-level metadata ---------------------------------------------------

# Re-export for documentation / introspection.
COLUMNS: tuple[str, ...] = _COLUMNS
SUPPORTED_ENGINES: frozenset[str] = _VALID_ENGINES
SCHEMA_FIELDS: tuple[str, ...] = tuple(f.name for f in fields(FitResult))

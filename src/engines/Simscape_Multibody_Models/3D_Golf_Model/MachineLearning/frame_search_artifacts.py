"""Read progress and checkpoint artifacts from a frame-by-frame search run.

Used by the GUI and analysis scripts to display live progress of an overnight
MATLAB frame-by-frame torque search. The progress CSV is written
incrementally by the MATLAB runner (one row per committed frame) and is safe
to read while the run is still in progress.
"""

from __future__ import annotations

import csv
import hashlib
import json
import logging
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

LOGGER = logging.getLogger(__name__)

PROGRESS_COLUMNS = (
    "frame_idx",
    "selected_candidate",
    "score",
    "wall_clock_s",
    "timestamp",
)


@dataclass(frozen=True)
class ProgressRow:
    """One row in a frame-search progress CSV."""

    frame_idx: int
    selected_candidate: int
    score: float
    wall_clock_s: float
    timestamp: str


@dataclass(frozen=True)
class RunStatus:
    """Snapshot of a (possibly in-progress) frame-search run."""

    run_dir: Path
    rows: tuple[ProgressRow, ...]
    last_frame_idx: int
    total_wall_clock_s: float
    is_stale: bool
    has_checkpoint: bool
    manifest_sha256: str | None


def manifest_sha256(manifest_path: Path) -> str:
    """Return the SHA-256 hex digest of the manifest JSON bytes.

    The hash is computed over the manifest file's exact bytes so that
    MATLAB and Python agree on the value.
    """
    if not manifest_path.is_file():
        raise FileNotFoundError(f"Manifest not found: {manifest_path}")
    return hashlib.sha256(manifest_path.read_bytes()).hexdigest()


def write_progress_header(progress_csv: Path) -> None:
    """Create a fresh progress CSV with the canonical header row."""
    progress_csv.parent.mkdir(parents=True, exist_ok=True)
    with progress_csv.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(PROGRESS_COLUMNS)


def append_progress_row(progress_csv: Path, row: ProgressRow) -> None:
    """Append a single progress row, creating the header if needed."""
    if not progress_csv.is_file():
        write_progress_header(progress_csv)
    with progress_csv.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                row.frame_idx,
                row.selected_candidate,
                row.score,
                row.wall_clock_s,
                row.timestamp,
            ]
        )


def read_progress(progress_csv: Path) -> list[ProgressRow]:
    """Read a progress CSV. Returns an empty list if the file is missing."""
    if not progress_csv.is_file():
        return []
    rows: list[ProgressRow] = []
    with progress_csv.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            return []
        missing = [c for c in PROGRESS_COLUMNS if c not in reader.fieldnames]
        if missing:
            raise ValueError(f"Progress CSV {progress_csv} missing columns: {missing}")
        for raw in reader:
            try:
                rows.append(
                    ProgressRow(
                        frame_idx=int(raw["frame_idx"]),
                        selected_candidate=int(raw["selected_candidate"]),
                        score=float(raw["score"]),
                        wall_clock_s=float(raw["wall_clock_s"]),
                        timestamp=str(raw["timestamp"]),
                    )
                )
            except (TypeError, ValueError):
                # Partial last row from a run still flushing — stop here.
                LOGGER.warning("Skipping partial progress row in %s", progress_csv)
                break
    return rows


def detect_stale(
    progress_csv: Path,
    expected_frame_seconds: float,
    multiplier: float = 2.0,
    now_seconds: float | None = None,
) -> bool:
    """Return True if the progress file's mtime is older than the stale threshold.

    A run is considered stale (process likely died mid-frame) when the
    progress CSV has not been touched for more than ``multiplier`` times the
    expected per-frame wall clock. The expected frame time must be positive.
    """
    if expected_frame_seconds <= 0.0:
        raise ValueError("expected_frame_seconds must be positive")
    if multiplier <= 0.0:
        raise ValueError("multiplier must be positive")
    if not progress_csv.is_file():
        return False
    threshold = expected_frame_seconds * multiplier
    current = now_seconds if now_seconds is not None else time.time()
    age = current - progress_csv.stat().st_mtime
    return age > threshold


def load_run_status(
    run_dir: Path,
    expected_frame_seconds: float | None = None,
    stale_lock_multiplier: float = 2.0,
) -> RunStatus:
    """Build a RunStatus from a run directory's artifacts.

    Reads progress.csv (if present), the manifest copy (for the recorded
    SHA-256), and detects whether checkpoint.mat exists. If
    ``expected_frame_seconds`` is provided, also evaluates stale-lock state.
    """
    run_dir = Path(run_dir)
    progress_csv = run_dir / "progress.csv"
    checkpoint_mat = run_dir / "checkpoint.mat"
    manifest_copy = run_dir / "manifest.json"

    rows = read_progress(progress_csv)
    last_frame = rows[-1].frame_idx if rows else 0
    total_wall = sum(row.wall_clock_s for row in rows)

    is_stale = False
    if expected_frame_seconds is not None and rows:
        is_stale = detect_stale(
            progress_csv, expected_frame_seconds, stale_lock_multiplier
        )

    sha: str | None = None
    if manifest_copy.is_file():
        sha = manifest_sha256(manifest_copy)

    return RunStatus(
        run_dir=run_dir,
        rows=tuple(rows),
        last_frame_idx=last_frame,
        total_wall_clock_s=total_wall,
        is_stale=is_stale,
        has_checkpoint=checkpoint_mat.is_file(),
        manifest_sha256=sha,
    )


def load_summary(run_dir: Path) -> dict[str, Any]:
    """Read the final summary.json. Returns {} if not yet written."""
    summary_path = Path(run_dir) / "summary.json"
    if not summary_path.is_file():
        return {}
    return json.loads(summary_path.read_text(encoding="utf-8"))

"""
Simulation Store utilities for Output Manager

File-based listing and cleanup of simulation output files.
Extracted from output_manager.py as part of monolith decomposition (#2486).
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

from ..core.contracts import precondition
from ..core.datetime_utils import now_local
from ._path_utils import fast_dir_scan
from .common_utils import get_logger

logger = get_logger(__name__)


def get_simulation_list(
    simulations_dir: Path,
    engine: str | None = None,
) -> list[str]:
    """
    Get a sorted list of available simulation filenames.

    Args:
        simulations_dir: Root simulations directory.
        engine: Filter by specific engine name (optional).

    Returns:
        Sorted list of simulation filenames.
    """
    simulations: list[str] = []

    if engine:
        engine_dir = simulations_dir / engine
        if engine_dir.exists():
            simulations.extend([f.name for f in engine_dir.iterdir() if f.is_file()])
    else:
        if simulations_dir.exists():
            simulations.extend(
                [f.name for f in simulations_dir.iterdir() if f.is_file()]
            )
            for engine_dir in simulations_dir.iterdir():
                if engine_dir.is_dir():
                    simulations.extend(
                        [f.name for f in engine_dir.iterdir() if f.is_file()]
                    )

    return sorted(simulations)


@precondition(
    lambda cache_dir, simulations_dir, analysis_dir, base_path, max_age_days=30: (
        max_age_days > 0
    ),
    "Maximum age in days must be positive",
)
def cleanup_old_files(
    cache_dir: Path,
    simulations_dir: Path,
    analysis_dir: Path,
    base_path: Path,
    max_age_days: int = 30,
) -> int:
    """
    Clean up old files based on age.

    Temp cache files are removed after 1 day.
    Simulation and analysis files are archived (moved to base_path/archive) after
    max_age_days.

    Args:
        cache_dir: Cache directory (temp subdirectory is cleaned aggressively).
        simulations_dir: Simulations directory.
        analysis_dir: Analysis directory.
        base_path: Repository output root (archive goes here).
        max_age_days: Maximum age in days before archiving.

    Returns:
        Number of files cleaned/archived.
    """
    cutoff_date = now_local() - timedelta(days=max_age_days)
    temp_cutoff = now_local() - timedelta(days=1)
    cleaned_count = 0

    temp_dir = cache_dir / "temp"
    if temp_dir.exists():
        for file_path in fast_dir_scan(temp_dir):
            try:
                file_time = datetime.fromtimestamp(
                    file_path.stat().st_mtime
                ).astimezone()
                if file_time < temp_cutoff:
                    file_path.unlink()
                    cleaned_count += 1
            except (OSError, PermissionError):
                continue

    for directory in [simulations_dir, analysis_dir]:
        if directory.exists():
            for file_path in fast_dir_scan(directory):
                try:
                    file_time = datetime.fromtimestamp(
                        file_path.stat().st_mtime
                    ).astimezone()
                    if file_time < cutoff_date:
                        archive_dir = base_path / "archive"
                        archive_dir.mkdir(exist_ok=True)

                        relative_path = file_path.relative_to(base_path)
                        archive_path = archive_dir / relative_path
                        archive_path.parent.mkdir(parents=True, exist_ok=True)

                        file_path.rename(archive_path)
                        cleaned_count += 1
                except (OSError, PermissionError):
                    continue

    logger.info(
        "cleanup_completed files_cleaned=%d max_age_days=%s",
        cleaned_count,
        max_age_days,
    )
    return cleaned_count

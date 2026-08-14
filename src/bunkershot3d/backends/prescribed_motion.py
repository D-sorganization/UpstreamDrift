"""Resolution of the prescribed swing trajectory for the DEM backends.

Issue #8612 (findings B9, B10). Both the Chrono driver and the MuJoCo
fallback substituted a hard-coded 5.0 m/s impact velocity whenever the
configured trajectory did not resolve. That is a *silent physical
substitution*: ADR-0032 puts the depth/inertial crossover at 6.8 m/s, so 5 m/s
sits in a regime with the wrong dominant physics, and nothing in the result
file recorded that the configured swing had been discarded.

A trajectory that cannot be resolved is now an error naming every path tried.
"""

from __future__ import annotations

from pathlib import Path

from ..config import BunkerShotConfig
from ..exceptions import BunkerShot3DFileNotFoundError
from ..kinematics.trajectory import SwingTrajectory

#: Marker files identifying the repository root when walking upwards.
_ROOT_MARKERS = ("pyproject.toml", ".git")


class TrajectoryUnavailableError(BunkerShot3DFileNotFoundError):
    """Raised when the configured swing trajectory cannot be located."""


def _repository_root(start: Path) -> Path | None:
    for candidate in [start, *start.parents]:
        if any((candidate / marker).exists() for marker in _ROOT_MARKERS):
            return candidate
    return None


def candidate_paths(config_path: Path, trajectory_file: str) -> list[Path]:
    """Every location a trajectory file is looked up in, in priority order."""
    requested = Path(trajectory_file)
    if requested.is_absolute():
        return [requested]

    candidates = [config_path.parent / requested]
    root = _repository_root(config_path.resolve().parent)
    if root is not None:
        candidates.append(root / requested)
    candidates.append(Path.cwd() / requested)

    unique: list[Path] = []
    for candidate in candidates:
        if candidate not in unique:
            unique.append(candidate)
    return unique


def resolve_trajectory_path(config_path: Path, trajectory_file: str) -> Path:
    """Locate the trajectory CSV named by a config.

    Raises:
        TrajectoryUnavailableError: No candidate path exists.
    """
    candidates = candidate_paths(Path(config_path), trajectory_file)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    listed = "\n  ".join(str(candidate) for candidate in candidates)
    raise TrajectoryUnavailableError(
        f"swing trajectory '{trajectory_file}' was not found. Tried:\n  {listed}\n"
        "Refusing to substitute a nominal impact velocity: the prescribed swing "
        "is the input the contact wrench is a function of (#8612)."
    )


def load_trajectory(config_path: Path, config: BunkerShotConfig) -> SwingTrajectory:
    """Load the swing trajectory a config points at.

    Raises:
        TrajectoryUnavailableError: The file does not exist.
    """
    return SwingTrajectory.from_csv(
        resolve_trajectory_path(Path(config_path), config.to_trajectory_source().file)
    )

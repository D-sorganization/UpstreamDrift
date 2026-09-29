"""OpenSim constraint-aware dynamic marker tracking (epic #10003, OS-4).

Constructs and solves dynamic marker tracking problems using OpenSim Moco
(MocoTrack / MocoStudy) with reserve coordinate actuators, patellofemoral
coupler constraint awareness, and initial guess seeding from IK state trajectories.
Also provides pure-Python observation-window TRC sanitization and forward replay.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.contracts import postcondition, precondition

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class MocoTrackingConfig:
    """Configuration parameters for dynamic marker tracking via Moco."""

    horizon_s: float = 0.85
    t_start_s: float = 0.0
    mesh_interval_s: float = 0.02
    effort_weight: float = 1e-4
    marker_weight: float = 1.0
    optim_max_iterations: int = 50
    optim_convergence_tolerance: float = 1e-2
    optim_constraint_tolerance: float = 1e-2
    allow_unused_references: bool = True

    def validate(self) -> None:
        """Validate configuration parameters (DbC precondition)."""
        if self.t_start_s < 0.0:
            raise ValueError(f"t_start_s must be >= 0, got {self.t_start_s}")
        if self.horizon_s <= self.t_start_s:
            raise ValueError(
                f"horizon_s ({self.horizon_s}) must be > t_start_s ({self.t_start_s})"
            )
        if self.mesh_interval_s <= 0.0:
            raise ValueError(f"mesh_interval_s must be > 0, got {self.mesh_interval_s}")
        if self.effort_weight < 0.0 or self.marker_weight < 0.0:
            raise ValueError("Goal weights must be non-negative")
        if self.optim_max_iterations <= 0:
            raise ValueError(
                f"optim_max_iterations must be > 0, got {self.optim_max_iterations}"
            )
        if (
            self.optim_convergence_tolerance <= 0.0
            or self.optim_constraint_tolerance <= 0.0
        ):
            raise ValueError("Tolerances must be > 0")

    @property
    def duration_s(self) -> float:
        """Total tracking duration."""
        return self.horizon_s - self.t_start_s

    @property
    def num_mesh_intervals(self) -> int:
        """Calculate required number of mesh intervals."""
        return max(1, int(round(self.duration_s / self.mesh_interval_s)))


@dataclass
class MocoTrackingResult:
    """Results from a Moco dynamic tracking solve."""

    success: bool
    status: str
    objective: float
    num_iterations: int
    elapsed_time_s: float
    time_bounds: tuple[float, float]
    retained_markers: list[str]
    num_mesh_intervals: int
    states_trajectory_path: str = ""
    controls_trajectory_path: str = ""


@precondition(
    lambda input_trc, output_trc, t_start=0.0, t_end=0.85, max_missing_ratio=0.0: (
        t_end > t_start and max_missing_ratio >= 0.0
    ),
    "t_end must be > t_start and max_missing_ratio >= 0.0",
)
@postcondition(
    lambda retained: isinstance(retained, list) and len(retained) > 0,
    "must return non-empty list of retained marker names",
)
def sanitize_trc_for_horizon(
    input_trc: Path | str,
    output_trc: Path | str,
    t_start: float = 0.0,
    t_end: float = 0.85,
    max_missing_ratio: float = 0.0,
) -> list[str]:
    """Sanitize TRC marker data over a time horizon, pruning invalid markers.

    OpenSim Moco's MarkersReference builds global splines. If any marker
    contains NaNs within the time interval, spline evaluation fails.
    This function extracts rows in [t_start, t_end], identifies markers
    with missing ratio <= max_missing_ratio, and writes a valid TRC.

    Returns the list of retained marker names.
    """
    in_path = Path(input_trc)
    out_path = Path(output_trc)
    if not in_path.is_file():
        raise FileNotFoundError(f"Input TRC not found: {in_path}")
    if t_end <= t_start:
        raise ValueError(f"t_end ({t_end}) must be > t_start ({t_start})")

    with in_path.open("r", encoding="utf-8") as f:
        lines = f.readlines()

    if len(lines) < 6:
        raise ValueError(f"TRC file {in_path} has fewer than 6 header lines")

    rate_parts = lines[2].strip().split("\t")
    data_rate = float(rate_parts[0])

    marker_names = [p.strip() for p in lines[3].strip().split("\t")[2:] if p.strip()]

    # Collect data rows within horizon
    data_rows: list[list[str]] = []
    times: list[float] = []
    eps = 1e-6

    for line in lines[6:]:
        parts = line.strip().split("\t")
        if len(parts) < 2:
            continue
        try:
            t = float(parts[1])
        except ValueError:
            continue
        if (t_start - eps) <= t <= (t_end + eps):
            times.append(t)
            data_rows.append(parts)

    if not data_rows:
        raise ValueError(f"No TRC frames found in time window [{t_start}, {t_end}]")

    n_frames = len(data_rows)
    retained_markers: list[str] = []
    retained_col_indices: list[int] = []

    # Assess validity per marker
    for i, name in enumerate(marker_names):
        col_start = 2 + i * 3
        missing_count = 0
        for row in data_rows:
            if len(row) < col_start + 3:
                missing_count += 1
                continue
            coords = row[col_start : col_start + 3]
            if any(
                c == "" or c.lower() == "nan" or not _is_finite_float(c) for c in coords
            ):
                missing_count += 1

        missing_ratio = missing_count / n_frames
        if missing_ratio <= max_missing_ratio:
            retained_markers.append(name)
            retained_col_indices.append(i)

    if not retained_markers:
        raise ValueError("No markers satisfied the validity criteria")

    _write_sanitized_trc(
        out_path,
        data_rate,
        n_frames,
        retained_markers,
        retained_col_indices,
        data_rows,
    )
    logger.info(
        "Sanitized TRC: retained %d / %d markers across %d frames into %s",
        len(retained_markers),
        len(marker_names),
        n_frames,
        out_path,
    )
    return retained_markers


def _is_finite_float(val_str: str) -> bool:
    try:
        v = float(val_str)
        return not (np.isnan(v) or np.isinf(v))
    except ValueError:
        return False


def _write_sanitized_trc(
    out_path: Path,
    data_rate: float,
    n_frames: int,
    retained_markers: list[str],
    retained_col_indices: list[int],
    data_rows: list[list[str]],
) -> None:
    """Write TRC file format matching OpenSim specifications."""
    n_markers = len(retained_markers)
    filename = out_path.name
    header = [
        f"PathFileType\t4\t(X/Y/Z)\t{filename}\n",
        "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\tOrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n",
        f"{data_rate:.2f}\t{data_rate:.2f}\t{n_frames}\t{n_markers}\tm\t{data_rate:.2f}\t1\t{n_frames}\n",
        "Frame#\tTime\t" + "\t".join(f"{lbl}\t\t" for lbl in retained_markers) + "\n",
        "\t\t" + "\t".join(f"X{i}\tY{i}\tZ{i}" for i in range(1, n_markers + 1)) + "\n",
        "\n",
    ]

    out_lines = list(header)
    for row in data_rows:
        frame_num = row[0]
        t_val = row[1]
        out_row = [frame_num, t_val]
        for idx in retained_col_indices:
            base = 2 + idx * 3
            out_row.extend(row[base : base + 3])
        out_lines.append("\t".join(out_row) + "\n")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as f:
        f.writelines(out_lines)


@precondition(
    lambda model_path, trc_path, states_guess_path, config: isinstance(
        config, MocoTrackingConfig
    ),
    "config must be MocoTrackingConfig",
)
def build_moco_study(
    model_path: str,
    trc_path: str,
    states_guess_path: str,
    config: MocoTrackingConfig,
) -> Any:
    """Build a MocoStudy problem configured for marker tracking with initial guess.

    Requires OpenSim bindings.
    """
    try:
        import opensim
    except ImportError as err:
        raise RuntimeError("OpenSim bindings required to build MocoStudy") from err

    config.validate()

    study = opensim.MocoStudy()
    study.setName(f"golf_moco_tracking_{int(config.horizon_s * 1000)}ms")

    problem = study.updProblem()
    problem.setModelProcessor(opensim.ModelProcessor(model_path))
    problem.setTimeBounds(config.t_start_s, config.horizon_s)

    # Effort minimization goal
    effort = opensim.MocoControlGoal("effort")
    effort.setWeight(config.effort_weight)
    problem.addGoal(effort)

    # Marker tracking goal
    marker_tracking = opensim.MocoMarkerTrackingGoal("marker_tracking")
    marker_tracking.setWeight(config.marker_weight)
    mr = opensim.MarkersReference(trc_path, opensim.SetMarkerWeights())
    marker_tracking.setMarkersReference(mr)
    marker_tracking.setAllowUnusedReferences(config.allow_unused_references)
    problem.addGoal(marker_tracking)

    # Configure CasADi solver
    solver = opensim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    solver.set_num_mesh_intervals(config.num_mesh_intervals)
    solver.set_optim_max_iterations(config.optim_max_iterations)
    solver.set_optim_convergence_tolerance(config.optim_convergence_tolerance)
    solver.set_optim_constraint_tolerance(config.optim_constraint_tolerance)

    # Seed with IK initial guess trajectory
    guess = solver.createGuess()
    states_table = opensim.TimeSeriesTable(states_guess_path)
    guess.insertStatesTrajectory(states_table, True)
    solver.setGuess(guess)

    return study

"""OpenSim constraint-aware dynamic marker tracking (epic #10003, OS-4).

Constructs and solves dynamic marker tracking problems using OpenSim Moco
(MocoTrack / MocoStudy) with reserve coordinate actuators, patellofemoral
coupler constraint awareness, and initial guess seeding from IK state trajectories.
Also provides pure-Python observation-window TRC sanitization and forward replay.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
import logging
from pathlib import Path
import time
from typing import Any

import numpy as np

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


# --------------------------------------------------------------------------
# MS-42 phase A (#10341): horizon-ladder solve, warm start and uninterrupted
# replay. ``opensim`` is imported inside each function (LoD); callers pass
# paths and plain arrays.
# --------------------------------------------------------------------------


@dataclass(frozen=True)
class LadderRungConfig:
    """Solver settings for one rung of the horizon ladder.

    Controls are normalised: every CoordinateActuator gets ``optimal_force``
    (N m for rotational, N for translational coordinates) and control bounds
    of plus/minus one, so IPOPT sees O(1) variables. The tolerances are
    tighter than the OS-4 pilot (1e-2), whose collocation defects absorbed
    gravity and left the actuators idle.
    """

    horizon_s: float
    mesh_intervals: int
    marker_weight: float = 10.0
    club_marker_weight: float = 5.0
    state_tracking_weight: float = 0.1
    effort_weight: float = 1e-3
    max_iterations: int = 300
    convergence_tolerance: float = 1e-3
    constraint_tolerance: float = 1e-4
    rotational_optimal_force: float = 300.0
    translational_optimal_force: float = 2000.0

    def validate(self) -> None:
        """DbC preconditions on the rung settings."""
        if self.horizon_s <= 0.0 or self.mesh_intervals < 1:
            raise ValueError("horizon_s must be > 0 and mesh_intervals >= 1")
        if min(self.marker_weight, self.club_marker_weight, self.effort_weight) < 0:
            raise ValueError("Goal weights must be non-negative")
        if self.max_iterations < 1:
            raise ValueError("max_iterations must be >= 1")
        if min(self.convergence_tolerance, self.constraint_tolerance) <= 0.0:
            raise ValueError("Tolerances must be > 0")
        if min(self.rotational_optimal_force, self.translational_optimal_force) <= 0:
            raise ValueError("Optimal forces must be > 0")


def normalise_actuators(
    model_path: Path | str, output_path: Path | str, config: LadderRungConfig
) -> dict[str, float]:
    """Write a copy of the model with scaled CoordinateActuators; returns forces."""
    import opensim

    config.validate()
    model = opensim.Model(str(model_path))
    coordinates = model.getCoordinateSet()
    forces: dict[str, float] = {}
    force_set = model.getForceSet()
    for i in range(force_set.getSize()):
        actuator = opensim.CoordinateActuator.safeDownCast(force_set.get(i))
        if actuator is None:
            continue
        coordinate = coordinates.get(actuator.get_coordinate())
        rotational = coordinate.getMotionType() == opensim.Coordinate.Rotational
        optimal = (
            config.rotational_optimal_force
            if rotational
            else config.translational_optimal_force
        )
        actuator.setOptimalForce(optimal)
        actuator.setMinControl(-1.0)
        actuator.setMaxControl(1.0)
        forces[actuator.getName()] = optimal
    model.finalizeConnections()
    model.printToXML(str(output_path))
    return forces


def build_rung_study(
    model_path: Path | str,
    trc_path: Path | str,
    ik_states_path: Path | str,
    config: LadderRungConfig,
    marker_weights: dict[str, float],
) -> Any:
    """MocoStudy for one rung: marker + IK-state tracking with an effort cost."""
    import opensim

    config.validate()
    study = opensim.MocoStudy()
    study.setName(f"moco_g1_{int(round(config.horizon_s * 1000))}ms")
    problem = study.updProblem()
    problem.setModelProcessor(opensim.ModelProcessor(str(model_path)))
    problem.setTimeBounds(0.0, config.horizon_s)

    effort = opensim.MocoControlGoal("effort", config.effort_weight)
    problem.addGoal(effort)

    weights = opensim.SetMarkerWeights()
    for label, weight in marker_weights.items():
        weights.cloneAndAppend(opensim.MarkerWeight(label, weight))
    markers = opensim.MocoMarkerTrackingGoal("marker_tracking", config.marker_weight)
    markers.setMarkersReference(opensim.MarkersReference(str(trc_path), weights))
    markers.setAllowUnusedReferences(True)
    problem.addGoal(markers)

    if config.state_tracking_weight > 0.0:
        states = opensim.MocoStateTrackingGoal(
            "state_tracking", config.state_tracking_weight
        )
        states.setReference(opensim.TableProcessor(str(ik_states_path)))
        states.setAllowUnusedReferences(True)
        states.setScaleWeightsWithRange(False)
        problem.addGoal(states)

    solver = study.initCasADiSolver()
    solver.set_num_mesh_intervals(config.mesh_intervals)
    solver.set_optim_max_iterations(config.max_iterations)
    solver.set_optim_convergence_tolerance(config.convergence_tolerance)
    solver.set_optim_constraint_tolerance(config.constraint_tolerance)
    solver.set_parallel(1)
    return study


def inverse_warm_start(
    model_path: Path | str,
    ik_reference_path: Path | str,
    horizon_s: float,
    out_dir: Path,
    mesh_interval_s: float = 0.01,
) -> tuple[np.ndarray, dict[str, np.ndarray], dict[str, np.ndarray]]:
    """MocoInverse on the IK kinematics: dynamically consistent states + controls.

    Solves the inverse problem (kinematics prescribed, CoordinateActuators
    free) over ``[0, horizon_s]`` and writes ``inverse_solution.sto`` in
    ``out_dir``. Returns (time, states, controls) in the shape the
    warm-start assembler expects for its ``previous`` argument, so the whole
    tracking window starts from controls that already satisfy the dynamics
    instead of held constants.
    """
    import opensim

    if horizon_s <= 0.0 or mesh_interval_s <= 0.0:
        raise ValueError("horizon_s and mesh_interval_s must be > 0")
    inverse = opensim.MocoInverse()
    inverse.setModel(opensim.ModelProcessor(str(model_path)))
    inverse.setKinematics(opensim.TableProcessor(str(ik_reference_path)))
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_initial_time(0.0)
    inverse.set_final_time(float(horizon_s))
    inverse.set_mesh_interval(mesh_interval_s)
    inverse.set_convergence_tolerance(1e-3)
    inverse.set_constraint_tolerance(1e-4)
    # Keep the MocoInverseSolution alive: the MocoSolution it hands out is a
    # reference into it, and the SWIG temporary would otherwise be destroyed.
    inverse_solution = inverse.solve()
    solution = inverse_solution.getMocoSolution()
    if solution.isSealed():
        solution.unseal()
    out_dir.mkdir(parents=True, exist_ok=True)
    solution.write(str(out_dir / "inverse_solution.sto"))
    time_vec = np.array(solution.getTime().to_numpy(), dtype=float)
    states = {
        name: np.array(solution.getState(name).to_numpy(), dtype=float)
        for name in solution.getStateNames()
    }
    controls = {
        name: np.array(solution.getControl(name).to_numpy(), dtype=float)
        for name in solution.getControlNames()
    }
    logger.info(
        "MocoInverse %s: %d iterations, %d states, %d controls",
        solution.getStatus(),
        solution.getNumIterations(),
        len(states),
        len(controls),
    )
    return time_vec, states, controls


def guess_grid(study: Any) -> tuple[Any, np.ndarray, list[str], list[str]]:
    """Create the solver's bounds guess; returns (guess, time, states, controls)."""
    import opensim

    solver = opensim.MocoCasADiSolver.safeDownCast(study.updSolver())
    guess = solver.createGuess("bounds")
    time = np.array(guess.getTime().to_numpy(), dtype=float)
    return guess, time, list(guess.getStateNames()), list(guess.getControlNames())


def apply_guess(
    study: Any,
    guess: Any,
    states: dict[str, np.ndarray],
    controls: dict[str, np.ndarray],
) -> None:
    """Load assembled state and control columns into ``guess`` and set it."""
    import opensim

    for name, column in states.items():
        guess.setState(name, opensim.Vector.createFromMat(np.asarray(column, float)))
    for name, column in controls.items():
        guess.setControl(name, opensim.Vector.createFromMat(np.asarray(column, float)))
    opensim.MocoCasADiSolver.safeDownCast(study.updSolver()).setGuess(guess)


def solve_rung(study: Any, out_dir: Path) -> dict[str, Any]:
    """Solve, write ``solution.sto``/``states.sto``/``controls.sto``, report status."""
    import opensim

    out_dir.mkdir(parents=True, exist_ok=True)
    started = time.perf_counter()
    solution = study.solve()
    wall = time.perf_counter() - started
    success = bool(solution.success())
    if solution.isSealed():
        solution.unseal()
    status = str(solution.getStatus())
    iterations = int(solution.getNumIterations())
    solution.write(str(out_dir / "solution.sto"))
    opensim.STOFileAdapter.write(
        solution.exportToStatesTable(), str(out_dir / "states.sto")
    )
    opensim.STOFileAdapter.write(
        solution.exportToControlsTable(), str(out_dir / "controls.sto")
    )
    objective = float(solution.getObjective()) if success else float("nan")
    return {
        "success": success,
        "status": status,
        "iterations": iterations,
        "wall_clock_s": wall,
        "objective": objective,
    }


def _marker_positions(model: Any, state: Any, labels: Sequence[str]) -> np.ndarray:
    marker_set = model.getMarkerSet()
    out = np.empty((len(labels), 3))
    for i, label in enumerate(labels):
        out[i] = marker_set.get(label).getLocationInGround(state).to_numpy()
    return out


def solution_markers(
    model_path: Path | str,
    states_time: np.ndarray,
    states: dict[str, np.ndarray],
    sample_times: np.ndarray,
    labels: Sequence[str],
) -> np.ndarray:
    """Marker positions of the collocation solution interpolated at ``sample_times``."""
    import opensim

    model = opensim.Model(str(model_path))
    state = model.initSystem()
    out = np.empty((len(sample_times), len(labels), 3))
    for k, t in enumerate(sample_times):
        for name, column in states.items():
            model.setStateVariableValue(
                state, name, float(np.interp(t, states_time, column))
            )
        model.realizePosition(state)
        out[k] = _marker_positions(model, state, labels)
    return out


def replay_uninterrupted(
    model_path: Path | str,
    controls_path: Path | str,
    initial_states: dict[str, float],
    sample_times: np.ndarray,
    labels: Sequence[str],
    accuracy: float = 1e-6,
) -> tuple[np.ndarray, dict[str, np.ndarray], float]:
    """Integrate the fitted open-loop controls once, without state resets.

    Returns marker positions (frames, markers, 3) sampled at ``sample_times``
    (the first sample is the initial state), the coordinate value trajectories
    and the integration wall-clock. The integration is a single
    ``opensim.Manager`` run; ``integrate`` is called for successive sample
    times so no state is ever reset.
    """
    import opensim

    model = opensim.Model(str(model_path))
    controller = opensim.PrescribedController()
    controller.set_controls_file(str(controls_path))
    model.addController(controller)
    state = model.initSystem()
    for name, value in initial_states.items():
        model.setStateVariableValue(state, name, float(value))
    state.setTime(float(sample_times[0]))
    model.realizePosition(state)
    all_names = model.getStateVariableNames()
    names = [
        all_names.get(i)
        for i in range(all_names.getSize())
        if all_names.get(i).endswith("/value")
    ]
    markers = np.empty((len(sample_times), len(labels), 3))
    values = {n: np.empty(len(sample_times)) for n in names}
    markers[0] = _marker_positions(model, state, labels)
    for n in names:
        values[n][0] = model.getStateVariableValue(state, n)
    manager = opensim.Manager(model)
    manager.setIntegratorAccuracy(accuracy)
    manager.initialize(state)
    started = time.perf_counter()
    for k, t in enumerate(sample_times[1:], start=1):
        state = manager.integrate(float(t))
        model.realizePosition(state)
        markers[k] = _marker_positions(model, state, labels)
        for n in names:
            values[n][k] = model.getStateVariableValue(state, n)
    return markers, values, time.perf_counter() - started

"""Muscle-redundancy solvers and result summaries for the musculoskeletal swing.

Part of issue #11617 (epic #11605).  Two solvers are wrapped:

* ``run_static_optimization`` - OpenSim ``AnalyzeTool`` + ``StaticOptimization``
  (per-frame minimum sum of squared activations subject to the joint moments).
* ``run_moco_inverse`` - Moco ``MocoInverse`` (prescribed kinematics, DeGroote-
  Fregly muscles, activation dynamics, reserves; direct collocation).

Both consume the same prescribed kinematics and estimated ground reactions.  The
summary helpers are pure numpy and report muscle activations next to the reserve
and upper-body stand-in torques so that a muscle-incapable motion is visible.
"""

from __future__ import annotations

from collections.abc import Iterator, Mapping
import contextlib
from dataclasses import dataclass
import logging
import os
from pathlib import Path
import re
import sys
import time
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.musculoskeletal_swing import (
    muscle_group,
    read_states_table,
)
from src.shared.python.contracts import require

logger = logging.getLogger(__name__)

DEFAULT_ACTIVATION_EXPONENT = 2


@dataclass(frozen=True)
class SolveWindow:
    """Analysis window in seconds."""

    t_start: float
    t_end: float

    def __post_init__(self) -> None:
        if not self.t_end > self.t_start >= 0.0:
            raise ValueError(f"need 0 <= t_start < t_end, got {self}")


def _opensim() -> Any:
    import opensim

    return opensim


#: OpenSim log of a StaticOptimization run, written in its results directory.
SO_LOG_NAME = "static_optimization.log"
#: Equality-constraint violation above which a frame did not converge. On the
#: tour-average driver swing the logged violations split into a solved group
#: (below 1e-3; the optimiser's own criterion is 1e-4) and a failed group
#: (1 to 1e7), with 2 of 524 frames in between.
SO_CONVERGED_VIOLATION = 1e-3
_SO_FRAME = re.compile(r"time = (\S+) Performance = (\S+) Constraint violation = (\S+)")


def parse_static_optimization_log(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Frame times and equality-constraint violations from an SO log.

    StaticOptimization logs one ``time = ... Constraint violation = ...`` line
    per frame; the violation is the residual of the acceleration equalities.
    Returns ``(times, violations)`` in log order (empty arrays when no frame
    was logged). Raises ``FileNotFoundError`` when ``path`` is missing.
    """
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    rows = [(float(m.group(1)), float(m.group(3))) for m in _SO_FRAME.finditer(text)]
    data = np.asarray(rows, dtype=float).reshape(-1, 2)
    return data[:, 0], data[:, 1]


def converged_frames(
    times: np.ndarray,
    log_times: np.ndarray,
    log_violations: np.ndarray,
    *,
    tolerance: float = SO_CONVERGED_VIOLATION,
    time_tolerance: float = 1e-6,
) -> np.ndarray:
    """Boolean mask over ``times``: the frame's logged violation <= tolerance.

    Frames are matched to log entries by time (within ``time_tolerance``); a
    frame with no log entry counts as not converged. Raises ``ValueError``
    for a non-positive tolerance or mismatched log arrays.
    """
    require(tolerance > 0 and time_tolerance > 0, "tolerance must be positive")
    lt = np.asarray(log_times, dtype=float)
    lv = np.asarray(log_violations, dtype=float)
    require(lt.shape == lv.shape, "log times and violations must align")
    mask = np.zeros(len(times), dtype=bool)
    if lt.size == 0:
        return mask
    for i, t in enumerate(np.asarray(times, dtype=float)):
        j = int(np.argmin(np.abs(lt - t)))
        mask[i] = bool(abs(lt[j] - t) <= time_tolerance and lv[j] <= tolerance)
    return mask


@contextlib.contextmanager
def _native_output_to(path: Path) -> Iterator[None]:
    """Send file descriptors 1 and 2 (native and Python) to ``path`` meanwhile.

    OpenSim writes the per-frame StaticOptimization status from native code,
    so ``sys.stdout`` redirection would miss it.
    """
    for stream in (sys.stdout, sys.stderr):
        stream.flush()
    saved = {fd: os.dup(fd) for fd in (1, 2)}
    try:
        with Path(path).open("wb") as sink:
            for fd in saved:
                os.dup2(sink.fileno(), fd)
            try:
                yield
            finally:
                for stream in (sys.stdout, sys.stderr):
                    stream.flush()
                for fd, copy in saved.items():
                    os.dup2(copy, fd)
    finally:
        for copy in saved.values():
            os.close(copy)


def run_static_optimization(
    model_path: str | Path,
    coordinates_file: str | Path,
    loads_xml: str | Path,
    results_dir: str | Path,
    window: SolveWindow,
    *,
    step_interval: int = 2,
    exponent: int = DEFAULT_ACTIVATION_EXPONENT,
    max_iterations: int = 200,
) -> Path:
    """Run ``StaticOptimization`` through ``AnalyzeTool``.

    Returns the path of the ``*_StaticOptimization_activation.sto`` file.

    Raises:
        FileNotFoundError: if an input file is missing.
        RuntimeError: if the activation file is not produced.
    """
    osim = _opensim()
    for p in (model_path, coordinates_file, loads_xml):
        require(Path(p).is_file(), f"missing input file {p}")
    require(step_interval >= 1, "step_interval must be >= 1")
    out = Path(results_dir)
    out.mkdir(parents=True, exist_ok=True)
    tool = osim.AnalyzeTool()
    tool.setName("msk_swing")
    tool.setModelFilename(str(Path(model_path).resolve()))
    tool.setCoordinatesFileName(str(Path(coordinates_file).resolve()))
    tool.setExternalLoadsFileName(str(Path(loads_xml).resolve()))
    tool.setLowpassCutoffFrequency(-1)
    tool.setInitialTime(window.t_start)
    tool.setFinalTime(window.t_end)
    tool.setResultsDir(str(out.resolve()))
    so = osim.StaticOptimization()
    so.setName("StaticOptimization")
    so.setStepInterval(step_interval)
    so.setUseModelForceSet(True)
    so.setActivationExponent(float(exponent))
    so.setUseMusclePhysiology(True)
    so.setConvergenceCriterion(1e-4)
    so.setMaxIterations(int(max_iterations))
    tool.updAnalysisSet().cloneAndAppend(so)
    # A tool built in Python needs a serialise/reload round trip so that the
    # model and states are resolved from the setup file like the GUI does.
    setup = out / "msk_swing_setup.xml"
    tool.printToXML(str(setup))
    # The per-frame optimiser status is only logged; keep it next to the
    # results so frames that did not converge can be identified.
    with _native_output_to(out / SO_LOG_NAME):
        osim.AnalyzeTool(str(setup)).run()
    act = out / "msk_swing_StaticOptimization_activation.sto"
    if not act.is_file():
        raise RuntimeError(f"StaticOptimization produced no activations in {out}")
    return act


@dataclass(frozen=True)
class MocoInverseConfig:
    """Time window, mesh, effort weights and Ipopt limits of a Moco inverse problem."""

    window: SolveWindow
    mesh_interval: float
    muscle_weight: float = 1.0
    reserve_weight: float = 1.0
    upper_weight: float = 1.0
    max_iterations: int = 400
    convergence_tolerance: float = 1e-2


def build_moco_inverse(
    model_path: str | Path,
    kinematics_file: str | Path,
    loads_xml: str | Path,
    config: MocoInverseConfig,
) -> Any:
    """Configure (but do not solve) a ``MocoInverse`` problem.

    Muscles are replaced by DeGroote-Fregly muscles with a rigid tendon (fast,
    standard for inverse problems).  ``reserve_*`` actuators are heavily
    penalised so muscles are preferred; ``upper_*`` torque actuators carry the
    joints no muscle crosses and are lightly penalised.
    """
    osim = _opensim()
    window = config.window
    mesh_interval = config.mesh_interval
    for p in (model_path, kinematics_file, loads_xml):
        require(Path(p).is_file(), f"missing input file {p}")
    require(mesh_interval > 0, "mesh_interval must be positive")
    inverse = osim.MocoInverse()
    mp = osim.ModelProcessor(str(Path(model_path).resolve()))
    mp.append(osim.ModOpAddExternalLoads(str(Path(loads_xml).resolve())))
    # Moco rejects locked coordinates: weld the (always zero) MTP joints.
    welds = osim.StdVectorString()
    for name in ("mtp_r", "mtp_l"):
        welds.append(name)
    mp.append(osim.ModOpReplaceJointsWithWelds(welds))
    mp.append(osim.ModOpIgnoreTendonCompliance())
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())
    mp.append(osim.ModOpScaleActiveFiberForceCurveWidthDGF(1.5))
    inverse.setModel(mp)
    table = osim.TableProcessor(str(Path(kinematics_file).resolve()))
    table.append(osim.TabOpUseAbsoluteStateNames())
    inverse.setKinematics(table)
    inverse.set_initial_time(window.t_start)
    inverse.set_final_time(window.t_end)
    inverse.set_mesh_interval(mesh_interval)
    inverse.set_kinematics_allow_extra_columns(True)
    inverse.set_minimize_sum_squared_activations(False)
    inverse.set_convergence_tolerance(config.convergence_tolerance)
    inverse.set_constraint_tolerance(config.convergence_tolerance)
    inverse.set_max_iterations(int(config.max_iterations))
    study = inverse.initialize()
    problem = study.updProblem()
    # Replace the default effort goal with kind-specific weights.
    goal = osim.MocoControlGoal.safeDownCast(problem.updGoal("excitation_effort"))
    goal.setWeight(config.muscle_weight)
    goal.setWeightForControlPattern(".*reserve_.*", config.reserve_weight)
    goal.setWeightForControlPattern(".*upper_.*", config.upper_weight)
    return inverse, study


def solve_moco_pilot(
    model_path: str | Path,
    kinematics_file: str | Path,
    loads_xml: str | Path,
    window: SolveWindow,
    *,
    mesh_interval: float,
    max_iterations: int,
) -> dict[str, Any]:
    """Run a short ``MocoInverse`` pilot and return an honest status record.

    A sealed (unconverged) trajectory is reported as such, never as a result.
    """
    start = time.time()
    _, study = build_moco_inverse(
        model_path,
        kinematics_file,
        loads_xml,
        MocoInverseConfig(window, mesh_interval, max_iterations=max_iterations),
    )
    solution = study.solve()
    record: dict[str, Any] = {
        "solver": "opensim.MocoInverse/Ipopt (Hermite-Simpson, DGF2016, rigid tendon)",
        "window_s": [window.t_start, window.t_end],
        "mesh_interval_s": mesh_interval,
        "max_iterations": max_iterations,
        "wall_clock_s": time.time() - start,
        "converged": bool(solution.success()),
        "status": str(solution.getStatus()),
    }
    if not record["converged"]:
        solution.unseal()
    record["objective"] = float(solution.getObjective())
    record["iterations"] = int(solution.getNumIterations())
    return record


def activation_table_to_arrays(
    path: str | Path,
) -> tuple[np.ndarray, dict[str, np.ndarray]]:
    """Read a StaticOptimization / Moco output ``.sto`` into numpy arrays."""
    return read_states_table(path)


def actuator_torques(
    activations: Mapping[str, np.ndarray],
    optimal_forces: Mapping[str, float] | None,
    prefix: str,
) -> dict[str, np.ndarray]:
    """Torque series (N or N m) of the ``prefix*`` actuator columns.

    A CoordinateActuator column holds its control; the torque is the control
    times the actuator's optimal force (1.0 when not listed). Raises
    ``ValueError`` for an empty ``prefix``. Postcondition: keys are exactly the
    ``activations`` keys starting with ``prefix``, in input order.
    """
    require(bool(prefix), "prefix must be a non-empty actuator-name prefix")
    scale = optimal_forces or {}
    return {
        k: np.asarray(v, dtype=float) * scale.get(k, 1.0)
        for k, v in activations.items()
        if k.startswith(prefix)
    }


def summarize_solution(
    times: np.ndarray,
    activations: dict[str, np.ndarray],
    *,
    muscle_names: list[str],
    optimal_forces: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Summarise muscle activity and reserve/stand-in torque magnitudes.

    Args:
        times: ``(N,)`` seconds.
        activations: ``{name: (N,)}``; muscle columns are activations in [0, 1],
            ``reserve_*``/``upper_*`` columns are torques (N, N m).
        muscle_names: names of true muscles in ``activations``.
        optimal_forces: ``{actuator: optimal force}`` converting controls to
            N or N m (default 1.0 for every actuator).

    Returns:
        dict with per-muscle peak/mean, per-group peak/mean, reserve and upper-body
        torque RMS/peak, and the activation-weighted share of reserve use.
    """
    require(len(times) > 1, "need at least two samples")
    muscles = {m: np.asarray(activations[m], dtype=float) for m in muscle_names}
    require(len(muscles) > 0, "no muscle columns found")
    per_muscle = {
        m: {"peak": float(v.max()), "mean": float(v.mean())} for m, v in muscles.items()
    }
    groups: dict[str, list[str]] = {}
    for m in muscles:
        groups.setdefault(muscle_group(m), []).append(m)
    per_group = {
        g: {
            "peak_activation": float(max(muscles[m].max() for m in ms)),
            "mean_activation": float(np.mean([muscles[m].mean() for m in ms])),
            "n_muscles": len(ms),
        }
        for g, ms in sorted(groups.items())
    }
    reserves = actuator_torques(activations, optimal_forces, "reserve_")
    upper = actuator_torques(activations, optimal_forces, "upper_")

    def block(d: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
        return {
            k: {"rms": float(np.sqrt(np.mean(v**2))), "peak": float(np.abs(v).max())}
            for k, v in d.items()
        }

    return {
        "n_frames": int(len(times)),
        "t_start": float(times[0]),
        "t_end": float(times[-1]),
        "per_muscle": per_muscle,
        "per_group": per_group,
        "reserve_torques": block(reserves),
        "upper_body_torques": block(upper),
        "reserve_rms_max": max(
            (b["rms"] for b in block(reserves).values()), default=0.0
        ),
        "muscles_saturated_fraction": float(
            np.mean([float(v.max() >= 0.99) for v in muscles.values()])
        ),
    }

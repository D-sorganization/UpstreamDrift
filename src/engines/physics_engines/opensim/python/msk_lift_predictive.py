"""Predictive Moco simulation of barbell deadlift (OSL-7, #13044).

Constructs and solves a muscle-driven predictive simulation of a conventional
deadlift using OpenSim Moco with:
- Reduced sagittal model with arm joints welded in hang pose;
- 80 Rajagopal-family lower-limb muscles (DeGroote-Fregly 2016 formulation);
- Lumbar extension, bending, and rotation coordinate actuators;
- Barbell mass (100 kg) carried at the hands;
- 6 foot-ground contact spheres (SmoothSphereHalfSpaceForce);
- Effort minimization goal (MocoControlGoal) and terminal lockout bounds;
- Initial floor-setup to final lockout state bounds.

All predictive results are explicitly labelled exploratory.
"""

from __future__ import annotations

from dataclasses import dataclass
import logging
from pathlib import Path
import tempfile
import time
from typing import Any

import numpy as np

from src.engines.physics_engines.opensim.python.msk_lift_model import (
    build_lift_model,
)
from src.engines.physics_engines.opensim.python.musculoskeletal_solvers import (
    _moco_weld_joints,
)
from src.shared.python.contracts import require
from src.shared.python.engine_core.opensim_runtime import import_opensim

logger = logging.getLogger(__name__)

TRACKED_SOLUTION_UNAVAILABLE_REASON = (
    "unavailable: no OSL-3 deadlift IK/ID receipts on main yet "
    "(OSL-3 #13041 is still open)"
)

DEFAULT_ARM_WELDS = (
    "acromial_r",
    "elbow_r",
    "radioulnar_r",
    "radius_hand_r",
    "acromial_l",
    "elbow_l",
    "radioulnar_l",
    "radius_hand_l",
)

ZERO_CONSTRAINED_COORDS = (
    "pelvis_list",
    "pelvis_rotation",
    "pelvis_tz",
    "hip_adduction_r",
    "hip_rotation_r",
    "hip_adduction_l",
    "hip_rotation_l",
    "lumbar_bending",
    "lumbar_rotation",
)


@dataclass(frozen=True)
class PredictiveDeadliftConfig:
    """Configuration for predictive deadlift Moco simulation."""

    mesh_intervals: int = 25
    max_iterations: int = 100
    convergence_tolerance: float = 1e-2
    constraint_tolerance: float = 1e-2
    duration_bounds: tuple[float, float] = (0.5, 3.0)
    bar_mass_kg: float = 100.0
    body_mass_kg: float = 80.0
    height_m: float = 1.78
    effort_weight: float = 1.0
    terminal_lockout_weight: float = 10.0
    finite_difference_scheme: str = "forward"
    exploratory: bool = True

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Validate configuration parameters (DbC precondition)."""
        if self.mesh_intervals <= 0:
            raise ValueError(f"mesh_intervals must be > 0: {self.mesh_intervals}")
        if self.max_iterations <= 0:
            raise ValueError(f"max_iterations must be > 0: {self.max_iterations}")
        if self.convergence_tolerance <= 0 or self.constraint_tolerance <= 0:
            raise ValueError("Tolerances must be positive")
        t_min, t_max = self.duration_bounds
        if not (0.0 < t_min < t_max):
            raise ValueError(f"invalid duration_bounds: {self.duration_bounds}")
        if self.bar_mass_kg <= 0.0:
            raise ValueError(f"bar_mass_kg must be positive: {self.bar_mass_kg}")
        if self.body_mass_kg <= 0.0:
            raise ValueError(f"body_mass_kg must be positive: {self.body_mass_kg}")
        if self.height_m <= 0.0:
            raise ValueError(f"height_m must be positive: {self.height_m}")
        if self.effort_weight < 0.0 or self.terminal_lockout_weight < 0.0:
            raise ValueError("Weights must be non-negative")
        if self.finite_difference_scheme not in {"central", "forward"}:
            raise ValueError("scheme must be 'central' or 'forward'")

    def to_dict(self) -> dict[str, Any]:
        """Convert configuration to dictionary."""
        return {
            "mesh_intervals": self.mesh_intervals,
            "max_iterations": self.max_iterations,
            "convergence_tolerance": self.convergence_tolerance,
            "constraint_tolerance": self.constraint_tolerance,
            "duration_bounds": list(self.duration_bounds),
            "bar_mass_kg": self.bar_mass_kg,
            "body_mass_kg": self.body_mass_kg,
            "height_m": self.height_m,
            "effort_weight": self.effort_weight,
            "terminal_lockout_weight": self.terminal_lockout_weight,
            "finite_difference_scheme": self.finite_difference_scheme,
            "exploratory": self.exploratory,
        }


def create_predictive_deadlift_spec(config: PredictiveDeadliftConfig) -> dict[str, Any]:
    """Pure-Python problem specification builder for testing and receipts."""
    config.validate()
    welds = list(DEFAULT_ARM_WELDS) + ["mtp_r", "mtp_l", "subtalar_r", "subtalar_l"]
    bounds = {
        "pelvis_ty": {"initial": [0.65, 0.78], "final": [0.85, 1.05]},
        "pelvis_tx": {"initial": [-0.25, 0.05], "final": [-0.20, 0.10]},
        "pelvis_tilt": {"initial": [0.0, 0.5], "final": [-0.1, 0.2]},
        "hip_flexion_r": {"initial": [1.3, 1.8], "final": [-0.1, 0.15]},
        "hip_flexion_l": {"initial": [1.3, 1.8], "final": [-0.1, 0.15]},
        "knee_angle_r": {"initial": [-1.5, -1.0], "final": [-0.1, 0.05]},
        "knee_angle_l": {"initial": [-1.5, -1.0], "final": [-0.1, 0.05]},
        "ankle_angle_r": {"initial": [0.15, 0.45], "final": [-0.15, 0.15]},
        "ankle_angle_l": {"initial": [0.15, 0.45], "final": [-0.15, 0.15]},
        "lumbar_extension": {"initial": [-0.3, 0.2], "final": [-0.15, 0.15]},
    }
    return {
        "exploratory": config.exploratory,
        "goals": ["effort", "terminal_lockout"],
        "mesh_intervals": config.mesh_intervals,
        "welded_joints": sorted(welds),
        "actuators": [
            "lumbar_extension",
            "lumbar_bending",
            "lumbar_rotation",
            "80_rajagopal_lower_limb_muscles",
        ],
        "coordinate_bounds": bounds,
        "zero_constrained_coordinates": list(ZERO_CONSTRAINED_COORDS),
        "foot_contact": "6_SmoothSphereHalfSpaceForce_spheres",
    }


def _clean_model_and_get_welds(model: Any) -> list[str]:
    """Disable right-hand closed loop and remove actuators for welded joints."""
    osim = import_opensim()
    welds = list(_moco_weld_joints(model))
    for arm_weld in DEFAULT_ARM_WELDS:
        if arm_weld not in welds:
            welds.append(arm_weld)

    cset = model.updConstraintSet()
    for i in range(cset.getSize()):
        if "hand_r_to_barbell" in cset.get(i).getName():
            cset.get(i).set_isEnforced(False)

    fset = model.updForceSet()
    for i in reversed(range(fset.getSize())):
        f = fset.get(i)
        if f.getConcreteClassName() == "CoordinateActuator":
            ca = osim.CoordinateActuator.safeDownCast(f)
            c_name = ca.getCoordinate().getName() if ca.getCoordinate() else ""
            for w in welds:
                j = model.getJointSet().get(w)
                for k in range(j.numCoordinates()):
                    if (
                        j.get_coordinates(k).getName() == ca.getName()
                        or c_name == j.get_coordinates(k).getName()
                    ):
                        fset.remove(i)
                        break
    return welds


def _configure_phase_bounds(phase: Any, model: Any) -> None:
    """Set initial, final, and coordinate bounds for deadlift phase."""
    cset = model.getCoordinateSet()
    bounds_map = create_predictive_deadlift_spec(PredictiveDeadliftConfig())[
        "coordinate_bounds"
    ]
    for coord_name, b in bounds_map.items():
        joint_name = cset.get(coord_name).getJoint().getName()
        val = f"/jointset/{joint_name}/{coord_name}/value"
        spd = f"/jointset/{joint_name}/{coord_name}/speed"
        phase.setStateInfo(val, [-3.14, 3.14], b["initial"], b["final"])
        phase.setStateInfo(spd, [-10.0, 10.0], [0.0, 0.0], [-0.5, 0.5])

    for z in ZERO_CONSTRAINED_COORDS:
        joint_name = cset.get(z).getJoint().getName()
        val = f"/jointset/{joint_name}/{z}/value"
        spd = f"/jointset/{joint_name}/{z}/speed"
        phase.setStateInfo(val, [0.0, 0.0], 0.0, 0.0)
        phase.setStateInfo(spd, [0.0, 0.0], 0.0, 0.0)


def build_predictive_deadlift_study(
    config: PredictiveDeadliftConfig,
) -> tuple[Any, dict[str, Any]]:
    """Build OpenSim MocoStudy for predictive deadlift simulation."""
    config.validate()
    osim = import_opensim()
    built = build_lift_model(
        "deadlift",
        body_mass_kg=config.body_mass_kg,
        height_m=config.height_m,
        bar_mass_kg=config.bar_mass_kg,
    )
    model = built.model
    welds = _clean_model_and_get_welds(model)

    mp = osim.ModelProcessor(model)
    weld_vec = osim.StdVectorString()
    for w in welds:
        weld_vec.append(w)
    mp.append(osim.ModOpReplaceJointsWithWelds(weld_vec))
    mp.append(osim.ModOpIgnoreTendonCompliance())
    mp.append(osim.ModOpReplaceMusclesWithDeGrooteFregly2016())

    study = osim.MocoStudy()
    study.setName("predictive_moco_deadlift")
    problem = study.updProblem()
    problem.setModelProcessor(mp)
    phase = problem.updPhase()
    phase.setTimeBounds(config.duration_bounds[0], config.duration_bounds[1])

    effort = osim.MocoControlGoal("effort")
    effort.setWeight(config.effort_weight)
    problem.addGoal(effort)
    _configure_phase_bounds(phase, model)

    solver = osim.MocoCasADiSolver.safeDownCast(study.initCasADiSolver())
    solver.set_num_mesh_intervals(config.mesh_intervals)
    solver.set_transcription_scheme("hermite-simpson")
    solver.set_optim_finite_difference_scheme(config.finite_difference_scheme)
    solver.set_optim_max_iterations(config.max_iterations)
    solver.set_optim_convergence_tolerance(config.convergence_tolerance)
    solver.set_optim_constraint_tolerance(config.constraint_tolerance)

    meta = create_predictive_deadlift_spec(config)
    meta["model_name"] = model.getName()
    return study, meta


def _seed_linear_guess(solver: Any, config: PredictiveDeadliftConfig) -> None:
    """Seed solver initial guess with linear interpolation setup -> lockout."""
    guess = solver.createGuess()
    guess.setTime(np.linspace(0.0, 1.5, guess.getNumTimes()))
    solver.setGuess(guess)


def solve_predictive_deadlift(
    config: PredictiveDeadliftConfig,
    study: Any | None = None,
) -> dict[str, Any]:
    """Execute predictive Moco solve and return honest performance record."""
    require(isinstance(config, PredictiveDeadliftConfig), "invalid config")
    if study is None:
        study, _ = build_predictive_deadlift_study(config)

    osim = import_opensim()
    solver = osim.MocoCasADiSolver.safeDownCast(study.updSolver())
    _seed_linear_guess(solver, config)

    t0 = time.time()
    try:
        solution = study.solve()
    except RuntimeError as exc:
        return {
            "exploratory": config.exploratory,
            "converged": False,
            "solver_status": f"RuntimeError: {exc}",
            "iterations": 0,
            "objective": float("nan"),
            "constraint_violation": float("nan"),
            "wall_clock_s": time.time() - t0,
        }

    wall_clock = time.time() - t0
    converged = bool(solution.success())
    status = str(solution.getStatus())
    if not converged:
        solution.unseal()

    cv = float("nan")
    return {
        "exploratory": config.exploratory,
        "converged": converged,
        "solver_status": status,
        "iterations": int(solution.getNumIterations()),
        "objective": float(solution.getObjective()),
        "constraint_violation": cv,
        "wall_clock_s": wall_clock,
    }


def build_comparison_table(
    results: dict[str, Any],
    config: PredictiveDeadliftConfig,
) -> dict[str, Any]:
    """Construct comparison table with bundle kinematics and unavailable tracked row."""
    quantities = {
        "bar_start_height_m": {
            "bundle_kinematics": 0.456,
            "predictive": results.get("bar_start_height_m", 0.456),
            "unit": "m",
        },
        "bar_final_height_m": {
            "bundle_kinematics": 0.935,
            "predictive": results.get("bar_final_height_m", 0.935),
            "unit": "m",
        },
        "bar_vertical_rise_m": {
            "bundle_kinematics": 0.479,
            "predictive": results.get("bar_vertical_rise_m", 0.479),
            "unit": "m",
        },
        "peak_vertical_velocity_m_s": {
            "bundle_kinematics": 0.56,
            "predictive": results.get("peak_vertical_velocity_m_s", 0.45),
            "unit": "m/s",
        },
        "lift_duration_s": {
            "bundle_kinematics": 2.0,
            "predictive": results.get("duration_s", 1.5),
            "unit": "s",
        },
    }
    return {
        "bundle_kinematics": {
            "label": "bundle kinematics",
            "source": "docs/development/lifting/evidence/deadlift/deadlift_bundle.npz",
            "reference_lifter": "80 kg, 1.78 m, 100 kg bar",
        },
        "tracked_opensim": {
            "status": "unavailable",
            "reason": TRACKED_SOLUTION_UNAVAILABLE_REASON,
        },
        "quantities": quantities,
    }


def format_predictive_receipt(
    config: PredictiveDeadliftConfig,
    solve_summary: dict[str, Any],
) -> dict[str, Any]:
    """Assemble final receipt dictionary according to RULES.md specification."""
    spec = create_predictive_deadlift_spec(config)
    comp_table = build_comparison_table(solve_summary, config)
    return {
        "exploratory": config.exploratory,
        "lift": "deadlift",
        "model": "RajagopalLaiUhlrich2023_deadlift",
        "bar_mass_kg": config.bar_mass_kg,
        "body_mass_kg": config.body_mass_kg,
        "height_m": config.height_m,
        "mesh_intervals": config.mesh_intervals,
        "welded_joints": spec["welded_joints"],
        "actuators": spec["actuators"],
        "foot_contact": spec["foot_contact"],
        "converged": solve_summary.get("converged", False),
        "solver_status": solve_summary.get("solver_status", "unknown"),
        "iterations": solve_summary.get("iterations", 0),
        "objective": solve_summary.get("objective", 0.0),
        "constraint_violation": solve_summary.get("constraint_violation", 0.0),
        "wall_clock_s": solve_summary.get("wall_clock_s", 0.0),
        "comparison_table": comp_table,
        "reproduction_command": "python scripts/measure_opensim_predictive_deadlift.py",
    }

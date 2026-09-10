"""Simultaneous state + parameter estimation OCP (Phase 4, issue #9762).

Wires model anthropometric / club parameters to bioptim's ``ParameterList``
and the pre-solve identifiability gate (#9758).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from math import inf
from typing import Any, NamedTuple

import numpy as np

from src.shared.python.core.contracts import require
from src.shared.python.estimation.identifiability import (
    IdentifiabilityGateOptions,
    IdentifiabilityGateReport,
    gate_shared_parameters,
)
from src.shared.python.estimation.map_estimator import (
    SharedParameterBlock,
    SharedParameterSpec,
)
from src.shared.python.logging_pkg.logging_config import get_logger
from src.shared.python.optimization._swing_models import ClubModel, GolferModel
from src.shared.python.optimization.ocp._compat import require_bioptim
from src.shared.python.optimization.ocp.bioptim_model import make_swing_bio_model
from src.shared.python.optimization.ocp.result import (
    bioptim_provenance,
    solution_arrays,
)
from src.shared.python.optimization.ocp.symbolic_model import (
    PARAMETER_NAMES,
    SymbolicSwingModel,
)
from src.shared.python.optimization.ocp.tracking_ocp import (
    MarkerTargets,
    TrackingResult,
    TrackingWeights,
    _marker_rms,
    _tracking_bounds,
    _tracking_objectives,
    _tracking_state_init,
    probe_marker_identifiability,
)

__all__ = [
    "ParameterBlockBundle",
    "ParameterOcpOptions",
    "add_parameter_block",
    "build_tracking_parameter_ocp",
    "solve_tracking_parameter_ocp",
]

logger = get_logger(__name__)


@dataclass(frozen=True)
class ParameterOcpOptions:
    """Execution options for the parameter-tracking OCP.

    ``max_iterations``: solver iteration limit.
    ``n_integration_steps``: RK4 sub-steps per shooting interval.
    ``n_threads``: worker threads for bioptim.
    ``gate``: pre-solve identifiability gate options (#9758).
    """

    max_iterations: int = 300
    n_integration_steps: int = 2
    n_threads: int = 1
    gate: IdentifiabilityGateOptions | None = None

    def __post_init__(self) -> None:
        require(self.max_iterations > 0, "max_iterations must be positive")
        require(self.n_integration_steps > 0, "n_integration_steps must be positive")
        require(self.n_threads > 0, "n_threads must be positive")


@dataclass(frozen=True)
class ParameterBlockBundle:
    """Container holding bioptim parameter structures and block metadata.

    Supports 4-tuple unpacking for bioptim OCP construction:
    ``params, bounds, init, objectives = bundle``.
    """

    parameters: Any
    parameter_bounds: Any
    parameter_init: Any
    parameter_objectives: Any
    block: SharedParameterBlock
    free_names: tuple[str, ...]
    locked_names: tuple[str, ...]

    def __iter__(self) -> Any:
        return iter(
            (
                self.parameters,
                self.parameter_bounds,
                self.parameter_init,
                self.parameter_objectives,
            )
        )


def _coerce_block(
    specs: Sequence[SharedParameterSpec] | SharedParameterBlock,
) -> SharedParameterBlock:
    """Validate parameter specifications and return a SharedParameterBlock."""
    block = (
        specs
        if isinstance(specs, SharedParameterBlock)
        else SharedParameterBlock.from_specs(specs)
    )
    unknown = [name for name in block.parameter_names if name not in PARAMETER_NAMES]
    if unknown:
        raise ValueError(
            f"unknown parameters {unknown}; allowed: {list(PARAMETER_NAMES)}"
        )
    return block


def add_parameter_block(
    model_or_ocp: Any,
    specs: Sequence[SharedParameterSpec] | SharedParameterBlock,
    *,
    bioptim: Any | None = None,
) -> ParameterBlockBundle:
    """Build bioptim ParameterList, Bounds, Init, and Objectives from specs.

    Args:
        model_or_ocp: A :class:`SwingBioModel` or :class:`OptimalControlProgram`.
        specs: Shared parameter specifications (locked specs remain numeric).
        bioptim: Optional bioptim module override.
    """
    block = _coerce_block(specs)
    biopt = bioptim or require_bioptim()

    model = (
        model_or_ocp
        if hasattr(model_or_ocp, "set_parameter")
        else getattr(getattr(model_or_ocp, "nlp", [None])[0], "model", model_or_ocp)
    )

    parameters = biopt.ParameterList(use_sx=False)
    parameter_bounds = biopt.BoundsList()
    parameter_init = biopt.InitialGuessList()
    parameter_objectives = biopt.ParameterObjectiveList()

    free_specs = block.free_specs
    free_names = block.free_parameter_names
    locked_names = tuple(s.name for s in block.specs if s.locked)

    for spec in free_specs:
        setter = getattr(model, "set_parameter", None)
        scaling = biopt.VariableScaling(spec.name, np.ones((1, 1)))
        parameters.add(
            name=spec.name,
            function=setter,
            size=1,
            scaling=scaling,
        )
        lower = -inf if spec.lower is None else float(spec.lower)
        upper = inf if spec.upper is None else float(spec.upper)
        parameter_bounds.add(
            spec.name,
            min_bound=np.array([[lower]]),
            max_bound=np.array([[upper]]),
            interpolation=biopt.InterpolationType.CONSTANT,
        )
        parameter_init.add(spec.name, np.array([[float(spec.initial)]]))

        if spec.prior is not None and spec.prior_scale is not None:
            param_obj = biopt.ObjectiveFcn.Parameter
            parameter_objectives.add(
                param_obj.MINIMIZE_PARAMETER,
                key=spec.name,
                weight=1.0 / (float(spec.prior_scale) ** 2),
                target=np.array([[float(spec.prior)]]),
                quadratic=True,
            )

    return ParameterBlockBundle(
        parameters=parameters,
        parameter_bounds=parameter_bounds,
        parameter_init=parameter_init,
        parameter_objectives=parameter_objectives,
        block=block,
        free_names=free_names,
        locked_names=locked_names,
    )


def _apply_gate(
    golfer: GolferModel,
    club: ClubModel,
    targets: MarkerTargets,
    block: SharedParameterBlock,
    q_guess: np.ndarray | None,
    options: IdentifiabilityGateOptions | None,
) -> tuple[SharedParameterBlock, IdentifiabilityGateReport | None]:
    """Evaluate marker sensitivity at initial guess and apply gate (#9758)."""
    gate_opts = options or IdentifiabilityGateOptions(policy="warn")
    if gate_opts.policy == "off" or block.free_size == 0:
        return block, None

    n_frames = targets.n_frames
    eval_q = (
        np.zeros((7, n_frames)) if q_guess is None else np.asarray(q_guess, dtype=float)
    )
    observed_mask = targets.weights > 0.0

    probe_model = SymbolicSwingModel(golfer, club, parameters=block.parameter_names)

    def residual_of_free(free_values: np.ndarray) -> np.ndarray:
        full_params = block.expand_free_vector(free_values)
        res_parts = []
        for k in range(n_frames):
            pred = np.asarray(probe_model.markers(eval_q[:, k], full_params))
            diff = pred - targets.positions[:, :, k]
            res_parts.append(diff[:, observed_mask[:, k]].reshape(-1))
        return np.concatenate(res_parts) if res_parts else np.zeros(0, dtype=float)

    report = gate_shared_parameters(residual_of_free, block, gate_opts)
    if report.locked_parameters:
        from dataclasses import replace

        locked_set = set(report.locked_parameters)
        new_specs = [
            replace(spec, locked=True) if spec.name in locked_set else spec
            for spec in block.specs
        ]
        return SharedParameterBlock.from_specs(new_specs), report
    return block, report


def build_tracking_parameter_ocp(
    targets: MarkerTargets,
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    parameters: Sequence[SharedParameterSpec] | SharedParameterBlock,
    weights: TrackingWeights | None = None,
    q_guess: np.ndarray | None = None,
    options: ParameterOcpOptions | None = None,
) -> tuple[Any, Any, ParameterBlockBundle, IdentifiabilityGateReport | None]:
    """Build the joint state + parameter tracking OCP."""
    biopt = require_bioptim()
    opts = options or ParameterOcpOptions()
    track_weights = weights or TrackingWeights()
    num_golfer = golfer or GolferModel()
    num_club = club or ClubModel()

    block = _coerce_block(parameters)
    gated_block, gate_report = _apply_gate(
        num_golfer, num_club, targets, block, q_guess, opts.gate
    )

    free_names = gated_block.free_parameter_names
    model = make_swing_bio_model(num_golfer, num_club, parameters=free_names)

    if tuple(targets.marker_names) != tuple(model.marker_names):
        raise ValueError("targets marker names must match model marker names")

    bundle = add_parameter_block(model, gated_block, bioptim=biopt)

    limits = model.symbolic.torque_limits()
    dynamics = biopt.DynamicsOptionsList()
    dynamics.add(
        biopt.DynamicsOptions(
            ode_solver=biopt.OdeSolver.RK4(
                n_integration_steps=opts.n_integration_steps
            ),
            expand_dynamics=True,
            phase_dynamics=biopt.PhaseDynamics.SHARED_DURING_THE_PHASE,
        )
    )
    x_bounds, u_bounds = _tracking_bounds(biopt, model, limits)
    n = model.nb_q

    u_init = biopt.InitialGuessList()
    u_init["tau"] = np.zeros(n)
    u_scaling = biopt.VariableScalingList()
    u_scaling.add("tau", scaling=limits)

    ocp = biopt.OptimalControlProgram(
        model,
        targets.n_frames - 1,
        targets.duration,
        dynamics=dynamics,
        x_bounds=x_bounds,
        u_bounds=u_bounds,
        x_init=_tracking_state_init(biopt, q_guess, targets, n),
        u_init=u_init,
        objective_functions=_tracking_objectives(biopt, targets, track_weights),
        parameters=bundle.parameters,
        parameter_bounds=bundle.parameter_bounds,
        parameter_init=bundle.parameter_init,
        parameter_objectives=bundle.parameter_objectives,
        u_scaling=u_scaling,
        use_sx=False,
        n_threads=opts.n_threads,
    )
    return ocp, model, bundle, gate_report


def solve_tracking_parameter_ocp(
    targets: MarkerTargets,
    golfer: GolferModel | None = None,
    club: ClubModel | None = None,
    *,
    parameters: Sequence[SharedParameterSpec] | SharedParameterBlock,
    weights: TrackingWeights | None = None,
    q_guess: np.ndarray | None = None,
    options: ParameterOcpOptions | None = None,
) -> TrackingResult:
    """Build and solve the simultaneous state and parameter estimation OCP."""
    biopt = require_bioptim()
    opts = options or ParameterOcpOptions()
    track_weights = weights or TrackingWeights()

    ocp, model, bundle, gate_report = build_tracking_parameter_ocp(
        targets,
        golfer,
        club,
        parameters=parameters,
        weights=track_weights,
        q_guess=q_guess,
        options=opts,
    )

    solver = biopt.Solver.IPOPT(show_online_optim=False)
    solver.set_print_level(0)
    solver.set_maximum_iterations(int(opts.max_iterations))
    if len(bundle.free_names) > 5:
        solver.set_hessian_approximation("limited-memory")

    start_time = time.perf_counter()
    sol = ocp.solve(solver=solver)
    wall = time.perf_counter() - start_time

    q, qdot, tau = solution_arrays(sol, biopt, n_nodes=targets.n_frames)

    solved_params: dict[str, float] = {
        spec.name: float(spec.initial) for spec in bundle.block.specs
    }
    if sol.status == 0:
        decision_params = sol.decision_parameters(scaled=False)
        for name in bundle.free_names:
            if name in decision_params:
                solved_params[name] = float(
                    np.asarray(decision_params[name]).ravel()[0]
                )

    param_vec = np.array(
        [solved_params[name] for name in bundle.free_names], dtype=float
    )
    marker_rms = _marker_rms(model, q, targets, param_vec)

    mid_pose = q[:, q.shape[1] // 2]
    identifiability = probe_marker_identifiability(
        model, mid_pose, parameters=param_vec
    )

    gate_locked = gate_report.locked_parameters if gate_report else ()
    all_locked = tuple(sorted(set(bundle.locked_names) | set(gate_locked)))

    settings = {
        "kind": "tracking_parameter",
        "n_shooting": targets.n_frames - 1,
        "final_time": targets.duration,
        "max_iterations": int(opts.max_iterations),
        "parameters": list(bundle.block.parameter_names),
        "free_parameters": list(bundle.free_names),
        "locked_parameters": list(all_locked),
    }

    return TrackingResult(
        time=targets.times.copy(),
        q=q,
        qdot=qdot,
        tau=tau,
        parameters=solved_params,
        marker_rms_m=marker_rms,
        status=int(sol.status),
        iterations=int(getattr(sol, "iterations", 0) or 0),
        cost=float(np.asarray(sol.cost).ravel()[0]),
        wall_time_s=wall,
        provenance=bioptim_provenance(model.golfer, model.club, settings),
        locked_by_gate=all_locked,
        identifiability=identifiability,
    )

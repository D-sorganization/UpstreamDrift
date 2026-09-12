"""Native Bernstein marker sensitivities checked against continuous replay."""

import json
from collections.abc import Callable, Mapping, Sequence
from typing import Any, NamedTuple, Protocol

import numpy as np
from numpy.typing import ArrayLike, NDArray

from src.engines.physics_engines.pinocchio.python.native_model import (
    NativeAccelerationDerivatives,
    NativeMarkerDerivatives,
    NativePinocchioModel,
)
from src.engines.physics_engines.pinocchio.python.native_replay import (
    NativeReplayEngine,
    NativeReplayResult,
    replay_candidate,
    replay_window,
)
from src.shared.python.motion_matching.forward_sensitivity import (
    integrate_sensitivities,
)
from src.shared.python.motion_matching.native_candidate import NativeReplayCandidate
from src.shared.python.motion_matching.native_effort_profile import NativeEffortProfile

Array = NDArray[np.float64]


class NativeSensitivityEngine(NativeReplayEngine, Protocol):
    """Native local derivative boundary required in addition to ordinary replay."""

    def acceleration_derivatives(
        self,
        coordinates: Mapping[str, float],
        rates: Mapping[str, float],
        primitive_efforts: Mapping[str, float],
    ) -> NativeAccelerationDerivatives: ...
    def marker_derivatives(
        self,
        coordinates: Mapping[str, float],
        bodies: Sequence[str],
        offsets: ArrayLike,
    ) -> NativeMarkerDerivatives: ...


class NativeMarkerSensitivityResult(NamedTuple):
    """Replay-qualified physical marker Jacobian, before optimizer scaling."""

    replay: NativeReplayResult
    marker_jacobian: Array
    sensitivity_elapsed_s: float
    sensitivity_evaluations: int
    primal_marker_max_abs_difference_m: float
    state_jacobian: Array


def replay_marker_sensitivities(
    model_bytes: bytes,
    candidate: NativeReplayCandidate,
    time_s: Array,
    *,
    first_control: int = 4,
    initial_state: Array | None = None,
    initial_sensitivity: Array | None = None,
    model_factory: Callable[
        [Mapping[str, Any]], NativeSensitivityEngine
    ] = NativePinocchioModel,
    rtol: float = 1e-10,
    atol: float = 1e-12,
    max_step: float = 0.00025,
) -> NativeMarkerSensitivityResult:
    """Integrate all selected native control columns with one initial state.

    Optional initial_sensitivity columns are state tangent directions at the
    supplied window initial_state; the caller must qualify their closure
    consistency. Those columns follow all effort columns. Returned state_jacobian
    supports shooting defects; the caller owns nonlinear node retraction.

    Columns are coordinate-major, ascending Bernstein index through six, in
    physical N/Nm. Ordinary replay validates model identity, full clock and weld;
    the augmented trajectory must additionally agree in marker positions within
    1e-7 m and sampled weld components within 1e-7. These are numerical checks,
    not C3D fit acceptance. Geometry and initial state are fixed.
    """
    if (
        isinstance(first_control, bool)
        or not isinstance(first_control, int)
        or not 0 <= first_control <= 6
    ):
        raise ValueError("Invalid first Bernstein control")
    if initial_sensitivity is not None and initial_state is None:
        raise ValueError("initial_sensitivity requires an explicit initial_state")
    if initial_state is None:
        primal = replay_candidate(
            model_bytes,
            candidate,
            time_s,
            model_factory=model_factory,
            rtol=1e-11,
            atol=1e-13,
            max_step=max_step,
        )
    else:
        primal = replay_window(
            model_bytes,
            candidate,
            time_s,
            initial_state,
            model_factory=model_factory,
            rtol=1e-11,
            atol=1e-13,
            max_step=max_step,
        )
    spec = json.loads(model_bytes)
    names = spec["coordinate_order"]
    n = len(names)
    effort_parameters = n * (7 - first_control)
    node_directions = (
        np.empty((2 * n, 0))
        if initial_sensitivity is None
        else np.asarray(initial_sensitivity, dtype=float)
    )
    if (
        node_directions.ndim != 2
        or node_directions.shape[0] != 2 * n
        or not np.isfinite(node_directions).all()
    ):
        raise ValueError("Invalid initial_sensitivity directions")
    parameters = effort_parameters + node_directions.shape[1]
    initial_jacobian = np.column_stack(
        (np.zeros((2 * n, effort_parameters)), node_directions)
    )
    doc = candidate.document
    root = next(joint for joint in spec["joints"] if joint["parent"] == "world")
    profile = NativeEffortProfile(
        names, doc["coefficients"], np.asarray(root["parent_to_base"])[:3, :3].T
    )
    engine = model_factory(spec)

    def mapping(values: Array) -> dict[str, float]:
        return dict(zip(names, map(float, values), strict=True))

    def linearize(t: float, state: Array) -> tuple[Array, Array, Array]:
        q, v = mapping(state[:n]), mapping(state[n:])
        effort = profile.evaluate(t)
        acceleration = engine.accelerations(q, v, effort)
        local = engine.acceleration_derivatives(q, v, effort)
        if local.names != tuple(names):
            raise ValueError("Native acceleration derivative coordinate order differs")
        a = np.block([[np.zeros((n, n)), np.eye(n)], [local.dq, local.dv]])
        inputs = profile.bernstein_control_jacobian(
            t, basis_duration_s=doc["duration_s"], first_control=first_control
        )
        b = np.vstack(
            (
                np.zeros((n, parameters)),
                np.column_stack(
                    (local.deffort @ inputs, np.zeros((n, node_directions.shape[1])))
                ),
            )
        )
        return np.concatenate((state[n:], [acceleration[name] for name in names])), a, b

    integrated = integrate_sensitivities(
        primal.integration.state[0],
        time_s - time_s[0],
        lambda t, state: linearize(t + float(time_s[0]), state),
        parameters,
        initial_sensitivity=initial_jacobian,
        rtol=rtol,
        atol=atol,
        max_step=max_step,
    )
    positions, derivatives = [], []
    for t, state, sensitivity in zip(
        time_s,
        integrated.integration.state,
        integrated.state_parameter_jacobian,
        strict=True,
    ):
        q, v = mapping(state[:n]), mapping(state[n:])
        marker = engine.marker_derivatives(
            q, doc["marker_bodies"], doc["marker_offsets_m"]
        )
        if marker.names != tuple(names):
            raise ValueError("Native marker derivative coordinate order differs")
        positions.append(marker.positions_m)
        derivatives.append(
            np.einsum("mcn,np->mcp", marker.dposition_dq, sensitivity[:n])
        )
        engine.accelerations(q, v, profile.evaluate(float(t)))
        closure = engine.closure_errors()
        if any(
            value.shape != (6,)
            or not np.isfinite(value).all()
            or np.max(np.abs(value)) > 1e-7
            for value in closure
        ):
            raise ValueError("Sensitivity trajectory violates sampled native closure")
    points = np.asarray(positions)
    jacobian = np.asarray(derivatives)
    if (
        points.shape != primal.markers_m.shape
        or jacobian.shape != (*points.shape, parameters)
        or not np.isfinite(points).all()
        or not np.isfinite(jacobian).all()
    ):
        raise ValueError("Invalid native marker sensitivity arrays")
    difference = float(np.max(np.abs(points - primal.markers_m)))
    if difference > 1e-7:
        raise ValueError(
            "Sensitivity primal marker replay exceeds numerical agreement bound"
        )
    jacobian.setflags(write=False)
    return NativeMarkerSensitivityResult(
        primal,
        jacobian,
        integrated.integration.elapsed_s,
        integrated.integration.evaluations,
        difference,
        integrated.state_parameter_jacobian,
    )

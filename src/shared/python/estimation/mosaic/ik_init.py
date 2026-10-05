"""Frame-parallel inverse-kinematics initialisation for MOSAIC.

Every frame is an independent small Gauss-Newton problem, so all frames are
solved simultaneously with batched linear algebra.  The result seeds the
kinematic spline; it is an initialiser, never the physical answer.
"""

from __future__ import annotations

from typing import Any, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

from src.shared.python.core.contracts import require

FloatArray: TypeAlias = npt.NDArray[np.float64]
_DEFAULT_DAMPING = 1e-9
_DEFAULT_ITERATIONS = 25
_DEFAULT_STEP_TOLERANCE = 1e-12


class MarkerModel(Protocol):
    """Minimal forward-kinematics surface required by the initialiser."""

    def marker_positions(self, q: FloatArray, markers: Any) -> FloatArray: ...

    def marker_jacobians(
        self, q: FloatArray, markers: Any
    ) -> tuple[FloatArray, FloatArray]: ...


def _gauss_newton_step(
    model: MarkerModel,
    markers: Any,
    observed: FloatArray,
    q: FloatArray,
    damping: float,
) -> FloatArray:
    residual = (model.marker_positions(q, markers) - observed).reshape(q.shape[0], -1)
    jac_q, _ = model.marker_jacobians(q, markers)
    jacobian = jac_q.reshape(q.shape[0], -1, q.shape[1])
    normal = np.einsum("nri,nrj->nij", jacobian, jacobian)
    normal += damping * np.eye(q.shape[1])[None]
    gradient = np.einsum("nri,nr->ni", jacobian, residual)
    return -np.linalg.solve(normal, gradient[..., None])[..., 0]


def initialize_joint_angles(
    model: MarkerModel,
    observed: FloatArray,
    markers: Any,
    initial_q: FloatArray,
    iterations: int = _DEFAULT_ITERATIONS,
    damping: float = _DEFAULT_DAMPING,
) -> FloatArray:
    """Return per-frame joint angles minimising marker error from ``initial_q``.

    Preconditions: ``observed`` is ``(T, n_markers, 2)``, ``initial_q`` is
    ``(T, n_dof)``, ``iterations >= 1`` and ``damping >= 0``.  Frames with
    non-finite observations are not supported here (mask upstream).
    """
    require(observed.ndim == 3, "observed must be (T, n_markers, 2)", observed.shape)
    require(
        initial_q.ndim == 2 and initial_q.shape[0] == observed.shape[0],
        "one q per frame",
    )
    require(iterations >= 1, "iterations must be >= 1", iterations)
    require(damping >= 0.0, "damping must be non-negative", damping)
    require(bool(np.all(np.isfinite(observed))), "observations must be finite")
    q = np.array(initial_q, dtype=np.float64, copy=True)
    for _ in range(iterations):
        step = _gauss_newton_step(model, markers, observed, q, damping)
        q += step
        if float(np.max(np.abs(step))) < _DEFAULT_STEP_TOLERANCE:
            break
    return q


def initialize_trajectory(
    model: MarkerModel,
    observed: FloatArray,
    markers: Any,
    first_frame_q: FloatArray,
    iterations: int = _DEFAULT_ITERATIONS,
    damping: float = _DEFAULT_DAMPING,
) -> FloatArray:
    """Sequentially warm-started IK over a trajectory, then a parallel polish.

    Frame ``t`` starts from the solution of frame ``t-1`` so the branch (elbow
    up/down, 2*pi wrap) is tracked continuously; the result is unwrapped along
    time and refined jointly with :func:`initialize_joint_angles`.  Returns a
    ``(T, n_dof)`` array.
    """
    require(first_frame_q.ndim == 1, "first_frame_q must be a single configuration")
    frames = observed.shape[0]
    trajectory = np.empty((frames, first_frame_q.size), dtype=np.float64)
    seed = first_frame_q[None, :]
    for frame in range(frames):
        seed = initialize_joint_angles(
            model, observed[frame : frame + 1], markers, seed, iterations, damping
        )
        trajectory[frame] = seed[0]
    unwrapped = np.asarray(np.unwrap(trajectory, axis=0), dtype=np.float64)
    return initialize_joint_angles(
        model, observed, markers, unwrapped, iterations, damping
    )

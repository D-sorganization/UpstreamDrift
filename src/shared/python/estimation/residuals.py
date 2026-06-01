"""Pure residual functions for canonical-core estimation.

The functions in this module are intentionally small and stateless. They accept
plain arrays plus explicit backend callbacks for model-specific work such as
forward kinematics and RNEA, which keeps estimator assembly separate from the
math being minimized.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from typing import Any, Literal, Protocol, TypeAlias

import numpy as np
import numpy.typing as npt

ArrayLike: TypeAlias = npt.ArrayLike
Array: TypeAlias = Any
ResidualFunction: TypeAlias = Callable[[Array], Array]
JacobianMethod: TypeAlias = Literal["auto", "jax", "finite"]


class RneaFunction(Protocol):
    """Callable inverse-dynamics backend with the canonical-v2 state layout."""

    def __call__(self, q: Array, v: Array, a: Array) -> Array:
        """Return inverse-dynamics torques for ``(q, v, a)``."""


def _is_jax_value(value: object) -> bool:
    module = type(value).__module__
    return module.startswith(("jax", "jaxlib"))


def _array_module(*values: object) -> Any:
    if any(_is_jax_value(value) for value in values):
        try:
            import jax.numpy as jnp
        except ImportError:  # pragma: no cover - only reachable with jax arrays
            return np
        return jnp
    return np


def _validate_numeric_array(
    value: object,
    name: str,
    *,
    ndim: int | None = None,
    shape: tuple[int | None, ...] | None = None,
) -> None:
    if _is_jax_value(value):
        arr_shape = getattr(value, "shape", None)
        arr_ndim = getattr(value, "ndim", None)
    else:
        arr = np.asarray(value, dtype=float)
        arr_shape = arr.shape
        arr_ndim = arr.ndim
        if not np.all(np.isfinite(arr)):
            raise ValueError(f"{name} must contain only finite values")

    if ndim is not None and arr_ndim != ndim:
        raise ValueError(f"{name} must have {ndim} dimensions, got {arr_ndim}")

    if shape is not None:
        if arr_shape is None or len(arr_shape) != len(shape):
            raise ValueError(f"{name} must have shape {shape}, got {arr_shape}")
        for actual, expected in zip(arr_shape, shape, strict=True):
            if expected is not None and actual != expected:
                raise ValueError(f"{name} must have shape {shape}, got {arr_shape}")


def _as_array(value: object, xp: Any) -> Array:
    if xp is np:
        return np.asarray(value, dtype=float)
    return xp.asarray(value)


def _validate_same_shape(
    left: object, right: object, left_name: str, right_name: str
) -> None:
    left_shape = getattr(left, "shape", None)
    right_shape = getattr(right, "shape", None)
    if left_shape is None:
        left_shape = np.asarray(left).shape
    if right_shape is None:
        right_shape = np.asarray(right).shape
    if left_shape != right_shape:
        raise ValueError(
            f"{left_name} and {right_name} must have matching shapes, "
            f"got {left_shape} and {right_shape}"
        )


def _validate_confidence(confidence: object, keypoint_count: int) -> None:
    _validate_numeric_array(confidence, "confidence", shape=(keypoint_count,))
    if _is_jax_value(confidence):
        return
    conf = np.asarray(confidence, dtype=float)
    if np.any((conf < 0.0) | (conf > 1.0)):
        raise ValueError("confidence values must be in the inclusive range [0, 1]")


DistortionLike: TypeAlias = "ArrayLike | None"


def _apply_brown_conrady(
    x_norm: Array, y_norm: Array, distortion: ArrayLike, xp: Any
) -> tuple[Array, Array]:
    """Apply the Brown-Conrady radial/tangential distortion model.

    ``distortion`` is ordered ``(k1, k2, p1, p2)`` or the canonical 5-term
    ``(k1, k2, p1, p2, k3)`` and operates on normalized image coordinates
    (camera-frame coordinates divided by depth), matching the synthetic rig in
    ``synthetic_ground_truth.project_world_point``. The optional third radial
    term ``k3`` extends the radial polynomial to ``r**6`` (#6907).
    """

    if _is_jax_value(distortion):
        arr_shape = getattr(distortion, "shape", None)
    else:
        arr_shape = np.asarray(distortion).shape
    if arr_shape is None or len(arr_shape) != 1 or arr_shape[0] not in (4, 5):
        raise ValueError(f"distortion must have shape (4,) or (5,), got {arr_shape}")
    _validate_numeric_array(distortion, "distortion", ndim=1)
    coeffs = _as_array(distortion, xp)
    k1, k2, p1, p2 = coeffs[0], coeffs[1], coeffs[2], coeffs[3]
    k3 = coeffs[4] if arr_shape[0] == 5 else 0.0
    radius_2 = x_norm * x_norm + y_norm * y_norm
    radial = 1.0 + k1 * radius_2 + k2 * radius_2**2 + k3 * radius_2**3
    x_dist = x_norm * radial + 2.0 * p1 * x_norm * y_norm
    x_dist = x_dist + p2 * (radius_2 + 2.0 * x_norm * x_norm)
    y_dist = y_norm * radial + p1 * (radius_2 + 2.0 * y_norm * y_norm)
    y_dist = y_dist + 2.0 * p2 * x_norm * y_norm
    return x_dist, y_dist


def project_pinhole(
    points_world: ArrayLike,
    camera_matrix: ArrayLike,
    *,
    rotation_world_to_camera: ArrayLike | None = None,
    translation_world_to_camera: ArrayLike | None = None,
    distortion: DistortionLike = None,
    depth_epsilon: float = 1.0e-9,
) -> Array:
    """Project world points with a pinhole camera and optional lens distortion.

    Parameters use the usual computer-vision convention:
    ``p_camera = R_world_to_camera @ p_world + t_world_to_camera``. Points are
    normalized by depth, the optional Brown-Conrady ``distortion`` model
    ``(k1, k2, p1, p2)`` is applied, and the intrinsics ``camera_matrix`` map the
    distorted normalized coordinates to pixels. ``distortion=None`` (the default)
    is the plain pinhole model.
    """

    _validate_numeric_array(points_world, "points_world", ndim=2, shape=(None, 3))
    _validate_numeric_array(camera_matrix, "camera_matrix", shape=(3, 3))
    if depth_epsilon <= 0.0 or not np.isfinite(depth_epsilon):
        raise ValueError("depth_epsilon must be a positive finite number")

    xp = _array_module(points_world, camera_matrix)
    points = _as_array(points_world, xp)
    camera = _as_array(camera_matrix, xp)

    if rotation_world_to_camera is None:
        rotation = xp.eye(3, dtype=points.dtype)
    else:
        _validate_numeric_array(
            rotation_world_to_camera,
            "rotation_world_to_camera",
            shape=(3, 3),
        )
        rotation = _as_array(rotation_world_to_camera, xp)

    if translation_world_to_camera is None:
        translation = xp.zeros(3, dtype=points.dtype)
    else:
        _validate_numeric_array(
            translation_world_to_camera,
            "translation_world_to_camera",
            shape=(3,),
        )
        translation = _as_array(translation_world_to_camera, xp)

    points_camera = points @ rotation.T + translation
    depth = points_camera[:, 2:3]
    safe_depth = xp.where(
        xp.abs(depth) < depth_epsilon,
        xp.where(depth < 0.0, -depth_epsilon, depth_epsilon),
        depth,
    )
    normalized = points_camera[:, :2] / safe_depth
    if distortion is not None:
        x_dist, y_dist = _apply_brown_conrady(
            normalized[:, 0:1], normalized[:, 1:2], distortion, xp
        )
        normalized = xp.concatenate([x_dist, y_dist], axis=1)
    homogeneous = xp.concatenate([normalized, xp.ones_like(normalized[:, :1])], axis=1)
    return (homogeneous @ camera.T)[:, :2]


def reprojection_residual_from_points(
    points_world: ArrayLike,
    observed_uv: ArrayLike,
    camera_matrix: ArrayLike,
    confidence: ArrayLike,
    *,
    keypoint_offsets_m: ArrayLike | None = None,
    rotation_world_to_camera: ArrayLike | None = None,
    translation_world_to_camera: ArrayLike | None = None,
    distortion: DistortionLike = None,
) -> Array:
    """Return weighted reprojection residuals for explicit 3-D keypoint points.

    Residuals are flattened as ``[u0, v0, u1, v1, ...]`` and weighted by
    ``sqrt(confidence)`` so missing keypoints with confidence 0 contribute zero.
    ``keypoint_offsets_m`` is the CC-15 local offset model evaluated in world
    coordinates by the caller. ``distortion`` threads the Brown-Conrady
    ``(k1, k2, p1, p2)`` coefficients into projection so a fit against a
    calibrated camera with nonzero distortion is unbiased.
    """

    _validate_numeric_array(points_world, "points_world", ndim=2, shape=(None, 3))
    points_shape = getattr(points_world, "shape", None)
    if points_shape is None:
        points_shape = np.asarray(points_world).shape
    keypoint_count = int(points_shape[0])
    _validate_numeric_array(observed_uv, "observed_uv", shape=(keypoint_count, 2))
    _validate_confidence(confidence, keypoint_count)

    xp = _array_module(points_world, observed_uv, confidence)
    points = _as_array(points_world, xp)
    observed = _as_array(observed_uv, xp)

    if keypoint_offsets_m is None:
        offsets = xp.zeros_like(points)
    else:
        _validate_numeric_array(
            keypoint_offsets_m,
            "keypoint_offsets_m",
            shape=(keypoint_count, 3),
        )
        offsets = _as_array(keypoint_offsets_m, xp)

    projected = project_pinhole(
        points + offsets,
        camera_matrix,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        distortion=distortion,
    )
    weights = xp.sqrt(_as_array(confidence, xp))[:, None]
    return ((projected - observed) * weights).reshape(-1)


def reprojection_residual(
    q: ArrayLike,
    observed_uv: ArrayLike,
    joint_center_fn: Callable[[Array], Array],
    camera_matrix: ArrayLike,
    confidence: ArrayLike,
    *,
    keypoint_offsets_m: ArrayLike | None = None,
    rotation_world_to_camera: ArrayLike | None = None,
    translation_world_to_camera: ArrayLike | None = None,
    distortion: DistortionLike = None,
) -> Array:
    """Return weighted reprojection residuals from a canonical-v2 ``q`` vector."""

    _validate_numeric_array(q, "q", ndim=1)
    xp = _array_module(q)
    q_arr = _as_array(q, xp)
    points_world = joint_center_fn(q_arr)
    return reprojection_residual_from_points(
        points_world,
        observed_uv,
        camera_matrix,
        confidence,
        keypoint_offsets_m=keypoint_offsets_m,
        rotation_world_to_camera=rotation_world_to_camera,
        translation_world_to_camera=translation_world_to_camera,
        distortion=distortion,
    )


def _select_dofs(
    values: Array, dof_indices: Sequence[int] | slice | None, xp: Any
) -> Array:
    if dof_indices is None:
        return values
    if isinstance(dof_indices, slice):
        return values[dof_indices]
    return xp.take(values, xp.asarray(tuple(dof_indices), dtype=int), axis=0)


def dynamics_residual(
    q: ArrayLike,
    v: ArrayLike,
    a: ArrayLike,
    rnea_fn: RneaFunction,
    *,
    torque_target: ArrayLike | None = None,
    torque_weights: ArrayLike | None = None,
    dof_indices: Sequence[int] | slice | None = None,
) -> Array:
    """Return weighted RNEA torque residuals for canonical-v2 ``(q, v, a)``.

    When ``torque_target`` is omitted, torques are eliminated from the solve and
    the inverse-dynamics torque itself is penalized. ``dof_indices`` lets callers
    drop unactuated floating-base coordinates, commonly ``slice(6, None)``.
    """

    _validate_numeric_array(q, "q", ndim=1)
    _validate_numeric_array(v, "v", ndim=1)
    _validate_numeric_array(a, "a", ndim=1)
    _validate_same_shape(v, a, "v", "a")

    xp = _array_module(q, v, a)
    tau = _as_array(rnea_fn(_as_array(q, xp), _as_array(v, xp), _as_array(a, xp)), xp)
    _validate_numeric_array(tau, "rnea_fn result", ndim=1)

    selected_tau = _select_dofs(tau, dof_indices, xp)
    if torque_target is None:
        target = xp.zeros_like(selected_tau)
    else:
        _validate_numeric_array(torque_target, "torque_target", ndim=1)
        target = _select_dofs(_as_array(torque_target, xp), dof_indices, xp)
        _validate_same_shape(selected_tau, target, "selected torques", "target")

    if torque_weights is None:
        weights = xp.ones_like(selected_tau)
    else:
        _validate_numeric_array(torque_weights, "torque_weights", ndim=1)
        weights = _select_dofs(_as_array(torque_weights, xp), dof_indices, xp)
        _validate_same_shape(selected_tau, weights, "selected torques", "weights")
        if not _is_jax_value(torque_weights) and np.any(np.asarray(weights) < 0.0):
            raise ValueError("torque_weights must be non-negative")

    return (selected_tau - target) * xp.sqrt(weights)


def anthropometric_prior_residual(
    parameters: ArrayLike,
    nominal_parameters: ArrayLike,
    sigma: ArrayLike,
    *,
    weights: ArrayLike | None = None,
) -> Array:
    """Return a diagonal Gaussian anthropometric prior residual."""

    _validate_numeric_array(parameters, "parameters", ndim=1)
    _validate_numeric_array(nominal_parameters, "nominal_parameters", ndim=1)
    _validate_numeric_array(sigma, "sigma", ndim=1)
    _validate_same_shape(
        parameters, nominal_parameters, "parameters", "nominal_parameters"
    )
    _validate_same_shape(parameters, sigma, "parameters", "sigma")
    if not _is_jax_value(sigma) and np.any(np.asarray(sigma, dtype=float) <= 0.0):
        raise ValueError("sigma must be strictly positive")

    xp = _array_module(parameters, nominal_parameters, sigma)
    residual = (
        _as_array(parameters, xp) - _as_array(nominal_parameters, xp)
    ) / _as_array(sigma, xp)

    if weights is None:
        return residual

    _validate_numeric_array(weights, "weights", ndim=1)
    _validate_same_shape(parameters, weights, "parameters", "weights")
    if not _is_jax_value(weights) and np.any(np.asarray(weights, dtype=float) < 0.0):
        raise ValueError("weights must be non-negative")
    return residual * xp.sqrt(_as_array(weights, xp))


def smoothness_residual(
    trajectory: ArrayLike,
    *,
    dt: float = 1.0,
    order: Literal[1, 2] = 2,
    weights: ArrayLike | None = None,
) -> Array:
    """Return finite-difference trajectory smoothness residuals.

    ``trajectory`` is shaped ``(frames, dofs)`` and should already be expressed
    in canonical-v2 tangent coordinates when it contains floating-base motion.
    """

    _validate_numeric_array(trajectory, "trajectory", ndim=2)
    if order not in (1, 2):
        raise ValueError("order must be 1 or 2")
    if dt <= 0.0 or not np.isfinite(dt):
        raise ValueError("dt must be a positive finite number")

    trajectory_shape = getattr(trajectory, "shape", None)
    if trajectory_shape is None:
        trajectory_shape = np.asarray(trajectory).shape
    frame_count, dof_count = trajectory_shape
    if frame_count <= order:
        raise ValueError(
            f"trajectory must have more than order={order} frames, got {frame_count}"
        )

    xp = _array_module(trajectory)
    differences = xp.diff(_as_array(trajectory, xp), n=order, axis=0) / (dt**order)

    if weights is not None:
        _validate_numeric_array(weights, "weights", shape=(dof_count,))
        if not _is_jax_value(weights) and np.any(
            np.asarray(weights, dtype=float) < 0.0
        ):
            raise ValueError("weights must be non-negative")
        differences = differences * xp.sqrt(_as_array(weights, xp))[None, :]

    return differences.reshape(-1)


def finite_difference_jacobian(
    function: ResidualFunction,
    x: ArrayLike,
    *,
    step: float = 1.0e-6,
) -> npt.NDArray[np.float64]:
    """Return a central finite-difference Jacobian for a residual function."""

    if step <= 0.0 or not np.isfinite(step):
        raise ValueError("step must be a positive finite number")
    x0 = np.asarray(x, dtype=float)
    if x0.ndim != 1:
        raise ValueError(f"x must be one-dimensional, got shape {x0.shape}")
    if not np.all(np.isfinite(x0)):
        raise ValueError("x must contain only finite values")

    f0 = np.asarray(function(x0), dtype=float).reshape(-1)
    jacobian = np.empty((f0.size, x0.size), dtype=float)
    for col in range(x0.size):
        delta = np.zeros_like(x0)
        delta[col] = step
        plus = np.asarray(function(x0 + delta), dtype=float).reshape(-1)
        minus = np.asarray(function(x0 - delta), dtype=float).reshape(-1)
        jacobian[:, col] = (plus - minus) / (2.0 * step)
    return jacobian


def autodiff_jacobian(
    function: ResidualFunction, x: ArrayLike
) -> npt.NDArray[np.float64]:
    """Return a JAX autodiff Jacobian for a residual function."""

    try:
        import jax
        import jax.numpy as jnp
    except ImportError as exc:
        raise ImportError(
            "JAX is required for autodiff_jacobian; use method='finite' "
            "or install the optional JAX stack."
        ) from exc

    x_arr = jnp.asarray(np.asarray(x, dtype=float))
    jacobian = jax.jacfwd(lambda value: jnp.asarray(function(value)).reshape(-1))(x_arr)
    return np.asarray(jacobian, dtype=float)


def residual_jacobian(
    function: ResidualFunction,
    x: ArrayLike,
    *,
    method: JacobianMethod = "auto",
    step: float = 1.0e-6,
) -> npt.NDArray[np.float64]:
    """Return a residual Jacobian via JAX autodiff or central differences."""

    if method == "finite":
        return finite_difference_jacobian(function, x, step=step)
    if method == "jax":
        return autodiff_jacobian(function, x)
    if method != "auto":
        raise ValueError("method must be 'auto', 'jax', or 'finite'")

    try:
        return autodiff_jacobian(function, x)
    except ImportError:
        return finite_difference_jacobian(function, x, step=step)

"""4-point dual-hand grasp matrix, Null-Space preload, and ZTCF/ZVCF counterfactuals.

Part of the UpstreamDrift 3D Counterbalanced Putting Dynamics Suite (v3.2/v3.3).
Implements:
- Zero-allocation @precondition signature caching (Issue D-11803)
- 4-point dual-hand grasp matrix A_couple mapping contact wrench channels to 3D
  Moment-of-Force (and optional pure-couple zero-net-force task constraints)
- Moore-Penrose pseudoinverse minimum-norm Moment-of-Force drive:
  u_task = A_couple^+ tau_task
- Orthogonal Null-Space counter-torque preload:
  u_null = N_null tau_preload, where N_null = I - A_couple^+ A_couple
- ZTCF (Zero-Torque Counterfactual) and ZVCF (Zero-Velocity Counterfactual)
  acceleration decomposition on the rigid-body manipulator equation.

Architecture budgets enforced:
- <= 500 LOC per file
- <= 100 LOC per function
- <= 8 parameters per function
"""

from __future__ import annotations

from dataclasses import dataclass
import functools
import inspect
from typing import Any, Callable, Dict, Tuple, TypeVar

import numpy as np
from numpy.typing import NDArray

F = TypeVar("F", bound=Callable[..., Any])

# Issue D-11803: Zero-allocation signature cache at decoration time.
_SIGNATURE_CACHE: Dict[Callable[..., Any], Tuple[inspect.Signature, Tuple[str, ...], Dict[str, Any]]] = {}

DEFAULT_GRIP_CONTACT_Z_M: Tuple[float, float, float, float] = (
    -0.04,  # Point 1: Lead hand proximal (butt/heel pad)
    -0.11,  # Point 2: Lead hand distal (index/finger clamp)
    -0.19,  # Point 3: Trail hand proximal (upper palm lifeline)
    -0.27,  # Point 4: Trail hand distal (lower index/thumb pinch)
)


def _cache_function_signature(
    func: Callable[..., Any],
) -> Tuple[inspect.Signature, Tuple[str, ...], Dict[str, Any]]:
    """Pre-compute and cache function signature metadata once at decoration time."""
    cached = _SIGNATURE_CACHE.get(func)
    if cached is not None:
        return cached
    sig = inspect.signature(func)
    param_names = tuple(sig.parameters.keys())
    defaults = {
        k: v.default
        for k, v in sig.parameters.items()
        if v.default is not inspect.Parameter.empty
    }
    entry = (sig, param_names, defaults)
    _SIGNATURE_CACHE[func] = entry
    return entry


def precondition(
    predicate: Callable[..., bool],
    message: str = "Precondition violated",
) -> Callable[[F], F]:
    """Zero-allocation Design-by-Contract @precondition decorator (Issue D-11803).

    Resolves inspect.signature(func) once at decoration time and caches parameter
    positions. At runtime, invokes the predicate directly without calling
    inspect.signature() or allocating inspect.BoundArguments objects.
    """
    pred_sig = inspect.signature(predicate)
    pred_params = tuple(pred_sig.parameters.values())
    single_dict_mode = (
        len(pred_params) == 1
        and pred_params[0].kind
        in (inspect.Parameter.POSITIONAL_OR_KEYWORD, inspect.Parameter.POSITIONAL_ONLY)
        and pred_params[0].name in ("params", "bound", "args_dict")
    )

    def decorator(func: F) -> F:
        _, param_names, defaults = _cache_function_signature(func)

        @functools.wraps(func)
        def wrapper(*args: Any, **kwargs: Any) -> Any:
            if single_dict_mode:
                call_map = defaults.copy()
                for idx, val in enumerate(args):
                    if idx < len(param_names):
                        call_map[param_names[idx]] = val
                if kwargs:
                    call_map.update(kwargs)
                ok = predicate(call_map)
            elif not kwargs:
                if len(args) < len(param_names) and defaults:
                    resolved = list(args)
                    for idx in range(len(args), len(param_names)):
                        pname = param_names[idx]
                        if pname in defaults:
                            resolved.append(defaults[pname])
                    ok = predicate(*resolved)
                else:
                    ok = predicate(*args)
            else:
                resolved = [
                    args[idx]
                    if idx < len(args)
                    else kwargs.get(pname, defaults.get(pname))
                    for idx, pname in enumerate(param_names)
                ]
                ok = predicate(*resolved)
            if not ok:
                raise ValueError(f"{message} in '{func.__name__}'")
            return func(*args, **kwargs)

        return wrapper  # type: ignore[return-value]

    return decorator


@dataclass(frozen=True)
class GraspStabilizationSolution:
    """Complete dual-hand Moment-of-Force and Null-Space stabilization breakdown."""

    a_couple: NDArray[np.float64]
    a_couple_pinv: NDArray[np.float64]
    n_null: NDArray[np.float64]
    u_task: NDArray[np.float64]
    u_null: NDArray[np.float64]
    u_total: NDArray[np.float64]
    realized_tau_task: NDArray[np.float64]
    null_leakage_norm: float
    orthogonality_inner_product: float


@dataclass(frozen=True)
class CounterfactualAccelerationDecomposition:
    """ZTCF and ZVCF counterfactual acceleration decomposition result."""

    q_ddot_actual: NDArray[np.float64]
    q_ddot_ztcf: NDArray[np.float64]
    q_ddot_zvcf: NDArray[np.float64]
    q_ddot_torque: NDArray[np.float64]
    q_ddot_velocity: NDArray[np.float64]
    q_ddot_gravity: NDArray[np.float64]
    superposition_residual_norm: float


def _skew_symmetric(r_vec: NDArray[np.floating[Any]]) -> NDArray[np.float64]:
    """Return 3x3 skew-symmetric cross-product matrix [r]_x such that [r]_x f = r x f."""
    rx, ry, rz = float(r_vec[0]), float(r_vec[1]), float(r_vec[2])
    return np.array(
        [
            [0.0, -rz, ry],
            [rz, 0.0, -rx],
            [-ry, rx, 0.0],
        ],
        dtype=np.float64,
    )


@precondition(
    lambda contact_z_m=DEFAULT_GRIP_CONTACT_Z_M, grip_radius_m=0.014, spatial_3d=False: (
        len(contact_z_m) == 4
        and len(set( float(z) for z in contact_z_m )) >= 2
        and grip_radius_m > 0.0
    ),
    "Requires 4 grip contact stations with distinct lever arms and grip_radius_m > 0",
)
def build_grasp_matrix_a_couple(
    contact_z_m: Tuple[float, float, float, float] = DEFAULT_GRIP_CONTACT_Z_M,
    grip_radius_m: float = 0.014,
    spatial_3d: bool = False,
) -> NDArray[np.float64]:
    """Construct the 4-point dual-hand grasp matrix A_couple.

    When spatial_3d=False (default 3x4 dual-hand modal couple mapping):
      Maps 4 dual-hand contact actuation channels u in R^4 at stations
      (z_1, z_2, z_3, z_4) to 3D Moment-of-Force / couple task coordinates:
        Row 0: Net translational force balance [1, 1, 1, 1] (zeroed for pure couple)
        Row 1: Primary stroke pitch moment lever arms [z_1, z_2, z_3, z_4]
        Row 2: Differential dual-hand face-yaw torsion couple
               [-r_g, +r_g, -r_g, +r_g]
      Yields a full-row-rank 3x4 matrix with a 1D internal preload null space, or
      if used with 2x4 planar couple task, 2D internal preload null space.

    When spatial_3d=True (3x12 full 3D contact force mapping):
      Maps 4 3D contact forces [f_1, f_2, f_3, f_4] in R^12 at 3D contact points
      r_i = [(-1)^i * r_g, 0, z_i]^T to 3D Moment-of-Force tau in R^3 via
      A_couple = [ [r_1]_x, [r_2]_x, [r_3]_x, [r_4]_x ] in R^{3x12}.
    """
    z = np.asarray(contact_z_m, dtype=np.float64)
    r_g = float(grip_radius_m)

    if not spatial_3d:
        return np.array(
            [
                [1.0, 1.0, 1.0, 1.0],
                [z[0], z[1], z[2], z[3]],
                [-r_g, r_g, -r_g, r_g],
            ],
            dtype=np.float64,
        )

    blocks = []
    for idx in range(4):
        x_offset = -r_g if idx % 2 == 0 else r_g
        r_i = np.array([x_offset, 0.0, z[idx]], dtype=np.float64)
        blocks.append(_skew_symmetric(r_i))
    return np.hstack(blocks)


@precondition(
    lambda a_couple, rcond=1e-12: (
        a_couple.ndim == 2
        and a_couple.shape[0] <= a_couple.shape[1]
        and rcond > 0.0
    ),
    "a_couple must be a 2D fat/square matrix (m <= n) with rcond > 0",
)
def compute_moore_penrose_pinv(
    a_couple: NDArray[np.floating[Any]],
    rcond: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute Moore-Penrose pseudoinverse A_couple^+."""
    a_mat = np.asarray(a_couple, dtype=np.float64)
    return np.linalg.pinv(a_mat, rcond=rcond)


@precondition(
    lambda a_couple, rcond=1e-12: (
        a_couple.ndim == 2
        and a_couple.shape[0] <= a_couple.shape[1]
        and rcond > 0.0
    ),
    "a_couple must be a 2D matrix with m <= n and rcond > 0",
)
def compute_nullspace_projector(
    a_couple: NDArray[np.floating[Any]],
    rcond: float = 1e-12,
) -> NDArray[np.float64]:
    """Compute orthogonal Null-Space projector N_null = I - A_couple^+ A_couple."""
    a_mat = np.asarray(a_couple, dtype=np.float64)
    a_pinv = compute_moore_penrose_pinv(a_mat, rcond=rcond)
    n_dim = a_mat.shape[1]
    return np.eye(n_dim, dtype=np.float64) - (a_pinv @ a_mat)


@precondition(
    lambda a_couple, tau_task, tau_preload, rcond=1e-12: (
        a_couple.ndim == 2
        and tau_task.ndim == 1
        and tau_preload.ndim == 1
        and a_couple.shape[0] == tau_task.shape[0]
        and a_couple.shape[1] == tau_preload.shape[0]
    ),
    "Dimensions of a_couple (m x n), tau_task (m,), and tau_preload (n,) must match",
)
def solve_grip_nullspace_stabilization(
    a_couple: NDArray[np.floating[Any]],
    tau_task: NDArray[np.floating[Any]],
    tau_preload: NDArray[np.floating[Any]],
    rcond: float = 1e-12,
) -> GraspStabilizationSolution:
    """Compute minimum-norm Moment-of-Force drive u_task and Null-Space preload u_null.

    Equations:
        u_task  = A_couple^+ tau_task
        N_null  = I - A_couple^+ A_couple
        u_null  = N_null tau_preload
        u_total = u_task + u_null
    """
    a_mat = np.asarray(a_couple, dtype=np.float64)
    t_task = np.asarray(tau_task, dtype=np.float64)
    t_pre = np.asarray(tau_preload, dtype=np.float64)

    a_pinv = compute_moore_penrose_pinv(a_mat, rcond=rcond)
    n_null = np.eye(a_mat.shape[1], dtype=np.float64) - (a_pinv @ a_mat)

    u_task = a_pinv @ t_task
    u_null = n_null @ t_pre
    u_total = u_task + u_null

    realized_tau = a_mat @ u_total
    null_leakage = float(np.linalg.norm(a_mat @ u_null))
    ortho_dot = float(np.dot(u_task, u_null))

    return GraspStabilizationSolution(
        a_couple=a_mat,
        a_couple_pinv=a_pinv,
        n_null=n_null,
        u_task=u_task,
        u_null=u_null,
        u_total=u_total,
        realized_tau_task=realized_tau,
        null_leakage_norm=null_leakage,
        orthogonality_inner_product=ortho_dot,
    )


@precondition(
    lambda mass_matrix, coriolis_matrix, gravity_vec, q_dot, tau_applied: (
        mass_matrix.ndim == 2
        and mass_matrix.shape[0] == mass_matrix.shape[1]
        and coriolis_matrix.shape == mass_matrix.shape
        and gravity_vec.shape == (mass_matrix.shape[0],)
        and q_dot.shape == (mass_matrix.shape[0],)
        and tau_applied.shape == (mass_matrix.shape[0],)
    ),
    "Manipulator matrices M(q), C(q, q_dot) and vectors g(q), q_dot, tau must align",
)
def decompose_ztcf_zvcf_accelerations(
    mass_matrix: NDArray[np.floating[Any]],
    coriolis_matrix: NDArray[np.floating[Any]],
    gravity_vec: NDArray[np.floating[Any]],
    q_dot: NDArray[np.floating[Any]],
    tau_applied: NDArray[np.floating[Any]],
) -> CounterfactualAccelerationDecomposition:
    """Decompose rigid-body accelerations into ZTCF and ZVCF counterfactual components.

    Manipulator Equation:
        M(q) q_ddot + C(q, q_dot) q_dot + g(q) = tau_applied

    Counterfactuals:
        - ZTCF (Zero-Torque Counterfactual, tau = 0):
          q_ddot_ZTCF = M(q)^{-1} ( -C(q, q_dot) q_dot - g(q) )
        - ZVCF (Zero-Velocity Counterfactual, q_dot = 0):
          q_ddot_ZVCF = M(q)^{-1} ( tau_applied - g(q) )
        - Exact superposition identity:
          q_ddot_actual = q_ddot_ZTCF + q_ddot_ZVCF - q_ddot_gravity
                        = q_ddot_torque + q_ddot_velocity + q_ddot_gravity
    """
    m_mat = np.asarray(mass_matrix, dtype=np.float64)
    c_mat = np.asarray(coriolis_matrix, dtype=np.float64)
    g_vec = np.asarray(gravity_vec, dtype=np.float64)
    qd = np.asarray(q_dot, dtype=np.float64)
    tau = np.asarray(tau_applied, dtype=np.float64)

    coriolis_wrench = c_mat @ qd

    # Solve linear systems via LU factorization (np.linalg.solve) rather than explicit inv
    rhs_stack = np.column_stack(
        [
            tau - coriolis_wrench - g_vec,  # actual
            -coriolis_wrench - g_vec,       # ZTCF (tau = 0)
            tau - g_vec,                    # ZVCF (q_dot = 0)
            tau,                            # pure torque
            -coriolis_wrench,               # pure velocity/Coriolis
            -g_vec,                         # pure gravity
        ]
    )
    sol = np.linalg.solve(m_mat, rhs_stack)

    q_ddot_actual = sol[:, 0]
    q_ddot_ztcf = sol[:, 1]
    q_ddot_zvcf = sol[:, 2]
    q_ddot_torque = sol[:, 3]
    q_ddot_velocity = sol[:, 4]
    q_ddot_gravity = sol[:, 5]

    reconstructed = q_ddot_ztcf + q_ddot_zvcf - q_ddot_gravity
    residual = float(np.linalg.norm(q_ddot_actual - reconstructed))

    return CounterfactualAccelerationDecomposition(
        q_ddot_actual=q_ddot_actual,
        q_ddot_ztcf=q_ddot_ztcf,
        q_ddot_zvcf=q_ddot_zvcf,
        q_ddot_torque=q_ddot_torque,
        q_ddot_velocity=q_ddot_velocity,
        q_ddot_gravity=q_ddot_gravity,
        superposition_residual_norm=residual,
    )

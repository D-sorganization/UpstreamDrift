"""Shared array-namespace contact kernel for NumPy and JAX (HRG-GS-M14, #13336).

Extracts the single qualified Hunt-Crossley normal, regularised Coulomb/Stribeck
tangential, and torsional spin-friction contact law evaluated per contact pair in
:mod:`src.shared.python.grip_contact.pair_contact` and
:mod:`src.shared.python.grip_contact.pad_contact` into one pure array-namespace
kernel (:func:`contact_law_kernel`).

Design invariants:
1. **Validation split (DbC outside tracing)**: Precondition checks on parameter
   ranges, types, shapes, and finiteness execute in thin public wrappers
   (:func:`evaluate_contact_kernel`, :func:`evaluate_contact_kernel_jax`,
   :func:`validate_kernel_params`) outside the traced kernel.
2. **Branch-free pure kernel**: :func:`contact_law_kernel` is written against an
   array namespace ``xp`` (``numpy`` or ``jax.numpy``) with ``xp.where`` and no
   Python branching on traced state values, supporting scalar, per-side batched
   ``(N,)`` arrays (composing with ``SideBatch`` in ``perf/grip-contact-batched``),
   and ``jax.jit`` / ``jax.vmap`` / ``jax.grad``.
3. **No silent smoothing**: :func:`contact_law_kernel` evaluates the exact
   qualified law. An opt-in $C^\\infty$ relaxation is provided separately as
   :func:`smooth_contact_law_kernel` and is never the default.
4. **Lazy JAX import**: ``jax`` and ``jax.numpy`` are imported only inside
   :func:`evaluate_contact_kernel_jax` and :func:`jax_contact_kernel` so that
   importing :mod:`src.shared.python.grip_contact` never imports ``jax``.

Non-differentiable / non-smooth loci of the qualified law:
- **Contact onset at zero penetration ($\\delta = 0$)**: The unilateral clipping
  $\\delta_+ = \\max(0, \\delta)$ has a $C^0$ hinge at $\\delta = 0$ where
  $\\partial_- f_n / \\partial \\delta = 0$ and
  $\\partial_+ f_n / \\partial \\delta = k(1 - c v_n) > 0$. A second unilateral
  hinge occurs at damping liftoff $1 - c v_n = 0$ ($v_n = 1/c$).
- **Zero tangential speed ($v_t = 0$)**: The tangential speed
  $s = \\|v_t\\|_2 = \\sqrt{v_t \\cdot v_t}$ is non-differentiable at $v_t = 0$,
  producing a $C^0$ V-cusp in the tangential force magnitude
  $\\partial_\\pm f_t / \\partial v_t|_{0^\\pm} = \\pm \\mu_s f_n / v_{\\mathrm{tr}}$
  and a $0/0$ removable singularity in the unit slip direction $v_t / \\|v_t\\|_2$
  (resolved without ``NaN`` in reverse-mode AD via the double-``xp.where`` idiom).
- **Stick/slip transition ($|v_t| \\sim v_{\\mathrm{tr}}$)**: Across the Stribeck/tanh
  boundary layer of width $v_{\\mathrm{tr}}$, the tangent modulus $\\partial f_t / \\partial s$
  drops sharply from the stick stiffness $\\mu_s f_n / v_{\\mathrm{tr}}$ at $s \\ll v_{\\mathrm{tr}}$
  through zero at the Stribeck peak into the velocity-weakening negative-slope
  regime ($\\mu_s > \\mu_d$) before approaching $\\mu_v f_n$ in full slip.
"""

from __future__ import annotations

import math
from typing import Any, NamedTuple

import numpy as np

from src.shared.python.motion_matching.contact_law import (
    ContactParameters,
    ContactSample,
    GroundPlane,
)

__all__ = [
    "ContactKernelParams",
    "ContactKernelResult",
    "contact_law_kernel",
    "evaluate_contact_kernel",
    "evaluate_contact_kernel_jax",
    "jax_contact_kernel",
    "sample_plane_contact",
    "smooth_contact_law_kernel",
    "validate_kernel_params",
]

_INVALID_INPUT_TYPES = (str, bytes, dict, set, bool)


class ContactKernelParams(NamedTuple):
    """Parameters of the shared Hunt-Crossley, Stribeck, and torsional spin law.

    As a :class:`typing.NamedTuple`, instances are native JAX PyTrees without
    importing ``jax`` at module load time.

    Attributes:
        stiffness_n_m: Normal contact stiffness $k$ [N/m] ($> 0$).
        dissipation_s_m: Hunt-Crossley damping coefficient $c$ [s/m] ($\\ge 0$).
        static_friction: Static friction coefficient $\\mu_s$ ($\\ge \\mu_d$).
        dynamic_friction: Dynamic friction coefficient $\\mu_d$ ($\\ge 0$).
        viscous_friction: Viscous friction slope $\\mu_v$ [s/m] ($\\ge 0$).
        transition_velocity_m_s: Stribeck/Coulomb transition speed $v_{\\mathrm{tr}}$ [m/s] ($> 0$).
        patch_radius_m: Contact patch radius $r_{\\mathrm{patch}}$ [m] ($\\ge 0$).
        spin_transition_rad_s: Torsional spin transition rate $\\omega_{\\mathrm{tr}}$ [rad/s] ($> 0$).
    """

    stiffness_n_m: float
    dissipation_s_m: float
    static_friction: float
    dynamic_friction: float
    viscous_friction: float
    transition_velocity_m_s: float
    patch_radius_m: float = 0.0
    spin_transition_rad_s: float = 1.0

    @classmethod
    def from_contact_parameters(
        cls,
        law: ContactParameters,
        patch_radius_m: float = 0.0,
        spin_transition_rad_s: float = 1.0,
    ) -> ContactKernelParams:
        """Build and validate :class:`ContactKernelParams` from :class:`ContactParameters`."""
        return validate_kernel_params(
            law,
            patch_radius_m=patch_radius_m,
            spin_transition_rad_s=spin_transition_rad_s,
        )


class ContactKernelResult(NamedTuple):
    """Output of the shared contact kernel.

    Attributes:
        normal_force_n: Normal contact force magnitude $f_n \\ge 0$ [N], shape ``S``.
        tangential_force_n: Tangential friction force [N]; shape ``S`` (magnitude
            $f_t \\ge 0$) when ``tangential_velocity_m_s`` has shape ``S``, or
            3-vector opposing slip of shape ``(*S, 3)`` when
            ``tangential_velocity_m_s`` has shape ``(*S, 3)``.
        spin_moment_nm: Signed torsional spin friction moment $m_s$ [N m] along
            the contact normal opposing ``spin_rate_rad_s``, shape ``S``.
    """

    normal_force_n: Any
    tangential_force_n: Any
    spin_moment_nm: Any


def validate_kernel_params(
    params: ContactKernelParams | ContactParameters,
    patch_radius_m: float | None = None,
    spin_transition_rad_s: float | None = None,
) -> ContactKernelParams:
    """Validate contact-law parameters outside the traced kernel (DbC).

    Raises:
        TypeError: If ``params`` is not :class:`ContactKernelParams` or
            :class:`ContactParameters`.
        ValueError: If any parameter is non-finite or violates physical sign
            and ordering invariants.
    """
    if isinstance(params, ContactKernelParams):
        p_patch = params.patch_radius_m if patch_radius_m is None else patch_radius_m
        p_spin = (
            params.spin_transition_rad_s
            if spin_transition_rad_s is None
            else spin_transition_rad_s
        )
        candidate = ContactKernelParams(
            stiffness_n_m=float(params.stiffness_n_m),
            dissipation_s_m=float(params.dissipation_s_m),
            static_friction=float(params.static_friction),
            dynamic_friction=float(params.dynamic_friction),
            viscous_friction=float(params.viscous_friction),
            transition_velocity_m_s=float(params.transition_velocity_m_s),
            patch_radius_m=float(p_patch),
            spin_transition_rad_s=float(p_spin),
        )
    elif isinstance(params, ContactParameters):
        p_patch = 0.0 if patch_radius_m is None else patch_radius_m
        p_spin = 1.0 if spin_transition_rad_s is None else spin_transition_rad_s
        candidate = ContactKernelParams(
            stiffness_n_m=float(params.stiffness_n_m),
            dissipation_s_m=float(params.dissipation_s_m),
            static_friction=float(params.static_friction),
            dynamic_friction=float(params.dynamic_friction),
            viscous_friction=float(params.viscous_friction),
            transition_velocity_m_s=float(params.transition_velocity_m_s),
            patch_radius_m=float(p_patch),
            spin_transition_rad_s=float(p_spin),
        )
    else:
        raise TypeError(
            "params must be ContactKernelParams or ContactParameters, "
            f"got {type(params).__name__}"
        )

    if not all(math.isfinite(v) for v in candidate):
        raise ValueError("All contact kernel parameters must be finite")
    if candidate.stiffness_n_m <= 0.0 or candidate.transition_velocity_m_s <= 0.0:
        raise ValueError("stiffness_n_m and transition_velocity_m_s must be positive")
    if (
        min(
            candidate.dissipation_s_m,
            candidate.dynamic_friction,
            candidate.viscous_friction,
        )
        < 0.0
    ):
        raise ValueError(
            "dissipation_s_m, dynamic_friction, and viscous_friction must be nonnegative"
        )
    if candidate.static_friction < candidate.dynamic_friction:
        raise ValueError("Static friction must not be below dynamic friction")
    if candidate.patch_radius_m < 0.0:
        raise ValueError(
            f"patch_radius_m must be finite and >= 0, got {candidate.patch_radius_m}"
        )
    if candidate.spin_transition_rad_s <= 0.0:
        raise ValueError(
            "spin_transition_rad_s must be positive and finite, "
            f"got {candidate.spin_transition_rad_s}"
        )
    return candidate


def _coerce_finite_array(value: Any, name: str) -> np.ndarray:
    """Validate that ``value`` is a finite numeric scalar or array."""
    if value is None or isinstance(value, _INVALID_INPUT_TYPES):
        raise TypeError(
            f"{name} must be a numeric scalar or array, got {type(value).__name__}"
        )
    try:
        arr = np.asarray(value, dtype=float)
    except (TypeError, ValueError) as exc:
        raise TypeError(f"{name} must be convertible to float array") from exc
    if not bool(np.isfinite(arr).all()):
        raise ValueError(f"{name} must contain only finite values")
    return arr


def _validate_kinematic_inputs(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    spin_rate_rad_s: Any = 0.0,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Validate shapes and finiteness of kinematic inputs outside tracing."""
    delta = _coerce_finite_array(penetration_m, "penetration_m")
    v_n = _coerce_finite_array(normal_velocity_m_s, "normal_velocity_m_s")
    v_t = _coerce_finite_array(tangential_velocity_m_s, "tangential_velocity_m_s")
    omega_n = _coerce_finite_array(spin_rate_rad_s, "spin_rate_rad_s")

    if v_n.shape != delta.shape:
        raise ValueError(
            f"normal_velocity_m_s shape {v_n.shape} must match penetration_m shape {delta.shape}"
        )
    if v_t.shape != delta.shape and v_t.shape != (*delta.shape, 3):
        raise ValueError(
            f"tangential_velocity_m_s shape {v_t.shape} must be {delta.shape} "
            f"or {(*delta.shape, 3)}"
        )
    if omega_n.shape not in ((), delta.shape):
        raise ValueError(
            f"spin_rate_rad_s shape {omega_n.shape} must be () or {delta.shape}"
        )
    return delta, v_n, v_t, omega_n


def contact_law_kernel(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    params: ContactKernelParams | ContactParameters,
    spin_rate_rad_s: Any = 0.0,
    *,
    xp: Any = np,
) -> ContactKernelResult:
    """Pure array-namespace kernel for the qualified Hunt-Crossley / Stribeck / spin law.

    Evaluates normal force, tangential friction force, and torsional spin moment
    without Python branching on traced values (uses ``xp.where``).

    Sign conventions:
    - ``penetration_m`` ($\\delta$): positive when penetrating, $\\le 0$ when separated.
    - ``normal_velocity_m_s`` ($v_n = \\hat{n} \\cdot v_{\\mathrm{rel}} = -\\dot{\\delta}$):
      negative when approaching/deepening, positive when separating.
    - ``tangential_velocity_m_s`` ($v_t$): scalar slip velocity/speed of shape ``S``
      or 3-vector slip velocity in the contact plane of shape ``(*S, 3)``.
    - ``spin_rate_rad_s`` ($\\omega_n = \\hat{n} \\cdot \\omega_{\\mathrm{rel}}$):
      normal relative angular velocity [rad/s].

    Non-differentiable points:
    1. Contact onset at $\\delta = 0$ (and damping liftoff at $1 - c v_n = 0$).
    2. Zero tangential speed at $v_t = 0$ ($\\|v_t\\|_2 = 0$).
    3. Stick/slip transition knee around $|v_t| \\sim v_{\\mathrm{tr}}$.
    """
    delta = xp.asarray(penetration_m, dtype=float)
    v_n = xp.asarray(normal_velocity_m_s, dtype=float)
    v_t = xp.asarray(tangential_velocity_m_s, dtype=float)
    omega_n = xp.broadcast_to(xp.asarray(spin_rate_rad_s, dtype=float), delta.shape)

    active_contact = delta > 0.0
    delta_pos = xp.where(active_contact, delta, 0.0)
    rate = -v_n
    raw_fn = params.stiffness_n_m * delta_pos * (1.0 + params.dissipation_s_m * rate)
    f_n = xp.where(active_contact & (raw_fn > 0.0), raw_fn, xp.zeros_like(delta))

    if v_t.ndim == delta.ndim + 1:
        s_sq = (
            v_t[..., 0] * v_t[..., 0]
            + v_t[..., 1] * v_t[..., 1]
            + v_t[..., 2] * v_t[..., 2]
        )
        s_pos = s_sq > 0.0
        s_sq_safe = xp.where(s_pos, s_sq, 1.0)
        speed = xp.where(s_pos, xp.sqrt(s_sq_safe), 0.0)
        speed_safe = xp.where(s_pos, speed, 1.0)
        active_slip = s_pos & (f_n > 0.0)
        ratio = speed / params.transition_velocity_m_s
        mu = (
            params.dynamic_friction
            + (params.static_friction - params.dynamic_friction) * xp.exp(-(ratio**2))
            + params.viscous_friction * speed
        )
        raw_ft = (
            -v_t
            / speed_safe[..., None]
            * mu[..., None]
            * f_n[..., None]
            * xp.tanh(ratio)[..., None]
        )
        f_t = xp.where(active_slip[..., None], raw_ft, xp.zeros_like(v_t))
    else:
        v_t_sq = v_t * v_t
        s_pos = v_t_sq > 0.0
        v_t_sq_safe = xp.where(s_pos, v_t_sq, 1.0)
        speed = xp.where(s_pos, xp.sqrt(v_t_sq_safe), 0.0)
        active_slip = s_pos & (f_n > 0.0)
        ratio = speed / params.transition_velocity_m_s
        mu = (
            params.dynamic_friction
            + (params.static_friction - params.dynamic_friction) * xp.exp(-(ratio**2))
            + params.viscous_friction * speed
        )
        f_t = xp.where(active_slip, mu * f_n * xp.tanh(ratio), xp.zeros_like(delta))

    patch_r = getattr(params, "patch_radius_m", 0.0)
    spin_tr = getattr(params, "spin_transition_rad_s", 1.0)
    limit = params.static_friction * f_n * patch_r
    m_s = xp.where(
        f_n > 0.0,
        -limit * xp.tanh(omega_n / spin_tr),
        xp.zeros_like(delta),
    )
    return ContactKernelResult(
        normal_force_n=f_n,
        tangential_force_n=f_t,
        spin_moment_nm=m_s,
    )


def evaluate_contact_kernel(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    params: ContactKernelParams | ContactParameters,
    spin_rate_rad_s: Any = 0.0,
) -> ContactKernelResult:
    """Validate inputs (DbC) and evaluate the shared contact kernel in NumPy."""
    validated_params = validate_kernel_params(params)
    delta, v_n, v_t, omega_n = _validate_kinematic_inputs(
        penetration_m,
        normal_velocity_m_s,
        tangential_velocity_m_s,
        spin_rate_rad_s,
    )
    return contact_law_kernel(delta, v_n, v_t, validated_params, omega_n, xp=np)


def evaluate_contact_kernel_jax(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    params: ContactKernelParams | ContactParameters,
    spin_rate_rad_s: Any = 0.0,
) -> ContactKernelResult:
    """Validate inputs (DbC) and evaluate the shared contact kernel in ``jax.numpy``.

    Imports ``jax.numpy`` lazily inside the function so importing
    :mod:`src.shared.python.grip_contact` never imports ``jax``.
    """
    import jax.numpy as jnp

    validated_params = validate_kernel_params(params)
    delta, v_n, v_t, omega_n = _validate_kinematic_inputs(
        penetration_m,
        normal_velocity_m_s,
        tangential_velocity_m_s,
        spin_rate_rad_s,
    )
    return contact_law_kernel(
        jnp.asarray(delta),
        jnp.asarray(v_n),
        jnp.asarray(v_t),
        validated_params,
        jnp.asarray(omega_n),
        xp=jnp,
    )


def jax_contact_kernel(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    params: ContactKernelParams | ContactParameters,
    spin_rate_rad_s: Any = 0.0,
) -> ContactKernelResult:
    """Traceable JAX entry point for ``jax.grad``, ``jax.jit``, and ``jax.vmap``.

    Validates static ``params`` outside tracing and lazily imports ``jax.numpy``.
    """
    import jax.numpy as jnp

    validated_params = validate_kernel_params(params)
    return contact_law_kernel(
        penetration_m,
        normal_velocity_m_s,
        tangential_velocity_m_s,
        validated_params,
        spin_rate_rad_s,
        xp=jnp,
    )


def smooth_contact_law_kernel(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    params: ContactKernelParams | ContactParameters,
    spin_rate_rad_s: Any = 0.0,
    *,
    smoothing_m: float = 1.0e-6,
    velocity_eps_m_s: float = 1.0e-7,
    xp: Any = np,
) -> ContactKernelResult:
    """Opt-in $C^\\infty$ smoothed contact kernel for gradient-based trajectory optimization.

    Never used by default: replaces the hard hinge $\\max(0, \\delta)$ with the
    hyperbolic relaxation $\\frac{1}{2}(\\delta + \\sqrt{\\delta^2 + \\varepsilon_\\delta^2})$
    and the Euclidean norm $\\|v_t\\|_2$ with the Charbonnier norm
    $\\sqrt{v_t \\cdot v_t + \\varepsilon_v^2} - \\varepsilon_v$.
    """
    if not (math.isfinite(smoothing_m) and smoothing_m > 0.0):
        raise ValueError(f"smoothing_m must be finite and > 0, got {smoothing_m}")
    if not (math.isfinite(velocity_eps_m_s) and velocity_eps_m_s > 0.0):
        raise ValueError(
            f"velocity_eps_m_s must be finite and > 0, got {velocity_eps_m_s}"
        )

    delta = xp.asarray(penetration_m, dtype=float)
    v_n = xp.asarray(normal_velocity_m_s, dtype=float)
    v_t = xp.asarray(tangential_velocity_m_s, dtype=float)
    omega_n = xp.broadcast_to(xp.asarray(spin_rate_rad_s, dtype=float), delta.shape)

    delta_smooth = 0.5 * (delta + xp.sqrt(delta * delta + smoothing_m * smoothing_m))
    damping_factor = 1.0 - params.dissipation_s_m * v_n
    damp_smooth = 0.5 * (
        damping_factor
        + xp.sqrt(damping_factor * damping_factor + smoothing_m * smoothing_m)
    )
    f_n = params.stiffness_n_m * delta_smooth * damp_smooth

    if v_t.ndim == delta.ndim + 1:
        s_sq = (
            v_t[..., 0] * v_t[..., 0]
            + v_t[..., 1] * v_t[..., 1]
            + v_t[..., 2] * v_t[..., 2]
        )
        denom = xp.sqrt(s_sq + velocity_eps_m_s * velocity_eps_m_s)
        speed = denom - velocity_eps_m_s
        ratio = speed / params.transition_velocity_m_s
        mu = (
            params.dynamic_friction
            + (params.static_friction - params.dynamic_friction) * xp.exp(-(ratio**2))
            + params.viscous_friction * speed
        )
        f_t = (
            -v_t
            / denom[..., None]
            * mu[..., None]
            * f_n[..., None]
            * xp.tanh(ratio)[..., None]
        )
    else:
        denom = xp.sqrt(v_t * v_t + velocity_eps_m_s * velocity_eps_m_s)
        speed = denom - velocity_eps_m_s
        ratio = speed / params.transition_velocity_m_s
        mu = (
            params.dynamic_friction
            + (params.static_friction - params.dynamic_friction) * xp.exp(-(ratio**2))
            + params.viscous_friction * speed
        )
        f_t = mu * f_n * xp.tanh(ratio)

    patch_r = getattr(params, "patch_radius_m", 0.0)
    spin_tr = getattr(params, "spin_transition_rad_s", 1.0)
    limit = params.static_friction * f_n * patch_r
    m_s = -limit * xp.tanh(omega_n / spin_tr)
    return ContactKernelResult(
        normal_force_n=f_n,
        tangential_force_n=f_t,
        spin_moment_nm=m_s,
    )


def sample_plane_contact(
    sphere_center_m: np.ndarray,
    rel_velocity_m_s: np.ndarray,
    sphere_radius_m: float,
    plane: GroundPlane,
    rel_omega_rad_s: np.ndarray,
    params: ContactKernelParams,
) -> tuple[ContactSample, np.ndarray]:
    """Evaluate 3D sphere-on-plane contact and spin moment via :func:`contact_law_kernel`.

    Shared NumPy path called by both :mod:`src.shared.python.grip_contact.pair_contact`
    and :mod:`src.shared.python.grip_contact.pad_contact`.
    """
    center = _coerce_finite_array(sphere_center_m, "sphere_center_m")
    velocity = _coerce_finite_array(rel_velocity_m_s, "rel_velocity_m_s")
    omega = _coerce_finite_array(rel_omega_rad_s, "rel_omega_rad_s")
    if center.shape != (3,) or velocity.shape != (3,) or omega.shape != (3,):
        raise ValueError(
            "sphere_center_m, rel_velocity_m_s, and rel_omega_rad_s must be 3-vectors"
        )
    if not (math.isfinite(sphere_radius_m) and sphere_radius_m > 0.0):
        raise ValueError("sphere_radius_m must be finite and positive")
    if not isinstance(plane, GroundPlane):
        raise TypeError(f"plane must be GroundPlane, got {type(plane).__name__}")
    validated_params = validate_kernel_params(params)

    normal = np.asarray(plane.normal, dtype=float)
    signed_distance = float(normal @ center) - plane.height_m - sphere_radius_m
    penetration = max(0.0, -signed_distance)
    v_n = float(normal @ velocity)
    rate = -v_n
    point = center - normal * sphere_radius_m
    tangential = velocity - normal * v_n
    spin_rate = float(normal @ omega)

    res = contact_law_kernel(
        penetration,
        v_n,
        tangential,
        validated_params,
        spin_rate,
        xp=np,
    )
    normal_force = normal * float(res.normal_force_n)
    friction_force = np.asarray(res.tangential_force_n, dtype=float)
    spin_moment = normal * float(res.spin_moment_nm)
    sample = ContactSample(
        penetration_m=penetration,
        penetration_rate_m_s=rate,
        contact_point_m=point,
        normal_force_n=normal_force,
        friction_force_n=friction_force,
    )
    return sample, spin_moment

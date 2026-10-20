"""Shared array-namespace contact kernel for NumPy and JAX (HRG-GS-M14, #13336).

Provides the single implementation of the qualified Hunt-Crossley normal,
regularised Coulomb/Stribeck tangential, and torsional spin-friction contact law
shared across:
- :mod:`src.shared.python.motion_matching.contact_law` (:func:`sphere_ground_contact`
  and :func:`torsional_friction_moment`, called with ``xp=numpy``),
- :mod:`src.shared.python.motion_matching.jax_contact` (:func:`sphere_ground_contact_jax`,
  called with ``xp=jax.numpy``), and
- :mod:`src.shared.python.grip_contact.pair_contact` and
  :mod:`src.shared.python.grip_contact.pad_contact` (which delegate through
  :mod:`src.shared.python.motion_matching.contact_law`).

Design invariants:
1. **Single formula (DRY)**: :func:`contact_law_kernel` and
   :func:`torsional_moment_kernel` hold the sole implementation of the normal,
   tangential, and spin-friction equations. When ``xp is numpy`` and scalar
   inputs are passed, the kernel dispatches its operations (`where`, `exp`,
   `tanh`, `sqrt`) through :class:`_ScalarMathOps` (backed by :mod:`math`)
   within the same function body so the hot scalar path in
   ``sphere_ground_contact`` avoids 0-D ``ndarray`` allocation overhead while
   executing the identical mathematical law.
2. **Validation split (DbC outside tracing)**: Precondition checks on parameter
   ranges, types, shapes, and finiteness execute in thin public wrappers
   (:func:`evaluate_contact_kernel`, :func:`evaluate_contact_kernel_jax`,
   :func:`validate_kernel_params`, :func:`sphere_ground_contact`,
   :func:`torsional_friction_moment`, :func:`sphere_ground_contact_jax`)
   outside the traced kernel.
3. **Branch-free pure kernel on traced values**: State masking uses
   ``ops.where`` with no Python branching on traced state values, supporting
   scalar, per-side batched ``(N,)`` arrays (composing with ``SideBatch``),
   and ``jax.jit`` / ``jax.vmap`` / ``jax.grad``.
4. **No silent smoothing**: :func:`contact_law_kernel` evaluates the exact
   qualified law. An opt-in $C^\\infty$ relaxation is provided separately as
   :func:`smooth_contact_law_kernel` and is never the default.
5. **Lazy JAX import**: ``jax`` and ``jax.numpy`` are imported only inside
   :func:`evaluate_contact_kernel_jax` and :func:`jax_contact_kernel` so that
   importing :mod:`src.shared.python.motion_matching` or
   :mod:`src.shared.python.grip_contact` never imports ``jax``.

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
from typing import TYPE_CHECKING, Any, NamedTuple

import numpy as np

if TYPE_CHECKING:
    from src.shared.python.motion_matching.contact_law import ContactParameters

__all__ = [
    "ContactKernelParams",
    "ContactKernelResult",
    "contact_law_kernel",
    "evaluate_contact_kernel",
    "evaluate_contact_kernel_jax",
    "friction_secant_gain",
    "jax_contact_kernel",
    "smooth_contact_law_kernel",
    "torsional_moment_kernel",
    "validate_kernel_params",
]

_INVALID_INPUT_TYPES = (str, bytes, dict, set, bool)
_SCALAR_TYPES = (int, float, np.floating)


class _ScalarMathOps:
    """Scalar math namespace matching the array-namespace subset used by the kernel."""

    exp = staticmethod(math.exp)
    tanh = staticmethod(math.tanh)
    sqrt = staticmethod(math.sqrt)

    @staticmethod
    def where(cond: Any, x: Any, y: Any) -> Any:
        return x if cond else y


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
    from src.shared.python.motion_matching.contact_law import ContactParameters

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


def torsional_moment_kernel(
    normal_load_n: Any,
    spin_rate_rad_s: Any,
    *,
    friction: float,
    patch_radius_m: float,
    transition_rad_s: float,
    xp: Any = np,
) -> Any:
    """Pure array-namespace kernel for the signed torsional spin friction moment.

    Evaluates ``m_s = -friction * max(0, normal_load_n) * patch_radius_m * tanh(spin_rate_rad_s / transition_rad_s)``
    along the contact normal without Python branching on traced values.
    """
    if (
        xp is np
        and isinstance(normal_load_n, _SCALAR_TYPES)
        and isinstance(spin_rate_rad_s, _SCALAR_TYPES)
    ):
        ops: Any = _ScalarMathOps
        f_n: Any = float(normal_load_n)
        omega_n: Any = float(spin_rate_rad_s)
        zero: Any = 0.0
    else:
        ops = xp
        f_n = (
            xp.asarray(normal_load_n, dtype=float)
            if xp is np
            else xp.asarray(normal_load_n)
        )
        omega_n = xp.broadcast_to(
            xp.asarray(spin_rate_rad_s, dtype=float)
            if xp is np
            else xp.asarray(spin_rate_rad_s, dtype=f_n.dtype),
            f_n.shape,
        )
        zero = xp.zeros_like(f_n)

    load_pos = f_n > 0.0
    load = ops.where(load_pos, f_n, zero)
    limit = friction * load * patch_radius_m
    return ops.where(load_pos, -limit * ops.tanh(omega_n / transition_rad_s), zero)


def _prepare_kernel_arrays(
    penetration_m: Any,
    normal_velocity_m_s: Any,
    tangential_velocity_m_s: Any,
    spin_rate_rad_s: Any,
    xp: Any,
) -> tuple[Any, Any, Any, Any, Any, Any, bool, Any]:
    """Coerce kernel inputs to either scalar fast-path or array-namespace tensors."""
    if (
        xp is np
        and isinstance(penetration_m, _SCALAR_TYPES)
        and isinstance(normal_velocity_m_s, _SCALAR_TYPES)
        and isinstance(spin_rate_rad_s, _SCALAR_TYPES)
        and (
            isinstance(tangential_velocity_m_s, _SCALAR_TYPES)
            or (
                isinstance(tangential_velocity_m_s, np.ndarray)
                and tangential_velocity_m_s.shape == (3,)
            )
        )
    ):
        ops: Any = _ScalarMathOps
        delta: Any = float(penetration_m)
        v_n: Any = float(normal_velocity_m_s)
        omega_n: Any = float(spin_rate_rad_s)
        zero: Any = 0.0
        if isinstance(tangential_velocity_m_s, np.ndarray):
            v_t: Any = tangential_velocity_m_s
            is_vec = True
            zero_vt: Any = np.zeros(3)
        else:
            v_t = float(tangential_velocity_m_s)
            is_vec = False
            zero_vt = 0.0
    else:
        ops = xp
        if xp is np:
            delta = xp.asarray(penetration_m, dtype=float)
            v_n = xp.asarray(normal_velocity_m_s, dtype=float)
            v_t = xp.asarray(tangential_velocity_m_s, dtype=float)
            omega_n = xp.broadcast_to(
                xp.asarray(spin_rate_rad_s, dtype=float), delta.shape
            )
        else:
            delta = xp.asarray(penetration_m)
            v_n = xp.asarray(normal_velocity_m_s, dtype=delta.dtype)
            v_t = xp.asarray(tangential_velocity_m_s, dtype=delta.dtype)
            omega_n = xp.broadcast_to(
                xp.asarray(spin_rate_rad_s, dtype=delta.dtype), delta.shape
            )
        zero = xp.zeros_like(delta)
        is_vec = v_t.ndim == delta.ndim + 1
        zero_vt = xp.zeros_like(v_t) if is_vec else zero
    return ops, delta, v_n, v_t, omega_n, zero, is_vec, zero_vt


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
    without Python branching on traced values (uses ``ops.where``).

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
    ops, delta, v_n, v_t, omega_n, zero, is_vec, zero_vt = _prepare_kernel_arrays(
        penetration_m, normal_velocity_m_s, tangential_velocity_m_s, spin_rate_rad_s, xp
    )

    active_contact = delta > 0.0
    delta_pos = ops.where(active_contact, delta, 0.0)
    rate = -v_n
    raw_fn = params.stiffness_n_m * delta_pos * (1.0 + params.dissipation_s_m * rate)
    f_n = ops.where(active_contact & (raw_fn > 0.0), raw_fn, zero)

    if is_vec:
        s_sq = (
            float(
                v_t.dot(v_t)
            )  # ⚡ Bolt: math.sqrt(dot) is faster than np.linalg.norm for small 1D arrays
            if ops is _ScalarMathOps
            else v_t[..., 0] * v_t[..., 0]
            + v_t[..., 1] * v_t[..., 1]
            + v_t[..., 2] * v_t[..., 2]
        )
        s_pos = s_sq > 0.0
        s_sq_safe = ops.where(s_pos, s_sq, 1.0)
        speed = ops.where(s_pos, ops.sqrt(s_sq_safe), 0.0)
    else:
        speed = ops.where(v_t >= 0.0, v_t, -v_t)
        s_pos = speed > 0.0

    active_slip = s_pos & (f_n > 0.0)
    ratio = speed / params.transition_velocity_m_s
    mu = (
        params.dynamic_friction
        + (params.static_friction - params.dynamic_friction) * ops.exp(-(ratio**2))
        + params.viscous_friction * speed
    )
    tanh_ratio = ops.tanh(ratio)

    if is_vec:
        speed_safe = ops.where(s_pos, speed, 1.0)
        if ops is _ScalarMathOps:
            raw_ft = -v_t / speed_safe * mu * f_n * tanh_ratio
            f_t = ops.where(active_slip, raw_ft, zero_vt)
        else:
            raw_ft = (
                -v_t
                / speed_safe[..., None]
                * mu[..., None]
                * f_n[..., None]
                * tanh_ratio[..., None]
            )
            f_t = ops.where(active_slip[..., None], raw_ft, zero_vt)
    else:
        f_t = ops.where(active_slip, mu * f_n * tanh_ratio, zero)

    patch_r = getattr(params, "patch_radius_m", 0.0)
    spin_tr = getattr(params, "spin_transition_rad_s", 1.0)
    m_s = torsional_moment_kernel(
        f_n,
        omega_n,
        friction=params.static_friction,
        patch_radius_m=patch_r,
        transition_rad_s=spin_tr,
        xp=xp,
    )
    return ContactKernelResult(
        normal_force_n=f_n,
        tangential_force_n=f_t,
        spin_moment_nm=m_s,
    )


def friction_secant_gain(
    speed_m_s: Any, params: ContactKernelParams | ContactParameters
) -> Any:
    """Secant slip gain $g(v) = \\mu(v)\\tanh(v/v_{tr})/v$ of the friction law [s/m].

    ``contact_law_kernel`` gives a friction magnitude
    $|f_t| = g(v)\\,f_n\\,v$ at slip speed $v$, so a viscous friction element of
    gain $g(v) f_n$ reproduces the law at that speed (#13306). At rest the gain
    is the static slope $\\mu_s / v_{tr}$.

    Preconditions: every speed is finite and nonnegative (``ValueError``).
    Postconditions: same shape as ``speed_m_s`` (a ``float`` for a scalar),
    positive, and nonincreasing in speed.
    """
    speed = np.asarray(speed_m_s, dtype=float)
    if not np.all(np.isfinite(speed)) or np.any(speed < 0.0):
        raise ValueError("speed_m_s must be finite and nonnegative")
    v_tr = params.transition_velocity_m_s
    ratio = speed / v_tr
    mu = (
        params.dynamic_friction
        + (params.static_friction - params.dynamic_friction) * np.exp(-(ratio**2))
        + params.viscous_friction * speed
    )
    small = ratio < 1e-6  # tanh(r)/r = 1 - r**2/3 + ...
    safe = np.where(small, 1.0, ratio)
    tanh_over_ratio = np.where(small, 1.0 - ratio**2 / 3.0, np.tanh(safe) / safe)
    gain = mu * tanh_over_ratio / v_tr
    return float(gain) if gain.ndim == 0 else gain


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
    :mod:`src.shared.python.motion_matching` or :mod:`src.shared.python.grip_contact`
    never imports ``jax``.
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
    else:
        denom = xp.sqrt(v_t * v_t + velocity_eps_m_s * velocity_eps_m_s)

    speed = denom - velocity_eps_m_s
    ratio = speed / params.transition_velocity_m_s
    mu = (
        params.dynamic_friction
        + (params.static_friction - params.dynamic_friction) * xp.exp(-(ratio**2))
        + params.viscous_friction * speed
    )
    f_t_mag = mu * f_n * xp.tanh(ratio)

    if v_t.ndim == delta.ndim + 1:
        f_t = -v_t / denom[..., None] * f_t_mag[..., None]
    else:
        f_t = f_t_mag

    patch_r = getattr(params, "patch_radius_m", 0.0)
    spin_tr = getattr(params, "spin_transition_rad_s", 1.0)
    m_s = torsional_moment_kernel(
        f_n,
        omega_n,
        friction=params.static_friction,
        patch_radius_m=patch_r,
        transition_rad_s=spin_tr,
        xp=xp,
    )
    return ContactKernelResult(
        normal_force_n=f_n,
        tangential_force_n=f_t,
        spin_moment_nm=m_s,
    )

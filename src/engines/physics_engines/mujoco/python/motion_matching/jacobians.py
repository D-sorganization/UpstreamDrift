"""Analytical Jacobians for ``fit_swing_mujoco`` (issue #4175).

This module implements ``∂q(t) / ∂θ`` along a polynomial-torque rollout via:

1. The closed-form polynomial chain rule ``∂u(t) / ∂θ``: at time ``t`` and
   joint ``j``, the column of ``θ_{j,k}`` in ``∂u_j / ∂θ`` is just the
   monomial ``t^k`` (ascending-power layout, column 0 = constant term;
   canonical cross-engine convention, #7688). Every other entry is zero.
2. MuJoCo's ``mjd_transitionFD``, which finite-differences the discrete
   state-transition map and returns the Jacobian matrices

       A = ∂x_next / ∂x      (shape ``(2*nv+na, 2*nv+na)``)
       B = ∂x_next / ∂u      (shape ``(2*nv+na, nu)``)

   at the current ``(x, u)`` linearisation point. The state ``x`` is the
   stacked vector ``[dqpos (in tangent space, nv); qvel (nv); act (na)]``,
   so the leading ``nv`` rows of the resulting sensitivity tell us how
   ``qpos`` changes with respect to the parameter.
3. A first-order propagation along the rollout:

       S_{k+1} = A_k @ S_k + B_k @ J_u(t_k)

   where ``S_k = ∂x_k / ∂θ`` and ``J_u(t_k) = ∂u(t_k) / ∂θ``. The trajectory
   ``∂q / ∂θ`` is the leading ``nv`` rows of ``S`` at every output frame.

The driver caches the model, the per-frame ``A`` / ``B`` allocations, and
``J_u`` so that repeated Jacobian evaluations during an optimisation reuse
all heavy buffers — only the linearisation-point step actually advances
MuJoCo. See ``MUJOCO_PARITY_SPEC.md`` §6.2 for the design rationale.

Threading
---------
The polynomial torque is applied by writing directly to ``data.ctrl`` here
(no ``mjcb_control`` callback), so this module is safe to use from any
thread that owns its own ``MjData``. ``simulate_with_coefficients`` itself
is still process-global because of its ``mjcb_control`` driver — do not
mix the two in the same process across threads.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from .simulate import (
    _CLUBHEAD_BODY_BY_VARIANT,
    _GRIP_BODY_BY_VARIANT,
    SimOptions,
    _apply_initial_pose,
    _load_model_xml,
    _output_grid,
    _resolve_body_id,
)

__all__ = [
    "JacobianCache",
    "compute_cost_gradient_analytical",
    "compute_qpos_jacobian",
    "polynomial_du_dtheta",
]


# --- Closed-form polynomial chain rule --------------------------------------


def polynomial_du_dtheta(
    n_joints: int,
    t: float,
) -> NDArray[np.float64]:
    """Return ``∂u(t) / ∂θ`` for the per-joint 6th-order polynomial driver.

    Layout matches :class:`PolynomialTorqueDriver`: ``θ`` is flattened
    row-major as ``θ.reshape(n_joints, 7)``, where column ``k`` is the
    coefficient of ``t^k`` (ascending power; canonical cross-engine
    convention, #7688). Therefore

        ∂u_j / ∂θ_{j', k} = δ_{j, j'} · t^k

    Args:
        n_joints: number of actuated joints (``model.nu``).
        t: time in seconds (relative to the polynomial reference ``t0``).

    Returns:
        ``(n_joints, n_joints * 7)`` matrix; row ``j`` has the seven
        non-zero monomials in columns ``j*7 .. j*7+7``.
    """
    if n_joints <= 0:
        raise ValueError(f"n_joints must be > 0; got {n_joints}")
    powers = np.array([1.0, t, t**2, t**3, t**4, t**5, t**6], dtype=np.float64)
    out = np.zeros((n_joints, n_joints * 7), dtype=np.float64)
    for j in range(n_joints):
        out[j, j * 7 : j * 7 + 7] = powers
    return out


# --- Cache for repeated calls ------------------------------------------------


@dataclass
class JacobianCache:
    """Holds reusable allocations across repeated Jacobian evaluations.

    The optimizer hammers ``compute_qpos_jacobian`` once per outer
    iteration, and rebuilding the model / data / scratch arrays each time
    is wasteful. Callers may opt into reusing a :class:`JacobianCache`
    across calls; if ``None`` is passed the function allocates a fresh
    one and discards it.

    Attributes are populated lazily on first use; mutating fields
    in-place is safe because the cache is single-threaded by contract.
    """

    model: Any = None
    data: Any = None
    A: NDArray[np.float64] | None = None
    B: NDArray[np.float64] | None = None
    grip_bid: int = -1
    head_bid: int = -1
    variant: str = ""

    def ensure_model(self, sim_opts: SimOptions) -> None:
        """Compile / re-compile the model if the variant has changed.

        ``mjd_transitionFD`` does not support the RK4 integrator, so we
        force-downgrade to Euler when needed. The forward-sim wrappers in
        this package use their own model instances, so this override is
        scoped to the Jacobian rollout only.
        """
        import mujoco

        if self.model is not None and self.variant == sim_opts.variant:
            if sim_opts.dt is not None:
                self.model.opt.timestep = float(sim_opts.dt)
            return
        xml = _load_model_xml(sim_opts.variant)
        self.model = mujoco.MjModel.from_xml_string(xml)
        if sim_opts.dt is not None:
            if sim_opts.dt <= 0:
                raise ValueError(f"dt must be > 0; got {sim_opts.dt}")
            self.model.opt.timestep = float(sim_opts.dt)
        if int(self.model.opt.integrator) == int(mujoco.mjtIntegrator.mjINT_RK4):
            # Drop to semi-implicit Euler — supported by mjd_transitionFD.
            self.model.opt.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
        self.data = mujoco.MjData(self.model)
        nx = 2 * int(self.model.nv) + int(self.model.na)
        nu = int(self.model.nu)
        self.A = np.zeros((nx, nx), dtype=np.float64)
        self.B = np.zeros((nx, nu), dtype=np.float64)
        self.grip_bid = _resolve_body_id(
            self.model, _GRIP_BODY_BY_VARIANT[sim_opts.variant]
        )
        self.head_bid = _resolve_body_id(
            self.model, _CLUBHEAD_BODY_BY_VARIANT[sim_opts.variant]
        )
        self.variant = sim_opts.variant


# --- Main entry point --------------------------------------------------------


def _evaluate_polynomial_ctrl(
    theta: NDArray[np.float64],
    t: float,
) -> NDArray[np.float64]:
    """Evaluate the polynomial torque at scalar time ``t`` (Horner).

    Coefficients are ascending in power (column k = t^k), so Horner folds
    from the highest power (column 6) down to the constant term (column 0),
    matching :func:`torque_driver._evaluate_polynomial` and the canonical
    cross-engine convention (#7688).
    """
    out = theta[:, -1].astype(np.float64, copy=True)
    for k in range(theta.shape[1] - 2, -1, -1):
        out = out * t + theta[:, k]
    return out


def _apply_ctrl_clip(
    ctrl: NDArray[np.float64],
    model: Any,
    do_clip: bool,
) -> NDArray[np.float64]:
    """Clip ``ctrl`` to the model's actuator ctrlrange when ``do_clip``."""
    if not do_clip:
        return ctrl
    limited = np.asarray(model.actuator_ctrllimited).astype(bool)
    if not np.any(limited):
        return ctrl
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    lo = np.where(limited, ranges[:, 0], -np.inf)
    hi = np.where(limited, ranges[:, 1], np.inf)
    return np.clip(ctrl, lo, hi)


@dataclass(frozen=True)
class _RolloutContext:
    """Immutable inputs shared by every substep of a Jacobian rollout.

    ``A`` and ``B`` are the (mutable) scratch matrices owned by the
    :class:`JacobianCache`; the context only carries references to them.
    """

    model: Any
    theta_mat: NDArray[np.float64]
    t0: float
    do_clip: bool
    eps: float
    flg_centered: int
    A: NDArray[np.float64]
    B: NDArray[np.float64]


def _coerce_theta_matrix(theta: NDArray[np.float64], nu: int) -> NDArray[np.float64]:
    """Return ``theta`` as a finite ``(nu, 7)`` matrix or raise ``ValueError``."""
    n_theta = nu * 7
    flat = np.asarray(theta, dtype=np.float64)
    if flat.ndim == 1:
        if flat.shape[0] != n_theta:
            raise ValueError(
                f"flat theta must have length nu*7 = {n_theta}; got {flat.shape[0]}"
            )
        theta_mat = flat.reshape(nu, 7)
    elif flat.shape == (nu, 7):
        theta_mat = flat
    else:
        raise ValueError(f"theta must have shape ({nu}, 7); got {flat.shape}")
    if not np.all(np.isfinite(theta_mat)):
        raise ValueError("theta must be finite")
    return theta_mat


def _saturation_masked_du_dtheta(
    model: Any,
    theta_mat: NDArray[np.float64],
    t_now: float,
    do_clip: bool,
) -> NDArray[np.float64]:
    """Return ``∂u/∂θ`` at ``t_now`` with saturated actuator rows zeroed.

    If the unclipped ctrl would have hit the box, the row is zero (not
    strictly true at the boundary, but a reasonable subgradient — and zero
    measure for any well-conditioned θ on a continuous problem).
    """
    nu = int(model.nu)
    J_u = polynomial_du_dtheta(nu, t_now)
    if not do_clip:
        return J_u
    limited = np.asarray(model.actuator_ctrllimited).astype(bool)
    if not np.any(limited):
        return J_u
    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
    raw = _evaluate_polynomial_ctrl(theta_mat, t_now)
    saturated = (limited & (raw <= ranges[:, 0])) | (limited & (raw >= ranges[:, 1]))
    if np.any(saturated):
        J_u = J_u.copy()
        J_u[saturated, :] = 0.0
    return J_u


def _apply_polynomial_ctrl(
    model: Any,
    data: Any,
    theta_mat: NDArray[np.float64],
    t_now: float,
    do_clip: bool,
) -> None:
    """Write the (optionally clipped) polynomial torque at ``t_now`` to ``data.ctrl``."""
    ctrl = _evaluate_polynomial_ctrl(theta_mat, t_now)
    data.ctrl[:] = _apply_ctrl_clip(ctrl, model, do_clip)


def _linearised_substep(
    ctx: _RolloutContext,
    data: Any,
    S: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Linearise at the current ``(x, u)``, propagate ``S`` and advance one step.

    ``mjd_transitionFD`` does NOT advance the simulation — it perturbs and
    restores state to compute ``A`` and ``B``; the explicit ``mj_step``
    afterwards advances the actual simulation to match the linearisation.
    """
    import mujoco

    model = ctx.model
    t_now = float(data.time) - ctx.t0
    _apply_polynomial_ctrl(model, data, ctx.theta_mat, t_now, ctx.do_clip)
    mujoco.mjd_transitionFD(
        model, data, ctx.eps, ctx.flg_centered, ctx.A, ctx.B, None, None
    )
    J_u = _saturation_masked_du_dtheta(model, ctx.theta_mat, t_now, ctx.do_clip)
    S = ctx.A @ S + ctx.B @ J_u
    mujoco.mj_step(model, data)
    return S


def _propagate_sensitivity_to(
    ctx: _RolloutContext,
    data: Any,
    S: NDArray[np.float64],
    t_target: float,
    max_substeps: int,
    frame: int,
) -> NDArray[np.float64]:
    """Advance ``data`` to ``t_target`` propagating ``S``; raise on divergence."""
    substeps = 0
    while data.time + 1e-12 < t_target and substeps < max_substeps:
        S = _linearised_substep(ctx, data, S)
        substeps += 1
        if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
            raise ValueError(
                f"rollout diverged at frame {frame}, substep {substeps}; "
                "θ may be near a stiff edge of the bound box"
            )
    return S


def compute_qpos_jacobian(
    theta: NDArray[np.float64],
    sim_opts: SimOptions,
    initial_pose: NDArray[np.float64] | None = None,
    cache: JacobianCache | None = None,
    eps: float = 1e-6,
    flg_centered: int = 1,
) -> NDArray[np.float64]:
    """Compute ``∂q(t_k) / ∂θ`` along the rollout via ``mjd_transitionFD``.

    The state-transition Jacobian ``A`` and the input Jacobian ``B`` are
    populated by ``mujoco.mjd_transitionFD`` at every step; we walk the
    rollout forward and propagate

        S_{k+1} = A_k @ S_k + B_k @ J_u(t_k)

    starting from ``S_0 = 0`` (the initial state is independent of ``θ``).
    ``S`` has the MuJoCo state layout ``[dqpos; qvel; act]``; the leading
    ``nv`` rows are the qpos sensitivities in the joint tangent space.

    Args:
        theta: ``(n_joints, 7)`` matrix or flat ``(n_joints * 7,)`` vector.
        sim_opts: forward-sim options. The output grid ``t_grid`` matches
            :func:`simulate_with_coefficients`.
        initial_pose: optional initial ``qpos`` (``<= nq`` entries).
        cache: optional :class:`JacobianCache` for amortising compile and
            scratch buffers across calls.
        eps: finite-difference step size passed to ``mjd_transitionFD``.
        flg_centered: 1 for centred differences (recommended), 0 for
            forward differences (faster but noisier).

    Returns:
        ``(N, nv, n_joints * 7)`` array where ``N`` is the number of
        output frames.

    Raises:
        ValueError: if the rollout diverges or sizes mismatch.
    """
    import mujoco

    if cache is None:
        cache = JacobianCache()
    cache.ensure_model(sim_opts)
    model = cache.model
    data = cache.data
    assert model is not None and data is not None
    assert cache.A is not None and cache.B is not None

    nu = int(model.nu)
    nv = int(model.nv)
    n_theta = nu * 7
    theta_mat = _coerce_theta_matrix(theta, nu)

    # Reset state for this rollout.
    mujoco.mj_resetData(model, data)
    _apply_initial_pose(data, initial_pose, model.nq)
    mujoco.mj_forward(model, data)

    t_grid = _output_grid(sim_opts.T_s, sim_opts.output_rate_hz)
    n_out = t_grid.size
    ctx = _RolloutContext(
        model=model,
        theta_mat=theta_mat,
        t0=float(sim_opts.t0),
        do_clip=bool(sim_opts.clip_torque_to_ctrlrange),
        eps=eps,
        flg_centered=flg_centered,
        A=cache.A,
        B=cache.B,
    )
    max_substeps = int(np.ceil(sim_opts.T_s / model.opt.timestep) + 16)

    # S[k] = ∂x / ∂θ at output frame k. Initial state independent of θ.
    out_dq_dtheta = np.zeros((n_out, nv, n_theta), dtype=np.float64)
    S = np.zeros((2 * nv + int(model.na), n_theta), dtype=np.float64)
    out_dq_dtheta[0] = S[:nv, :]
    for k in range(1, n_out):
        S = _propagate_sensitivity_to(ctx, data, S, float(t_grid[k]), max_substeps, k)
        out_dq_dtheta[k] = S[:nv, :]

    return out_dq_dtheta


# --- Cost gradient via analytical dq/dθ -------------------------------------


@dataclass
class _BodyTrack:
    """Grip / clubhead world positions and translational Jacobians per frame."""

    grip_pos: NDArray[np.float64]
    head_pos: NDArray[np.float64]
    grip_jac_p: NDArray[np.float64]
    head_jac_p: NDArray[np.float64]

    @classmethod
    def allocate(cls, n_out: int, nv: int) -> _BodyTrack:
        """Allocate zeroed tracks for ``n_out`` frames and ``nv`` dofs."""
        if n_out <= 0 or nv <= 0:
            raise ValueError(f"n_out and nv must be > 0; got {n_out}, {nv}")
        return cls(
            grip_pos=np.zeros((n_out, 3), dtype=np.float64),
            head_pos=np.zeros((n_out, 3), dtype=np.float64),
            grip_jac_p=np.zeros((n_out, 3, nv), dtype=np.float64),
            head_jac_p=np.zeros((n_out, 3, nv), dtype=np.float64),
        )

    def record(self, k: int, model: Any, data: Any, cache: JacobianCache) -> None:
        """Store body positions and ``mj_jacBody`` Jacobians for frame ``k``."""
        import mujoco

        jp = np.zeros((3, int(model.nv)), dtype=np.float64)
        self.grip_pos[k] = data.xpos[cache.grip_bid]
        self.head_pos[k] = data.xpos[cache.head_bid]
        mujoco.mj_jacBody(model, data, jp, None, cache.grip_bid)
        self.grip_jac_p[k] = jp
        mujoco.mj_jacBody(model, data, jp, None, cache.head_bid)
        self.head_jac_p[k] = jp


def _walk_body_track(
    cache: JacobianCache,
    sim_opts: SimOptions,
    initial_pose: NDArray[np.float64] | None,
    theta_mat: NDArray[np.float64],
    n_out: int,
) -> _BodyTrack:
    """Re-walk the rollout recording body positions and Jacobians per frame.

    Re-uses the cache's model and a *separate* ``MjData`` so the
    linearisation walk is not disturbed.
    """
    import mujoco

    model = cache.model
    data_walk = mujoco.MjData(model)
    _apply_initial_pose(data_walk, initial_pose, model.nq)
    mujoco.mj_forward(model, data_walk)

    track = _BodyTrack.allocate(n_out, int(model.nv))
    track.record(0, model, data_walk, cache)

    t0 = float(sim_opts.t0)
    t_grid = _output_grid(sim_opts.T_s, sim_opts.output_rate_hz)
    do_clip = bool(sim_opts.clip_torque_to_ctrlrange)
    max_substeps = int(np.ceil(sim_opts.T_s / model.opt.timestep) + 16)

    for k in range(1, n_out):
        t_target = float(t_grid[k])
        substeps = 0
        while data_walk.time + 1e-12 < t_target and substeps < max_substeps:
            t_now = float(data_walk.time) - t0
            _apply_polynomial_ctrl(model, data_walk, theta_mat, t_now, do_clip)
            mujoco.mj_step(model, data_walk)
            substeps += 1
        track.record(k, model, data_walk, cache)
    return track


def _assemble_cost_gradient(
    track: _BodyTrack,
    dq_dtheta: NDArray[np.float64],
    target: Any,
    cost_opts: Any,
    flat_theta: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Assemble ``∂J/∂θ`` from the body tracks and ``∂q/∂θ``.

    The cost is mean over frames, so the accumulator is divided by N at
    the end.

    Position term (mean over frames):
      J_pos = w_pos / N · Σ_k (||grip - g*||^2 + ||head - h*||^2)
      ∂J_pos / ∂θ = 2*w_pos/N · Σ_k [
          (grip - g*)^T (Jp_grip @ dq/dθ_k)
        + (head - h*)^T (Jp_head @ dq/dθ_k)
      ]
    """
    n_out = dq_dtheta.shape[0]
    target_grip = np.asarray(target.butt, dtype=np.float64)
    target_head = np.asarray(target.clubhead, dtype=np.float64)
    if target_grip.shape != (n_out, 3) or target_head.shape != (n_out, 3):
        raise ValueError(
            f"target shapes {target_grip.shape}/{target_head.shape} do not "
            f"match rollout length {n_out}"
        )
    grad = np.zeros(flat_theta.size, dtype=np.float64)
    w_pos = float(cost_opts.w_position)
    for k in range(n_out):
        # dgrip/dθ = Jp_grip @ dq/dθ  (3 x n_theta)
        d_grip = track.grip_jac_p[k] @ dq_dtheta[k]
        d_head = track.head_jac_p[k] @ dq_dtheta[k]
        diff_g = track.grip_pos[k] - target_grip[k]
        diff_h = track.head_pos[k] - target_head[k]
        grad += diff_g @ d_grip
        grad += diff_h @ d_head
    grad *= 2.0 * w_pos / n_out

    # Impact-anchor term (only at impact_idx-1 in 0-based):
    #   J_anc = w_anc · ||head[k_imp] - target_head[k_imp]||^2
    #   ∂J_anc / ∂θ = 2*w_anc · (head - target_head) · Jp_head · dq/dθ
    k_imp = int(target.impact_idx) - 1
    if 0 <= k_imp < n_out:
        d_head_imp = track.head_jac_p[k_imp] @ dq_dtheta[k_imp]
        diff_imp = track.head_pos[k_imp] - target_head[k_imp]
        grad += 2.0 * float(cost_opts.w_anchor_impact) * (diff_imp @ d_head_imp)

    # coeff_l2 regularizer: closed form. ∂(λ θ^T θ)/∂θ = 2 λ θ.
    if getattr(cost_opts, "regularizer", "") == "coeff_l2":
        grad += 2.0 * float(cost_opts.lambda_) * flat_theta
    return grad


def compute_cost_gradient_analytical(
    theta: NDArray[np.float64],
    target: Any,
    sim_opts: SimOptions,
    cost_opts: Any,
    initial_pose: NDArray[np.float64] | None = None,
    cache: JacobianCache | None = None,
) -> NDArray[np.float64]:
    """Closed-form ``∂J / ∂θ`` for the position + impact-anchor cost terms.

    Implements the analytical gradient path tracked by issue #4175. The
    position term is the dominant signal in the cost landscape (per
    ``MUJOCO_PARITY_SPEC.md`` §6.2); we close the chain rule

        ∂J_pos / ∂θ = (2 / N) · Σ_k [
            (grip_k - target_grip_k) · ∂grip_k / ∂q_k · ∂q_k / ∂θ
          + (head_k - target_head_k) · ∂head_k / ∂q_k · ∂q_k / ∂θ
        ]

    using ``mj_jacBody`` for ``∂x_body / ∂q`` and :func:`compute_qpos_jacobian`
    for ``∂q / ∂θ``. The orientation, ``coeff_l2`` regularizer, and
    impact-anchor terms are handled in closed form where their derivatives
    are simple (``coeff_l2``: ``2 · λ · θ``; impact-anchor: same chain rule
    as position but only at the impact frame). Other regularizers fall back
    to the position-only gradient — the optimizer still benefits from the
    cheap-to-compute dominant-term direction.

    Args:
        theta: flat ``(n_theta,)`` parameter vector.
        target: :class:`ClubTarget`.
        sim_opts: forward-sim options (must produce ``len(target.time)``
            output frames).
        cost_opts: :class:`CostOptions` driving the weights and regularizer.
        initial_pose: optional initial qpos.
        cache: optional :class:`JacobianCache` for reuse across calls.

    Returns:
        ``(n_theta,)`` gradient vector.
    """
    if cache is None:
        cache = JacobianCache()
    cache.ensure_model(sim_opts)
    assert cache.model is not None and cache.data is not None

    nu = int(cache.model.nu)
    n_theta = nu * 7

    flat = np.asarray(theta, dtype=np.float64).reshape(-1)
    if flat.size != n_theta:
        raise ValueError(f"theta must have length {n_theta}; got {flat.size}")

    # 1. Trajectory-Jacobian: dq/dθ at every frame (N, nv, n_theta).
    dq_dtheta = compute_qpos_jacobian(
        flat, sim_opts, initial_pose=initial_pose, cache=cache
    )

    # 2. Re-walk the rollout for body positions and body Jacobians.
    track = _walk_body_track(
        cache, sim_opts, initial_pose, flat.reshape(nu, 7), dq_dtheta.shape[0]
    )

    # 3. Assemble the gradient.
    return _assemble_cost_gradient(track, dq_dtheta, target, cost_opts, flat)

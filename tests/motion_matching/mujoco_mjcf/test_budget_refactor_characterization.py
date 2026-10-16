"""Bit-identical characterization of the MuJoCo motion-matching kernels (#12842).

The ``legacy_*`` functions below are verbatim copies of the pre-refactor
``simulate_with_coefficients``, ``compute_qpos_jacobian`` and
``compute_cost_gradient_analytical`` bodies.  The refactored public
functions must reproduce them exactly (``np.array_equal``), including the
error behaviour pinned at the end of the module.

Marked ``requires_mujoco``; skipped when the ``mujoco`` package is missing.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from numpy.typing import NDArray

pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.motion_matching import (  # noqa: E402
    jacobians as jac_mod,
)
from src.engines.physics_engines.mujoco.python.motion_matching.jacobians import (  # noqa: E402
    JacobianCache,
    _apply_ctrl_clip,
    _evaluate_polynomial_ctrl,
    compute_cost_gradient_analytical,
    compute_qpos_jacobian,
    polynomial_du_dtheta,
)
from src.engines.physics_engines.mujoco.python.motion_matching.simulate import (  # noqa: E402
    _CLUBHEAD_BODY_BY_VARIANT,
    _GRIP_BODY_BY_VARIANT,
    SimOptions,
    SimOut,
    _apply_initial_pose,
    _load_model_xml,
    _output_grid,
    _resolve_body_id,
    simulate_with_coefficients,
)
from src.engines.physics_engines.mujoco.python.motion_matching.torque_driver import (  # noqa: E402
    PolynomialTorqueDriver,
)
from src.shared.python.motion_matching.club_target import (  # noqa: E402
    ClubTarget,
    SourceProvenance,
)
from src.shared.python.motion_matching.final_cost import CostOptions  # noqa: E402
from src.shared.python.motion_matching.validate_theta import (  # noqa: E402
    validate_theta,
)

pytestmark = [pytest.mark.requires_mujoco, pytest.mark.unit]


# --- Legacy oracles (verbatim pre-refactor bodies) --------------------------


def legacy_simulate(
    theta: NDArray[np.float64],
    options: SimOptions | None = None,
    initial_pose: NDArray[np.float64] | None = None,
) -> SimOut:
    """Roll out the MuJoCo model under the polynomial-torque driver.

    Args:
        theta: ``(n_joints * 7,)`` flat vector, or ``(n_joints, 7)`` matrix.
            ``n_joints`` must equal the compiled model's ``nu``. Layout per
            joint is ascending in power (column k = ``t^k``):
            ``[A, B, C, D, E, F, G]`` for
            ``A + B*t + C*t^2 + D*t^3 + E*t^4 + F*t^5 + G*t^6`` (canonical
            cross-engine convention, #7688).
        options: :class:`SimOptions`.
        initial_pose: optional ``(<= nq,)`` initial generalised
            coordinates. ``None`` uses the MJCF default.

    Returns:
        :class:`SimOut` with shapes documented on the dataclass.

    Raises:
        ValueError: if ``theta`` is malformed or options are invalid.
        RuntimeError: if model compilation fails or required bodies are
            missing.
    """
    if options is None:
        options = SimOptions()
    if not isinstance(options, SimOptions):
        raise TypeError(
            f"options must be a SimOptions instance; got {type(options).__name__}"
        )

    import mujoco

    if options.xml_path is not None:
        xml_p = Path(options.xml_path)
        if not xml_p.exists():
            raise FileNotFoundError(f"model XML not found at {xml_p}")
        model = mujoco.MjModel.from_xml_path(str(xml_p))
    else:
        xml = _load_model_xml(options.variant)
        model = mujoco.MjModel.from_xml_string(xml)

    # Optional override of the model timestep.
    if options.dt is not None:
        if options.dt <= 0:
            raise ValueError(f"dt must be > 0; got {options.dt}")
        model.opt.timestep = float(options.dt)

    if options.compute_energy:
        model.opt.enableflags |= mujoco.mjtEnableBit.mjENBL_ENERGY

    # Spec §2.2: enforce length + finiteness against the compiled model's
    # actuator count (``nu``). Bounds are enforced separately by the
    # ``PolynomialTorqueDriver`` clip path; the validator focuses on the
    # two checks that prevent silent failures / numerical divergence.
    theta = validate_theta(theta, n_joints=int(model.nu))

    data = mujoco.MjData(model)
    _apply_initial_pose(data, initial_pose, model.nq)
    mujoco.mj_forward(model, data)

    # Resolve grip and clubhead: prefer canonical sites 'mid_hands' and 'clubhead',
    # falling back to candidate bodies per variant or general names.
    grip_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "mid_hands")
    club_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, "clubhead")

    grip_bid = -1
    head_bid = -1
    if grip_site_id < 0 or club_site_id < 0:
        variant_key = (
            options.variant if options.variant in _GRIP_BODY_BY_VARIANT else "full"
        )
        grip_body_name = _GRIP_BODY_BY_VARIANT.get(variant_key, "club")
        head_body_name = _CLUBHEAD_BODY_BY_VARIANT.get(variant_key, "clubhead")
        try:
            grip_bid = _resolve_body_id(model, grip_body_name)
        except RuntimeError:
            for alt in ("club_grip", "hand_right", "hand_left"):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, alt)
                if bid >= 0:
                    grip_bid = bid
                    break
        try:
            head_bid = _resolve_body_id(model, head_body_name)
        except RuntimeError:
            for alt in ("club_head", "clubhead", "club_shaft"):
                bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, alt)
                if bid >= 0:
                    head_bid = bid
                    break

    # Output grid + per-frame sample stride.
    t_grid = _output_grid(options.T_s, options.output_rate_hz)
    n_out = t_grid.size
    nq = int(model.nq)
    nv = int(model.nv)
    nu = int(model.nu)

    out_q = np.zeros((n_out, nq), dtype=np.float64)
    out_qd = np.zeros((n_out, nv), dtype=np.float64)
    out_qdd = np.zeros((n_out, nv), dtype=np.float64)
    out_tau = np.zeros((n_out, nu), dtype=np.float64)
    out_grip = np.zeros((n_out, 3), dtype=np.float64)
    out_grip_q = np.zeros((n_out, 4), dtype=np.float64)
    out_head = np.zeros((n_out, 3), dtype=np.float64)
    out_head_q = np.zeros((n_out, 4), dtype=np.float64)
    out_ke = np.zeros(n_out, dtype=np.float64)
    out_pe = np.zeros(n_out, dtype=np.float64)

    def _sample_kinematics(dest_idx: int) -> None:
        if grip_site_id >= 0:
            out_grip[dest_idx] = data.site_xpos[grip_site_id]
            mujoco.mju_mat2Quat(out_grip_q[dest_idx], data.site_xmat[grip_site_id])
        elif grip_bid >= 0:
            out_grip[dest_idx] = data.xpos[grip_bid]
            out_grip_q[dest_idx] = data.xquat[grip_bid]

        if club_site_id >= 0:
            out_head[dest_idx] = data.site_xpos[club_site_id]
            mujoco.mju_mat2Quat(out_head_q[dest_idx], data.site_xmat[club_site_id])
        elif head_bid >= 0:
            out_head[dest_idx] = data.xpos[head_bid]
            out_head_q[dest_idx] = data.xquat[head_bid]

        if options.compute_energy:
            mujoco.mj_energyPos(model, data)
            mujoco.mj_energyVel(model, data)
            out_pe[dest_idx] = data.energy[0]
            out_ke[dest_idx] = data.energy[1]

    # Capture frame 0 (post mj_forward, pre any mj_step).
    out_q[0] = data.qpos
    out_qd[0] = data.qvel
    if options.compute_qdd:
        out_qdd[0] = data.qacc
    _sample_kinematics(0)

    driver = PolynomialTorqueDriver(
        model,
        theta,
        t0=options.t0,
        clip_to_ctrlrange=options.clip_torque_to_ctrlrange,
        basis=options.basis,
        T_s=options.T_s,
    )
    out_tau[0] = driver.evaluate(0.0)
    if options.clip_torque_to_ctrlrange:
        np.clip(
            out_tau[0],
            np.where(
                model.actuator_ctrllimited.astype(bool),
                model.actuator_ctrlrange[:, 0],
                -np.inf,
            ),
            np.where(
                model.actuator_ctrllimited.astype(bool),
                model.actuator_ctrlrange[:, 1],
                np.inf,
            ),
            out=out_tau[0],
        )

    solver_status = "success"
    t_start = time.perf_counter()
    try:
        with driver:
            for i in range(1, n_out):
                t_target = float(t_grid[i])
                max_substeps = int(np.ceil(options.T_s / model.opt.timestep) + 16)
                substeps = 0
                while data.time + 1e-12 < t_target and substeps < max_substeps:
                    mujoco.mj_step(model, data)
                    substeps += 1
                    if not np.all(np.isfinite(data.qpos)) or not np.all(
                        np.isfinite(data.qvel)
                    ):
                        solver_status = "failed"
                        break
                if solver_status != "success":
                    break
                out_q[i] = data.qpos
                out_qd[i] = data.qvel
                if options.compute_qdd:
                    out_qdd[i] = data.qacc
                out_tau[i] = data.ctrl
                _sample_kinematics(i)
    finally:
        driver.uninstall()

    duration_s = time.perf_counter() - t_start

    meta = {
        "n_joints": nu,
        "model_nq": nq,
        "model_nv": nv,
        "model_nu": nu,
        "variant": options.variant if options.xml_path is None else "custom",
        "xml_path": str(options.xml_path) if options.xml_path is not None else None,
        "grip_site_id": int(grip_site_id),
        "club_site_id": int(club_site_id),
        "grip_body_id": int(grip_bid),
        "head_body_id": int(head_bid),
        "timestep": float(model.opt.timestep),
        "compute_energy": bool(options.compute_energy),
        "basis": str(options.basis),
        "duration_s": float(duration_s),
    }

    return SimOut(
        time=t_grid,
        q=out_q,
        qd=out_qd,
        qdd=out_qdd,
        tau=out_tau,
        grip=out_grip,
        grip_quat=out_grip_q,
        clubhead=out_head,
        club_quat=out_head_q,
        solver_status=solver_status,
        duration_s=duration_s,
        kinetic_energy=out_ke,
        potential_energy=out_pe,
        meta=meta,
    )


def legacy_qpos_jacobian(
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
    A = cache.A
    B = cache.B
    assert model is not None and data is not None
    assert A is not None and B is not None

    nu = int(model.nu)
    nv = int(model.nv)
    na = int(model.na)
    nx = 2 * nv + na
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

    # Reset state for this rollout.
    mujoco.mj_resetData(model, data)
    _apply_initial_pose(data, initial_pose, model.nq)
    mujoco.mj_forward(model, data)

    t_grid = _output_grid(sim_opts.T_s, sim_opts.output_rate_hz)
    n_out = t_grid.size

    # S[k] = ∂x / ∂θ at output frame k. Initial state independent of θ.
    out_dq_dtheta = np.zeros((n_out, nv, n_theta), dtype=np.float64)
    S = np.zeros((nx, n_theta), dtype=np.float64)
    out_dq_dtheta[0] = S[:nv, :]

    do_clip = bool(sim_opts.clip_torque_to_ctrlrange)
    t0 = float(sim_opts.t0)
    max_substeps = int(np.ceil(sim_opts.T_s / model.opt.timestep) + 16)

    for k in range(1, n_out):
        t_target = float(t_grid[k])
        substeps = 0
        while data.time + 1e-12 < t_target and substeps < max_substeps:
            t_now = float(data.time) - t0
            ctrl = _evaluate_polynomial_ctrl(theta_mat, t_now)
            ctrl = _apply_ctrl_clip(ctrl, model, do_clip)
            data.ctrl[:] = ctrl

            # Linearise around the current (x, u). mjd_transitionFD does
            # NOT advance the simulation — it perturbs and restores state
            # to compute A and B; we still need an explicit mj_step.
            mujoco.mjd_transitionFD(model, data, eps, flg_centered, A, B, None, None)

            # ∂u / ∂θ at the linearisation time. If the unclipped ctrl
            # would have hit the box, the row is zero (not strictly true
            # at the boundary, but a reasonable subgradient — and zero
            # measure for any well-conditioned θ on a continuous problem).
            J_u = polynomial_du_dtheta(nu, t_now)
            if do_clip:
                limited = np.asarray(model.actuator_ctrllimited).astype(bool)
                if np.any(limited):
                    ranges = np.asarray(model.actuator_ctrlrange, dtype=np.float64)
                    raw = _evaluate_polynomial_ctrl(theta_mat, t_now)
                    at_lo = limited & (raw <= ranges[:, 0])
                    at_hi = limited & (raw >= ranges[:, 1])
                    saturated = at_lo | at_hi
                    if np.any(saturated):
                        J_u = J_u.copy()
                        J_u[saturated, :] = 0.0

            S = A @ S + B @ J_u

            # Now advance the actual simulation to match the linearisation.
            mujoco.mj_step(model, data)

            substeps += 1
            if not np.all(np.isfinite(data.qpos)) or not np.all(np.isfinite(data.qvel)):
                raise ValueError(
                    f"rollout diverged at frame {k}, substep {substeps}; "
                    "θ may be near a stiff edge of the bound box"
                )
        out_dq_dtheta[k] = S[:nv, :]

    return out_dq_dtheta


def legacy_cost_gradient(
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
    import mujoco

    if cache is None:
        cache = JacobianCache()
    cache.ensure_model(sim_opts)
    model = cache.model
    data = cache.data
    assert model is not None and data is not None

    nv = int(model.nv)
    nu = int(model.nu)
    n_theta = nu * 7

    flat = np.asarray(theta, dtype=np.float64).reshape(-1)
    if flat.size != n_theta:
        raise ValueError(f"theta must have length {n_theta}; got {flat.size}")

    # 1. Trajectory-Jacobian: dq/dθ at every frame (N, nv, n_theta).
    dq_dtheta = legacy_qpos_jacobian(
        flat, sim_opts, initial_pose=initial_pose, cache=cache
    )
    n_out = dq_dtheta.shape[0]

    # 2. Re-walk the rollout to evaluate body positions and the
    # body Jacobians at each output frame. We re-use the cache's model
    # and a *separate* mjData to avoid disturbing the linearisation walk.
    data_walk = mujoco.MjData(model)
    _apply_initial_pose(data_walk, initial_pose, model.nq)
    mujoco.mj_forward(model, data_walk)

    grip_bid = cache.grip_bid
    head_bid = cache.head_bid
    grip_pos = np.zeros((n_out, 3), dtype=np.float64)
    head_pos = np.zeros((n_out, 3), dtype=np.float64)
    grip_jac_p = np.zeros((n_out, 3, nv), dtype=np.float64)
    head_jac_p = np.zeros((n_out, 3, nv), dtype=np.float64)

    grip_pos[0] = data_walk.xpos[grip_bid]
    head_pos[0] = data_walk.xpos[head_bid]
    jp = np.zeros((3, nv), dtype=np.float64)
    mujoco.mj_jacBody(model, data_walk, jp, None, grip_bid)
    grip_jac_p[0] = jp.copy()
    mujoco.mj_jacBody(model, data_walk, jp, None, head_bid)
    head_jac_p[0] = jp.copy()

    theta_mat = flat.reshape(nu, 7)
    t0 = float(sim_opts.t0)
    t_grid = _output_grid(sim_opts.T_s, sim_opts.output_rate_hz)
    do_clip = bool(sim_opts.clip_torque_to_ctrlrange)
    max_substeps = int(np.ceil(sim_opts.T_s / model.opt.timestep) + 16)

    for k in range(1, n_out):
        t_target = float(t_grid[k])
        substeps = 0
        while data_walk.time + 1e-12 < t_target and substeps < max_substeps:
            t_now = float(data_walk.time) - t0
            ctrl = _evaluate_polynomial_ctrl(theta_mat, t_now)
            ctrl = _apply_ctrl_clip(ctrl, model, do_clip)
            data_walk.ctrl[:] = ctrl
            mujoco.mj_step(model, data_walk)
            substeps += 1
        grip_pos[k] = data_walk.xpos[grip_bid]
        head_pos[k] = data_walk.xpos[head_bid]
        mujoco.mj_jacBody(model, data_walk, jp, None, grip_bid)
        grip_jac_p[k] = jp.copy()
        mujoco.mj_jacBody(model, data_walk, jp, None, head_bid)
        head_jac_p[k] = jp.copy()

    # 3. Assemble gradient. The cost is mean over frames, so the
    # accumulator is divided by N at the end.
    #
    # Position term (mean over frames):
    #   J_pos = w_pos / N · Σ_k (||grip - g*||^2 + ||head - h*||^2)
    #   ∂J_pos / ∂θ = 2*w_pos/N · Σ_k [
    #       (grip - g*)^T (Jp_grip @ dq/dθ_k)
    #     + (head - h*)^T (Jp_head @ dq/dθ_k)
    #   ]
    target_grip = np.asarray(target.butt, dtype=np.float64)
    target_head = np.asarray(target.clubhead, dtype=np.float64)
    if target_grip.shape != (n_out, 3) or target_head.shape != (n_out, 3):
        raise ValueError(
            f"target shapes {target_grip.shape}/{target_head.shape} do not "
            f"match rollout length {n_out}"
        )
    grad = np.zeros(n_theta, dtype=np.float64)
    w_pos = float(cost_opts.w_position)
    for k in range(n_out):
        # dgrip/dθ = Jp_grip @ dq/dθ  (3 x n_theta)
        d_grip = grip_jac_p[k] @ dq_dtheta[k]
        d_head = head_jac_p[k] @ dq_dtheta[k]
        diff_g = grip_pos[k] - target_grip[k]
        diff_h = head_pos[k] - target_head[k]
        grad += diff_g @ d_grip
        grad += diff_h @ d_head
    grad *= 2.0 * w_pos / n_out

    # Impact-anchor term (only at impact_idx-1 in 0-based):
    #   J_anc = w_anc · ||head[k_imp] - target_head[k_imp]||^2
    #   ∂J_anc / ∂θ = 2*w_anc · (head - target_head) · Jp_head · dq/dθ
    k_imp = int(target.impact_idx) - 1
    if 0 <= k_imp < n_out:
        d_head_imp = head_jac_p[k_imp] @ dq_dtheta[k_imp]
        diff_imp = head_pos[k_imp] - target_head[k_imp]
        grad += 2.0 * float(cost_opts.w_anchor_impact) * (diff_imp @ d_head_imp)

    # coeff_l2 regularizer: closed form. ∂(λ θ^T θ)/∂θ = 2 λ θ.
    if getattr(cost_opts, "regularizer", "") == "coeff_l2":
        grad += 2.0 * float(cost_opts.lambda_) * flat

    return grad


# --- Helpers ----------------------------------------------------------------


def _nu(variant: str = "upper") -> int:
    import mujoco

    return int(mujoco.MjModel.from_xml_string(_load_model_xml(variant)).nu)


def _theta(variant: str = "upper", seed: int = 0, scale: float = 0.5) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return rng.uniform(-1.0, 1.0, size=_nu(variant) * 7) * scale


def _assert_same_simout(new: SimOut, old: SimOut) -> None:
    for name in (
        "time",
        "q",
        "qd",
        "qdd",
        "tau",
        "grip",
        "grip_quat",
        "clubhead",
        "club_quat",
        "kinetic_energy",
        "potential_energy",
    ):
        a, b = getattr(new, name), getattr(old, name)
        assert a.shape == b.shape, name
        assert np.array_equal(a, b), name
    assert new.solver_status == old.solver_status
    new_meta = {k: v for k, v in new.meta.items() if k != "duration_s"}
    old_meta = {k: v for k, v in old.meta.items() if k != "duration_s"}
    assert new_meta == old_meta


def _target(n: int, seed: int = 1, impact_idx: int | None = None) -> ClubTarget:
    rng = np.random.default_rng(seed)
    return ClubTarget(
        time=np.linspace(0.0, 0.05, n),
        butt=rng.normal(size=(n, 3)),
        clubhead=rng.normal(size=(n, 3)),
        club_quat=np.tile([1.0, 0.0, 0.0, 0.0], (n, 1)),
        impact_idx=n - 2 if impact_idx is None else impact_idx,
        source=SourceProvenance(
            filename="<synthetic>",
            format="synthetic",
            subject_id="s",
            trial_id="t",
            sha256="0" * 64,
        ),
    )


_UPPER = SimOptions(variant="upper", T_s=0.05, output_rate_hz=200.0)
_N_UPPER = int(round(_UPPER.T_s * _UPPER.output_rate_hz)) + 1


# --- simulate_with_coefficients ---------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "opts",
    [
        SimOptions(variant="upper", T_s=0.05, output_rate_hz=200.0),
        SimOptions(
            variant="upper",
            T_s=0.04,
            output_rate_hz=250.0,
            compute_energy=False,
            compute_qdd=False,
            clip_torque_to_ctrlrange=False,
            basis="bernstein",
            t0=0.01,
        ),
        SimOptions(variant="upper", T_s=0.03, output_rate_hz=300.0, dt=0.0005),
        SimOptions(variant="full", T_s=0.02, output_rate_hz=200.0),
        SimOptions(variant="canonical", T_s=0.02, output_rate_hz=200.0),
    ],
    ids=["upper", "upper-bernstein-noclip", "upper-dt", "full", "canonical"],
)
def test_simulate_bit_identical_to_legacy(opts: SimOptions) -> None:
    theta = _theta(opts.variant)
    new = simulate_with_coefficients(theta, opts)
    old = legacy_simulate(theta, opts)
    _assert_same_simout(new, old)


@pytest.mark.unit
def test_simulate_initial_pose_and_matrix_theta_bit_identical() -> None:
    theta = _theta().reshape(_nu(), 7)
    pose = np.full(3, 0.05)
    new = simulate_with_coefficients(theta, _UPPER, pose)
    old = legacy_simulate(theta, _UPPER, pose)
    _assert_same_simout(new, old)


@pytest.mark.unit
def test_simulate_xml_path_bit_identical(tmp_path: Path) -> None:
    xml = tmp_path / "m.xml"
    xml.write_text(_load_model_xml("upper"), encoding="utf-8")
    opts = SimOptions(xml_path=xml, T_s=0.03, output_rate_hz=200.0)
    new = simulate_with_coefficients(_theta(), opts)
    old = legacy_simulate(_theta(), opts)
    _assert_same_simout(new, old)
    assert new.meta["variant"] == "custom"
    assert new.meta["xml_path"] == str(xml)


@pytest.mark.unit
def test_simulate_divergence_matches_legacy() -> None:
    theta = _theta(scale=1.0e12)
    opts = SimOptions(
        variant="upper",
        T_s=0.05,
        output_rate_hz=200.0,
        clip_torque_to_ctrlrange=False,
    )
    try:
        old = legacy_simulate(theta, opts)
    except Exception as exc:  # noqa: BLE001 - postcondition raised on non-finite state
        with pytest.raises(type(exc)):
            simulate_with_coefficients(theta, opts)
        return
    new = simulate_with_coefficients(theta, opts)
    _assert_same_simout(new, old)


@pytest.mark.unit
def test_simulate_error_contract() -> None:
    with pytest.raises(TypeError, match="SimOptions"):
        simulate_with_coefficients(_theta(), {"T_s": 0.1})  # type: ignore[arg-type]
    with pytest.raises(FileNotFoundError, match="model XML not found"):
        simulate_with_coefficients(
            _theta(), SimOptions(xml_path="/nonexistent/model.xml")
        )
    with pytest.raises(ValueError, match="dt must be > 0"):
        simulate_with_coefficients(_theta(), SimOptions(variant="upper", dt=-1.0))
    with pytest.raises(ValueError, match="unknown variant"):
        simulate_with_coefficients(_theta(), SimOptions(variant="nope"))  # type: ignore[arg-type]
    with pytest.raises(Exception, match="multiple of 7"):
        simulate_with_coefficients(np.zeros(8), _UPPER)
    with pytest.raises(ValueError, match="initial_pose"):
        simulate_with_coefficients(_theta(), _UPPER, np.zeros(10_000))
    with pytest.raises(ValueError, match="finite"):
        simulate_with_coefficients(_theta(), _UPPER, np.array([np.nan]))


# --- compute_qpos_jacobian --------------------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize(
    "opts",
    [
        SimOptions(variant="upper", T_s=0.03, output_rate_hz=200.0),
        SimOptions(
            variant="upper",
            T_s=0.03,
            output_rate_hz=200.0,
            clip_torque_to_ctrlrange=False,
            t0=0.005,
            dt=0.001,
        ),
    ],
    ids=["clip", "noclip-dt-t0"],
)
def test_qpos_jacobian_bit_identical_to_legacy(opts: SimOptions) -> None:
    theta = _theta(scale=2.0)
    new = compute_qpos_jacobian(theta, opts)
    old = legacy_qpos_jacobian(theta, opts)
    assert new.shape == old.shape
    assert np.array_equal(new, old)


@pytest.mark.unit
def test_qpos_jacobian_saturated_rows_bit_identical() -> None:
    theta = _theta(scale=1.0e6)  # drives actuators into their ctrlrange
    new = compute_qpos_jacobian(theta, _UPPER)
    old = legacy_qpos_jacobian(theta, _UPPER)
    assert np.array_equal(new, old)


@pytest.mark.unit
def test_qpos_jacobian_options_cache_pose_bit_identical() -> None:
    theta = _theta().reshape(_nu(), 7)
    pose = np.full(2, 0.02)
    new = compute_qpos_jacobian(theta, _UPPER, pose, JacobianCache(), 1e-5, 0)
    old = legacy_qpos_jacobian(theta, _UPPER, pose, JacobianCache(), 1e-5, 0)
    assert np.array_equal(new, old)


@pytest.mark.unit
def test_qpos_jacobian_cache_is_reused_across_calls() -> None:
    cache = JacobianCache()
    first = compute_qpos_jacobian(_theta(), _UPPER, cache=cache)
    model = cache.model
    second = compute_qpos_jacobian(_theta(), _UPPER, cache=cache)
    assert cache.model is model
    assert np.array_equal(first, second)


@pytest.mark.unit
def test_qpos_jacobian_error_contract() -> None:
    nu = _nu()
    with pytest.raises(ValueError, match="flat theta must have length"):
        compute_qpos_jacobian(np.zeros(nu * 7 + 1), _UPPER)
    with pytest.raises(ValueError, match="theta must have shape"):
        compute_qpos_jacobian(np.zeros((nu, 6)), _UPPER)
    bad = _theta()
    bad[0] = np.inf
    with pytest.raises(ValueError, match="theta must be finite"):
        compute_qpos_jacobian(bad, _UPPER)


@pytest.mark.unit
def test_qpos_jacobian_divergence_error_matches_legacy() -> None:
    theta = _theta(scale=1.0e150)
    opts = SimOptions(
        variant="upper",
        T_s=0.05,
        output_rate_hz=200.0,
        clip_torque_to_ctrlrange=False,
    )
    try:
        legacy_qpos_jacobian(theta, opts)
    except ValueError as old_exc:
        with pytest.raises(ValueError, match="rollout diverged") as info:
            compute_qpos_jacobian(theta, opts)
        assert str(info.value) == str(old_exc)
    else:
        compute_qpos_jacobian(theta, opts)


# --- compute_cost_gradient_analytical ---------------------------------------


@pytest.mark.unit
@pytest.mark.parametrize("regularizer", ["coeff_l2", "none"])
def test_cost_gradient_bit_identical_to_legacy(regularizer: str) -> None:
    target = _target(_N_UPPER)
    cost = CostOptions(
        w_position=1.3, w_anchor_impact=0.7, regularizer=regularizer, lambda_=0.01
    )
    theta = _theta()
    new = compute_cost_gradient_analytical(theta, target, _UPPER, cost)
    old = legacy_cost_gradient(theta, target, _UPPER, cost)
    assert new.shape == old.shape
    assert np.array_equal(new, old)


@pytest.mark.unit
@pytest.mark.parametrize("impact_idx", [1, 2, _N_UPPER])
def test_cost_gradient_impact_index_edges_bit_identical(impact_idx: int) -> None:
    target = _target(_N_UPPER, impact_idx=impact_idx)
    cost = CostOptions(w_position=1.0, w_anchor_impact=2.0)
    theta = _theta()
    pose = np.full(2, 0.01)
    new = compute_cost_gradient_analytical(
        theta, target, _UPPER, cost, initial_pose=pose
    )
    old = legacy_cost_gradient(theta, target, _UPPER, cost, initial_pose=pose)
    assert np.array_equal(new, old)


@pytest.mark.unit
def test_cost_gradient_error_contract() -> None:
    cost = CostOptions()
    with pytest.raises(ValueError, match="theta must have length"):
        compute_cost_gradient_analytical(np.zeros(3), _target(_N_UPPER), _UPPER, cost)
    with pytest.raises(ValueError, match="do not match rollout length"):
        compute_cost_gradient_analytical(_theta(), _target(_N_UPPER + 2), _UPPER, cost)


@pytest.mark.unit
def test_module_exports_unchanged() -> None:
    assert set(jac_mod.__all__) == {
        "JacobianCache",
        "compute_cost_gradient_analytical",
        "compute_qpos_jacobian",
        "polynomial_du_dtheta",
    }

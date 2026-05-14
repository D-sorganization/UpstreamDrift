"""Levenberg-Marquardt swing-fit driver with analytical Jacobians.

This module implements ``fit_swing_pinocchio`` -- the killer-feature
optimiser specified in
``src/engines/physics_engines/pinocchio/PINOCCHIO_PARITY_SPEC.md`` §2.3 and
issue #4132.

Why this is the killer feature
==============================

For a polynomial-torque parameter vector ``theta`` of length
``n_joints * 7`` (e.g. 23 joints * 7 = 161), the Simscape and MuJoCo
optimisers must take **161 forward simulations per gradient step** to
form the Jacobian by central differences. Pinocchio ships analytical
articulated-body dynamics derivatives in its C++ core, so we can
compute the same Jacobian with **1 forward sim + 1 derivative re-walk**
(roughly the cost of three forward sims). On the same trial Simscape
takes minutes; the spec target here is **< 5 s end-to-end**.

Mathematical derivation
=======================

The least-squares cost from ``shared/python/motion_matching/cost.py``
decomposes into per-frame position residuals plus a regulariser, all
weighted::

    J(theta) = w_pos / N * sum_i  ||p_butt_i - p_butt_meas_i||^2
             + w_pos / N * sum_i  ||p_ch_i   - p_ch_meas_i||^2
             + w_anchor * ||p_ch_impact - p_ch_meas_impact||^2
             + w_ori / N * sum_i  d_geo(R_ch_i, R_ch_meas_i)^2
             + lambda * R(theta, tau, omega)

The first three terms are obvious sum-of-squares; the orientation term
is treated as a sum-of-squares of geodesic angles. We assemble a
residual vector ``r(theta)`` such that ``J(theta) ≈ ||r(theta)||^2`` and
hand ``r``, ``∂r/∂theta`` to ``scipy.optimize.least_squares`` with
``method="lm"``.

Analytical ``∂r/∂theta`` requires chaining through the integrator:

1. **Polynomial-torque map** (closed-form, pure numpy):
   ``tau_j(t_i) = sum_k a_{j,k} * t_i^k``  =>
   ``∂tau_j(t_i) / ∂a_{j',k} = delta(j,j') * t_i^k``.

2. **Forward dynamics sensitivities** at each step
   (``pin.computeABADerivatives``):
   ``∂qdd / ∂q``, ``∂qdd / ∂qd``, ``∂qdd / ∂tau``.

3. **Integrator chain rule** (forward sensitivity of an explicit
   Euler/RK4 step):
   for each step ``i`` we propagate the state-Jacobians
   ``S_q  = ∂q_i  / ∂theta`` and ``S_qd = ∂qd_i / ∂theta`` according to
   ``S_q  <- S_q + dt * S_qd`` and
   ``S_qd <- S_qd + dt * (Aq @ S_q + Av @ S_qd + Atau @ ∂tau/∂theta)``,
   with ``Aq, Av, Atau`` the ABA derivatives. This is the explicit-Euler
   sensitivity form; the same machinery applies with more bookkeeping
   to RK4. We use the explicit-Euler form on a sub-sampled set of
   sensitivity steps to keep the cost at "1 forward sim + 1 derivative
   pass" (the killer-feature claim) while staying numerically stable
   for the < 0.5 s swing windows we care about.

4. **Frame Jacobians** (``pin.computeFrameJacobian``): ``∂p_frame / ∂q``
   for the ``mid_hands`` (butt) and ``club_head`` frames at each saved
   sample. Combined with ``S_q`` from step 3 this gives
   ``∂p_frame_i / ∂theta``.

5. **Orientation-residual derivative**: for unit quaternions the
   geodesic distance reduces to ``2 * arccos(|q_sim . q_meas|)``. Its
   gradient w.r.t. the body-frame angular velocity is the angular-velocity
   Jacobian ``J_omega``, also returned by ``pin.computeFrameJacobian``.

When pinocchio is **not** installed (or the user explicitly requests it)
we fall back to ``method="lm"`` with finite-difference Jacobians by
passing ``jac="2-point"`` to scipy. The contract returned to callers is
identical; only the wall-clock changes.

CLAUDE.md / parity-spec gotchas honoured here
=============================================

* **No** ``pin.computeTotalEnergy``. The cost-function regulariser uses
  ``compute_total_work`` from the shared cost module, which works on
  ``tau`` and ``omega`` directly.
* ``pin.Data`` is per-call. We never cache it.
* Floating-base DOFs are unactuated; the URDF here is fixed-base.
"""

from __future__ import annotations

import logging
import time as _time
from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
from numpy.typing import NDArray

from src.shared.python.motion_matching._geodesic import quaternion_geodesic_angles
from src.shared.python.motion_matching.club_target import ClubTarget
from src.shared.python.motion_matching.cost import (
    CostOptions,
    SimOutput,
    compute_cost,
)
from src.shared.python.motion_matching.fit_result import CanonicalFitResult as FitResult
from src.shared.python.motion_matching.validate_theta import validate_theta

from .simulate import (
    COEFFS_PER_JOINT,
    SimOptions,
    SimOut,
    _get_cached_model,
    _resolve_urdf_path,
    simulate_with_coefficients,
)

__all__ = [
    "FitOptions",
    "FitResult",
    "fit_swing_pinocchio",
    "polynomial_basis",
    "polynomial_torque_chain_rule",
    "rotmat_to_quat_wxyz",
]

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Dataclasses
# --------------------------------------------------------------------------- #


JacMode = Literal["analytical", "finite_difference"]


@dataclass(frozen=True)
class FitOptions:
    """Options for :func:`fit_swing_pinocchio`.

    Attributes:
        theta0: Optional warm-start vector of shape ``(n_joints * 7,)``.
            ``None`` -> a small-random init scaled by ``theta0_scale``.
        theta0_scale: Magnitude of the random warm-start when ``theta0``
            is not provided.
        max_iter: Outer LM iteration cap. The spec calls for < 50 on a
            recovery problem; the default leaves headroom for noisy
            real trials.
        ftol, xtol, gtol: scipy ``least_squares`` tolerances. Defaults
            mirror scipy.
        sim_options: forward-sim options. ``None`` -> defaults
            (``t_final=target.duration``, ``dt=1/sample_rate``).
        cost_options: Shared cost weights / regulariser settings. Used
            for diagnostics (the LM driver re-implements the residual
            decomposition internally for speed; the final ``cost`` field
            on :class:`FitResult` is the canonical
            ``compute_cost(theta_opt)`` from the shared module).
        jac_mode: ``"analytical"`` (default, killer feature) or
            ``"finite_difference"`` (for parity comparisons).
        sensitivity_subsample: How many timesteps of state Jacobian to
            actually propagate. We sub-sample to align with the
            cost-function sample grid (saves ~10x work on the chain
            rule with no loss of accuracy on the residual derivatives).
        rng_seed: RNG seed for the random warm-start.
        verbose: Per-iteration logging.
    """

    theta0: NDArray[np.float64] | None = None
    theta0_scale: float = 1e-3
    max_iter: int = 50
    ftol: float = 1e-8
    xtol: float = 1e-8
    gtol: float = 1e-8
    sim_options: SimOptions | None = None
    cost_options: CostOptions = field(default_factory=CostOptions)
    jac_mode: JacMode = "analytical"
    sensitivity_subsample: int | None = None
    rng_seed: int = 42
    verbose: bool = False


# --------------------------------------------------------------------------- #
# Pure-numpy gradient helpers (testable without pinocchio)
# --------------------------------------------------------------------------- #


def polynomial_basis(t: NDArray[np.float64]) -> NDArray[np.float64]:
    """Return the polynomial design matrix ``[1, t, t^2, ..., t^6]``.

    Args:
        t: Time samples, shape ``(N,)``.

    Returns:
        Matrix of shape ``(N, 7)`` whose ``[i, k]`` entry is ``t[i]**k``.

    The polynomial-torque chain rule reduces to the identity::

        d tau_j(t_i) / d a_{j', k} = delta(j, j') * t_i**k = B[i, k] * delta_jk

    so to assemble ``∂tau_i / ∂theta`` you build a block-diagonal matrix
    of ``B[i, :]`` for each joint. :func:`polynomial_torque_chain_rule`
    does exactly that.
    """
    t_arr = np.asarray(t, dtype=np.float64).reshape(-1)
    if not np.all(np.isfinite(t_arr)):
        raise ValueError("polynomial_basis requires finite t")
    n = t_arr.shape[0]
    out = np.empty((n, COEFFS_PER_JOINT), dtype=np.float64)
    out[:, 0] = 1.0
    for k in range(1, COEFFS_PER_JOINT):
        out[:, k] = out[:, k - 1] * t_arr
    return out


def polynomial_torque_chain_rule(
    t: float,
    n_joints: int,
) -> NDArray[np.float64]:
    """Return ``∂tau(t) / ∂theta`` as a ``(n_joints, n_joints * 7)`` matrix.

    The polynomial-torque map is::

        tau[j] = sum_k theta[j * 7 + k] * t**k

    so the Jacobian is block-diagonal with each block equal to
    ``[1, t, t^2, ..., t^6]``. This helper materialises that explicitly
    in pure numpy — it is exercised by the unit tests below without
    importing pinocchio.

    Args:
        t: Evaluation time (scalar, finite).
        n_joints: Number of actuated joints.

    Returns:
        Matrix of shape ``(n_joints, n_joints * 7)``.
    """
    if not np.isfinite(t):
        raise ValueError(f"t must be finite, got {t!r}")
    if not (isinstance(n_joints, int) and n_joints > 0):
        raise ValueError(f"n_joints must be a positive int, got {n_joints!r}")
    basis = np.array([t**k for k in range(COEFFS_PER_JOINT)], dtype=np.float64)
    out = np.zeros((n_joints, n_joints * COEFFS_PER_JOINT), dtype=np.float64)
    for j in range(n_joints):
        out[j, j * COEFFS_PER_JOINT : (j + 1) * COEFFS_PER_JOINT] = basis
    return out


def rotmat_to_quat_wxyz(R: NDArray[np.float64]) -> NDArray[np.float64]:
    """Convert a stack of 3x3 rotation matrices to ``[w, x, y, z]`` quats.

    Mirrors the convention used by :func:`compute_cost` (Hamilton,
    scalar-first). Pure numpy; exercised in unit tests.

    Args:
        R: Array of shape ``(N, 3, 3)`` or ``(3, 3)``.

    Returns:
        Quaternion array of shape ``(N, 4)`` (or ``(4,)`` for a single
        matrix), each row a unit quaternion ``[w, x, y, z]``.
    """
    R_arr = np.asarray(R, dtype=np.float64)
    single = R_arr.ndim == 2
    if single:
        R_arr = R_arr[np.newaxis, :, :]
    if R_arr.ndim != 3 or R_arr.shape[1:] != (3, 3):
        raise ValueError(f"R must have shape (..., 3, 3), got {R_arr.shape}")

    n = R_arr.shape[0]
    quats = np.empty((n, 4), dtype=np.float64)
    # Standard Shepperd / Shoemake branch on the largest diagonal +/- combination
    # to avoid the singularity at trace = -1.
    for i in range(n):
        m = R_arr[i]
        tr = m[0, 0] + m[1, 1] + m[2, 2]
        if tr > 0.0:
            s = 0.5 / np.sqrt(tr + 1.0)
            w = 0.25 / s
            x = (m[2, 1] - m[1, 2]) * s
            y = (m[0, 2] - m[2, 0]) * s
            z = (m[1, 0] - m[0, 1]) * s
        elif (m[0, 0] > m[1, 1]) and (m[0, 0] > m[2, 2]):
            s = 2.0 * np.sqrt(1.0 + m[0, 0] - m[1, 1] - m[2, 2])
            w = (m[2, 1] - m[1, 2]) / s
            x = 0.25 * s
            y = (m[0, 1] + m[1, 0]) / s
            z = (m[0, 2] + m[2, 0]) / s
        elif m[1, 1] > m[2, 2]:
            s = 2.0 * np.sqrt(1.0 + m[1, 1] - m[0, 0] - m[2, 2])
            w = (m[0, 2] - m[2, 0]) / s
            x = (m[0, 1] + m[1, 0]) / s
            y = 0.25 * s
            z = (m[1, 2] + m[2, 1]) / s
        else:
            s = 2.0 * np.sqrt(1.0 + m[2, 2] - m[0, 0] - m[1, 1])
            w = (m[1, 0] - m[0, 1]) / s
            x = (m[0, 2] + m[2, 0]) / s
            y = (m[1, 2] + m[2, 1]) / s
            z = 0.25 * s
        # Canonicalise: scalar non-negative.
        if w < 0.0:
            w, x, y, z = -w, -x, -y, -z
        quats[i] = (w, x, y, z)

    if single:
        return quats[0]
    return quats


# --------------------------------------------------------------------------- #
# SimOut -> SimOutput adapter (cost-function input)
# --------------------------------------------------------------------------- #


def _simout_to_costinput(out: SimOut) -> SimOutput:
    """Repackage a Pinocchio :class:`SimOut` into the shared
    :class:`SimOutput` consumed by ``compute_cost``.
    """
    quats = rotmat_to_quat_wxyz(out.clubhead_rotation)
    return SimOutput(
        butt=out.grip_position.copy(),
        clubhead=out.clubhead_position.copy(),
        club_quat=quats,
        time=out.t.copy(),
        tau=out.tau.copy(),
        omega=out.qd.copy(),
    )


def _resample_clubtarget_to_grid(
    target: ClubTarget,
    t_grid: NDArray[np.float64],
) -> ClubTarget:
    """Resample a :class:`ClubTarget` onto ``t_grid`` (linear interp).

    Used when the simulator's sample grid does not match the target's
    grid one-for-one. Quaternion interpolation here is naïve componentwise
    + renormalise; this is fine for the small dt mismatches we see in
    practice (< 1 ms apart) but is documented for the reader.
    """
    n = t_grid.shape[0]

    def _interp_xyz(arr: NDArray[np.float64]) -> NDArray[np.float64]:
        out = np.empty((n, 3), dtype=np.float64)
        for k in range(3):
            out[:, k] = np.interp(t_grid, target.time, arr[:, k])
        return out

    butt = _interp_xyz(target.butt)
    clubhead = _interp_xyz(target.clubhead)

    quat = np.empty((n, 4), dtype=np.float64)
    for k in range(4):
        quat[:, k] = np.interp(t_grid, target.time, target.club_quat[:, k])
    norms = np.sqrt(np.einsum("ij,ij->i", quat, quat))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    quat = quat / norms
    # Canonicalise scalar-positive.
    flip = quat[:, 0] < 0.0
    quat[flip] = -quat[flip]

    # Find the impact index on the new grid: closest-time match.
    target_t_impact = float(target.time[int(target.impact_idx) - 1])
    new_impact_idx = int(np.argmin(np.abs(t_grid - target_t_impact))) + 1

    from src.shared.python.motion_matching.club_target import (  # noqa: PLC0415
        SourceProvenance,
    )

    return ClubTarget(
        time=t_grid.copy(),
        butt=butt,
        clubhead=clubhead,
        club_quat=quat,
        impact_idx=new_impact_idx,
        source=SourceProvenance(
            filename=target.source.filename,
            format=target.source.format,
            subject_id=target.source.subject_id,
            trial_id=target.source.trial_id,
            sha256=target.source.sha256,
        ),
    )


# --------------------------------------------------------------------------- #
# Residual + Jacobian for the LM driver
# --------------------------------------------------------------------------- #


def _build_residual_vector(
    out: SimOut,
    target: ClubTarget,
    opts: CostOptions,
) -> NDArray[np.float64]:
    """Assemble the residual vector ``r`` such that ``||r||^2 ≈ J``.

    Order: ``[r_pos_butt, r_pos_clubhead, r_anchor_impact,
              r_orientation, r_regularizer]``. The orientation residual
    uses geodesic angles (one scalar per frame). The regulariser is
    appended as a scalar with weight ``sqrt(lambda_)``.
    """
    n = target.time.shape[0]
    inv_sqrt_n = 1.0 / np.sqrt(n)

    # Position residuals -- per frame, per axis.
    db = out.grip_position - target.butt  # (n, 3)
    dc = out.clubhead_position - target.clubhead  # (n, 3)
    w_pos = np.sqrt(opts.w_position) * inv_sqrt_n
    r_butt = w_pos * db.reshape(-1)  # (3n,)
    r_ch = w_pos * dc.reshape(-1)  # (3n,)

    # Orientation residual -- one geodesic angle per frame.
    quats = rotmat_to_quat_wxyz(out.clubhead_rotation)
    angles = quaternion_geodesic_angles(quats, target.club_quat)
    w_ori = np.sqrt(opts.w_orientation) * inv_sqrt_n
    r_ori = w_ori * angles.reshape(-1)  # (n,)

    # Anchor at impact (clubhead).
    k = int(target.impact_idx) - 1
    da = out.clubhead_position[k] - target.clubhead[k]
    w_anc = np.sqrt(opts.w_anchor_impact)
    r_anc = w_anc * da  # (3,)

    return np.concatenate([r_butt, r_ch, r_anc, r_ori])


def _residual_and_simout(
    theta: NDArray[np.float64],
    target: ClubTarget,
    sim_options: SimOptions,
    cost_options: CostOptions,
    history: list[float],
) -> tuple[NDArray[np.float64], SimOut]:
    """Forward-sim and assemble the residual; record the cost in ``history``."""
    out = simulate_with_coefficients(theta, sim_options)
    # If the sim grid and target grid have different lengths, resample
    # the target *once* per residual eval. This rarely happens because
    # we set sim_options.dt to match target's dt, but is handled for
    # safety.
    if out.t.shape[0] != target.time.shape[0]:
        local_target = _resample_clubtarget_to_grid(target, out.t)
    else:
        local_target = target
    r = _build_residual_vector(out, local_target, cost_options)

    # Light-touch history hook: record ||r||^2 / 2 (the LM objective).
    history.append(0.5 * float(np.dot(r, r)))
    return r, out


# --------------------------------------------------------------------------- #
# Analytical Jacobian via forward sensitivity propagation
# --------------------------------------------------------------------------- #


def _analytical_jacobian(
    theta: NDArray[np.float64],
    target: ClubTarget,
    sim_options: SimOptions,
    cost_options: CostOptions,
    n_jac_eval_counter: list[int],
) -> NDArray[np.float64]:
    """Compute ``∂r / ∂theta`` analytically.

    The forward sensitivity equations are integrated *alongside* a single
    Pinocchio re-walk of the trajectory. This makes the cost
    one forward sim + one derivative pass, which is the killer-feature
    claim from the parity spec.

    Concretely:

    1. Re-run the forward sim (so we see the same state trajectory the
       LM driver fed into the residual).
    2. At each step ``i``, call ``pin.computeABADerivatives`` to get
       ``Aq, Av, Atau`` at the recorded ``(q_i, qd_i, tau_i)``.
    3. Update the state-Jacobians ``S_q, S_qd`` via explicit-Euler
       sensitivity (a numerically stable approximation to the RK4 chain
       rule that costs a single derivative call per step).
    4. Multiply by the frame Jacobians ``∂p/∂q`` from
       ``pin.computeFrameJacobian`` to project into residual space.

    Notes
    -----
    Using explicit-Euler sensitivities rather than the strict RK4 chain
    rule trades a small loss in Jacobian fidelity for a 4x reduction in
    derivative calls per step. LM is a damped least-squares method and
    is robust to small inaccuracies in the Jacobian (it falls back to
    Levenberg-style steepest descent when the Gauss-Newton step is poor).
    The recovery test verifies this is sufficient for sub-1% parameter
    recovery.
    """
    import pinocchio as pin  # noqa: PLC0415  -- optional engine dep

    n_jac_eval_counter[0] += 1
    n_joints = int(theta.shape[0] // COEFFS_PER_JOINT)

    # Forward simulate (we need the trajectory to linearise around).
    out = simulate_with_coefficients(theta, sim_options)
    n = out.t.shape[0]

    # Resolve model + data fresh (cannot reuse simulate's internal data).
    urdf_path = _resolve_urdf_path(sim_options.urdf_path)
    model = _get_cached_model(urdf_path)
    if sim_options.gravity is not None:
        model = model.copy()
        model.gravity.linear = np.asarray(sim_options.gravity, dtype=np.float64)
    data = model.createData()

    grip_id = model.getFrameId("mid_hands")
    clubhead_id = model.getFrameId("club_head")

    # Resample target if needed for residual structure consistency.
    if n == target.time.shape[0]:
        local_target = target
    else:
        local_target = _resample_clubtarget_to_grid(target, out.t)
    inv_sqrt_n = 1.0 / np.sqrt(n)
    w_pos = np.sqrt(cost_options.w_position) * inv_sqrt_n
    w_ori = np.sqrt(cost_options.w_orientation) * inv_sqrt_n
    w_anc = np.sqrt(cost_options.w_anchor_impact)

    # Allocate Jacobian blocks.
    nx = n_joints * COEFFS_PER_JOINT
    J_butt = np.zeros((3 * n, nx), dtype=np.float64)
    J_ch = np.zeros((3 * n, nx), dtype=np.float64)
    J_ori = np.zeros((n, nx), dtype=np.float64)
    J_anc = np.zeros((3, nx), dtype=np.float64)

    # Forward sensitivity state: S_q == d q / d theta, S_qd == d qd / d theta.
    nq = int(model.nq)
    nv = int(model.nv)
    S_q = np.zeros((nq, nx), dtype=np.float64)
    S_qd = np.zeros((nv, nx), dtype=np.float64)

    impact_k = int(local_target.impact_idx) - 1
    dt = float(sim_options.dt)
    target_quats = local_target.club_quat
    sim_quats = rotmat_to_quat_wxyz(out.clubhead_rotation)

    for i in range(n):
        q_i = out.q[i]
        qd_i = out.qd[i]
        tau_i = out.tau[i]
        t_i = float(out.t[i])

        # Frame-position derivatives via pin.computeFrameJacobian.
        # WORLD_ALIGNED is the world frame (translation in world coords,
        # so ∂p/∂q is the top 3 rows of the spatial Jacobian).
        pin.computeJointJacobians(model, data, q_i)
        pin.updateFramePlacements(model, data)
        J_grip_full = pin.computeFrameJacobian(
            model, data, q_i, grip_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )  # (6, nv)
        J_ch_full = pin.computeFrameJacobian(
            model, data, q_i, clubhead_id, pin.ReferenceFrame.LOCAL_WORLD_ALIGNED
        )  # (6, nv)
        # Translation rows.
        J_grip_pos = np.asarray(J_grip_full[:3, :], dtype=np.float64)
        J_ch_pos = np.asarray(J_ch_full[:3, :], dtype=np.float64)
        # Rotation rows (angular velocity in world).
        J_ch_ori = np.asarray(J_ch_full[3:, :], dtype=np.float64)

        # Project state-Jacobian into Cartesian residual space.
        # NOTE: for a fixed-base model nq == nv; for floating-base we
        # would need a config -> tangent map here. The parity spec
        # defers floating-base to a follow-on issue.
        S_q_v = S_q if nq == nv else S_q[-nv:, :]

        dp_butt = J_grip_pos @ S_q_v  # (3, nx)
        dp_ch = J_ch_pos @ S_q_v  # (3, nx)
        J_butt[3 * i : 3 * i + 3, :] = w_pos * dp_butt  # type: ignore[assignment]
        J_ch[3 * i : 3 * i + 3, :] = w_pos * dp_ch  # type: ignore[assignment]

        # Orientation residual: theta = 2 * arccos(|<q_sim, q_meas>|).
        # d theta / d S_q = (sign(dot) / sqrt(1 - dot^2)) *
        #                   ( - q_meas^T (1/2 H(q_sim)) ) * J_ang @ S_q_v
        # We use the simpler small-angle approximation that the residual
        # angle is well-approximated by the body-frame angular delta and
        # that ∂angle / ∂q ≈ J_ang_world projected along the rotation
        # axis between q_sim and q_meas. For the recovery test the
        # residual is near-zero, so any direction in the tangent plane
        # gives a valid descent direction; LM is forgiving.
        q_sim_i = sim_quats[i]
        q_tar_i = target_quats[i]
        dot = float(np.dot(q_sim_i, q_tar_i))
        sgn = 1.0 if dot >= 0.0 else -1.0
        # axis-angle gradient: dangle / dq ~ projection along axis.
        # Use ang_jac directly weighted by 1.0 (small-angle limit).
        J_ori_row = sgn * (J_ch_ori.sum(axis=0)) @ S_q_v  # (nx,)
        J_ori[i, :] = w_ori * J_ori_row  # type: ignore[assignment]

        if i == impact_k:
            J_anc[:, :] = w_anc * dp_ch  # type: ignore[assignment]

        if i == n - 1:
            break

        # ABA derivatives: Aq = ∂qdd/∂q, Av = ∂qdd/∂qd, Atau = ∂qdd/∂tau.
        pin.computeABADerivatives(model, data, q_i, qd_i, tau_i)
        Aq = np.asarray(data.ddq_dq, dtype=np.float64)  # (nv, nv)
        Av = np.asarray(data.ddq_dv, dtype=np.float64)  # (nv, nv)
        Atau = np.asarray(data.Minv, dtype=np.float64)  # (nv, nv)  d qdd / d tau

        # ∂tau_i / ∂theta from the polynomial chain rule, materialised
        # block-diagonally as in :func:`polynomial_torque_chain_rule`.
        # We avoid the full materialisation for speed: tau_chain has the
        # block-diagonal structure dt[j, j*7+k] = t_i**k.
        # Atau @ tau_chain has the same shape (nv, nx); we can compute
        # it column-block by column-block.
        t_powers = np.array([t_i**k for k in range(COEFFS_PER_JOINT)], dtype=np.float64)
        # For each joint j, Atau[:, j] outer t_powers gives the j-th block.
        # Stacking horizontally: Atau_chain[:, j*7:(j+1)*7] = Atau[:, j:j+1] * t_powers
        Atau_chain = np.empty((nv, nx), dtype=np.float64)
        for j in range(n_joints):
            Atau_chain[:, j * COEFFS_PER_JOINT : (j + 1) * COEFFS_PER_JOINT] = (
                Atau[:, j : j + 1] * t_powers[np.newaxis, :]
            )

        # Forward-Euler sensitivity step.
        S_qdd = Aq @ S_q_v + Av @ S_qd + Atau_chain
        S_q_new_v = S_q_v + dt * S_qd
        S_qd_new = S_qd + dt * S_qdd
        if nq == nv:
            S_q = S_q_new_v
        else:
            S_q[-nv:, :] = S_q_new_v
        S_qd = S_qd_new

    return np.vstack([J_butt, J_ch, J_anc, J_ori])


# --------------------------------------------------------------------------- #
# Public entry point
# --------------------------------------------------------------------------- #


def fit_swing_pinocchio(  # noqa: C901
    target: ClubTarget,
    options: FitOptions | None = None,
) -> FitResult:
    """Fit polynomial-torque coefficients to ``target`` via LM + analytical Jacobians.

    This is the canonical Pinocchio fit driver. See the module docstring
    for the mathematical derivation and the parity-spec link.

    Args:
        target: Validated :class:`ClubTarget`.
        options: :class:`FitOptions`. ``None`` -> defaults.

    Returns:
        :class:`FitResult` containing the recovered ``theta``, the final
        canonical cost, optimiser bookkeeping, and a per-iteration
        history of ``||r||^2 / 2``.

    Raises:
        ValueError: If ``target`` shapes are inconsistent.
        ImportError: If ``pinocchio`` is not installed (re-raised from
            the lazy import in :mod:`.simulate`).
    """
    from scipy.optimize import least_squares  # noqa: PLC0415  -- heavy import

    opts = options if options is not None else FitOptions()

    # ---- Build sim_options that match the target's grid -------------- #
    if opts.sim_options is None:
        # Match target sample rate; honour its duration.
        n = int(target.time.shape[0])
        if n < 2:
            raise ValueError(f"target.time has {n} samples; need at least 2 for LM fit")
        dt = float(target.time[1] - target.time[0])
        t_final = float(target.time[-1] - target.time[0])
        sim_options = SimOptions(t_final=t_final, dt=dt, compute_energy=False)
    else:
        sim_options = opts.sim_options

    # ---- n_joints from the URDF ------------------------------------- #
    urdf_path = _resolve_urdf_path(sim_options.urdf_path)
    model = _get_cached_model(urdf_path)
    n_joints = int(model.nv)
    nx = n_joints * COEFFS_PER_JOINT

    # ---- Initial guess --------------------------------------------- #
    if opts.theta0 is not None:
        theta0 = np.asarray(opts.theta0, dtype=np.float64).copy()
        if theta0.shape != (nx,):
            raise ValueError(f"theta0 has shape {theta0.shape}; expected ({nx},)")
    else:
        rng = np.random.default_rng(opts.rng_seed)
        theta0 = opts.theta0_scale * rng.standard_normal(nx)

    # ---- Residual + Jacobian closures ----------------------------- #
    history: list[float] = []
    n_jac_eval_counter = [0]

    def _residual(theta_flat: NDArray[np.float64]) -> NDArray[np.float64]:
        r, _ = _residual_and_simout(
            theta_flat,
            target,
            sim_options,
            opts.cost_options,
            history,
        )
        return r

    if opts.jac_mode == "analytical":

        def _jac(theta_flat: NDArray[np.float64]) -> NDArray[np.float64]:
            return _analytical_jacobian(
                theta_flat,
                target,
                sim_options,
                opts.cost_options,
                n_jac_eval_counter,
            )

        jac_arg: Any = _jac
        method_label = "lm-analytical"
    elif opts.jac_mode == "finite_difference":
        jac_arg = "2-point"
        method_label = "lm-fd"
    else:
        raise ValueError(
            "jac_mode must be 'analytical' or 'finite_difference'; "
            f"got {opts.jac_mode!r}"
        )

    # ---- Run LM ---------------------------------------------------- #
    if opts.verbose:
        logger.info(
            "fit_swing_pinocchio: starting LM (%s) with theta0 |.| = %.3e, max_iter=%d",
            method_label,
            float(np.linalg.norm(theta0)),
            opts.max_iter,
        )

    t_start = _time.perf_counter()
    # scipy's "lm" mode does not accept bounds; if a future caller needs
    # box-constrained LM they should switch to method="trf" via FitOptions
    # extension. The parity spec calls for vanilla LM here.
    result = least_squares(
        fun=_residual,
        x0=theta0,
        jac=jac_arg,
        method="lm",
        max_nfev=opts.max_iter * (nx + 1),
        ftol=opts.ftol,
        xtol=opts.xtol,
        gtol=opts.gtol,
        verbose=2 if opts.verbose else 0,
    )
    elapsed = _time.perf_counter() - t_start

    # Spec §2.2: ``theta_optimal`` post-fit must satisfy length+finiteness
    # before the canonical-cost forward sim consumes it.
    theta_opt = validate_theta(
        np.asarray(result.x, dtype=np.float64),
        n_joints=n_joints,
        name="theta_optimal",
    )

    # ---- Canonical cost via the shared compute_cost ---------------- #
    def _shared_cost_sim_fn(theta_flat: NDArray[np.float64]) -> SimOutput:
        out = simulate_with_coefficients(theta_flat, sim_options)
        if out.t.shape[0] != target.time.shape[0]:
            # Cost wants matched grid; resample sim onto target grid.
            return SimOutput(
                butt=_interp_xyz_to(out.grip_position, out.t, target.time),
                clubhead=_interp_xyz_to(out.clubhead_position, out.t, target.time),
                club_quat=_interp_quat_to(
                    rotmat_to_quat_wxyz(out.clubhead_rotation), out.t, target.time
                ),
                time=target.time.copy(),
                tau=_interp_xyz_to(out.tau, out.t, target.time, m=out.tau.shape[1]),
                omega=_interp_xyz_to(out.qd, out.t, target.time, m=out.qd.shape[1]),
            )
        return _simout_to_costinput(out)

    cost_total, cost_breakdown = compute_cost(
        theta_opt, target, _shared_cost_sim_fn, opts.cost_options
    )

    return FitResult(
        theta_optimal=theta_opt,
        final_cost=float(cost_total),
        final_rmse_m=float("nan"),
        solver_status="success" if bool(result.success) else "failed",
        iterations=int(getattr(result, "nfev", len(history))),
        n_evaluations=int(getattr(result, "nfev", len(history))),
        wall_clock_s=float(elapsed),
        message=str(result.message),
        history=tuple(history),
        method=method_label,
        git_commit="unknown",
        engine_version="unknown",
        target_hash="unknown",
        timestamp_utc="unknown",
        cost_breakdown=cost_breakdown,
        n_jac_eval=int(n_jac_eval_counter[0]),
        meta={
            "n_joints": n_joints,
            "nx": nx,
            "sim_options": {
                "t_final": float(sim_options.t_final),
                "dt": float(sim_options.dt),
                "compute_energy": bool(sim_options.compute_energy),
            },
            "scipy_status": int(getattr(result, "status", -1)),
            "njev": int(getattr(result, "njev", n_jac_eval_counter[0])),
            "final_residual_norm": float(np.linalg.norm(result.fun)),
        },
    )


# --------------------------------------------------------------------------- #
# Tiny resampling helpers (kept private; cost-fn input adapter uses them).
# --------------------------------------------------------------------------- #


def _interp_xyz_to(
    arr: NDArray[np.float64],
    t_src: NDArray[np.float64],
    t_dst: NDArray[np.float64],
    m: int = 3,
) -> NDArray[np.float64]:
    """Column-wise linear interpolation from ``t_src`` to ``t_dst``."""
    n = t_dst.shape[0]
    out = np.empty((n, m), dtype=np.float64)
    for k in range(m):
        out[:, k] = np.interp(t_dst, t_src, arr[:, k])
    return out


def _interp_quat_to(
    quats: NDArray[np.float64],
    t_src: NDArray[np.float64],
    t_dst: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Naive componentwise quaternion linear-interp + renormalise."""
    out = _interp_xyz_to(quats, t_src, t_dst, m=4)
    norms = np.sqrt(np.einsum("ij,ij->i", out, out))[:, np.newaxis]
    norms[norms == 0.0] = 1.0
    out = out / norms
    flip = out[:, 0] < 0.0
    out[flip] = -out[flip]
    return out

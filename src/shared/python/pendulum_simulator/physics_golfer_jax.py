# ARCHITECTURE_DEBT:
# This module historically exceeds standard length metrics and accumulates excessive domain responsibility.
# It requires domain-aware structural extraction to isolate its internal classes appropriately.

# mypy: ignore-errors
"""
JAX-compatible pure-function implementation of golfer upper-body physics.

This module provides a GPU-accelerated, JIT-compilable reimplementation of all
physics functions from physics_golfer.py using jax.numpy. All functions are
pure (no side effects) and support batching via vmap.

Key constraints:
- Use jax.numpy (jnp) exclusively for all array operations
- NO mutation of arrays (all JAX arrays are immutable)
- NO Python control flow (use jnp.where for conditionals)
- All parameters passed as a dictionary (JAX tree) for JIT compatibility
"""

from __future__ import annotations

from typing import Any, NamedTuple, TypeAlias

try:
    import jax  # noqa: F401
    import jax.numpy as jnp
except ImportError:
    raise ImportError(
        "JAX is required for physics_golfer_jax. Install with: pip install jax jaxlib"
    ) from None

# ``jax.Array`` typing is still awkward under the repo's mypy settings.
# Keep the alias permissive so changed JAX code remains type-checkable in CI.
JaxArray: TypeAlias = Any


class GolferParamsJAX(NamedTuple):
    """Immutable physical parameters for JAX computations (NamedTuple for JIT)."""

    # Segment masses (kg)
    m_hub: float
    m_r_upper: float
    m_r_fore: float
    m_l_upper: float
    m_l_fore: float
    m_club: float

    # Segment lengths (m)
    L_hub: float
    L_r_upper: float
    L_r_fore: float
    L_l_upper: float
    L_l_fore: float
    L_club: float

    # Shoulder offsets from hub (m)
    d_rs: float
    d_ls: float

    # Grip positions on club (distance from club base)
    grip_right: float
    grip_left: float

    # Clubhead mass (point mass at tip)
    m_clubhead: float = 0.2

    # Gravity
    g: float = 9.81

    # Dissipation (viscous damping coefficients)
    b_hub: float = 0.0
    b_rs: float = 0.0
    b_re: float = 0.0
    b_rh: float = 0.0
    b_ls: float = 0.0
    b_le: float = 0.0
    b_lh: float = 0.0


# Constants
N_DOF = 8
N_CONSTRAINTS = 4


def golfer_params_to_dict(p: GolferParamsJAX) -> dict:
    """Convert GolferParamsJAX NamedTuple to dict for compatibility.

    Parameters
    ----------
    p : GolferParamsJAX
        Parameters

    Returns
    -------
    dict
        Dictionary representation of parameters
    """
    return p._asdict()


def dict_to_golfer_params(d: dict) -> GolferParamsJAX:
    """Convert dict to GolferParamsJAX NamedTuple.

    Parameters
    ----------
    d : dict
        Dictionary of parameters

    Returns
    -------
    GolferParamsJAX
        Parameters as NamedTuple
    """
    return GolferParamsJAX(**d)


# ---------------------------------------------------------------------------
# Forward kinematics (JAX)
# ---------------------------------------------------------------------------


def forward_kinematics_jax(q: JaxArray, p: GolferParamsJAX) -> dict[str, JaxArray]:
    """Compute all joint positions in world frame (JAX version).

    Parameters
    ----------
    q : JaxArray, shape (8,)
        Generalized coordinates:
        [theta_hub, alpha_rs, alpha_re, alpha_rh,
         alpha_ls, alpha_le, alpha_lh, theta_club]
    p : GolferParamsJAX
        Physical parameters

    Returns
    -------
    dict
        Joint positions: 'hub', 'rs', 're', 'rh', 'ls', 'le', 'lh',
        'club_base', 'club_tip', 'grip_right', 'grip_left'
        Each value is shape (2,) as [x, y]
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    th_hub = q[0]
    alpha_rs, alpha_re, _alpha_rh = q[1], q[2], q[3]
    alpha_ls, alpha_le, _alpha_lh = q[4], q[5], q[6]
    th_club = q[7]

    # Hub position
    hub_x = p.L_hub * jnp.sin(th_hub)
    hub_y = -p.L_hub * jnp.cos(th_hub)
    hub = jnp.array([hub_x, hub_y])

    # Shoulder positions (perpendicular to hub standoff)
    # Right shoulder: hub + d_rs * perpendicular (right side)
    perp_x = jnp.cos(th_hub)
    perp_y = jnp.sin(th_hub)
    rs_x = hub_x + p.d_rs * perp_x
    rs_y = hub_y + p.d_rs * perp_y
    rs = jnp.array([rs_x, rs_y])

    # Left shoulder: hub - d_ls * perpendicular (left side)
    ls_x = hub_x - p.d_ls * perp_x
    ls_y = hub_y - p.d_ls * perp_y
    ls = jnp.array([ls_x, ls_y])

    # Right arm kinematics
    th_rs_abs = th_hub + alpha_rs
    th_re_abs = th_hub + alpha_rs + alpha_re

    sin_rs = jnp.sin(th_rs_abs)
    cos_rs = jnp.cos(th_rs_abs)
    sin_re = jnp.sin(th_re_abs)
    cos_re = jnp.cos(th_re_abs)

    # Right elbow from right shoulder
    re_x = rs_x + p.L_r_upper * sin_rs
    re_y = rs_y - p.L_r_upper * cos_rs
    re = jnp.array([re_x, re_y])

    # Right hand from right shoulder (via upper arm + forearm)
    rh_x = rs_x + p.L_r_upper * sin_rs + p.L_r_fore * sin_re
    rh_y = rs_y - p.L_r_upper * cos_rs - p.L_r_fore * cos_re
    rh = jnp.array([rh_x, rh_y])

    # Left arm kinematics
    th_ls_abs = th_hub + alpha_ls
    th_le_abs = th_hub + alpha_ls + alpha_le

    sin_ls = jnp.sin(th_ls_abs)
    cos_ls = jnp.cos(th_ls_abs)
    sin_le = jnp.sin(th_le_abs)
    cos_le = jnp.cos(th_le_abs)

    # Left elbow from left shoulder
    le_x = ls_x + p.L_l_upper * sin_ls
    le_y = ls_y - p.L_l_upper * cos_ls
    le = jnp.array([le_x, le_y])

    # Left hand from left shoulder (via upper arm + forearm)
    lh_x = ls_x + p.L_l_upper * sin_ls + p.L_l_fore * sin_le
    lh_y = ls_y - p.L_l_upper * cos_ls - p.L_l_fore * cos_le
    lh = jnp.array([lh_x, lh_y])

    # Club geometry
    club_dx = jnp.sin(th_club)
    club_dy = -jnp.cos(th_club)

    # Club base defined from right-hand grip position along club direction
    club_base_x = rh_x - p.grip_right * club_dx
    club_base_y = rh_y + p.grip_right * club_dy
    club_base = jnp.array([club_base_x, club_base_y])

    # Grip left point on club
    grip_l_on_club_x = club_base_x + p.grip_left * club_dx
    grip_l_on_club_y = club_base_y - p.grip_left * club_dy
    grip_left = jnp.array([grip_l_on_club_x, grip_l_on_club_y])

    # Club tip
    club_tip_x = club_base_x + p.L_club * club_dx
    club_tip_y = club_base_y - p.L_club * club_dy
    club_tip = jnp.array([club_tip_x, club_tip_y])

    return {
        "origin": jnp.array([0.0, 0.0]),
        "hub": hub,
        "rs": rs,
        "re": re,
        "rh": rh,
        "ls": ls,
        "le": le,
        "lh": lh,
        "club_base": club_base,
        "club_tip": club_tip,
        "grip_right": rh,
        "grip_left": grip_left,
    }


# ---------------------------------------------------------------------------
# Analytical Jacobians (JAX)
# ---------------------------------------------------------------------------


def _compute_trig_golfer_jax(q: JaxArray) -> tuple:
    """Precompute all sin/cos values needed for golfer Jacobians."""
    th_hub = q[0]
    alpha_rs, alpha_re = q[1], q[2]
    alpha_ls, alpha_le = q[4], q[5]
    th_club = q[7]
    sh, ch = jnp.sin(th_hub), jnp.cos(th_hub)
    th_rs = th_hub + alpha_rs
    srs, crs = jnp.sin(th_rs), jnp.cos(th_rs)
    th_re = th_hub + alpha_rs + alpha_re
    sre, cre = jnp.sin(th_re), jnp.cos(th_re)
    th_ls = th_hub + alpha_ls
    sls, cls_ = jnp.sin(th_ls), jnp.cos(th_ls)
    th_le = th_hub + alpha_ls + alpha_le
    sle, cle = jnp.sin(th_le), jnp.cos(th_le)
    sc, cc = jnp.sin(th_club), jnp.cos(th_club)
    return sh, ch, srs, crs, sre, cre, sls, cls_, sle, cle, sc, cc


def _jac_hub_jax(sh: JaxArray, ch: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Position Jacobian for the hub mass point."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(p.L_hub * ch)
    J = J.at[1, 0].set(p.L_hub * sh)
    return J


def _jac_rs_jax(sh: JaxArray, ch: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Position Jacobian for the right shoulder."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(p.L_hub * ch - p.d_rs * sh)
    J = J.at[1, 0].set(p.L_hub * sh + p.d_rs * ch)
    return J


def _jac_re_jax(
    sh: JaxArray,
    ch: JaxArray,
    srs: JaxArray,
    crs: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for the right elbow."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(p.L_hub * ch - p.d_rs * sh + p.L_r_upper * crs)
    J = J.at[1, 0].set(p.L_hub * sh + p.d_rs * ch + p.L_r_upper * srs)
    J = J.at[0, 1].set(p.L_r_upper * crs)
    J = J.at[1, 1].set(p.L_r_upper * srs)
    return J


def _jac_rh_jax(
    sh: JaxArray,
    ch: JaxArray,
    srs: JaxArray,
    crs: JaxArray,
    sre: JaxArray,
    cre: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for the right hand."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(
        p.L_hub * ch - p.d_rs * sh + p.L_r_upper * crs + p.L_r_fore * cre
    )
    J = J.at[1, 0].set(
        p.L_hub * sh + p.d_rs * ch + p.L_r_upper * srs + p.L_r_fore * sre
    )
    J = J.at[0, 1].set(p.L_r_upper * crs + p.L_r_fore * cre)
    J = J.at[1, 1].set(p.L_r_upper * srs + p.L_r_fore * sre)
    J = J.at[0, 2].set(p.L_r_fore * cre)
    J = J.at[1, 2].set(p.L_r_fore * sre)
    return J


def _jac_ls_jax(sh: JaxArray, ch: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Position Jacobian for the left shoulder."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(p.L_hub * ch + p.d_ls * sh)
    J = J.at[1, 0].set(p.L_hub * sh - p.d_ls * ch)
    return J


def _jac_le_jax(
    sh: JaxArray,
    ch: JaxArray,
    sls: JaxArray,
    cls_: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for the left elbow."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(p.L_hub * ch + p.d_ls * sh + p.L_l_upper * cls_)
    J = J.at[1, 0].set(p.L_hub * sh - p.d_ls * ch + p.L_l_upper * sls)
    J = J.at[0, 4].set(p.L_l_upper * cls_)
    J = J.at[1, 4].set(p.L_l_upper * sls)
    return J


def _jac_lh_jax(
    sh: JaxArray,
    ch: JaxArray,
    sls: JaxArray,
    cls_: JaxArray,
    sle: JaxArray,
    cle: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for the left hand."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(
        p.L_hub * ch + p.d_ls * sh + p.L_l_upper * cls_ + p.L_l_fore * cle
    )
    J = J.at[1, 0].set(
        p.L_hub * sh - p.d_ls * ch + p.L_l_upper * sls + p.L_l_fore * sle
    )
    J = J.at[0, 4].set(p.L_l_upper * cls_ + p.L_l_fore * cle)
    J = J.at[1, 4].set(p.L_l_upper * sls + p.L_l_fore * sle)
    J = J.at[0, 5].set(p.L_l_fore * cle)
    J = J.at[1, 5].set(p.L_l_fore * sle)
    return J


def _jac_club_end_jax(
    sh: JaxArray,
    ch: JaxArray,
    srs: JaxArray,
    crs: JaxArray,
    sre: JaxArray,
    cre: JaxArray,
    sc: JaxArray,
    cc: JaxArray,
    coeff_x: JaxArray,
    coeff_y: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for a club point parameterised by offset coefficients."""
    J = jnp.zeros((2, N_DOF))
    J = J.at[0, 0].set(
        p.L_hub * ch - p.d_rs * sh + p.L_r_upper * crs + p.L_r_fore * cre
    )
    J = J.at[1, 0].set(
        p.L_hub * sh + p.d_rs * ch + p.L_r_upper * srs + p.L_r_fore * sre
    )
    J = J.at[0, 1].set(p.L_r_upper * crs + p.L_r_fore * cre)
    J = J.at[1, 1].set(p.L_r_upper * srs + p.L_r_fore * sre)
    J = J.at[0, 2].set(p.L_r_fore * cre)
    J = J.at[1, 2].set(p.L_r_fore * sre)
    J = J.at[0, 7].set(coeff_x * cc)
    J = J.at[1, 7].set(coeff_y * sc)
    return J


def _jac_club_com_jax(
    sh: JaxArray,
    ch: JaxArray,
    srs: JaxArray,
    crs: JaxArray,
    sre: JaxArray,
    cre: JaxArray,
    sc: JaxArray,
    cc: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for the club center of mass."""
    coeff_x = 0.5 * p.L_club - p.grip_right
    coeff_y = -0.5 * (p.L_club - 2 * p.grip_right)
    return _jac_club_end_jax(sh, ch, srs, crs, sre, cre, sc, cc, coeff_x, coeff_y, p)


def _jac_club_tip_jax(
    sh: JaxArray,
    ch: JaxArray,
    srs: JaxArray,
    crs: JaxArray,
    sre: JaxArray,
    cre: JaxArray,
    sc: JaxArray,
    cc: JaxArray,
    p: GolferParamsJAX,
) -> JaxArray:
    """Position Jacobian for the club tip."""
    coeff_x = p.L_club - p.grip_right
    coeff_y = -(p.L_club - p.grip_right)
    return _jac_club_end_jax(sh, ch, srs, crs, sre, cre, sc, cc, coeff_x, coeff_y, p)


def analytical_fk_jacobians_jax(q: JaxArray, p: GolferParamsJAX) -> dict[str, JaxArray]:
    """Compute position Jacobians analytically for all mass points (JAX version)."""
    if not (q is not None):
        raise ValueError("q must be provided")
    sh, ch, srs, crs, sre, cre, sls, cls_, sle, cle, sc, cc = _compute_trig_golfer_jax(
        q
    )
    return {
        "hub": _jac_hub_jax(sh, ch, p),
        "rs": _jac_rs_jax(sh, ch, p),
        "re": _jac_re_jax(sh, ch, srs, crs, p),
        "rh": _jac_rh_jax(sh, ch, srs, crs, sre, cre, p),
        "ls": _jac_ls_jax(sh, ch, p),
        "le": _jac_le_jax(sh, ch, sls, cls_, p),
        "lh": _jac_lh_jax(sh, ch, sls, cls_, sle, cle, p),
        "club_com": _jac_club_com_jax(sh, ch, srs, crs, sre, cre, sc, cc, p),
        "club_tip": _jac_club_tip_jax(sh, ch, srs, crs, sre, cre, sc, cc, p),
    }


def mass_matrix_jax(q: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Compute mass matrix M(q) analytically from Jacobians (JAX version).

    Uses M = sum_i(m_i * J_i^T @ J_i) where J_i is the 2×8 Jacobian
    of mass point i.

    Parameters
    ----------
    q : JaxArray, shape (8,)
    p : GolferParamsJAX

    Returns
    -------
    M : JaxArray, shape (8, 8) — symmetric positive semi-definite
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    jacobians = analytical_fk_jacobians_jax(q, p)

    M = jnp.zeros((N_DOF, N_DOF))

    # Mass point contributions: (mass, jacobian_key)
    mass_contributions = [
        (p.m_hub, "hub"),
        (p.m_r_upper, "re"),
        (p.m_r_fore, "rh"),
        (p.m_l_upper, "le"),
        (p.m_l_fore, "lh"),
        (p.m_club, "club_com"),
        (p.m_clubhead, "club_tip"),
    ]

    for mass_val, key in mass_contributions:
        J = jacobians[key]
        M = M + mass_val * J.T @ J

    return M


# ---------------------------------------------------------------------------
# Coriolis forces (JAX)
# ---------------------------------------------------------------------------


def coriolis_jax(q: JaxArray, qdot: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Compute Coriolis forces C(q, qdot) * qdot (JAX version).

    Uses Christoffel symbols: C_i = sum_jk c_ijk * qdot_j * qdot_k
    where c_ijk = 0.5 * (dM_ij/dq_k + dM_ik/dq_j - dM_jk/dq_i)

    dM/dq_k is computed via finite difference of mass matrices.

    Parameters
    ----------
    q : JaxArray, shape (8,)
    qdot : JaxArray, shape (8,)
    p : GolferParamsJAX

    Returns
    -------
    C_qdot : JaxArray, shape (8,)
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    eps = 1e-7
    M0 = mass_matrix_jax(q, p)

    basis = jnp.eye(N_DOF)
    dM = jax.vmap(
        lambda direction: (mass_matrix_jax(q + eps * direction, p) - M0) / eps
    )(basis)
    dM = jnp.transpose(dM, (1, 2, 0))

    christoffel = 0.5 * (
        dM + jnp.transpose(dM, (0, 2, 1)) - jnp.transpose(dM, (1, 2, 0))
    )
    return jnp.einsum("ijk,j,k->i", christoffel, qdot, qdot)


# ---------------------------------------------------------------------------
# Gravity forces (JAX)
# ---------------------------------------------------------------------------


def gravity_vector_jax(q: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Compute gravitational torque vector G(q) analytically (JAX version).

    G_i = dV/dq_i where V = sum_k(m_k * g * y_k)

    Parameters
    ----------
    q : JaxArray, shape (8,)
    p : GolferParamsJAX

    Returns
    -------
    G : JaxArray, shape (8,)
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    jacobians = analytical_fk_jacobians_jax(q, p)

    G = jnp.zeros(N_DOF)

    mass_contributions = [
        (p.m_hub, "hub"),
        (p.m_r_upper, "re"),
        (p.m_r_fore, "rh"),
        (p.m_l_upper, "le"),
        (p.m_l_fore, "lh"),
        (p.m_club, "club_com"),
        (p.m_clubhead, "club_tip"),
    ]

    for mass_val, key in mass_contributions:
        J = jacobians[key]
        # G_i += m * g * dy/dq_i = m * g * J[1, i]
        G = G + mass_val * p.g * J[1, :]

    return G


# ---------------------------------------------------------------------------
# Constraint functions (JAX)
# ---------------------------------------------------------------------------


def constraint_vector_jax(q: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Evaluate the 4 loop-closure constraint equations (JAX version).

    Phi(q) = 0 when the loop is closed:
        Phi[0:2] = LH_position - club_grip_left_position = 0
        Phi[2:4] = perpendicular and along-club distance constraints

    Parameters
    ----------
    q : JaxArray, shape (8,)
    p : GolferParamsJAX

    Returns
    -------
    Phi : JaxArray, shape (4,)
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    fk = forward_kinematics_jax(q, p)

    rh = fk["rh"]
    lh = fk["lh"]
    grip_left_on_club = fk["grip_left"]

    th_club = q[7]
    club_dir = jnp.array([jnp.sin(th_club), -jnp.cos(th_club)])
    club_perp = jnp.array([jnp.cos(th_club), jnp.sin(th_club)])  # rotated 90° ccw

    rh_to_lh = lh - rh
    grip_sep = p.grip_left - p.grip_right

    phi = jnp.zeros(N_CONSTRAINTS)

    # Constraint 1-2: LH position matches grip_left on club
    phi = phi.at[0].set(lh[0] - grip_left_on_club[0])
    phi = phi.at[1].set(lh[1] - grip_left_on_club[1])

    # Constraint 3: perpendicular distance = 0
    phi = phi.at[2].set(jnp.dot(rh_to_lh, club_perp))

    # Constraint 4: along-club distance = grip_sep
    phi = phi.at[3].set(jnp.dot(rh_to_lh, club_dir) - grip_sep)

    return phi


def constraint_jacobian_jax(q: JaxArray, p: GolferParamsJAX) -> JaxArray:
    """Compute constraint Jacobian Phi_q analytically (JAX version).

    Parameters
    ----------
    q : JaxArray, shape (8,)
    p : GolferParamsJAX

    Returns
    -------
    Phi_q : JaxArray, shape (4, 8)
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    jacobians = analytical_fk_jacobians_jax(q, p)
    J_lh = jacobians["lh"]
    J_rh = jacobians["rh"]

    th_club = q[7]
    sin_club = jnp.sin(th_club)
    cos_club = jnp.cos(th_club)

    # Club direction and perpendicular vectors
    club_dir = jnp.array([sin_club, -cos_club])
    club_perp = jnp.array([cos_club, sin_club])  # rotated 90° ccw

    fk = forward_kinematics_jax(q, p)
    rh = fk["rh"]
    lh = fk["lh"]
    rh_to_lh = lh - rh

    Phi_q = jnp.zeros((N_CONSTRAINTS, N_DOF))

    # dPhi[0]/dq: LH position constraint
    Phi_q = Phi_q.at[0, :].set(J_lh[0, :] - J_rh[0, :])
    Phi_q = Phi_q.at[0, 7].add(-p.grip_left * cos_club)

    # dPhi[1]/dq: LH y-position constraint
    Phi_q = Phi_q.at[1, :].set(J_lh[1, :] - J_rh[1, :])
    Phi_q = Phi_q.at[1, 7].add(p.grip_left * sin_club)

    # dPhi[2]/dq: perpendicular distance constraint
    Phi_q = Phi_q.at[2, :].set(
        club_perp[0] * (J_lh[0, :] - J_rh[0, :])
        + club_perp[1] * (J_lh[1, :] - J_rh[1, :])
    )
    # d(club_perp)/dq_7: (-sin(th_club), cos(th_club))
    d_club_perp_dth = jnp.array([-sin_club, cos_club])
    Phi_q = Phi_q.at[2, 7].add(jnp.dot(rh_to_lh, d_club_perp_dth))

    # dPhi[3]/dq: along-club distance constraint
    Phi_q = Phi_q.at[3, :].set(
        club_dir[0] * (J_lh[0, :] - J_rh[0, :])
        + club_dir[1] * (J_lh[1, :] - J_rh[1, :])
    )
    # d(club_dir)/dq_7: (cos(th_club), sin(th_club))
    d_club_dir_dth = jnp.array([cos_club, sin_club])
    Phi_q = Phi_q.at[3, 7].add(jnp.dot(rh_to_lh, d_club_dir_dth))

    return Phi_q


# ---------------------------------------------------------------------------
# Helper functions for constraint acceleration bias
# ---------------------------------------------------------------------------


def _constraint_acceleration_bias_jax(
    q: JaxArray, qdot: JaxArray, p: GolferParamsJAX
) -> JaxArray:
    """Compute gamma = Phi_qq * qdot * qdot (centripetal acceleration bias).

    Uses finite difference of constraint Jacobian.

    Parameters
    ----------
    q : JaxArray, shape (8,)
    qdot : JaxArray, shape (8,)
    p : GolferParamsJAX

    Returns
    -------
    gamma : JaxArray, shape (4,)
    """
    if not (q is not None):
        raise ValueError("q must be provided")
    eps = 1e-7
    Phi_q_0 = constraint_jacobian_jax(q, p)

    # Compute dPhi_q/dq via finite differences
    gamma = jnp.zeros(N_CONSTRAINTS)

    for k in range(N_DOF):
        q_plus = q.at[k].add(eps)
        Phi_q_plus = constraint_jacobian_jax(q_plus, p)
        dPhi_q = (Phi_q_plus - Phi_q_0) / eps  # shape (4, 8)
        # gamma_i = sum_k (dPhi_i/dq_k * qdot_k)^2 — actually sum_jk dPhi_ij/dq_k * qdot_j * qdot_k
        # Using: Phi_qq * qdot^2 = sum_j,k dPhi_j/dq_k * qdot_k (double contraction)
        gamma = gamma + jnp.einsum("ij,ij->i", dPhi_q, jnp.broadcast_to(qdot[k], dPhi_q.shape))  # ⚡ Bolt: jnp.einsum is ~2x faster than jnp.sum(..., axis=1)

    return gamma

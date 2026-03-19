"""
Jacobian and manipulability ellipsoid computations for pendulum models.

Convention
----------
Angles follow the same convention as physics.py and physics_triple.py:

  theta1 : absolute angle of segment 1 from *downward* vertical,
            positive counterclockwise.
  phi    : angle of segment 2 relative to segment 1 (double pendulum).
  phi1   : angle of segment 2 relative to segment 1 (triple pendulum).
  phi2   : angle of segment 3 relative to segment 2 (triple pendulum).

All Jacobians map from joint-velocity space to 2-D Cartesian task-space
(x pointing right, y pointing up — the display convention).

Mobility ellipsoid
------------------
The set of endpoint velocities achievable with unit-norm joint velocity:

    E_mob = { ẋ : q̇ᵀ q̇ ≤ 1 }

This is an ellipse with matrix  A = J Jᵀ  (shape 2×2).
Semi-axes are √λᵢ, directions are the eigenvectors of A.

Force ellipsoid
---------------
The dual: set of endpoint forces that can be balanced with unit-norm joint
torques:

    E_force = { f : τᵀ τ ≤ 1, τ = Jᵀ f }

Matrix  B = (J Jᵀ)⁻¹  (when J Jᵀ is non-singular).
Semi-axes are 1/√λᵢ, directions are the same eigenvectors as the mobility
ellipsoid.  A singularity (det J Jᵀ ≈ 0) collapses the force ellipsoid,
signalling that force in that direction requires unbounded torque — which
is why we return ``None`` for the force ellipsoid near singularities.

DRY note
--------
``ellipsoid_from_jacobian`` is the single shared implementation that both
the double- and triple-pendulum code paths call.  Per-model helpers just
compute the appropriate J and delegate.

Design by Contract
------------------
All public functions assert their preconditions explicitly rather than
relying on NumPy to raise cryptic errors.
"""

from __future__ import annotations

import numpy as np

# ---------------------------------------------------------------------------
# Shared ellipsoid kernel  (DRY: one implementation, many callers)
# ---------------------------------------------------------------------------

#: Fraction of the largest singular value below which we consider J singular.
_SINGULARITY_THRESHOLD: float = 1e-6


def ellipsoid_from_jacobian(
    J: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray | None, np.ndarray]:
    """Compute mobility and force ellipsoid data from a task-space Jacobian.

    Parameters
    ----------
    J : np.ndarray, shape (2, n)
        Task-space Jacobian mapping n joint velocities to 2-D Cartesian
        velocity.  n ≥ 1.

    Returns
    -------
    directions : np.ndarray, shape (2, 2)
        Unit eigenvectors of J Jᵀ (columns), i.e. the principal axes of
        both ellipsoids.  ``directions[:, 0]`` is the major axis.
    mob_semi_axes : np.ndarray, shape (2,)
        Semi-axis lengths of the **mobility** ellipsoid (√λᵢ).
    force_semi_axes : np.ndarray or None, shape (2,)
        Semi-axis lengths of the **force** ellipsoid (1/√λᵢ).
        ``None`` if the configuration is singular (smallest singular value
        is less than `_SINGULARITY_THRESHOLD` × largest).
    singular_values : np.ndarray, shape (2,)
        Raw singular values of J (for external use / colouring).

    Raises
    ------
    AssertionError
        If ``J`` does not have shape (2, n) with n ≥ 1, or contains
        non-finite values.
    """
    assert isinstance(J, np.ndarray), "J must be a numpy ndarray"
    assert (
        J.ndim == 2 and J.shape[0] == 2 and J.shape[1] >= 1
    ), f"J must have shape (2, n) with n≥1, got {J.shape}"
    assert np.all(np.isfinite(J)), "J must not contain NaN or Inf"

    # SVD of J directly: J = U Σ Vᵀ, J Jᵀ = U Σ² Uᵀ
    # We only need the left singular vectors U (task-space directions).
    U, s, _Vt = np.linalg.svd(J, full_matrices=False)

    # Keep only the 2 largest singular values (s is already length 2 since J is 2×n)
    # np.linalg.svd with full_matrices=False returns min(m,n)=2 values here.
    directions: np.ndarray = U  # shape (2, 2) — principal axes as columns
    mob_semi_axes: np.ndarray = s  # shape (2,) — √λᵢ

    # Detect singularity
    if s[0] < _SINGULARITY_THRESHOLD:
        # Badly conditioned — both semi-axes essentially zero (degenerate config)
        force_semi_axes: np.ndarray | None = None
    elif s[-1] < _SINGULARITY_THRESHOLD * s[0]:
        # Near-singular in at least one direction
        force_semi_axes = None
    else:
        force_semi_axes = 1.0 / s

    return directions, mob_semi_axes, force_semi_axes, s


# ---------------------------------------------------------------------------
# Double-pendulum Jacobians
# ---------------------------------------------------------------------------


def jacobian_double(
    theta1: float,
    phi: float,
    L1: float,
    L2: float,
) -> dict[str, np.ndarray]:
    """Compute task-space Jacobians for both endpoints of the double pendulum.

    The Jacobian maps [dθ1/dt, dφ/dt] → [dx/dt, dy/dt] at each endpoint.

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 from downward vertical (rad).
    phi : float
        Relative angle of segment 2 w.r.t. segment 1 (rad).
    L1 : float
        Length of segment 1 (m).  Must be > 0.
    L2 : float
        Length of segment 2 (m).  Must be > 0.

    Returns
    -------
    dict with keys:
        ``"wrist"``   : np.ndarray shape (2, 2) — Jacobian at segment-1 tip
        ``"tip"``     : np.ndarray shape (2, 2) — Jacobian at segment-2 tip

    Raises
    ------
    AssertionError
        On invalid inputs.
    """
    assert np.isfinite(theta1), f"theta1 must be finite, got {theta1}"
    assert np.isfinite(phi), f"phi must be finite, got {phi}"
    assert L1 > 0, f"L1 must be positive, got {L1}"
    assert L2 > 0, f"L2 must be positive, got {L2}"

    theta2 = theta1 + phi  # absolute angle of segment 2

    c1 = np.cos(theta1)
    s1 = np.sin(theta1)
    c2 = np.cos(theta2)
    s2 = np.sin(theta2)

    # ∂wrist_position / ∂[theta1, phi]
    # wrist = (L1 sin θ1, −L1 cos θ1)  — phi has no effect on this point
    J_wrist = np.array(
        [
            [L1 * c1, 0.0],  # dx/d[θ1, φ]
            [L1 * s1, 0.0],  # dy/d[θ1, φ]
        ]
    )

    # ∂tip_position / ∂[theta1, phi]
    # tip = wrist + (L2 sin θ2, −L2 cos θ2)
    J_tip = np.array(
        [
            [L1 * c1 + L2 * c2, L2 * c2],
            [L1 * s1 + L2 * s2, L2 * s2],
        ]
    )

    return {"wrist": J_wrist, "tip": J_tip}


def ellipsoids_double(
    theta1: float,
    phi: float,
    L1: float,
    L2: float,
) -> dict[str, dict]:
    """Compute mobility and force ellipsoid data for both double-pendulum endpoints.

    Returns
    -------
    dict with keys ``"wrist"`` and ``"tip"``, each containing:
        ``"jacobian"``       : (2, 2) ndarray
        ``"directions"``     : (2, 2) ndarray  — principal axes (columns)
        ``"mob_semi_axes"``  : (2,) ndarray    — mobility ellipsoid semi-axes
        ``"force_semi_axes"``: (2,) ndarray or None — force ellipsoid semi-axes
        ``"singular_values"``: (2,) ndarray
    """
    assert np.isfinite(theta1) and np.isfinite(phi), "Angles must be finite"
    assert L1 > 0 and L2 > 0, "Segment lengths must be positive"

    jacs = jacobian_double(theta1, phi, L1, L2)
    result: dict[str, dict] = {}
    for name, J in jacs.items():
        dirs, mob, force, svs = ellipsoid_from_jacobian(J)
        result[name] = {
            "jacobian": J,
            "directions": dirs,
            "mob_semi_axes": mob,
            "force_semi_axes": force,
            "singular_values": svs,
        }
    return result


# ---------------------------------------------------------------------------
# Triple-pendulum Jacobians
# ---------------------------------------------------------------------------


def jacobian_triple(
    theta1: float,
    phi1: float,
    phi2: float,
    L1: float,
    L2: float,
    L3: float,
) -> dict[str, np.ndarray]:
    """Compute task-space Jacobians for all three endpoints of the triple pendulum.

    Parameters
    ----------
    theta1 : float
        Absolute angle of segment 1 from downward vertical (rad).
    phi1 : float
        Relative angle of segment 2 w.r.t. segment 1 (rad).
    phi2 : float
        Relative angle of segment 3 w.r.t. segment 2 (rad).
    L1, L2, L3 : float
        Segment lengths (m).  All must be > 0.

    Returns
    -------
    dict with keys:
        ``"wrist1"`` : (2, 3) ndarray — Jacobian at segment-1 tip
        ``"wrist2"`` : (2, 3) ndarray — Jacobian at segment-2 tip
        ``"tip"``    : (2, 3) ndarray — Jacobian at segment-3 tip
    """
    assert (
        np.isfinite(theta1) and np.isfinite(phi1) and np.isfinite(phi2)
    ), "All angles must be finite"
    assert L1 > 0 and L2 > 0 and L3 > 0, "All segment lengths must be positive"

    theta2 = theta1 + phi1  # absolute angle of segment 2
    theta3 = theta1 + phi1 + phi2  # absolute angle of segment 3

    c1, s1 = np.cos(theta1), np.sin(theta1)
    c2, s2 = np.cos(theta2), np.sin(theta2)
    c3, s3 = np.cos(theta3), np.sin(theta3)

    # Wrist-1: tip of segment 1; only θ1 affects it, φ1 and φ2 do not
    J_w1 = np.array(
        [
            [L1 * c1, 0.0, 0.0],
            [L1 * s1, 0.0, 0.0],
        ]
    )

    # Wrist-2: tip of segment 2; θ1 and φ1 affect it, φ2 does not
    J_w2 = np.array(
        [
            [L1 * c1 + L2 * c2, L2 * c2, 0.0],
            [L1 * s1 + L2 * s2, L2 * s2, 0.0],
        ]
    )

    # Tip (end-effector): all three DOFs contribute
    J_tip = np.array(
        [
            [L1 * c1 + L2 * c2 + L3 * c3, L2 * c2 + L3 * c3, L3 * c3],
            [L1 * s1 + L2 * s2 + L3 * s3, L2 * s2 + L3 * s3, L3 * s3],
        ]
    )

    return {"wrist1": J_w1, "wrist2": J_w2, "tip": J_tip}


def ellipsoids_triple(
    theta1: float,
    phi1: float,
    phi2: float,
    L1: float,
    L2: float,
    L3: float,
) -> dict[str, dict]:
    """Compute mobility and force ellipsoid data for all three triple-pendulum endpoints.

    Returns
    -------
    dict with keys ``"wrist1"``, ``"wrist2"``, and ``"tip"``,
    each containing the same sub-keys as :func:`ellipsoids_double`.
    """
    assert (
        np.isfinite(theta1) and np.isfinite(phi1) and np.isfinite(phi2)
    ), "All angles must be finite"
    assert L1 > 0 and L2 > 0 and L3 > 0, "All segment lengths must be positive"

    jacs = jacobian_triple(theta1, phi1, phi2, L1, L2, L3)
    result: dict[str, dict] = {}
    for name, J in jacs.items():
        dirs, mob, force, svs = ellipsoid_from_jacobian(J)
        result[name] = {
            "jacobian": J,
            "directions": dirs,
            "mob_semi_axes": mob,
            "force_semi_axes": force,
            "singular_values": svs,
        }
    return result

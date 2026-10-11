"""Physics checks of the two-hand internal force (issue #11739, OSV-7).

A flat bound on the internal force ``(F_L - F_R)/2`` cannot tell a real,
couple-carrying force pair from the "fighting hands" artefact of inconsistent
kinematics.  Two checks replace it, both built on
:func:`~src.shared.python.grip_contact.force_decomposition.decompose_hand_forces`:

1. **Squeeze.**  The axial part of the internal force (along the inter-hand
   line ``u``, positive in compression) must stay small: the club needs no
   squeeze, so a large one means the hands are pulled against each other.

2. **Couple consistency.**  The transverse internal pair must be the pair the
   club's rigid-body dynamics require.  Definitions, with the hand-acting-on-
   the-club convention, midpoint ``P = (p_L + p_R)/2``, spacing
   ``d = |p_R - p_L|`` and ``u = (p_R - p_L)/d``:

   * Newton-Euler of the club gives the total moment that the hands must apply
     about ``P``::

         M_hands,P = I w' + w x (I w) + (c - P) x m (a_c - g)

     with ``I`` the world inertia about the centre of mass ``c``, ``w`` the
     angular velocity, ``w'`` its rate, ``a_c`` the centre-of-mass
     acceleration and ``g`` gravity, all from the simulated club motion (the
     engine's realised accelerations), not from the hand forces.
   * The hands apply that moment as their free torques ``tau_L + tau_R`` plus
     the moment of their contact forces about ``P``.  The net force has no
     moment about the midpoint, so the contact-force moment is carried by the
     internal pair alone: ``M_contact = M_hands,P - tau_L - tau_R`` and
     ``M_contact = -d u x F_int``.
   * Hence the prediction ``|F_int,perp|_pred = |M_contact,perp| / d``, where
     ``perp`` is the component normal to ``u`` (a pair along ``u`` carries no
     moment).  The check compares it with the measured ``|F_int,perp|`` from
     :func:`decompose_hand_forces` and passes when the relative error is at
     most 5 %.  Samples with ``|M_contact,perp|`` below ``noise_floor_nm``
     (default 2 N m) are skipped, because a relative error of a near-zero
     couple is meaningless.

   Scope.  Evaluated with the club's own realised motion this is a
   Newton-Euler closure: it verifies that the extracted per-hand wrenches,
   their frames and signs and the spec inertia account for the club's motion,
   i.e. that the transverse pair is couple-carrying and not an unexplained
   load.  It does not say whether the club motion is the one the input swing
   demands; for that, evaluate :func:`required_hand_moment_nm` with the
   rigid-club kinematics (the club welded to the prescribed hand).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.force_decomposition import decompose_hand_forces

#: Samples whose contact-force couple ``|M_contact,perp|`` is below this are
#: skipped (N m): 2 % of the swing peak (about 100 N m) and above the
#: address/backswing level, where the relative error of a vanishing couple is
#: meaningless.  About 80 % of the 0-1.8 s samples are checked.
DEFAULT_COUPLE_NOISE_FLOOR_NM = 2.0


@dataclass(frozen=True)
class CoupleConsistency:
    """Result of :func:`couple_consistency` (time series, world frame)."""

    predicted_transverse_n: np.ndarray  # |M_contact,perp| / d, shape (n,)
    actual_transverse_n: np.ndarray  # |F_int,perp| from the decomposition
    contact_moment_nm: np.ndarray  # M_contact, shape (n, 3)
    checked: np.ndarray  # bool mask, couple above the noise floor
    noise_floor_nm: float

    def relative_error(self) -> np.ndarray:
        """``|actual - predicted| / predicted`` on the checked samples."""
        pred = self.predicted_transverse_n[self.checked]
        return np.asarray(np.abs(self.actual_transverse_n[self.checked] - pred) / pred)

    def max_relative_error(self) -> float:
        """Largest relative error over the checked samples (0 when none)."""
        err = self.relative_error()
        return float(err.max()) if err.size else 0.0


def _series(name: str, value: object, n: int | None = None) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    if arr.ndim != 2 or arr.shape[1] != 3 or (n is not None and arr.shape[0] != n):
        raise ValueError(f"{name} must have shape (n, 3), got {arr.shape}")
    if not np.all(np.isfinite(arr)):
        raise ValueError(f"{name} must be finite")
    return arr


@dataclass(frozen=True)
class ClubKinematics:
    """Rigid-body motion of the club, world frame, shape ``(n, ...)``.

    ``rotation`` is world <- club body; ``alpha_rad_s2`` and
    ``com_acceleration_m_s2`` are the engine's realised accelerations.
    """

    rotation: np.ndarray  # (n, 3, 3)
    omega_rad_s: np.ndarray  # (n, 3)
    alpha_rad_s2: np.ndarray  # (n, 3)
    com_m: np.ndarray  # (n, 3)
    com_acceleration_m_s2: np.ndarray  # (n, 3)

    def __post_init__(self) -> None:
        w = _series("omega_rad_s", self.omega_rad_s)
        n = w.shape[0]
        for name in ("alpha_rad_s2", "com_m", "com_acceleration_m_s2"):
            object.__setattr__(self, name, _series(name, getattr(self, name), n))
        rot = np.asarray(self.rotation, dtype=float)
        if rot.shape != (n, 3, 3) or not np.all(np.isfinite(rot)):
            raise ValueError("rotation must be a finite (n, 3, 3) array")
        object.__setattr__(self, "rotation", rot)
        object.__setattr__(self, "omega_rad_s", w)

    def subset(self, keep: np.ndarray) -> ClubKinematics:
        """The samples selected by the boolean or index array ``keep``."""
        return ClubKinematics(
            self.rotation[keep],
            self.omega_rad_s[keep],
            self.alpha_rad_s2[keep],
            self.com_m[keep],
            self.com_acceleration_m_s2[keep],
        )


def required_hand_moment_nm(
    club: ClubKinematics,
    dynamics: ClubDynamics,
    gravity_m_s2: np.ndarray,
    point_m: np.ndarray,
) -> np.ndarray:
    """Moment the hands must apply about ``point_m`` (Newton-Euler), (n, 3).

    Uses the realised accelerations in ``club`` (no finite differences: the
    bushing modes, 400 to 950 Hz, lie above the Nyquist rate of 2 ms samples).

    Raises:
        ValueError: on a malformed ``point_m`` or ``gravity_m_s2``.
    """
    n = club.omega_rad_s.shape[0]
    p = _series("point_m", point_m, n)
    g = np.asarray(gravity_m_s2, dtype=float)
    if g.shape != (3,) or not np.all(np.isfinite(g)):
        raise ValueError("gravity_m_s2 must be a finite 3-vector")
    rot, w = club.rotation, club.omega_rad_s
    inertia = np.einsum("nij,jk,nlk->nil", rot, dynamics.inertia_com_kg_m2, rot)
    i_w = np.einsum("nij,nj->ni", inertia, w)
    euler = np.einsum("nij,nj->ni", inertia, club.alpha_rad_s2) + np.cross(w, i_w)
    accel = dynamics.mass_kg * (club.com_acceleration_m_s2 - g)
    return np.asarray(euler + np.cross(club.com_m - p, accel))


def peak_squeeze_n(
    force_left_n: object,
    force_right_n: object,
    point_left_m: object,
    point_right_m: object,
) -> float:
    """Largest |axial internal force| (squeeze or traction) over the series."""
    dec = decompose_hand_forces(
        force_left_n, force_right_n, point_left_m, point_right_m
    )
    return float(np.max(np.abs(np.atleast_1d(dec.internal_axial_n))))


def couple_consistency(
    force_n: tuple[np.ndarray, np.ndarray],
    point_m: tuple[np.ndarray, np.ndarray],
    free_torque_nm: tuple[np.ndarray, np.ndarray],
    required_moment_at_midpoint_nm: np.ndarray,
    noise_floor_nm: float = DEFAULT_COUPLE_NOISE_FLOOR_NM,
) -> CoupleConsistency:
    """Compare the transverse internal pair with the couple the club needs.

    Each pair argument is ``(left, right)`` of ``(n, 3)`` world series; the
    required moment comes from :func:`required_hand_moment_nm` evaluated at
    the hand midpoint.  See the module docstring for the definitions.

    Raises:
        ValueError: on mismatched shapes, non-finite input, coincident hand
            points or a negative noise floor.
    """
    if not np.isfinite(noise_floor_nm) or noise_floor_nm < 0.0:
        raise ValueError("noise_floor_nm must be a finite non-negative number")
    f_l, f_r = (_series(f"force_n[{k}]", f) for k, f in enumerate(force_n))
    n = f_l.shape[0]
    p_l, p_r = (_series(f"point_m[{k}]", p, n) for k, p in enumerate(point_m))
    t_l, t_r = (
        _series(f"free_torque_nm[{k}]", t, n) for k, t in enumerate(free_torque_nm)
    )
    m_req = _series("required_moment_at_midpoint_nm", required_moment_at_midpoint_nm, n)
    dec = decompose_hand_forces(f_l, _series("force_n[1]", f_r, n), p_l, p_r)
    line = p_r - p_l
    d = np.linalg.norm(line, axis=1)
    u = line / d[:, None]
    m_contact = m_req - t_l - t_r
    m_perp = m_contact - np.sum(m_contact * u, axis=1)[:, None] * u
    m_perp_norm = np.linalg.norm(m_perp, axis=1)
    return CoupleConsistency(
        predicted_transverse_n=m_perp_norm / d,
        actual_transverse_n=np.linalg.norm(dec.internal_transverse_n, axis=1),
        contact_moment_nm=m_contact,
        checked=m_perp_norm >= noise_floor_nm,
        noise_floor_nm=float(noise_floor_nm),
    )

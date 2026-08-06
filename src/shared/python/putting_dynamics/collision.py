"""Two-body putter-ball collision with loft and hosel twist (#8345, P3).

Reused shared infrastructure (AGENTS.md section A discovery):

* Ball constants from ``src.shared.python.core.physics_constants``
  (USGA Rule 5 mass/radius; single source).
* DbC helpers from ``src.shared.python.contracts``.
* The lofted-face decomposition and the 2/7 tangential rolling cap
  restate the derivation in Tools ``swing_sim.putting.impact`` (branch
  ``feat/putting-vertical``, epic Tools#4125 H3) — the vendored Tools
  snapshot predates that branch, so the formulas are restated with
  credit instead of imported (UD vendors Tools; no import cycle).
* The ground restitution used by the post-impact hop settle is the
  green-contact default in
  ``rust_core/upstream-physics/src/contact.rs`` (``cor = 0.78``).

Model
-----
The face is a plane tilted back by the dynamic loft ``delta``; the
head travels horizontally at ``V``.  Contact is sub-millisecond, so
gravity and turf reaction are ignored during the impulse (restated
Tools assumption).

**Normal direction** — two-body impulse with a *finite effective
mass*.  For a strike offset ``d`` toe-ward of the head CG the head
responds with::

    1 / m_eff = 1 / M + d**2 / I

(standard rigid-body effective mass at an offset contact; ``I`` is the
head MOI about the vertical axis).  With reduced mass
``m_red = m_eff m_b / (m_eff + m_b)`` the normal impulse is::

    J_n = (1 + e) m_red V cos(delta)

giving ball launch ``J_n / m_b`` along the normal and a putter CG
slowdown ``J_x / M`` (momentum conservation is test-pinned).  The
normal-direction energy loss is ``(1/2) m_red (V cos d)^2 (1 - e^2)``
(test-pinned COR audit).

**Tangential direction** — resolving the horizontal face velocity
along the face tangent gives a down-and-forward contact velocity of
magnitude ``u = V sin(delta)`` for positive loft.  Impulsive friction
spins the ball to the no-slip cap, transferring ``v_t = (2/7) u``
down the face and backspin ``omega r = (5/7) u``.  This uses the same
2/7 sphere cap as Tools ``swing_sim.putting.impact`` but corrects that
module's inconsistent vector recomposition: a backspin-producing
tangential impulse must point down the face, not up it.  The tangential slip dissipates
``m_b u^2 / 7`` (work ``(2/7) m u^2`` minus ball tangential KE gain
``(1/7) m u^2``).

**Hosel twist (face-balanced vs toe-hang spectrum)** — the shaft
attaches at ``(hosel_forward_m, hosel_toe_m)`` relative to the head
CG on the top surface.  The impact impulse acts at the strike point
``(0, impact_toe_m)``; its moment about the vertical shaft axis is::

    M_shaft = J_x * (impact_toe_m - hosel_toe_m)

(the forward hosel offset drops out because the normal impulse has no
lateral component — it still enters the full wrench below).  The
face-twist impulse is ``M_shaft / I``; positive twist swings the toe
backward (face opening toward the toe side).  A centered attachment
with a centered strike therefore produces zero twist, and a heel-side
(anser-style) attachment with a center strike produces positive twist
(both test-pinned, with toe-mirror antisymmetry).

This is a rigid-body *impulse* treatment.  The documented seam for the
upcoming finite-interval twist-dynamics epic is the attachment-point
impulse wrench reported on :class:`~.types.CollisionReport`
(``attachment_impulse_n_s`` / ``attachment_moment_n_m_s``, head frame:
x forward, y toe-ward, z up; the vertical lever from top surface to
face center is omitted here and belongs to that epic).
"""

from __future__ import annotations

import math
from dataclasses import replace

from src.shared.python.contracts import ensure, require, require_finite
from src.shared.python.core.physics_constants import (
    GOLF_BALL_MASS_KG,
    GOLF_BALL_RADIUS_M,
)

from .types import CollisionReport, PutterState

__all__ = [
    "GROUND_RESTITUTION",
    "LOFT_SWEEP_MAX_DEG",
    "LOFT_SWEEP_MIN_DEG",
    "effective_head_mass",
    "strike",
    "sweep_dynamic_loft",
]

#: Green-contact restitution for the post-impact hop settle — the
#: default in ``rust_core/upstream-physics/src/contact.rs``.
GROUND_RESTITUTION = 0.78

#: Dynamic-loft sweep bounds [deg] (epic #8345 P3).
LOFT_SWEEP_MIN_DEG = -4.0
LOFT_SWEEP_MAX_DEG = 8.0

#: 2/7 tangential rolling cap for a struck sphere (restated from Tools
#: ``swing_sim.putting.impact`` / ``swing_sim.impact``).
_ROLLING_CAP = 2.0 / 7.0

#: Contact-duration proxy [s] for visualization pacing (documented
#: sub-millisecond scale; not a dynamics quantity).
_CONTACT_TIME_PROXY_S = 0.0005


def effective_head_mass(putter: PutterState, impact_toe_m: float) -> float:
    """Head effective mass at an offset strike point.

    ``1/m_eff = 1/M + d^2/I`` (module derivation); equals the head
    mass for a centered strike.

    Args:
        putter: Putter head state.
        impact_toe_m: Strike offset toe-ward of the head CG [m].

    Returns:
        Effective mass [kg], in ``(0, head_mass_kg]``.

    Raises:
        ValueError: If the offset is out of range.
    """
    require_finite(impact_toe_m, "impact_toe_m")
    require(
        abs(impact_toe_m) <= 0.08,
        "impact offset within +/- 0.08 m",
        impact_toe_m,
    )
    m_eff = 1.0 / (1.0 / putter.head_mass_kg + impact_toe_m**2 / putter.moi_kg_m2)
    ensure(0.0 < m_eff <= putter.head_mass_kg, "m_eff in (0, M]", m_eff)
    return m_eff


def strike(putter: PutterState, impact_toe_m: float = 0.0) -> CollisionReport:
    """Solve one putter-ball impact (module derivation).

    Args:
        putter: Putter head at impact (speed, loft, COR, hosel).
        impact_toe_m: Strike offset toe-ward of the head CG [m];
            0 is a centered strike.

    Returns:
        The full :class:`~.types.CollisionReport`.

    Raises:
        ValueError: If inputs are out of physical range.
    """
    delta = math.radians(putter.loft_deg)
    m_ball = float(GOLF_BALL_MASS_KG)
    m_eff = effective_head_mass(putter, impact_toe_m)
    m_red = m_eff * m_ball / (m_eff + m_ball)

    v_normal_rel = putter.speed_mps * math.cos(delta)
    impulse_n = (1.0 + putter.cor) * m_red * v_normal_rel
    ball_normal = impulse_n / m_ball

    u_tangential = putter.speed_mps * math.sin(delta)
    ball_tangential = _ROLLING_CAP * u_tangential
    impulse_t = _ROLLING_CAP * m_ball * u_tangential
    # Backspin: contact-surface speed (5/7) u, topspin-positive sign
    # (restated Tools convention).
    spin_rad_s = -(1.0 - _ROLLING_CAP) * u_tangential / GOLF_BALL_RADIUS_M

    # World components (x forward, z up); n = (cos d, sin d), while
    # the backspin-producing tangential impulse points down the face,
    # -t = (sin d, -cos d).
    horizontal = ball_normal * math.cos(delta) + ball_tangential * math.sin(delta)
    vertical = ball_normal * math.sin(delta) - ball_tangential * math.cos(delta)
    ball_speed = math.hypot(horizontal, vertical)

    impulse_x = impulse_n * math.cos(delta) + impulse_t * math.sin(delta)
    impulse_z = impulse_n * math.sin(delta) - impulse_t * math.cos(delta)
    putter_dv = impulse_x / putter.head_mass_kg

    energy_loss = (
        0.5 * m_red * v_normal_rel**2 * (1.0 - putter.cor**2)
        + m_ball * u_tangential**2 / 7.0
    )

    twist_moment = impulse_x * (impact_toe_m - putter.hosel_toe_m)
    shaft_axis_moi = putter.moi_kg_m2 + putter.head_mass_kg * (
        putter.hosel_toe_m**2 + putter.hosel_forward_m**2
    )
    face_twist = twist_moment / shaft_axis_moi

    # Impulse wrench on the putter at the shaft attachment (head
    # frame: x forward, y toe-ward, z up); vertical lever omitted —
    # documented seam for the finite-interval twist epic.
    lever_f = 0.0 - putter.hosel_forward_m
    lever_t = impact_toe_m - putter.hosel_toe_m
    force = (-impulse_x, 0.0, -impulse_z)
    moment = (
        lever_t * force[2],
        -lever_f * force[2],
        lever_f * 0.0 - lever_t * force[0],
    )

    ensure(ball_speed > 0.0, "ball must leave the face", ball_speed)
    ensure(
        ball_speed <= 2.0 * putter.speed_mps,
        "smash factor bounded by 2 (equal-mass elastic limit)",
        ball_speed / putter.speed_mps,
    )
    ensure(putter_dv >= 0.0, "the head cannot speed up", putter_dv)
    ensure(energy_loss >= 0.0, "impact cannot create energy", energy_loss)
    return CollisionReport(
        ball_speed_mps=ball_speed,
        launch_angle_deg=math.degrees(math.atan2(vertical, horizontal)),
        horizontal_speed_mps=horizontal,
        vertical_speed_mps=vertical,
        spin_rad_s=spin_rad_s,
        effective_loft_deg=putter.loft_deg,
        putter_dv_mps=putter_dv,
        impulse_n_s=impulse_n,
        contact_time_proxy_s=_CONTACT_TIME_PROXY_S,
        kinetic_energy_loss_j=energy_loss,
        face_twist_rad_s=face_twist,
        twist_moment_n_m_s=twist_moment,
        attachment_impulse_n_s=force,
        attachment_moment_n_m_s=moment,
    )


def sweep_dynamic_loft(
    putter: PutterState,
    loft_min_deg: float = LOFT_SWEEP_MIN_DEG,
    loft_max_deg: float = LOFT_SWEEP_MAX_DEG,
    step_deg: float = 1.0,
    impact_toe_m: float = 0.0,
) -> tuple[CollisionReport, ...]:
    """Dynamic-loft sweep helper (P3 acceptance: -4 to +8 deg).

    Args:
        putter: Base putter state; its loft is replaced per sample.
        loft_min_deg: Sweep start [deg].
        loft_max_deg: Sweep end, inclusive [deg].
        step_deg: Sweep step [deg], > 0.
        impact_toe_m: Strike offset passed to every sample.

    Returns:
        Reports ordered by increasing loft.

    Raises:
        ValueError: If the sweep bounds are invalid.
    """
    require_finite(loft_min_deg, "loft_min_deg")
    require_finite(loft_max_deg, "loft_max_deg")
    require(loft_min_deg < loft_max_deg, "sweep must ascend", loft_min_deg)
    require_finite(step_deg, "step_deg")
    require(0.0 < step_deg <= 5.0, "step in (0, 5] deg", step_deg)
    count = int(math.floor((loft_max_deg - loft_min_deg) / step_deg + 1e-9)) + 1
    reports = tuple(
        strike(
            replace(putter, loft_deg=loft_min_deg + i * step_deg),
            impact_toe_m=impact_toe_m,
        )
        for i in range(count)
    )
    ensure(len(reports) >= 2, "sweep must contain >= 2 samples", len(reports))
    return reports

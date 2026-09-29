"""Head deceleration, peak loads, and head twist under sand load (issue #8614).

============================== ===========================================================
Quantity                       Definition
============================== ===========================================================
Deceleration                   ``-d|v_cg|/dt`` [m/s^2]; the rate the head loses *speed*,
                               not the magnitude of its acceleration, so a pure direction
                               change scores zero.
Peak deceleration              Maximum of that over the window, also reported in g.
Mean deceleration              ``(speed_start - speed_end) / (t_end - t_start)`` -- the
                               time-average of the instantaneous rate, evaluated in the
                               closed form so it is exact rather than quadrature-limited.
Peak resultant force           ``max |F|`` over the window [N].
Peak resultant moment          ``max |M_cg|`` over the window [N.m], about the centre of
                               mass, including the ``(r - r_cg) x F`` transport term that
                               finding B5b showed was missing.
Shaft-axis moment              ``M_cg . e_shaft`` [N.m] -- the couple that opens or closes
                               the face under sand load.
Free face rotation             The rotation the head would take about the shaft axis if
                               the shaft applied no restoring couple: double integration
                               of ``M_shaft / I_shaft``. An upper bound, not a prediction.
============================== ===========================================================

**Why the twist is here at all.** Between entry and the ball the sand load acts
on the sole -- below and forward of the CG -- so it generates both a pitching
couple and a couple about the shaft axis. That is why bounce and relief exist. A
literature search for this epic found it **quantified nowhere in public work**,
so it is an output of this tool rather than a reproduction of someone else's.

**Axis convention.** Three orthonormal axes are built per sample:

* ``e_shaft`` -- the head's shaft axis, pointing from the head up toward the grip.
* ``e_travel`` -- the scene's travel axis, made perpendicular to ``e_shaft``.
* ``e_loft = e_shaft x e_travel`` -- completes a right-handed triad.

Each reported component is the right-handed moment about the named axis. Worked
example fixing the face-opening sign: take world ``+x`` toward the target,
``+z`` up, ``+y = z x x``, and a right-handed player, so the toe lies on the
``+y`` side of the shaft axis and ``e_shaft ~ +z``. Sand retards the sole with a
force along ``-x`` applied at an offset ``+d y``, giving
``M = (0, d, 0) x (-F, 0, 0) = (0, 0, +dF)``: a **positive** shaft-axis moment,
and a positive rotation about ``+z`` carries the face normal from ``+x`` toward
``+y``. Positive shaft-axis moment therefore **opens** the face. A left-handed
player mirrors the scene and flips the sign; that is a property of the scene the
caller builds, not a hidden constant here.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.integrate import cumulative_trapezoid

from src.shared.python.core.contracts import require

from .trace import (
    STANDARD_GRAVITY_MPS2,
    HeadModel,
    StrikeScene,
    StrikeTrace,
    centre_of_mass_moment_Nm,
)

__all__ = [
    "HeadLoadMetrics",
    "HeadTwistMetrics",
    "head_load_metrics",
    "head_twist_metrics",
    "shaft_travel_loft_axes",
]


def _resolve_window(trace: StrikeTrace, window: slice | None) -> slice:
    """Return the sample window to reduce over, defaulting to the whole trace.

    Args:
        trace: Strike trace.
        window: Requested window, or ``None``.

    Returns:
        The window.

    Raises:
        ValueError: If the window selects fewer than three samples, which cannot
            support a second-order rate.
    """
    selection = slice(None) if window is None else window
    if trace.time_s[selection].size < 3:
        raise ValueError("the load window needs at least 3 samples")
    return selection


@dataclass(frozen=True)
class HeadLoadMetrics:
    """What the sand did to the head, as a designer reads it.

    Attributes:
        peak_deceleration_mps2: Maximum rate of speed loss over the window.
        peak_deceleration_g: The same in multiples of standard gravity. The
            research addendum's end-to-end smoke test is ~527 g on a 0.30 kg
            head for a 20x80 mm sole at 25 m/s.
        peak_deceleration_time_s: When the peak occurred.
        mean_deceleration_mps2: Speed lost divided by window duration.
        entry_speed_mps: Head centre-of-mass speed at the window start.
        exit_speed_mps: Head centre-of-mass speed at the window end.
        peak_resultant_force_N: ``max |F|``.
        peak_resultant_force_time_s: When it occurred.
        mean_resultant_force_N: Time-average of ``|F|`` over the window.
        peak_resultant_moment_Nm: ``max |M_cg|``.
        peak_resultant_moment_time_s: When it occurred.
        linear_impulse_Ns: ``(3,) integral of F dt``.
        angular_impulse_Nms: ``(3,) integral of M_cg dt``.
    """

    peak_deceleration_mps2: float
    peak_deceleration_g: float
    peak_deceleration_time_s: float
    mean_deceleration_mps2: float
    entry_speed_mps: float
    exit_speed_mps: float
    peak_resultant_force_N: float
    peak_resultant_force_time_s: float
    mean_resultant_force_N: float
    peak_resultant_moment_Nm: float
    peak_resultant_moment_time_s: float
    linear_impulse_Ns: np.ndarray
    angular_impulse_Nms: np.ndarray


def head_load_metrics(
    trace: StrikeTrace,
    head: HeadModel,
    *,
    window: slice | None = None,
) -> HeadLoadMetrics:
    """Reduce the wrench and the head path to peak and mean loads.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        window: Optional sample window; defaults to the whole trace.

    Returns:
        The load metrics.

    Raises:
        ValueError: If the window holds fewer than two samples.
    """
    selection = _resolve_window(trace, window)
    times = trace.time_s[selection]
    speed = np.linalg.norm(
        trace.point_velocity_mps(head.centre_of_mass_body_m)[selection], axis=1
    )
    deceleration = -np.gradient(speed, times, edge_order=2)
    peak_index = int(np.argmax(deceleration))
    force = trace.sand_force_N[selection]
    moment = centre_of_mass_moment_Nm(trace, head)[selection]
    # ⚡ Bolt: np.sqrt(np.einsum) is ~2.4x faster than np.linalg.norm(..., axis=1)
    force_magnitude = np.sqrt(np.einsum("ij,ij->i", force, force))
    moment_magnitude = np.sqrt(np.einsum("ij,ij->i", moment, moment))
    duration_s = float(times[-1] - times[0])
    peak_deceleration = float(deceleration[peak_index])
    return HeadLoadMetrics(
        peak_deceleration_mps2=peak_deceleration,
        peak_deceleration_g=peak_deceleration / STANDARD_GRAVITY_MPS2,
        peak_deceleration_time_s=float(times[peak_index]),
        mean_deceleration_mps2=float(speed[0] - speed[-1]) / duration_s,
        entry_speed_mps=float(speed[0]),
        exit_speed_mps=float(speed[-1]),
        peak_resultant_force_N=float(force_magnitude.max()),
        peak_resultant_force_time_s=float(times[int(np.argmax(force_magnitude))]),
        mean_resultant_force_N=float(np.trapezoid(force_magnitude, times)) / duration_s,
        peak_resultant_moment_Nm=float(moment_magnitude.max()),
        peak_resultant_moment_time_s=float(times[int(np.argmax(moment_magnitude))]),
        linear_impulse_Ns=np.trapezoid(force, times, axis=0),
        angular_impulse_Nms=np.trapezoid(moment, times, axis=0),
    )


def shaft_travel_loft_axes(
    trace: StrikeTrace, head: HeadModel, scene: StrikeScene
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Return the per-sample ``(e_shaft, e_travel, e_loft)`` orthonormal triad.

    Args:
        trace: Strike trace.
        head: Head supplying the body-frame shaft axis.
        scene: Scene supplying the travel axis.

    Returns:
        Three ``(T, 3)`` arrays of unit world vectors.

    Raises:
        ValueError: If the shaft axis is parallel to the travel axis at any
            sample, which leaves the perpendicular component undefined. A wedge
            at a 64 deg lie is nowhere near that, so it means the head model and
            the scene disagree about their frames.
    """
    shaft = trace.body_axis_world(head.shaft_axis_body)
    projection = (shaft @ scene.travel_axis)[:, None]
    travel = scene.travel_axis - projection * shaft
    norms = np.sqrt(
        np.einsum("ij,ij->i", travel, travel)
    )  # ⚡ Bolt: np.sqrt(np.einsum) avoids temporary allocations and is ~2.4x faster than np.linalg.norm(..., axis=1)
    if float(norms.min()) < 1e-6:
        raise ValueError(
            "the shaft axis is parallel to the travel axis, so the twist triad "
            "is undefined; check that head and scene use the same frame"
        )
    travel = travel / norms[:, None]
    return shaft, travel, np.cross(shaft, travel)


@dataclass(frozen=True)
class HeadTwistMetrics:
    """Head rotation under sand load -- the metric nobody has published.

    Attributes:
        shaft_axis_moment_Nm: ``(T,)`` moment about the shaft axis; positive
            opens the face under the convention documented in this module.
        travel_axis_moment_Nm: ``(T,)`` moment about the travel axis made
            perpendicular to the shaft -- the toe-down/toe-up droop couple.
        loft_axis_moment_Nm: ``(T,)`` moment about ``e_shaft x e_travel`` -- the
            couple that changes dynamic loft.
        peak_shaft_axis_moment_Nm: Largest-magnitude shaft-axis moment, signed.
        mean_shaft_axis_moment_Nm: Time-average of the shaft-axis moment.
        shaft_axis_angular_impulse_Nms: ``integral of M_shaft dt``.
        peak_travel_axis_moment_Nm: Largest-magnitude droop couple, signed.
        peak_loft_axis_moment_Nm: Largest-magnitude loft couple, signed.
        peak_resultant_moment_Nm: ``max |M_cg|``.
        shaft_axis_inertia_kg_m2: ``a . I . a`` about the CG, or ``None`` when
            the head carries no inertia tensor.
        free_face_rate_radps: Shaft-axis angular velocity the head would have
            gained if the shaft applied no restoring couple, or ``None``.
        free_face_rotation_rad: Rotation from double-integrating that rate, or
            ``None``. An **upper bound** on face opening: a gripped club is
            restrained, so the real rotation is smaller.
    """

    shaft_axis_moment_Nm: np.ndarray
    travel_axis_moment_Nm: np.ndarray
    loft_axis_moment_Nm: np.ndarray
    peak_shaft_axis_moment_Nm: float
    mean_shaft_axis_moment_Nm: float
    shaft_axis_angular_impulse_Nms: float
    peak_travel_axis_moment_Nm: float
    peak_loft_axis_moment_Nm: float
    peak_resultant_moment_Nm: float
    shaft_axis_inertia_kg_m2: float | None
    free_face_rate_radps: float | None
    free_face_rotation_rad: float | None

    @property
    def free_face_rotation_deg(self) -> float | None:
        """Free-head face rotation in degrees, or ``None`` when unknown."""
        if self.free_face_rotation_rad is None:
            return None
        return float(np.degrees(self.free_face_rotation_rad))


def _signed_peak(values: np.ndarray) -> float:
    """Return the largest-magnitude element of ``values``, keeping its sign."""
    return float(values[int(np.argmax(np.abs(values)))])


def _free_face_response(
    times: np.ndarray, moment: np.ndarray, inertia_kg_m2: float
) -> tuple[float, float]:
    """Return the free-head shaft-axis rate change and rotation.

    Args:
        times: ``(T,)`` sample times [s].
        moment: ``(T,)`` shaft-axis moment [N.m].
        inertia_kg_m2: Moment of inertia about the shaft axis through the CG.

    Returns:
        ``(rate_radps, rotation_rad)``.
    """
    rate = cumulative_trapezoid(moment, times, initial=0) / inertia_kg_m2  # type: ignore[call-overload]
    rotation = float(np.trapezoid(rate, times))
    return float(rate[-1]), rotation


def head_twist_metrics(
    trace: StrikeTrace,
    head: HeadModel,
    scene: StrikeScene,
    *,
    window: slice | None = None,
) -> HeadTwistMetrics:
    """Resolve the sand moment onto the shaft, travel and loft axes.

    Args:
        trace: Strike trace.
        head: Head the trace was recorded for.
        scene: Scene supplying the travel axis.
        window: Optional sample window; defaults to the whole trace.

    Returns:
        The twist metrics. The free-head rotation fields are ``None`` unless the
        head carries an inertia tensor -- a rotation is not guessed from a
        default inertia.

    Raises:
        ValueError: If the window holds fewer than two samples or the twist
            triad is degenerate.
    """
    selection = _resolve_window(trace, window)
    times = trace.time_s[selection]
    moment = centre_of_mass_moment_Nm(trace, head)[selection]
    axes = shaft_travel_loft_axes(trace, head, scene)
    shaft, travel, loft = (axes[0][selection], axes[1][selection], axes[2][selection])
    shaft_moment = np.einsum(
        "ij,ij->i", moment, shaft
    )  # ⚡ Bolt: np.einsum avoids temporary arrays and is ~2.5x faster than np.sum(a * b, axis=1)
    travel_moment = np.einsum(
        "ij,ij->i", moment, travel
    )  # ⚡ Bolt: np.einsum avoids temporary arrays and is ~2.5x faster than np.sum(a * b, axis=1)
    loft_moment = np.einsum(
        "ij,ij->i", moment, loft
    )  # ⚡ Bolt: np.einsum avoids temporary arrays and is ~2.5x faster than np.sum(a * b, axis=1)
    duration_s = float(times[-1] - times[0])
    angular_impulse = float(np.trapezoid(shaft_moment, times))
    inertia: float | None = None
    rate: float | None = None
    rotation: float | None = None
    if head.inertia_body_kg_m2 is not None:
        inertia = head.shaft_axis_moment_of_inertia()
        require(inertia > 0.0, "shaft-axis inertia must be positive", value=inertia)
        rate, rotation = _free_face_response(times, shaft_moment, inertia)
    return HeadTwistMetrics(
        shaft_axis_moment_Nm=shaft_moment,
        travel_axis_moment_Nm=travel_moment,
        loft_axis_moment_Nm=loft_moment,
        peak_shaft_axis_moment_Nm=_signed_peak(shaft_moment),
        mean_shaft_axis_moment_Nm=angular_impulse / duration_s,
        shaft_axis_angular_impulse_Nms=angular_impulse,
        peak_travel_axis_moment_Nm=_signed_peak(travel_moment),
        peak_loft_axis_moment_Nm=_signed_peak(loft_moment),
        peak_resultant_moment_Nm=float(
            np.sqrt(np.einsum("ij,ij->i", moment, moment).max())
        ),  # ⚡ Bolt: np.sqrt(np.einsum(...).max()) avoids intermediate allocations and square root calculations and is ~2x faster
        shaft_axis_inertia_kg_m2=inertia,
        free_face_rate_radps=rate,
        free_face_rotation_rad=rotation,
    )

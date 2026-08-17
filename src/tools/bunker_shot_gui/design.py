"""Designer-facing inputs for the BunkerShot3D workbench (issue #8618, W11).

This module is the vocabulary layer: it turns the things a wedge designer
actually types -- a grind preset, a marketed bounce angle, a sole width, a
playing condition, a delivery -- into the value objects ADR-0032 defines
(:class:`~bunkershot3d.geometry.wedge.WedgeGeometry`,
:class:`~bunkershot3d.sand.state.SandState`,
:class:`~bunkershot3d.domain.swing.SwingCondition`).

It imports **no GUI toolkit**, so every input can be built, validated and
tested without Qt, and the same objects can back a Tauri/React front end.

Two conventions are load-bearing and are chosen here, once:

* **Bounce is entered in the marketed convention.** ADR-0032 structural
  decision 2 forbids mixing the patent's geometric bounce (measured to the
  true trailing contact point, >20 deg) with marketed bounce (measured to
  the ground-contact plane, 4-14 deg). A designer thinks in the marketed
  number, so that is what :class:`WedgeDesign` carries, and the geometric
  angle is *always* re-derived from it with
  :func:`~bunkershot3d.geometry.bounce.geometric_from_marketed`. Changing
  the sole width therefore keeps the marketed bounce the designer asked for
  rather than silently moving it.
* **Attack angle is negative for a descending blow**, matching
  :class:`~bunkershot3d.geometry.delivery.DeliveryCondition` and the -2 to
  -12 deg tour delivery band.
"""

from __future__ import annotations

import dataclasses
import math
from dataclasses import dataclass

from bunkershot3d.domain.swing import SwingCondition
from bunkershot3d.geometry import (
    DeliveryCondition,
    MarketedBounce,
    WedgeGeometry,
    geometric_from_marketed,
    get_preset,
    preset_names,
)
from bunkershot3d.sand import (
    PlayingCondition,
    SandState,
    playing_condition,
    usga_reference_sand,
)

__all__ = [
    "DEFAULT_GRIND_PRESET",
    "FIRMNESS_RANGE_KG_PER_CM2",
    "SandCondition",
    "SolverSetup",
    "SwingSetup",
    "WedgeDesign",
    "WorkbenchInputError",
    "grind_preset_names",
    "playing_condition_names",
]

DEFAULT_GRIND_PRESET = "sm9_58_m"
"""The 58 deg crescent-sole preset: the archetypal greenside bunker wedge."""

FIRMNESS_RANGE_KG_PER_CM2 = (1.6, 2.8)
"""The published penetrometer sweep, loose to dense (USGA firmness bands)."""

_MIN_STATIONS = 5
"""Mirrors ``bunkershot3d.geometry.lofting``: fewer cannot loft a head."""

_MIN_PROFILE_POINTS = 12
"""Mirrors ``bunkershot3d.geometry.profile``: fewer cannot sample a sole."""


class WorkbenchInputError(ValueError):
    """A designer input does not describe a constructible wedge or shot.

    Raised instead of letting the underlying value object's ``ValueError``
    escape unlabelled, so the caller can tell "you asked for an impossible
    sole" apart from "the solver refused this query" -- two very different
    things that must never be shown to a designer with the same wording.
    """


def grind_preset_names() -> tuple[str, ...]:
    """Every named grind preset, sorted.

    Returns:
        The preset names accepted by :attr:`WedgeDesign.grind_preset`.
    """
    return preset_names()


def playing_condition_names() -> tuple[str, ...]:
    """Every named playing condition, in the package's declaration order.

    Returns:
        The condition names accepted by :attr:`SandCondition.preset`.
    """
    return tuple(condition.value for condition in PlayingCondition)


@dataclass(frozen=True, slots=True)
class WedgeDesign:
    """One candidate sole, expressed in the W2 design vocabulary.

    A design starts from a named grind preset and overrides the parameters a
    designer moves. ``None`` means "keep the preset's value", so a design is
    always a complete, provenance-carrying geometry rather than a partially
    specified one.

    Attributes:
        name: Label used in the A/B comparison. Must be non-empty.
        grind_preset: Name of the starting preset.
        loft_deg: Static loft, or ``None`` for the preset's.
        marketed_bounce_deg: Bounce in the **marketed** convention, or
            ``None`` to keep the preset's marketed bounce.
        sole_width_mm: Patent ``d1``, leading edge to trailing contact.
        entry_height_mm: Patent ``d3``, the sole drop over the 1.2 mm datum.
        leading_edge_radius_mm: Patent ``rho1``.
        camber_area_mm2: Area between the sole and the LE/TC chord.
        heel_relief_fraction: Share of the sole width relieved at the heel.
        toe_relief_fraction: Share of the sole width relieved at the toe.
    """

    name: str
    grind_preset: str = DEFAULT_GRIND_PRESET
    loft_deg: float | None = None
    marketed_bounce_deg: float | None = None
    sole_width_mm: float | None = None
    entry_height_mm: float | None = None
    leading_edge_radius_mm: float | None = None
    camber_area_mm2: float | None = None
    heel_relief_fraction: float | None = None
    toe_relief_fraction: float | None = None

    def __post_init__(self) -> None:
        """Validate the label and the preset name.

        Raises:
            WorkbenchInputError: If the name is blank or the preset is
                unknown. Both are typing mistakes, not physics.
        """
        if not str(self.name).strip():
            raise WorkbenchInputError(
                "a design needs a non-empty name; the A/B comparison ranks "
                "designs by name and an unnamed column cannot be reported"
            )
        if self.grind_preset not in grind_preset_names():
            raise WorkbenchInputError(
                f"unknown grind preset {self.grind_preset!r}; available: "
                + ", ".join(grind_preset_names())
            )

    def geometry(self) -> WedgeGeometry:
        """Resolve the design into a full parametric wedge geometry.

        The geometric (patent) bounce is re-derived from the marketed bounce
        every time, so the two conventions can never drift apart.

        Returns:
            The resolved design vector.

        Raises:
            WorkbenchInputError: If the combination is not a constructible
                sole -- for example a relief fraction that moves the trailing
                contact point in front of the sole entry.
        """
        base = get_preset(self.grind_preset).geometry
        sole_width_m = _resolve_mm(self.sole_width_mm, base.sole_width_m)
        entry_height_m = _resolve_mm(self.entry_height_mm, base.entry_height_m)
        marketed_deg = (
            base.marketed_bounce.angle_deg
            if self.marketed_bounce_deg is None
            else float(self.marketed_bounce_deg)
        )
        try:
            geometric = geometric_from_marketed(
                MarketedBounce(marketed_deg),
                sole_width_m=sole_width_m,
                entry_height_m=entry_height_m,
                datum_offset_m=base.datum_offset_m,
            )
            return dataclasses.replace(
                base,
                loft_deg=_resolve(self.loft_deg, base.loft_deg),
                geometric_bounce=geometric,
                sole_width_m=sole_width_m,
                entry_height_m=entry_height_m,
                leading_edge_radius_m=_resolve_mm(
                    self.leading_edge_radius_mm, base.leading_edge_radius_m
                ),
                sole_camber_area_m2=_resolve(
                    None
                    if self.camber_area_mm2 is None
                    else float(self.camber_area_mm2) * 1e-6,
                    base.sole_camber_area_m2,
                ),
                heel_relief_fraction=_resolve(
                    self.heel_relief_fraction, base.heel_relief_fraction
                ),
                toe_relief_fraction=_resolve(
                    self.toe_relief_fraction, base.toe_relief_fraction
                ),
            )
        except (ValueError, TypeError) as error:
            raise WorkbenchInputError(
                f"design {self.name!r} does not describe a constructible sole: {error}"
            ) from error


@dataclass(frozen=True, slots=True)
class SandCondition:
    """The W3 playing condition, with an optional firmness override.

    Attributes:
        preset: One of firm, fluffy, wet or plugged.
        firmness_kg_per_cm2: Penetrometer reading in the published unit, or
            ``None`` to keep the preset's. Overriding it holds the preset's
            gradation, moisture and grain shape fixed, which is exactly what
            the published firmness sweep isolates.
    """

    preset: PlayingCondition = PlayingCondition.FIRM
    firmness_kg_per_cm2: float | None = None

    def __post_init__(self) -> None:
        """Validate the condition.

        Raises:
            WorkbenchInputError: If the preset is unknown or the firmness is
                not a positive finite reading.
        """
        try:
            object.__setattr__(self, "preset", PlayingCondition(self.preset))
        except ValueError as error:
            raise WorkbenchInputError(
                f"unknown playing condition {self.preset!r}; expected one of "
                + ", ".join(playing_condition_names())
            ) from error
        if self.firmness_kg_per_cm2 is None:
            return
        value = float(self.firmness_kg_per_cm2)
        if not math.isfinite(value) or value <= 0.0:
            raise WorkbenchInputError(
                f"firmness must be a positive penetrometer reading in kg/cm^2, "
                f"got {self.firmness_kg_per_cm2!r}"
            )
        object.__setattr__(self, "firmness_kg_per_cm2", value)

    def sand_state(self) -> SandState:
        """Build the sand state this condition describes.

        Returns:
            The USGA-referenced state, carrying its provenance.

        Raises:
            WorkbenchInputError: If the firmness override does not produce a
                physically packable bed.
        """
        preset_state = playing_condition(self.preset)
        if self.firmness_kg_per_cm2 is None:
            return preset_state
        try:
            return usga_reference_sand(
                name=f"usga-{self.preset.value}-{self.firmness_kg_per_cm2:g}",
                firmness_kg_per_cm2=self.firmness_kg_per_cm2,
                gravimetric_water_content=(
                    preset_state.moisture.gravimetric_water_content
                ),
                angularity=preset_state.angularity,
                psd=preset_state.psd,
                bed=preset_state.bed,
            )
        except ValueError as error:
            raise WorkbenchInputError(
                f"a firmness of {self.firmness_kg_per_cm2:g} kg/cm^2 does not "
                f"describe a packable bed: {error}"
            ) from error

    def with_firmness(self, firmness_kg_per_cm2: float) -> SandCondition:
        """Return the same condition at a different penetrometer reading.

        Args:
            firmness_kg_per_cm2: The new reading, published unit.

        Returns:
            A new condition; the original is unchanged.
        """
        return SandCondition(
            preset=self.preset, firmness_kg_per_cm2=float(firmness_kg_per_cm2)
        )


@dataclass(frozen=True, slots=True)
class SwingSetup:
    """How the head is delivered, plus where the ball sits.

    Attributes:
        clubhead_speed_mps: Speed at impact. Greenside delivery is 20-27 m/s,
            which is well above the 6.8 m/s depth/inertia crossover.
        attack_angle_deg: Club-path angle to the horizontal, **negative for a
            descending blow** (tour: -2 to -12 deg).
        face_open_deg: Rotation about the shaft axis; positive opens.
        shaft_lean_deg: Forward lean; positive de-lofts.
        entry_distance_behind_ball_m: Where the sole crosses the surface,
            measured behind the ball. Wivou et al. (2016) measured
            0.080-0.280 m; the registered sweep range is 0.025-0.150 m.
        ball_depth_m: How far the ball centre sits below the sand surface.
            Positive is buried, negative is sitting up.
        dynamic_terms_active: Whether the DRFT inertial term is applied.
            Switching it off yields quasi-static RFT, which the envelope then
            **refuses** above ``Fr ~ 1`` -- the honest behaviour, and the one
            path in this tool that produces a refusal at normal speeds.
    """

    clubhead_speed_mps: float = 25.0
    attack_angle_deg: float = -8.0
    face_open_deg: float = 10.0
    shaft_lean_deg: float = 6.0
    entry_distance_behind_ball_m: float = 0.080
    ball_depth_m: float = 0.005
    dynamic_terms_active: bool = True

    def __post_init__(self) -> None:
        """Validate the delivery.

        Raises:
            WorkbenchInputError: If any angle, speed or distance is outside
                the range its value object accepts.
        """
        if not math.isfinite(self.attack_angle_deg) or self.attack_angle_deg >= 0.0:
            raise WorkbenchInputError(
                "attack_angle_deg must be negative: a level or ascending head "
                "never enters the sand, so there is no bunker shot to solve. "
                f"Tour delivery is -2 to -12 deg; got {self.attack_angle_deg!r}"
            )
        if self.entry_distance_behind_ball_m <= 0.0 or not math.isfinite(
            self.entry_distance_behind_ball_m
        ):
            raise WorkbenchInputError(
                "entry_distance_behind_ball_m must be positive: the sole enters "
                "behind the ball in a splash shot, and a non-positive distance "
                f"describes a strike at or past it, got "
                f"{self.entry_distance_behind_ball_m!r}"
            )
        try:
            self.delivery()
            self.swing_condition()
        except ValueError as error:
            raise WorkbenchInputError(f"unusable swing condition: {error}") from error

    def delivery(self) -> DeliveryCondition:
        """The geometric half of the delivery.

        Returns:
            Face opening, shaft lean and attack angle.
        """
        return DeliveryCondition(
            face_open_deg=float(self.face_open_deg),
            shaft_lean_deg=float(self.shaft_lean_deg),
            attack_angle_deg=float(self.attack_angle_deg),
        )

    def swing_condition(self, duration_s: float = 0.030) -> SwingCondition:
        """The full kinematic delivery, speed included.

        Args:
            duration_s: Simulated time the shot covers.

        Returns:
            The swing condition.
        """
        return SwingCondition(
            clubhead_speed_mps=float(self.clubhead_speed_mps),
            duration_s=float(duration_s),
            delivery=self.delivery(),
        )

    def with_attack_angle(self, attack_angle_deg: float) -> SwingSetup:
        """Return the same delivery at a different attack angle.

        Args:
            attack_angle_deg: The new angle, negative for a descending blow.

        Returns:
            A new setup; the original is unchanged.
        """
        return dataclasses.replace(self, attack_angle_deg=float(attack_angle_deg))


@dataclass(frozen=True, slots=True)
class SolverSetup:
    """Discretisation, integration and study settings.

    These are not physics: they are how finely the tool resolves it, and they
    are exposed because the surface discretisation length feeds the validity
    envelope directly. Refining the mesh *raises* Askari and Kamrin's ``I_G``,
    so a finer mesh is not automatically a better answer.

    Attributes:
        n_profile_points: Sole samples per lofted cross-section.
        n_stations: Cross-sections from heel to toe.
        time_step_s: Fixed integration step.
        max_time_s: Hard stop for one shot, lead-in included. A nominal
            wedge strike is over in 12-18 ms, but a low-bounce grind digs
            for well over 100 ms before its sole comes back out, and a
            window that ends first is a truncated shot the solver refuses
            to hand on (issue #8700). It costs nothing when the shot is
            ordinary: the march stops at the exit, not at the wall.
        target_carry_m: Carry the playability window is measured against.
        carry_tolerance_fraction: Half-width of the acceptance band.
        playability_points: Stations per playability axis; the grid costs
            ``playability_points ** 2`` shots.
        flight_time_step_s: Ball-flight integration step.
    """

    n_profile_points: int = 24
    n_stations: int = 11
    time_step_s: float = 2.5e-4
    max_time_s: float = 0.200
    target_carry_m: float = 12.0
    carry_tolerance_fraction: float = 0.10
    playability_points: int = 5
    flight_time_step_s: float = 0.005

    def __post_init__(self) -> None:
        """Validate the settings.

        Raises:
            WorkbenchInputError: If a resolution is too coarse to represent a
                sole, a time is non-positive, or the tolerance is outside
                ``(0, 1]``.
        """
        if self.n_stations < _MIN_STATIONS:
            raise WorkbenchInputError(
                f"n_stations must be at least {_MIN_STATIONS}, got {self.n_stations}"
            )
        if self.n_profile_points < _MIN_PROFILE_POINTS:
            raise WorkbenchInputError(
                f"n_profile_points must be at least {_MIN_PROFILE_POINTS}, got "
                f"{self.n_profile_points}"
            )
        for label, value in (
            ("time_step_s", self.time_step_s),
            ("max_time_s", self.max_time_s),
            ("target_carry_m", self.target_carry_m),
            ("flight_time_step_s", self.flight_time_step_s),
        ):
            if not math.isfinite(value) or value <= 0.0:
                raise WorkbenchInputError(f"{label} must be positive, got {value!r}")
        if self.time_step_s > self.max_time_s:
            raise WorkbenchInputError(
                f"time_step_s {self.time_step_s} exceeds max_time_s {self.max_time_s}"
            )
        if not 0.0 < self.carry_tolerance_fraction <= 1.0:
            raise WorkbenchInputError(
                "carry_tolerance_fraction must lie in (0, 1], got "
                f"{self.carry_tolerance_fraction!r}"
            )
        if self.playability_points < 2:
            raise WorkbenchInputError(
                "a playability axis needs at least 2 stations, got "
                f"{self.playability_points}"
            )


def _resolve(override: float | None, fallback: float) -> float:
    """Return the override when given, otherwise the preset's value."""
    return fallback if override is None else float(override)


def _resolve_mm(override_mm: float | None, fallback_m: float) -> float:
    """Return an override quoted in millimetres, in metres."""
    return fallback_m if override_mm is None else float(override_mm) * 1e-3

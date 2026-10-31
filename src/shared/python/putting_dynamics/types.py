"""Core dataclasses for the putting-dynamics package (#8345, P2+P3).

Reused shared infrastructure (AGENTS.md section A discovery):

* DbC helpers from ``src.shared.python.contracts``.
* Golf-ball constants from ``src.shared.python.core.physics_constants``
  (USGA Rule 5 values; single source, not re-declared here).
* Layout mirrors ``src.shared.python.physics.impact_model``
  (types / solver / utils split with a curated façade).

All quantities are SI internally; display conversions (yards default
per fleet direction) live in :mod:`.utils` and are applied only at the
edges.

Frame convention: ``x`` along the initial putt line, ``y`` to the
left, ``z`` up.  ``spin_rad_s`` is about the transverse (left) axis
with topspin positive, matching Tools ``swing_sim.putting`` (restated
convention; a freshly struck putt has backspin, negative).
"""

from __future__ import annotations

import enum
import math
from dataclasses import dataclass

from src.shared.python.contracts import require, require_finite

__all__ = [
    "BallState",
    "CollisionReport",
    "Mode",
    "PuttResult",
    "PutterState",
    "TrajectorySample",
]


class Mode(enum.Enum):
    """Ball motion mode used by the solver's mode machine."""

    AIRBORNE = "airborne"
    SLIDE = "slide"
    ROLL = "roll"
    REST = "rest"


@dataclass(frozen=True)
class BallState:
    """Instantaneous ball state.

    Attributes:
        x_m: Position along the putt line [m].
        y_m: Lateral position, left positive [m].
        height_m: Height above the local surface [m]; 0 on-surface.
        vx_mps: Velocity along x [m/s].
        vy_mps: Velocity along y [m/s].
        vz_mps: Vertical velocity [m/s]; nonzero only while airborne.
        spin_rad_s: Spin about the transverse axis [rad/s], topspin
            positive.
    """

    x_m: float
    y_m: float
    height_m: float = 0.0
    vx_mps: float = 0.0
    vy_mps: float = 0.0
    vz_mps: float = 0.0
    spin_rad_s: float = 0.0

    def __post_init__(self) -> None:
        for name in ("x_m", "y_m", "height_m", "vx_mps", "vy_mps", "vz_mps"):
            require_finite(getattr(self, name), name)
        require_finite(self.spin_rad_s, "spin_rad_s")
        require(self.height_m >= 0.0, "height must be >= 0", self.height_m)
        require(
            abs(self.x_m) <= 200.0 and abs(self.y_m) <= 200.0,
            "position must be within +/- 200 m",
            (self.x_m, self.y_m),
        )
        require(
            math.hypot(self.vx_mps, self.vy_mps) <= 20.0,
            "putt speeds must be <= 20 m/s",
            (self.vx_mps, self.vy_mps),
        )

    @property
    def speed_mps(self) -> float:
        """Horizontal speed magnitude [m/s]."""
        return math.hypot(self.vx_mps, self.vy_mps)


#: Head of :meth:`PutterState.typical` (uncited engineering defaults: a
#: common blade/mallet mass, MOI and loft, not a measured putter).
TYPICAL_HEAD_MASS_KG = 0.35
TYPICAL_HEAD_MOI_KG_M2 = 4.5e-4
TYPICAL_LOFT_DEG = 3.0


@dataclass(frozen=True)
class PutterState:
    """Putter head at impact.

    The shaft attachment (hosel) is freely positionable on the head's
    top surface relative to the head CG: ``hosel_toe_m`` (heel
    negative, toe positive) and ``hosel_forward_m`` (toward the target
    positive).  An impact impulse that is not collinear with the
    attachment/CG line produces a twisting moment about the shaft axis
    — the face-balanced vs toe-hang spectrum; see
    :func:`~.collision.strike`.

    Attributes:
        head_mass_kg: Head mass [kg]; putters span ~0.30-0.40.
        moi_kg_m2: Scalar head moment of inertia about the vertical
            (shaft-parallel) axis through the CG [kg m^2]; typical
            putters are ~3e-4 to 6e-4 (0.3-0.6 g m^2 in catalogue
            units). Used for the off-center effective mass; the
            shaft-axis twist proxy applies the parallel-axis theorem
            using the hosel offsets.
        loft_deg: Dynamic loft presented at impact [deg]; the sweep
            helper spans -4 to +8.
        speed_mps: Head speed at impact along the putt line [m/s].
        cor: Face coefficient of restitution at putt speeds.
        hosel_toe_m: Shaft-attachment offset toward the toe [m]
            (negative = heel-side, anser-style).
        hosel_forward_m: Shaft-attachment offset toward the target [m]
            (negative = behind the CG).
    """

    head_mass_kg: float
    moi_kg_m2: float
    loft_deg: float
    speed_mps: float
    cor: float = 0.78
    hosel_toe_m: float = 0.0
    hosel_forward_m: float = 0.0

    def __post_init__(self) -> None:
        require_finite(self.head_mass_kg, "head_mass_kg")
        require(
            0.1 <= self.head_mass_kg <= 1.0,
            "head mass must be plausible [kg]",
            self.head_mass_kg,
        )
        require_finite(self.moi_kg_m2, "moi_kg_m2")
        require(
            1e-5 <= self.moi_kg_m2 <= 1e-2,
            "head MOI must be plausible [kg m^2]",
            self.moi_kg_m2,
        )
        require_finite(self.loft_deg, "loft_deg")
        require(
            -6.0 <= self.loft_deg <= 10.0,
            "dynamic loft must be in [-6, 10] deg",
            self.loft_deg,
        )
        require_finite(self.speed_mps, "speed_mps")
        require(
            0.0 < self.speed_mps <= 10.0,
            "head speed must be in (0, 10] m/s",
            self.speed_mps,
        )
        require_finite(self.cor, "cor")
        require(0.0 < self.cor < 1.0, "COR must be in (0, 1)", self.cor)
        require_finite(self.hosel_toe_m, "hosel_toe_m")
        require(
            abs(self.hosel_toe_m) <= 0.08,
            "hosel toe offset within +/- 0.08 m",
            self.hosel_toe_m,
        )
        require_finite(self.hosel_forward_m, "hosel_forward_m")
        require(
            abs(self.hosel_forward_m) <= 0.05,
            "hosel forward offset within +/- 0.05 m",
            self.hosel_forward_m,
        )

    @classmethod
    def typical(cls, speed_mps: float) -> PutterState:
        """A typical putter head swung at ``speed_mps``, for callers that
        have only a stroke speed (API routes, GUI scene builder; #14141 7d).
        """
        return cls(
            head_mass_kg=TYPICAL_HEAD_MASS_KG,
            moi_kg_m2=TYPICAL_HEAD_MOI_KG_M2,
            loft_deg=TYPICAL_LOFT_DEG,
            speed_mps=speed_mps,
        )


@dataclass(frozen=True)
class CollisionReport:
    """Everything the P1 visualization needs from one impact.

    The attachment-point *impulse wrench* (force impulse + moment
    impulse, both at the shaft attachment) is the documented seam for
    the upcoming finite-interval twist-dynamics epic: a richer model
    can consume the same wrench without changing this report's shape.

    Attributes:
        ball_speed_mps: Ball launch speed magnitude [m/s].
        launch_angle_deg: Launch angle above horizontal [deg].
        horizontal_speed_mps: Ground-plane launch speed [m/s].
        vertical_speed_mps: Upward launch speed [m/s].
        spin_rad_s: Launch spin, topspin positive [rad/s].
        effective_loft_deg: Dynamic loft presented at impact [deg].
        putter_dv_mps: Putter-head slowdown along its travel [m/s].
        impulse_n_s: Normal contact impulse magnitude [N s].
        contact_time_proxy_s: Contact-duration proxy [s] (fixed
            sub-millisecond scale for visualization pacing).
        kinetic_energy_loss_j: Impact energy dissipated [J].
        face_twist_rad_s: Pinned-shaft face-twist velocity proxy
            [rad/s], computed from the moment impulse and the
            parallel-axis MOI about the shaft; positive rotates the
            toe backward (face opening for a strike toe-side of the
            attachment). The finite-interval model will replace this
            proxy with its boundary-condition-specific trajectory.
        twist_moment_n_m_s: Moment impulse about the shaft axis
            [N m s].
        attachment_impulse_n_s: Impulse wrench force part at the
            attachment, ``(x, y, z)`` [N s] (reaction on the putter).
        attachment_moment_n_m_s: Impulse wrench moment part at the
            attachment, ``(x, y, z)`` [N m s].
    """

    ball_speed_mps: float
    launch_angle_deg: float
    horizontal_speed_mps: float
    vertical_speed_mps: float
    spin_rad_s: float
    effective_loft_deg: float
    putter_dv_mps: float
    impulse_n_s: float
    contact_time_proxy_s: float
    kinetic_energy_loss_j: float
    face_twist_rad_s: float
    twist_moment_n_m_s: float
    attachment_impulse_n_s: tuple[float, float, float]
    attachment_moment_n_m_s: tuple[float, float, float]

    def __post_init__(self) -> None:
        """Cheap invariants on the impact report (characterization).

        Postconditions: every field is finite; the launch and
        putter-slowdown speeds and both dissipation/impulse magnitudes
        are non-negative; the attachment wrench components are 3-tuples
        of finite floats.
        """
        for name in (
            "ball_speed_mps",
            "launch_angle_deg",
            "horizontal_speed_mps",
            "vertical_speed_mps",
            "spin_rad_s",
            "effective_loft_deg",
            "putter_dv_mps",
            "impulse_n_s",
            "contact_time_proxy_s",
            "kinetic_energy_loss_j",
            "face_twist_rad_s",
            "twist_moment_n_m_s",
        ):
            require_finite(getattr(self, name), name)
        require(
            self.ball_speed_mps >= 0.0, "ball speed must be >= 0", self.ball_speed_mps
        )
        require(self.putter_dv_mps >= 0.0, "putter_dv must be >= 0", self.putter_dv_mps)
        require(self.impulse_n_s >= 0.0, "impulse must be >= 0", self.impulse_n_s)
        require(
            self.kinetic_energy_loss_j >= 0.0,
            "energy loss must be >= 0",
            self.kinetic_energy_loss_j,
        )
        for name in ("attachment_impulse_n_s", "attachment_moment_n_m_s"):
            wrench = getattr(self, name)
            require(len(wrench) == 3, f"{name} must have 3 components", wrench)
            for component in wrench:
                require_finite(component, name)


@dataclass(frozen=True)
class TrajectorySample:
    """One solver output sample.

    Attributes:
        t_s: Sample time [s].
        x_m: Position along the putt line [m].
        y_m: Lateral position, left positive [m].
        height_m: Height above the local surface [m].
        speed_mps: Horizontal speed [m/s].
        spin_rad_s: Spin, topspin positive [rad/s].
        mode: Motion mode at the sample.
    """

    t_s: float
    x_m: float
    y_m: float
    height_m: float
    speed_mps: float
    spin_rad_s: float
    mode: Mode

    def __post_init__(self) -> None:
        """Cheap invariants on one solver sample (characterization).

        Postconditions: all numeric fields are finite; sample time and
        height are non-negative; ground-plane speed is non-negative;
        ``mode`` is a :class:`Mode` member.
        """
        for name in ("t_s", "x_m", "y_m", "height_m", "speed_mps", "spin_rad_s"):
            require_finite(getattr(self, name), name)
        require(self.t_s >= 0.0, "sample time must be >= 0", self.t_s)
        require(self.height_m >= 0.0, "height must be >= 0", self.height_m)
        require(self.speed_mps >= 0.0, "speed must be >= 0", self.speed_mps)
        require(isinstance(self.mode, Mode), "mode must be a Mode member", self.mode)


@dataclass(frozen=True)
class PuttResult:
    """One integrated putt.

    Attributes:
        samples: Time-ordered trajectory samples.
        collision: Impact report when the putt started from a strike;
            None when integrated from a given ball state.
        holed: Whether the ball was captured.
        rest_x_m: Final x position [m].
        rest_y_m: Final y position [m].
        total_distance_m: Ground-path length [m].
        time_s: Time to rest or capture [s].
        skid_distance_m: Ground covered before pure roll [m].
        speed_at_hole_mps: Speed when first over the hole mouth [m/s];
            None when the ball never crossed it.
    """

    samples: tuple[TrajectorySample, ...]
    collision: CollisionReport | None
    holed: bool
    rest_x_m: float
    rest_y_m: float
    total_distance_m: float
    time_s: float
    skid_distance_m: float
    speed_at_hole_mps: float | None

    def __post_init__(self) -> None:
        """Cheap invariants on one integrated putt (characterization).

        Postconditions: at least one sample exists; the rest position,
        total distance, elapsed time and skid distance are finite; the
        distances and time are non-negative; skid never exceeds the
        total path length; a reported hole-crossing speed is finite
        and non-negative.
        """
        require(len(self.samples) > 0, "a putt must have at least one sample")
        for name in (
            "rest_x_m",
            "rest_y_m",
            "total_distance_m",
            "time_s",
            "skid_distance_m",
        ):
            require_finite(getattr(self, name), name)
        require(
            self.total_distance_m >= 0.0, "distance must be >= 0", self.total_distance_m
        )
        require(self.time_s >= 0.0, "time must be >= 0", self.time_s)
        require(
            self.skid_distance_m <= self.total_distance_m + 1e-9,
            "skid cannot exceed the total path",
            (self.skid_distance_m, self.total_distance_m),
        )
        if self.speed_at_hole_mps is not None:
            require_finite(self.speed_at_hole_mps, "speed_at_hole_mps")
            require(
                self.speed_at_hole_mps >= 0.0,
                "speed_at_hole must be >= 0",
                self.speed_at_hole_mps,
            )

    @property
    def final_mode(self) -> Mode:
        """Mode of the last sample (REST when the ball stopped)."""
        return self.samples[-1].mode

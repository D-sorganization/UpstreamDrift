"""How the head is delivered, kinematically (issue #8608, ADR-0032).

The *geometric* half of a delivery -- face open, shaft lean, attack angle --
was built by #8609 as
:class:`~bunkershot3d.geometry.delivery.DeliveryCondition`, together with the
closed-form effective loft/bounce/aim it implies. :class:`SwingCondition`
composes that object rather than restating its three angles, and adds the one
thing it does not carry: how fast the head is moving.

Speed is not a detail. ADR-0032 shows the DRFT depth term and the inertial term
crossing at 6.8 m/s, with a greenside bunker shot delivered at 20-27 m/s, so
the inertial term carries roughly 90 % of the load. The existing code's
hard-coded 5 m/s sat *below* the crossover, in a regime with the wrong dominant
physics; :attr:`SwingCondition.is_inertially_dominated` makes that visible
instead of implicit.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from ..exceptions import DomainInvariantError
from ..geometry.delivery import DeliveryCondition
from ._validate import require_positive

__all__ = [
    "DRFT_INERTIAL_CROSSOVER_MPS",
    "MAX_PLAUSIBLE_CLUBHEAD_SPEED_MPS",
    "SwingCondition",
    "TrajectorySource",
]

#: Speed at which the DRFT inertial term ``lambda rho v_n^2`` overtakes the
#: depth term ``alpha |z|`` for medium sand at a 40 mm divot depth (ADR-0032,
#: with alpha_z ~ 2.02 N/cm^3, lambda ~ 1.1, rho ~ 1600 kg/m^3).
DRFT_INERTIAL_CROSSOVER_MPS = 6.8

#: Refusal threshold for a clubhead speed. Long-drive competitors reach about
#: 65 m/s; anything past 100 m/s is a unit error (mph read as m/s), not a swing.
MAX_PLAUSIBLE_CLUBHEAD_SPEED_MPS = 100.0


@dataclass(frozen=True, slots=True)
class TrajectorySource:
    """Where a prescribed swing's kinematics are read from.

    Kept separate from :class:`SwingCondition`: this is bookkeeping about a
    file, not physics, and it is the only part of the swing the legacy
    configuration schema actually carries.

    Attributes:
        file: Path to the trajectory CSV, as authored in the configuration.
        duration_s: Simulated time the run should cover.
    """

    file: str
    duration_s: float

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            DomainInvariantError: The file name is blank or the duration is not
                a positive finite time.
        """
        text = str(self.file).strip()
        if not text:
            raise DomainInvariantError(
                "file must name a trajectory source; got a blank string. A "
                "missing swing is an error, not a 5 m/s default (defect B33)."
            )
        object.__setattr__(self, "file", text)
        object.__setattr__(
            self, "duration_s", require_positive(self.duration_s, "duration_s")
        )


@dataclass(frozen=True, slots=True)
class SwingCondition:
    """The kinematic delivery of the head at impact.

    Attributes:
        clubhead_speed_mps: Head speed at impact.
        duration_s: Simulated time the shot covers.
        delivery: The geometric delivery (#8609). Defaults to square and level,
            which is a documented neutral reference, not a measurement.
    """

    clubhead_speed_mps: float
    duration_s: float
    delivery: DeliveryCondition = field(default_factory=DeliveryCondition)

    def __post_init__(self) -> None:
        """Validate.

        Raises:
            DomainInvariantError: The speed or duration is non-positive or
                non-finite, or the speed is physically implausible.
        """
        speed = require_positive(self.clubhead_speed_mps, "clubhead_speed_mps")
        if speed > MAX_PLAUSIBLE_CLUBHEAD_SPEED_MPS:
            raise DomainInvariantError(
                f"clubhead_speed_mps of {speed!r} m/s exceeds the "
                f"{MAX_PLAUSIBLE_CLUBHEAD_SPEED_MPS} m/s plausibility ceiling; "
                "this is usually miles per hour read as metres per second."
            )
        object.__setattr__(self, "clubhead_speed_mps", speed)
        object.__setattr__(
            self, "duration_s", require_positive(self.duration_s, "duration_s")
        )

    @property
    def is_inertially_dominated(self) -> bool:
        """True when the DRFT inertial term outweighs the depth term.

        Below :data:`DRFT_INERTIAL_CROSSOVER_MPS` a quasi-static solver is at
        least arguable; above it, one is wrong by an order of magnitude.
        """
        return self.clubhead_speed_mps > DRFT_INERTIAL_CROSSOVER_MPS

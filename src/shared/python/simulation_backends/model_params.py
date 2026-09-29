"""Single source of truth for the golf double-pendulum model parameters.

``GolfModelParams`` is a typed, validated (:mod:`pydantic`) description of the
planar driven double pendulum (shoulder + wrist/club). It is the *one* model
from which every backend is derived:

* :meth:`GolfModelParams.to_double_pendulum_parameters` renders the analytical
  EOM parameters (the existing :class:`DoublePendulumParameters` dataclass).
* :func:`simulation_backends.mjcf.params_to_mjcf` renders the MuJoCo MJCF XML.

Because both renderers consume the *same* instance — and the default constants
are imported from the analytical module rather than duplicated — the two
derivations cannot silently drift apart. A regression test asserts that
perturbing a parameter changes both outputs (epic task M2.3).

Units are SI and annotated on every field.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING

import yaml
from pydantic import BaseModel, ConfigDict, Field, field_validator

if TYPE_CHECKING:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DoublePendulumParameters,
    )


def _default_plane_inclination() -> float:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DEFAULT_PLANE_INCLINATION_DEG,
    )

    return DEFAULT_PLANE_INCLINATION_DEG


def _default_damping_shoulder() -> float:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DEFAULT_DAMPING_SHOULDER,
    )

    return DEFAULT_DAMPING_SHOULDER


def _default_damping_wrist() -> float:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        DEFAULT_DAMPING_WRIST,
    )

    return DEFAULT_DAMPING_WRIST


def _default_gravity() -> float:
    from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
        GRAVITATIONAL_ACCELERATION,
    )

    return float(GRAVITATIONAL_ACCELERATION)


class UpperSegmentParams(BaseModel):
    """Upper segment (combined arms) — modelled as a single rigid link.

    Attributes:
        length_m: Segment length from shoulder to wrist [m].
        mass_kg: Segment mass [kg].
        center_of_mass_ratio: COM distance as a fraction of length, in ``(0, 1]``.
        inertia_about_com_kg_m2: Rotational inertia about the COM [kg*m^2]. If
            ``None``, a uniform-rod value ``(1/12) m L^2`` is used.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    length_m: float = Field(gt=0.0)
    mass_kg: float = Field(gt=0.0)
    center_of_mass_ratio: float = Field(gt=0.0, le=1.0)
    inertia_about_com_kg_m2: float | None = Field(default=None, ge=0.0)

    @property
    def effective_inertia_about_com(self) -> float:
        """Resolved inertia about COM, defaulting to a uniform rod."""
        if self.inertia_about_com_kg_m2 is not None:
            return self.inertia_about_com_kg_m2
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
            DEFAULT_ARM_INERTIA_SCALING,
        )

        return DEFAULT_ARM_INERTIA_SCALING * self.mass_kg * self.length_m**2


class LowerSegmentParams(BaseModel):
    """Lower segment (golf club) — composite shaft + clubhead point mass.

    Attributes:
        length_m: Grip-to-clubhead length [m].
        shaft_mass_kg: Distributed shaft + grip mass [kg].
        clubhead_mass_kg: Clubhead point mass at the distal end [kg].
        shaft_com_ratio: Shaft COM as a fraction of length, in ``(0, 1]``.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    length_m: float = Field(gt=0.0)
    shaft_mass_kg: float = Field(gt=0.0)
    clubhead_mass_kg: float = Field(gt=0.0)
    shaft_com_ratio: float = Field(gt=0.0, le=1.0)


class GolfModelParams(BaseModel):
    """Validated, immutable description of the golf double-pendulum mechanism.

    This is the canonical model definition; do not hand-edit derived
    representations (analytical params or MJCF) independently.

    Attributes:
        upper: Upper-segment (arms) properties.
        lower: Lower-segment (club) properties.
        plane_inclination_deg: Swing-plane tilt from vertical [deg], ``[-90, 90]``.
        damping_shoulder: Viscous damping at the shoulder joint [N*m*s/rad].
        damping_wrist: Viscous damping at the wrist joint [N*m*s/rad].
        gravity_m_s2: Gravitational acceleration magnitude [m/s^2].
        gravity_enabled: Whether gravity acts at all.
        constrained_to_plane: Whether gravity is projected onto the swing plane.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    upper: UpperSegmentParams
    lower: LowerSegmentParams
    plane_inclination_deg: float = Field(
        default_factory=_default_plane_inclination, ge=-90.0, le=90.0
    )
    damping_shoulder: float = Field(default_factory=_default_damping_shoulder, ge=0.0)
    damping_wrist: float = Field(default_factory=_default_damping_wrist, ge=0.0)
    gravity_m_s2: float = Field(default_factory=_default_gravity, gt=0.0)
    gravity_enabled: bool = True
    constrained_to_plane: bool = True

    @field_validator("plane_inclination_deg")
    @classmethod
    def _finite_inclination(cls, value: float) -> float:
        """Reject NaN/inf inclination (DbC precondition)."""
        if not math.isfinite(value):
            raise ValueError("plane_inclination_deg must be finite")
        return value

    @property
    def num_joints(self) -> int:
        """Number of actuated revolute joints (shoulder, wrist)."""
        return 2

    @property
    def state_dim(self) -> int:
        """Dimension of the in-plane state vector ``[q (2), v (2)]``."""
        return 4

    @property
    def projected_gravity(self) -> float:
        """Gravity magnitude after optional projection onto the swing plane.

        Mirrors :attr:`DoublePendulumParameters.projected_gravity` so the MJCF
        renderer and analytical model agree on the effective in-plane gravity.
        """
        if not self.gravity_enabled:
            return 0.0
        if not self.constrained_to_plane:
            return self.gravity_m_s2
        return self.gravity_m_s2 * math.cos(math.radians(self.plane_inclination_deg))

    @classmethod
    def default(cls) -> GolfModelParams:
        """Construct the canonical golf-swing defaults.

        Numerically identical to :meth:`DoublePendulumParameters.default` by
        construction (shares the same imported constants), guaranteeing the
        analytical and MJCF renderers start from the same baseline.
        """
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
            DEFAULT_ARM_CENTER_OF_MASS_RATIO,
            DEFAULT_ARM_INERTIA_SCALING,
            DEFAULT_ARM_LENGTH_M,
            DEFAULT_ARM_MASS_KG,
            DEFAULT_CLUBHEAD_MASS_KG,
            DEFAULT_SHAFT_COM_RATIO,
            DEFAULT_SHAFT_LENGTH_M,
            DEFAULT_SHAFT_MASS_KG,
        )

        return cls(
            upper=UpperSegmentParams(
                length_m=DEFAULT_ARM_LENGTH_M,
                mass_kg=DEFAULT_ARM_MASS_KG,
                center_of_mass_ratio=DEFAULT_ARM_CENTER_OF_MASS_RATIO,
                inertia_about_com_kg_m2=(
                    DEFAULT_ARM_INERTIA_SCALING
                    * DEFAULT_ARM_MASS_KG
                    * DEFAULT_ARM_LENGTH_M**2
                ),
            ),
            lower=LowerSegmentParams(
                length_m=DEFAULT_SHAFT_LENGTH_M,
                shaft_mass_kg=DEFAULT_SHAFT_MASS_KG,
                clubhead_mass_kg=DEFAULT_CLUBHEAD_MASS_KG,
                shaft_com_ratio=DEFAULT_SHAFT_COM_RATIO,
            ),
        )

    def to_double_pendulum_parameters(self) -> DoublePendulumParameters:
        """Render the analytical-EOM parameter dataclass from this model.

        This is the bridge that keeps the analytical model and this single
        source of truth in lock-step (epic task M2.3).
        """
        from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (
            DoublePendulumParameters,
            LowerSegmentProperties,
            SegmentProperties,
        )

        upper = SegmentProperties(
            length_m=self.upper.length_m,
            mass_kg=self.upper.mass_kg,
            center_of_mass_ratio=self.upper.center_of_mass_ratio,
            inertia_about_com=self.upper.effective_inertia_about_com,
        )
        lower = LowerSegmentProperties(
            length_m=self.lower.length_m,
            shaft_mass_kg=self.lower.shaft_mass_kg,
            clubhead_mass_kg=self.lower.clubhead_mass_kg,
            shaft_com_ratio=self.lower.shaft_com_ratio,
        )
        return DoublePendulumParameters(
            upper_segment=upper,
            lower_segment=lower,
            plane_inclination_deg=self.plane_inclination_deg,
            damping_shoulder=self.damping_shoulder,
            damping_wrist=self.damping_wrist,
            gravity_m_s2=self.gravity_m_s2,
            gravity_enabled=self.gravity_enabled,
            constrained_to_plane=self.constrained_to_plane,
        )

    def to_yaml(self) -> str:
        """Serialise to a YAML document string."""
        return yaml.safe_dump(self.model_dump(), sort_keys=True)

    @classmethod
    def from_yaml(cls, text: str) -> GolfModelParams:
        """Parse a YAML document string into a validated model.

        Args:
            text: YAML produced by :meth:`to_yaml` (or compatible).

        Raises:
            ValueError: If ``text`` is empty or not a mapping.
        """
        if not text or not text.strip():
            raise ValueError("YAML text must be non-empty")
        data = yaml.safe_load(text)
        if not isinstance(data, dict):
            raise ValueError("YAML must decode to a mapping of parameters")
        return cls.model_validate(data)

"""Lie and rough interaction: a parametric modifier on impact conditions (#14569).

Grass, sand or hardpan between the face and the ball changes what leaves the
clubface.  This module maps a lie (and, for rough, a severity in [0, 1]) to an
:class:`ImpactPerturbation`: a mean speed loss, spin loss and face twist plus
explicit uncertainty and a flyer mixture.  It is a plug-in in the style of
``mud_ball.py``: consumers (the shot model, the optimiser) call
:func:`lie_perturbation` and apply it to their own impact conditions, so no
contact solver is forked.  Fairway is the exact identity.

**Every number here is an assumption.**  Rough values are the existing
course-conditions estimates (``sg_optimizer.course.conditions.RoughModel``,
documented as ESTIMATE in ``docs/sg_optimizer/data_sources.md``); the other
lies are qualitative priors from the literature on grass and sand between
face and ball (for example Penner, "The physics of golf", Rep. Prog. Phys.
66, 2003, and launch-monitor lie studies), with deliberately wide uncertainty.
:data:`LIE_PARAMETERS` records, per parameter, whether it is ``cited`` or an
``assumption`` and why; none is currently ``cited`` from a primary table.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import ensure, require
from src.shared.python.sg_optimizer.course.conditions import RoughModel
from src.shared.python.sg_optimizer.course.rasterize import LIE_CODES
from src.shared.python.sg_optimizer.shot_model.skill_family import IMPACT_FIELDS

Array = NDArray[np.float64]


class LieType(str, Enum):
    FAIRWAY = "fairway"
    FIRST_CUT = "first_cut"
    ROUGH = "rough"
    BUNKER = "bunker"
    HARDPAN = "hardpan"


@dataclass(frozen=True)
class ParameterRecord:
    """Provenance of one modifier parameter."""

    value: float
    basis: str  # "cited" or "assumption"
    source: str
    rationale: str = ""


_ASSUMED = "Assumed prior; qualitative direction from Penner (2003) Rep. Prog. Phys."
LIE_PARAMETERS: dict[str, ParameterRecord] = {
    "first_cut.speed_factor": ParameterRecord(
        0.99, "assumption", _ASSUMED, "short grass: a few percent of speed at most"
    ),
    "first_cut.spin_factor": ParameterRecord(
        0.92, "assumption", _ASSUMED, "light grass film lowers spin slightly"
    ),
    "bunker.speed_factor": ParameterRecord(
        0.80, "assumption", _ASSUMED, "sand cushions the strike; wide uncertainty"
    ),
    "bunker.spin_factor": ParameterRecord(
        0.70, "assumption", _ASSUMED, "sand between face and ball cuts spin"
    ),
    "hardpan.speed_factor": ParameterRecord(
        0.97, "assumption", _ASSUMED, "thin strike risk, small mean loss"
    ),
    "hardpan.spin_factor": ParameterRecord(
        0.95, "assumption", _ASSUMED, "firm, clean contact keeps most spin"
    ),
    "rough.face_twist_deg_per_severity": ParameterRecord(
        -3.0,
        "assumption",
        _ASSUMED,
        "long grass wraps the hosel and closes the face as severity rises",
    ),
    "rough.speed_sd_per_severity": ParameterRecord(
        0.06, "assumption", _ASSUMED, "lie-to-lie speed scatter grows with depth"
    ),
    "flyer.speed_boost": ParameterRecord(
        1.08,
        "assumption",
        _ASSUMED,
        "flyers launch hot because little spin and friction remain",
    ),
    "flyer.spin_factor": ParameterRecord(
        0.55, "assumption", _ASSUMED, "flyer spin is a fraction of normal rough spin"
    ),
}


def _v(key: str) -> float:
    return LIE_PARAMETERS[key].value


@dataclass(frozen=True)
class PerturbationSamples:
    """Per-shot draws of the perturbation (arrays of equal length)."""

    speed_factor: Array
    spin_factor: Array
    face_twist_deg: Array
    is_flyer: NDArray[np.bool_]


@dataclass(frozen=True)
class ImpactPerturbation:
    """Mean lie effect on impact conditions with its uncertainty.

    ``speed_factor`` and ``spin_factor`` multiply ball-speed-at-impact and
    backspin; ``face_twist_deg`` is added to the face angle (+ open, - closed).
    The ``*_sd`` fields are the shot-to-shot standard deviations (speed and
    spin as fractions, twist in degrees).  A flyer replaces the spin and
    boosts the speed of that shot.
    """

    speed_factor: float = 1.0
    spin_factor: float = 1.0
    face_twist_deg: float = 0.0
    speed_sd: float = 0.0
    spin_sd: float = 0.0
    twist_sd_deg: float = 0.0
    flyer_probability: float = 0.0
    flyer_speed_boost: float = 1.0
    flyer_spin_factor: float = 1.0

    def is_identity(self) -> bool:
        return (
            self.speed_factor == 1.0
            and self.spin_factor == 1.0
            and self.face_twist_deg == 0.0
            and self.speed_sd == self.spin_sd == self.twist_sd_deg == 0.0
            and self.flyer_probability == 0.0
        )

    def apply(
        self, speed_mph: float, backspin_rpm: float, face_deg: float
    ) -> tuple[float, float, float]:
        """Mean impact conditions after the lie (no sampling)."""
        require(speed_mph >= 0.0, "speed_mph must be >= 0", speed_mph)
        require(backspin_rpm >= 0.0, "backspin_rpm must be >= 0", backspin_rpm)
        return (
            speed_mph * self.speed_factor,
            backspin_rpm * self.spin_factor,
            face_deg + self.face_twist_deg,
        )

    def sample(self, rng: np.random.Generator, n: int) -> PerturbationSamples:
        """Draw ``n`` shots; an identity perturbation returns exact ones/zeros."""
        require(n >= 1, "n must be >= 1", n)
        speed = self.speed_factor + self.speed_sd * rng.standard_normal(n)
        spin = self.spin_factor + self.spin_sd * rng.standard_normal(n)
        twist = self.face_twist_deg + self.twist_sd_deg * rng.standard_normal(n)
        flyer = rng.random(n) < self.flyer_probability
        speed = np.where(flyer, speed * self.flyer_speed_boost, speed)
        spin = np.where(flyer, spin * self.flyer_spin_factor, spin)
        out = PerturbationSamples(
            np.clip(speed, 0.0, None), np.clip(spin, 0.0, None), twist, flyer
        )
        ensure(out.speed_factor.shape == (n,), "one draw per shot")
        return out


def lie_perturbation(lie: LieType, severity: float = 0.0) -> ImpactPerturbation:
    """Impact perturbation for ``lie``; ``severity`` in [0, 1] applies to rough."""
    require(0.0 <= severity <= 1.0, "severity must lie in [0, 1]", severity)
    if lie is LieType.FAIRWAY:
        return ImpactPerturbation()
    if lie is LieType.ROUGH:
        rough = RoughModel(severity)
        return ImpactPerturbation(
            speed_factor=rough.distance_multiplier(),
            spin_factor=1.0 - rough.spin_reduction(),
            face_twist_deg=_v("rough.face_twist_deg_per_severity") * severity,
            speed_sd=_v("rough.speed_sd_per_severity") * severity,
            spin_sd=0.5 * _v("rough.speed_sd_per_severity") * severity,
            twist_sd_deg=1.5 * severity,
            flyer_probability=rough.flyer_probability(),
            flyer_speed_boost=_v("flyer.speed_boost"),
            flyer_spin_factor=_v("flyer.spin_factor"),
        )
    key = lie.value
    speed, spin = _v(f"{key}.speed_factor"), _v(f"{key}.spin_factor")
    spread = 1.0 - speed
    return ImpactPerturbation(
        speed_factor=speed,
        spin_factor=spin,
        speed_sd=0.25 * spread,
        spin_sd=0.25 * (1.0 - spin),
        twist_sd_deg=0.5,
    )


def effective_face_friction(mu_dry: float, perturbation: ImpactPerturbation) -> float:
    """Face-ball Coulomb friction after the lie (feeds ``check_friction_cone``).

    Assumption: friction falls in proportion to the spin that survives, so the
    result is ``mu_dry * spin_factor`` and the fairway returns ``mu_dry``.
    """
    require(mu_dry >= 0.0, "mu_dry must be >= 0", mu_dry)
    return mu_dry * perturbation.spin_factor


_CODE_TO_LIE: dict[int, LieType] = {
    LIE_CODES["tee"]: LieType.FAIRWAY,
    LIE_CODES["fairway"]: LieType.FAIRWAY,
    LIE_CODES["rough"]: LieType.ROUGH,
    LIE_CODES["sand"]: LieType.BUNKER,
}
_SPEED = IMPACT_FIELDS.index("clubhead_speed_mph")
_FACE = IMPACT_FIELDS.index("face_angle_deg")


def lie_from_code(code: int) -> LieType:
    """The :class:`LieType` of a course ``LIE_CODES`` value (the B3 lie dimension).

    Only tee, fairway, rough and sand modify the strike; any other code (water,
    trees, green...) is not a full-swing lie and is rejected.
    """
    require(code in _CODE_TO_LIE, "lie code has no impact modifier", code)
    return _CODE_TO_LIE[code]


def perturb_impact_rows(
    rows: Array, perturbation: ImpactPerturbation, rng: np.random.Generator
) -> tuple[Array, Array]:
    """Apply ``perturbation`` to ``IMPACT_FIELDS`` rows; return ``(rows, spin_factor)``.

    Clubhead speed is scaled and the face angle shifted per shot; every other
    column is untouched and ``rows`` is not modified.  Spin is not an impact
    field, so its per-shot factor is returned for the caller to apply.
    Assumption: ball speed is proportional to clubhead speed, so the speed factor
    acts on clubhead speed.
    """
    require(rows.ndim == 2 and rows.shape[1] == len(IMPACT_FIELDS), "impact rows")
    draws = perturbation.sample(rng, len(rows))
    out = rows.copy()
    out[:, _SPEED] *= draws.speed_factor
    out[:, _FACE] += draws.face_twist_deg
    ensure(out.shape == rows.shape, "row shape is preserved")
    return out, draws.spin_factor

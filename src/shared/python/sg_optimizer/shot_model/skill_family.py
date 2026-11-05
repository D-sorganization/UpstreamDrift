"""``SkillFamily``: per-shot-type impact-condition distributions with provenance.

A player's skill is **data**, not subclasses: for each shot type (``SKILL_NAMES``)
a mean and covariance over the impact conditions that drive the ball-flight
physics, plus where the numbers came from. The variance sits on impact
conditions (speed, face, path, AoA, strike), never on joint torques (epic
#14564 panel, Argon).

Vocabulary: ``IMPACT_FIELDS`` are attribute names of
``impact_parameters.ImpactParameters`` (single vocabulary, sign conventions
GCV-15: +face open, +path in-to-out, +AoA upward; ``toe_mm``/``high_mm`` are the
strike offsets as extracted). Units: mph, degrees, millimetres (serialized as
``units`` and checked on load). The putt link is not a skill here: putting stays
in ``putting_provider``.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import yaml
from numpy.typing import NDArray

from src.shared.python.contracts import require

IMPACT_FIELDS: tuple[str, ...] = (
    "clubhead_speed_mph",
    "face_angle_deg",
    "club_path_deg",
    "attack_angle_deg",
    "toe_mm",
    "high_mm",
)
SKILL_NAMES: tuple[str, ...] = (
    "driver",
    "fairway_wood",
    "long_iron",
    "mid_iron",
    "short_iron",
    "wedge",
    "pitch",
    "chip",
    "bunker",
)
# Unit of each field, taken from its name suffix; serialized and checked on load.
UNITS: dict[str, str] = {f: f.rsplit("_", 1)[1] for f in IMPACT_FIELDS}
PROVENANCE_KINDS = ("fit", "preset", "manual")
_N = len(IMPACT_FIELDS)
_PSD_TOL = 1e-9

Row = tuple[float, ...]


@dataclass(frozen=True)
class Provenance:
    """Where a distribution came from: a fit, a cited preset, or manual entry."""

    kind: str
    source: str
    version: str

    def __post_init__(self) -> None:
        require(
            self.kind in PROVENANCE_KINDS, f"kind must be one of {PROVENANCE_KINDS}"
        )
        require(bool(self.source.strip()), "provenance needs a source (or assumption)")
        require(bool(self.version.strip()), "provenance needs a version")


@dataclass(frozen=True)
class ImpactDistribution:
    """Gaussian over ``IMPACT_FIELDS``: mean, per-field sd, optional correlation."""

    mean: Row
    sd: Row
    corr: tuple[Row, ...] | None = None

    def __post_init__(self) -> None:
        require(len(self.mean) == _N and len(self.sd) == _N, f"need {_N} values")
        require(all(np.isfinite(self.mean)), "mean must be finite")
        require(self.mean[0] > 0, "clubhead speed must be > 0")
        require(all(np.isfinite(self.sd)) and min(self.sd) >= 0, "sd must be >= 0")
        if self.corr is not None:
            c = np.array(self.corr, dtype=float)
            require(c.shape == (_N, _N), "corr must be square")
            require(bool(np.allclose(c, c.T)), "corr must be symmetric")
            require(bool(np.allclose(np.diag(c), 1.0)), "corr diagonal must be 1")
            require(bool(np.all(np.abs(c) <= 1.0)), "corr entries must be in [-1, 1]")
            require(
                bool(np.linalg.eigvalsh(c).min() >= -_PSD_TOL),
                "corr must be PSD",
            )

    def covariance(self) -> NDArray[np.float64]:
        sd = np.array(self.sd, dtype=float)
        c = np.eye(_N) if self.corr is None else np.array(self.corr, dtype=float)
        return np.asarray(c * np.outer(sd, sd), dtype=np.float64)

    def sample(self, rng: np.random.Generator, n: int) -> NDArray[np.float64]:
        """``n`` impact-condition draws, shape ``(n, 6)`` in ``IMPACT_FIELDS`` order."""
        require(n > 0, "n must be > 0", n)
        draws = rng.multivariate_normal(self.mean, self.covariance(), size=n)
        return np.asarray(draws, dtype=np.float64)


@dataclass(frozen=True)
class SkillSpec:
    """One shot type's impact distribution and its provenance."""

    distribution: ImpactDistribution
    provenance: Provenance


@dataclass(frozen=True)
class SkillFamily:
    """A named set of skills (a player, or a preset skill level)."""

    name: str
    skills: dict[str, SkillSpec]
    provenance: Provenance

    def __post_init__(self) -> None:
        require(len(self.skills) > 0, "a skill family needs at least one skill")
        unknown = set(self.skills) - set(SKILL_NAMES)
        require(not unknown, f"unknown skills {sorted(unknown)}; have {SKILL_NAMES}")

    # --- I/O -------------------------------------------------------------
    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "units": dict(UNITS),
            "provenance": _prov_dict(self.provenance),
            "skills": {
                k: {
                    "mean": list(s.distribution.mean),
                    "sd": list(s.distribution.sd),
                    "corr": None
                    if s.distribution.corr is None
                    else [list(r) for r in s.distribution.corr],
                    "provenance": _prov_dict(s.provenance),
                }
                for k, s in self.skills.items()
            },
        }

    @classmethod
    def from_dict(cls, raw: dict[str, Any]) -> SkillFamily:
        require(
            "skills" in raw and "provenance" in raw, "family needs skills+provenance"
        )
        check_units(raw)
        skills = {k: _spec_from(v) for k, v in raw["skills"].items()}
        return cls(str(raw["name"]), skills, Provenance(**raw["provenance"]))

    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(
            yaml.safe_dump(self.to_dict(), sort_keys=False), encoding="utf-8"
        )

    @classmethod
    def from_yaml(cls, path: str | Path) -> SkillFamily:
        p = Path(path)
        require(p.exists(), f"skill family file not found: {p}")
        return cls.from_dict(yaml.safe_load(p.read_text(encoding="utf-8")))


def check_units(raw: dict[str, Any]) -> None:
    """Fail closed unless ``raw["units"]`` is exactly the expected ``UNITS``."""
    require(
        raw.get("units") == UNITS, f"units must equal {UNITS}, got {raw.get('units')}"
    )


def _prov_dict(p: Provenance) -> dict[str, str]:
    return {"kind": p.kind, "source": p.source, "version": p.version}


def _row(values: Any) -> Row:
    return tuple(float(v) for v in values)


def _spec_from(raw: dict[str, Any]) -> SkillSpec:
    corr = raw.get("corr")
    dist = ImpactDistribution(
        mean=_row(raw["mean"]),
        sd=_row(raw["sd"]),
        corr=None if corr is None else tuple(_row(r) for r in corr),
    )
    return SkillSpec(dist, Provenance(**raw["provenance"]))

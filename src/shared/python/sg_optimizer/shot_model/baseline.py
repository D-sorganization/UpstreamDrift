"""Baseline (tour / scratch / bogey) shot-dispersion reference data.

YAML-backed; cited to Broadie (2014) in ``docs/sg_optimizer/data_sources.md``.
Numerical values **must** be sourced; do not invent.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.distributions import (
    TiltedBivariateGaussian,
)


@dataclass(frozen=True)
class ClubBaseline:
    """Reference shot model for a single club in the baseline bag."""

    name: str
    carry_mean: float  # yards
    total_mean: float  # yards (carry + roll)
    sigma_long: float
    sigma_lat: float
    rho: float = 0.20
    bias_long: float = 0.0
    bias_lat: float = 0.0

    def __post_init__(self) -> None:
        require(self.carry_mean > 0, "carry_mean must be > 0", self.carry_mean)
        require(self.total_mean >= self.carry_mean, "total_mean must be ≥ carry_mean")

    def distribution(self) -> TiltedBivariateGaussian:
        return TiltedBivariateGaussian(
            sigma_long=self.sigma_long,
            sigma_lat=self.sigma_lat,
            rho=self.rho,
            bias_long=self.bias_long,
            bias_lat=self.bias_lat,
        )


@dataclass(frozen=True)
class BaselineBag:
    """Reference bag: a name, source citation, and per-club baselines."""

    name: str
    source: str
    clubs: dict[str, ClubBaseline]

    def __post_init__(self) -> None:
        require(len(self.clubs) > 0, "baseline bag must have at least one club")

    def get(self, club: str) -> ClubBaseline:
        require(club in self.clubs, f"unknown club {club!r}; have {sorted(self.clubs)}")
        return self.clubs[club]


def load_baseline(path: str | Path) -> BaselineBag:
    """Load a baseline bag from YAML.

    Schema: see ``data/sg_optimizer/baselines/pga_tour.yaml`` for the
    canonical example.
    """
    p = Path(path)
    require(p.exists(), f"baseline file not found: {p}")
    raw: dict[str, Any] = yaml.safe_load(p.read_text(encoding="utf-8"))
    require("clubs" in raw, "baseline YAML missing 'clubs' section")
    clubs = {
        name: ClubBaseline(
            name=name, **{k: v for k, v in spec.items() if not k.startswith("_")}
        )
        for name, spec in raw["clubs"].items()
    }
    return BaselineBag(
        name=str(raw.get("name", p.stem)),
        source=str(raw.get("source", "uncited")),
        clubs=clubs,
    )

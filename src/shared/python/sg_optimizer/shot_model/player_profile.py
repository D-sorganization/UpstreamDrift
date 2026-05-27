"""Player skill profile: multiplicative departures from a baseline bag.

Skill multipliers scale σ; bias terms are absolute yards (a chronic
push-right is 5 yards regardless of skill level — spec §1.2).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from src.shared.python.contracts import require
from src.shared.python.sg_optimizer.shot_model.baseline import (
    BaselineBag,
    load_baseline,
)
from src.shared.python.sg_optimizer.shot_model.distributions import (
    TiltedBivariateGaussian,
)


@dataclass(frozen=True)
class ClubSkill:
    """Per-club departures from a baseline (multiplicative + offsets)."""

    skill_mult_long: float = 1.0
    skill_mult_lat: float = 1.0
    distance_offset: float = 0.0
    bias_long: float = 0.0
    bias_lat: float = 0.0
    enabled: bool = True

    def __post_init__(self) -> None:
        require(
            self.skill_mult_long > 0,
            "skill_mult_long must be > 0",
            self.skill_mult_long,
        )
        require(
            self.skill_mult_lat > 0, "skill_mult_lat must be > 0", self.skill_mult_lat
        )


@dataclass(frozen=True)
class PuttingSkill:
    """Putting model: per-distance make-% multipliers + 3-putt avoidance."""

    make_pct_multipliers: dict[float, float] = field(default_factory=dict)
    three_putt_avoidance: float = 1.0

    def __post_init__(self) -> None:
        require(self.three_putt_avoidance > 0, "three_putt_avoidance must be > 0")
        for d, m in self.make_pct_multipliers.items():
            require(d >= 0 and m > 0, f"invalid putting entry {d!r}->{m!r}")

    def multiplier_at(self, distance_ft: float) -> float:
        """Piecewise-linear interpolation of make-% multipliers in distance."""
        if not self.make_pct_multipliers:
            return 1.0
        xs = sorted(self.make_pct_multipliers)
        if distance_ft <= xs[0]:
            return self.make_pct_multipliers[xs[0]]
        if distance_ft >= xs[-1]:
            return self.make_pct_multipliers[xs[-1]]
        for i in range(len(xs) - 1):
            a, b = xs[i], xs[i + 1]
            if a <= distance_ft <= b:
                t = (distance_ft - a) / (b - a)
                return (1 - t) * self.make_pct_multipliers[
                    a
                ] + t * self.make_pct_multipliers[b]
        return 1.0  # pragma: no cover - guarded by xs[-1] branch


@dataclass(frozen=True)
class PlayerProfile:
    """A named player with baseline reference + per-club skill departures."""

    name: str
    baseline: str  # path or stem of baseline YAML (resolved at load time)
    clubs: dict[str, ClubSkill] = field(default_factory=dict)
    putting: PuttingSkill = field(default_factory=PuttingSkill)
    short_game: dict[str, float] = field(default_factory=dict)
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    notes: str = ""

    def with_baseline(
        self, baseline_path: str | Path
    ) -> tuple[PlayerProfile, BaselineBag]:
        """Resolve and attach a concrete baseline bag, returning (self, bag)."""
        bag = load_baseline(baseline_path)
        return self, bag

    def effective_distribution(
        self, club: str, baseline: BaselineBag
    ) -> TiltedBivariateGaussian:
        """Compose the baseline ClubBaseline with this player's skill multipliers."""
        cb = baseline.get(club)
        skill = self.clubs.get(club, ClubSkill())
        require(skill.enabled, f"club {club!r} is disabled in profile {self.name!r}")
        return TiltedBivariateGaussian(
            sigma_long=cb.sigma_long * skill.skill_mult_long,
            sigma_lat=cb.sigma_lat * skill.skill_mult_lat,
            rho=cb.rho,
            bias_long=cb.bias_long + skill.bias_long,
            bias_lat=cb.bias_lat + skill.bias_lat,
        )

    def effective_distance(self, club: str, baseline: BaselineBag) -> float:
        cb = baseline.get(club)
        skill = self.clubs.get(club, ClubSkill())
        return cb.total_mean + skill.distance_offset

    # --- I/O -------------------------------------------------------------
    def to_yaml(self, path: str | Path) -> None:
        Path(path).write_text(_dump(self), encoding="utf-8")

    @classmethod
    def from_yaml(cls, path: str | Path) -> PlayerProfile:
        p = Path(path)
        require(p.exists(), f"profile file not found: {p}")
        return _load(p)


def _dump(profile: PlayerProfile) -> str:
    payload: dict[str, Any] = {
        "name": profile.name,
        "baseline": profile.baseline,
        "last_updated": profile.last_updated.isoformat(),
        "notes": profile.notes,
        "clubs": {k: asdict(v) for k, v in profile.clubs.items()},
        "putting": {
            "make_pct_multipliers": dict(profile.putting.make_pct_multipliers),
            "three_putt_avoidance": profile.putting.three_putt_avoidance,
        },
        "short_game": dict(profile.short_game),
    }
    return yaml.safe_dump(payload, sort_keys=False)


def _load(p: Path) -> PlayerProfile:
    raw = yaml.safe_load(p.read_text(encoding="utf-8"))
    clubs = {k: ClubSkill(**v) for k, v in (raw.get("clubs") or {}).items()}
    putting_raw = raw.get("putting") or {}
    putting = PuttingSkill(
        make_pct_multipliers={
            float(k): float(v)
            for k, v in (putting_raw.get("make_pct_multipliers") or {}).items()
        },
        three_putt_avoidance=float(putting_raw.get("three_putt_avoidance", 1.0)),
    )
    last_updated_raw = raw.get("last_updated")
    last_updated = (
        datetime.fromisoformat(last_updated_raw)
        if isinstance(last_updated_raw, str)
        else datetime.now(timezone.utc)
    )
    return PlayerProfile(
        name=str(raw["name"]),
        baseline=str(raw["baseline"]),
        clubs=clubs,
        putting=putting,
        short_game=dict(raw.get("short_game") or {}),
        last_updated=last_updated,
        notes=str(raw.get("notes", "")),
    )


# Suppress unused import warning under DBC=off.
_ = replace

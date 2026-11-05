"""Empirical-Bayes shrinkage of a player's club skills toward the tour baseline."""

from __future__ import annotations

from dataclasses import replace

from src.shared.python.contracts import ensure, require
from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
)


def shrinkage_weight(n_shots: int, strength: float) -> float:
    """Weight on the player's own estimate: ``n / (n + strength)`` in [0, 1).

    ``strength`` is the prior's pseudo-shot count: larger pulls harder to tour.
    """
    require(n_shots >= 0, "n_shots must be >= 0", n_shots)
    require(strength >= 0, "strength must be >= 0", strength)
    total = n_shots + strength
    w = n_shots / total if total > 0 else 0.0
    ensure(0.0 <= w <= 1.0, "weight must lie in [0, 1]")
    return w


def shrink_profile(
    profile: PlayerProfile, shots: dict[str, int], strength: float
) -> PlayerProfile:
    """Per-club posterior: neutral baseline (mult 1, offsets 0) + w * departure."""
    clubs = {}
    for club, s in profile.clubs.items():
        w = shrinkage_weight(shots.get(club, 0), strength)
        clubs[club] = ClubSkill(
            skill_mult_long=1.0 + w * (s.skill_mult_long - 1.0),
            skill_mult_lat=1.0 + w * (s.skill_mult_lat - 1.0),
            distance_offset=w * s.distance_offset,
            bias_long=w * s.bias_long,
            bias_lat=w * s.bias_lat,
            enabled=s.enabled,
        )
    return replace(profile, clubs=clubs)

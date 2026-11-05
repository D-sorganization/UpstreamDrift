"""Empirical-Bayes shrinkage of a player's club skills toward the tour baseline.

The weight ``n / (n + strength)`` is ``skill_fit.shrinkage_weight`` (one formula,
shared with the SkillFamily fit); only the per-club blend lives here.

Scope: this shrinks ``PlayerProfile`` club multipliers, which the *baseline* backend
applies. The ``physics`` backend ignores them (the table encodes the skill), so its
personalisation shrinks through ``skill_fit.fit_skill_family(prior_strength=...)``.
"""

from __future__ import annotations

from dataclasses import replace

from src.shared.python.sg_optimizer.shot_model.player_profile import (
    ClubSkill,
    PlayerProfile,
)
from src.shared.python.sg_optimizer.shot_model.skill_fit import shrinkage_weight


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

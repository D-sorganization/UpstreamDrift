"""Skill-conditioned putting analytics (epic #14168).

Physics-agnostic layer: a player's own expected-putts curve E_player(d)
(:mod:`.expected_putts`) and aim/pace optimisation under a strokes-gained or
must-make objective (:mod:`.optimize`). Ball physics and skill dispersion are
injected through the :class:`.expected_putts.PuttOutcomeModel` protocol, so
``putting_dynamics`` (#14141) stays the single physics source; the skill
profile and its sampler follow in #14169.
"""

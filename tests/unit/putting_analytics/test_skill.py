"""Skill profile contracts: validation, scaling and channel ablation."""

from __future__ import annotations

import pytest

from src.shared.python.putting_analytics.skill import (
    CHANNELS,
    PuttingSkillProfile,
    ReadDispersion,
    StrokeDispersion,
)

pytestmark = pytest.mark.unit


def _profile() -> PuttingSkillProfile:
    stroke = StrokeDispersion(1.0, 2.0, 0.1, 3.0, face_bias_deg=0.5, speed_bias=0.05)
    return PuttingSkillProfile("p", stroke, ReadDispersion(0.2, 1.5, 0.1, 0.1))


def test_scaled_multiplies_spreads_and_keeps_biases() -> None:
    half = _profile().scaled(0.5)
    assert (half.stroke.face_sd_deg, half.stroke.speed_cv) == (0.5, 0.05)
    assert (half.read.aspect_sd_deg, half.read.grade_rel_sd) == (0.75, 0.1)
    assert (half.stroke.face_bias_deg, half.read.grade_rel_bias) == (0.5, 0.1)


def test_without_zeroes_only_the_named_channels() -> None:
    ablated = _profile().without("path", "read_stimp")
    assert ablated.stroke.path_sd_deg == 0.0 and ablated.read.stimp_rel_sd == 0.0
    assert ablated.stroke.face_sd_deg == 1.0 and ablated.read.aspect_sd_deg == 1.5
    assert _profile().without(*CHANNELS).scaled(3.0).stroke.speed_cv == 0.0


@pytest.mark.parametrize(
    "build",
    [
        lambda: StrokeDispersion(-1.0, 1.0, 0.1),
        lambda: StrokeDispersion(1.0, 1.0, 0.1, speed_bias=-1.0),
        lambda: ReadDispersion(aspect_sd_deg=-0.1),
        lambda: PuttingSkillProfile("p", StrokeDispersion(1, 1, 0.1), face_weight=1.2),
        lambda: _profile().scaled(-1.0),
        lambda: _profile().without("grip"),
    ],
)
def test_invalid_inputs_are_rejected(build) -> None:
    with pytest.raises(ValueError):
        build()

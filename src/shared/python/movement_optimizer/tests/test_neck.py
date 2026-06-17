# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for neck rendering constants and visual properties."""

from __future__ import annotations

from movement_optimizer.constants import LENGTH_FRAC, NECK_MAX_ANGLE_DEG
from movement_optimizer.rendering import NECK_LENGTH_FRAC


class TestNeckRendering:
    def test_neck_length_in_constants(self) -> None:
        assert "neck" in LENGTH_FRAC
        assert 0.05 < LENGTH_FRAC["neck"] < 0.15

    def test_neck_max_angle(self) -> None:
        assert NECK_MAX_ANGLE_DEG == 45.0

    def test_neck_length_frac_matches_constants(self) -> None:
        assert LENGTH_FRAC["neck"] == NECK_LENGTH_FRAC

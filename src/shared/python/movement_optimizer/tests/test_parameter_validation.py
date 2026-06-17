# Copyright (c) 2026 D-Sorganization. All rights reserved.
"""Tests for movement_optimizer.validation (issue #400)."""

from __future__ import annotations

import math

import pytest

from movement_optimizer.validation import (
    BAR_MASS_RANGE,
    BODY_MASS_RANGE,
    SMOOTHNESS_RANGE,
    validate_all,
    validate_bar_mass,
    validate_body_mass,
    validate_duration,
    validate_height,
    validate_smoothness,
)


class TestBodyMass:
    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match=r"body_mass.*\[30\.0, 300\.0\]"):
            validate_body_mass(BODY_MASS_RANGE[0] - 0.01)

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError, match=r"body_mass.*\[30\.0, 300\.0\]"):
            validate_body_mass(BODY_MASS_RANGE[1] + 0.01)

    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_body_mass(-1.0)

    def test_zero_raises(self):
        with pytest.raises(ValueError):
            validate_body_mass(0.0)

    def test_at_lower_bound_ok(self):
        assert validate_body_mass(BODY_MASS_RANGE[0]) == BODY_MASS_RANGE[0]

    def test_at_upper_bound_ok(self):
        assert validate_body_mass(BODY_MASS_RANGE[1]) == BODY_MASS_RANGE[1]

    def test_typical_ok(self):
        assert validate_body_mass(80.0) == 80.0


class TestHeight:
    def test_zero_raises(self):
        with pytest.raises(ValueError):
            validate_height(0.0)

    def test_below_minimum_raises(self):
        with pytest.raises(ValueError, match=r"height.*\[1\.4, 2\.2\]"):
            validate_height(1.0)

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError):
            validate_height(3.0)

    def test_typical_ok(self):
        assert validate_height(1.75) == 1.75


class TestBarMass:
    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_bar_mass(-1.0)

    def test_zero_ok(self):
        # 0 = empty hand is valid
        assert validate_bar_mass(0.0) == 0.0

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError):
            validate_bar_mass(BAR_MASS_RANGE[1] + 0.01)


class TestDuration:
    def test_too_short_raises(self):
        with pytest.raises(ValueError, match=r"duration.*\[0\.5, 10\.0\]"):
            validate_duration(0.1)

    def test_too_long_raises(self):
        with pytest.raises(ValueError):
            validate_duration(20.0)

    def test_typical_ok(self):
        assert validate_duration(2.0) == 2.0


class TestSmoothness:
    def test_negative_raises(self):
        with pytest.raises(ValueError):
            validate_smoothness(-1.0)

    def test_zero_below_minimum_raises(self):
        # SMOOTHNESS_RANGE starts at 0.1 -- zero is rejected.
        with pytest.raises(ValueError):
            validate_smoothness(0.0)

    def test_above_maximum_raises(self):
        with pytest.raises(ValueError):
            validate_smoothness(SMOOTHNESS_RANGE[1] + 0.1)


class TestNonFinite:
    @pytest.mark.parametrize(
        "validator",
        [
            validate_body_mass,
            validate_height,
            validate_bar_mass,
            validate_duration,
            validate_smoothness,
        ],
    )
    def test_nan_raises(self, validator):
        with pytest.raises(ValueError, match="finite"):
            validator(float("nan"))

    @pytest.mark.parametrize(
        "validator",
        [
            validate_body_mass,
            validate_height,
            validate_bar_mass,
            validate_duration,
            validate_smoothness,
        ],
    )
    def test_inf_raises(self, validator):
        with pytest.raises(ValueError, match="finite"):
            validator(math.inf)

    @pytest.mark.parametrize(
        "validator",
        [
            validate_body_mass,
            validate_height,
            validate_bar_mass,
            validate_duration,
            validate_smoothness,
        ],
    )
    def test_neg_inf_raises(self, validator):
        with pytest.raises(ValueError, match="finite"):
            validator(-math.inf)

    def test_string_raises_type_error(self):
        with pytest.raises(TypeError):
            validate_body_mass("80")  # type: ignore[arg-type]

    def test_bool_rejected(self):
        # bool is technically int but should not be accepted as a real
        # number for these physical quantities.
        with pytest.raises(TypeError):
            validate_body_mass(True)  # type: ignore[arg-type]


class TestValidateAll:
    def test_valid_passes(self):
        # Should not raise
        validate_all(
            body_mass=80.0,
            height=1.75,
            bar_mass=60.0,
            duration=2.0,
            smoothness=1.0,
        )

    def test_one_invalid_raises(self):
        with pytest.raises(ValueError, match="body_mass"):
            validate_all(
                body_mass=-5.0,
                height=1.75,
                bar_mass=60.0,
                duration=2.0,
                smoothness=1.0,
            )

    def test_error_message_includes_range(self):
        with pytest.raises(ValueError, match=r"\[1\.4, 2\.2\]"):
            validate_all(
                body_mass=80.0,
                height=0.0,
                bar_mass=60.0,
                duration=2.0,
                smoothness=1.0,
            )

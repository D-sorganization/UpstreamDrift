"""Coverage for src.engines.tiers — tier policy helpers."""

from __future__ import annotations

import warnings

import pytest

from src.engines.tiers import (
    ALLOWED_TIERS,
    ENGINE_TIERS,
    ExperimentalTierWarning,
    get_engine_tier,
    warn_if_experimental,
)


class TestGetEngineTier:
    def test_known_engines_return_documented_tier(self) -> None:
        for name, tier in ENGINE_TIERS.items():
            assert get_engine_tier(name) == tier
            assert tier in ALLOWED_TIERS

    def test_normalizes_whitespace_and_case(self) -> None:
        assert get_engine_tier("  MuJoCo  ") == "core"

    def test_unknown_engine_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="Unknown engine tier metadata"):
            get_engine_tier("not_a_real_engine")

    def test_empty_string_raises_value_error(self) -> None:
        with pytest.raises(ValueError, match="must not be empty"):
            get_engine_tier("   ")

    def test_non_string_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="must be a string"):
            get_engine_tier(123)  # type: ignore[arg-type]


class TestWarnIfExperimental:
    def test_experimental_emits_warning(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_experimental("opensim", "OpenSim")
        assert any(issubclass(w.category, ExperimentalTierWarning) for w in caught)
        assert any("OpenSim" in str(w.message) for w in caught)

    def test_core_engine_does_not_warn(self) -> None:
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always")
            warn_if_experimental("mujoco", "MuJoCo")
        assert not any(issubclass(w.category, ExperimentalTierWarning) for w in caught)

    def test_non_string_display_name_raises_type_error(self) -> None:
        with pytest.raises(TypeError, match="display_name must be a string"):
            warn_if_experimental("opensim", 42)  # type: ignore[arg-type]

    def test_unknown_engine_propagates_value_error(self) -> None:
        with pytest.raises(ValueError):
            warn_if_experimental("does_not_exist", "Bogus")

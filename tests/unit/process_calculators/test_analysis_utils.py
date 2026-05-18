"""Tests for sidekick.process_calculators.analysis_utils (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.process_calculators.analysis_utils import (
    evaluate_output,
)


class _StubEngine:
    """Minimal engine stub: calculate(**params) returns a dict."""

    def calculate(self, **params):
        return {
            "efficiency": 0.75,
            "power": params.get("manual_hhv", 0.0) * 2.0,
            "state": {"temp": 300.0},
            "composition": {"H2": 0.5},
        }


class _FailingEngine:
    """Engine that always raises."""

    def calculate(self, **params):
        raise ValueError("Engine failure")


class _NonDictEngine:
    """Engine that returns a non-dict."""

    def calculate(self, **params):
        return 42.0


class TestEvaluateOutput:
    def test_analysis_utils_returns_tuple_of_three(self) -> None:
        result = evaluate_output(_StubEngine(), {}, 0.0, "efficiency")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_extracts_named_output(self) -> None:
        value, state, comp = evaluate_output(_StubEngine(), {}, 0.0, "efficiency")
        assert value == pytest.approx(0.75)

    def test_state_dict_returned(self) -> None:
        _, state, _ = evaluate_output(_StubEngine(), {}, 0.0, "efficiency")
        assert isinstance(state, dict)
        assert state.get("temp") == 300.0

    def test_composition_dict_returned(self) -> None:
        _, _, comp = evaluate_output(_StubEngine(), {}, 0.0, "efficiency")
        assert isinstance(comp, dict)

    def test_overrides_applied(self) -> None:
        value, _, _ = evaluate_output(
            _StubEngine(), {}, 10.0, "power", overrides={"x": 1}
        )
        # power = manual_hhv * 2 = 10 * 2 = 20
        assert value == pytest.approx(20.0)

    def test_missing_key_returns_zero(self) -> None:
        value, _, _ = evaluate_output(_StubEngine(), {}, 0.0, "nonexistent_key")
        assert value == pytest.approx(0.0)

    def test_failing_engine_returns_zeros(self) -> None:
        value, state, comp = evaluate_output(_FailingEngine(), {}, 0.0, "efficiency")
        assert value == 0.0
        assert state == {}
        assert comp == {}

    def test_non_dict_engine_returns_zeros(self) -> None:
        value, state, comp = evaluate_output(_NonDictEngine(), {}, 0.0, "efficiency")
        assert value == 0.0
        assert state == {}
        assert comp == {}

    def test_no_hhv_not_injected(self) -> None:
        # manual_hhv=0 means it won't be injected (condition: manual_hhv > 0)
        value, _, _ = evaluate_output(_StubEngine(), {}, 0.0, "power")
        assert value == pytest.approx(0.0)

    def test_overrides_merge_with_base_params(self) -> None:
        base = {"a": 1.0}
        overrides = {"b": 2.0}
        # Should not raise
        value, _, _ = evaluate_output(_StubEngine(), base, 0.0, "efficiency", overrides)
        assert value == pytest.approx(0.75)

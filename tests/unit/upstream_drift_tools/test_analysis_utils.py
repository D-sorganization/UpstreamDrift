"""Tests for src.shared.python.upstream_drift_tools.process_calculators.analysis_utils (Issues #1949, #1744)."""

from __future__ import annotations

from typing import Any

from src.shared.python.upstream_drift_tools.process_calculators.analysis_utils import (
    evaluate_output,
)

# ---------------------------------------------------------------------------
# Minimal mock engine
# ---------------------------------------------------------------------------


class _FixedEngine:
    """Returns a fixed dict with the given values."""

    def __init__(self, result: dict[str, Any]) -> None:
        self._result = result

    def calculate(self, **_kwargs: Any) -> dict[str, Any]:
        return self._result


class _RaisingEngine:
    """Always raises a TypeError."""

    def calculate(self, **_kwargs: Any) -> dict[str, Any]:
        msg = "engine error"
        raise TypeError(msg)


class _BadReturnEngine:
    """Returns a non-dict."""

    def calculate(self, **_kwargs: Any) -> str:
        return "not a dict"


# ---------------------------------------------------------------------------
# evaluate_output
# ---------------------------------------------------------------------------


class TestEvaluateOutput:
    _BASE = {"flow": 100.0, "temperature": 400.0}

    def test_analysis_utils_returns_tuple_of_three(self) -> None:
        engine = _FixedEngine({"efficiency": 0.85})
        result = evaluate_output(engine, self._BASE, 0.0, "efficiency")
        assert isinstance(result, tuple)
        assert len(result) == 3

    def test_extracts_named_output(self) -> None:
        engine = _FixedEngine({"efficiency": 0.85})
        value, _, _ = evaluate_output(engine, self._BASE, 0.0, "efficiency")
        assert abs(value - 0.85) < 1e-10

    def test_missing_key_returns_zero(self) -> None:
        engine = _FixedEngine({"other": 1.0})
        value, _, _ = evaluate_output(engine, self._BASE, 0.0, "efficiency")
        assert value == 0.0

    def test_engine_raises_returns_zero(self) -> None:
        value, state, comp = evaluate_output(_RaisingEngine(), self._BASE, 0.0, "x")
        assert value == 0.0
        assert state == {}
        assert comp == {}

    def test_non_dict_return_gives_zero(self) -> None:
        value, state, comp = evaluate_output(_BadReturnEngine(), self._BASE, 0.0, "x")
        assert value == 0.0

    def test_overrides_applied(self) -> None:
        """Engine should receive merged params."""
        received: dict[str, Any] = {}

        class _CapturingEngine:
            def calculate(self, **kwargs: Any) -> dict[str, Any]:
                received.update(kwargs)
                return {"out": kwargs.get("temperature", 0.0)}

        evaluate_output(
            _CapturingEngine(),
            {"temperature": 300.0},
            0.0,
            "out",
            overrides={"temperature": 600.0},
        )
        assert received["temperature"] == 600.0

    def test_manual_hhv_injected_when_positive(self) -> None:
        received: dict[str, Any] = {}

        class _CapturingEngine:
            def calculate(self, **kwargs: Any) -> dict[str, Any]:
                received.update(kwargs)
                return {"out": 1.0}

        evaluate_output(_CapturingEngine(), {}, 55.0, "out")
        assert received.get("manual_hhv") == 55.0

    def test_manual_hhv_zero_not_injected(self) -> None:
        received: dict[str, Any] = {}

        class _CapturingEngine:
            def calculate(self, **kwargs: Any) -> dict[str, Any]:
                received.update(kwargs)
                return {"out": 1.0}

        evaluate_output(_CapturingEngine(), {}, 0.0, "out")
        assert "manual_hhv" not in received

    def test_state_and_composition_extracted(self) -> None:
        engine = _FixedEngine(
            {
                "efficiency": 0.9,
                "state": {"x": 1.0},
                "composition": {"H2": 0.5},
            }
        )
        _, state, comp = evaluate_output(engine, self._BASE, 0.0, "efficiency")
        assert state == {"x": 1.0}
        assert comp == {"H2": 0.5}

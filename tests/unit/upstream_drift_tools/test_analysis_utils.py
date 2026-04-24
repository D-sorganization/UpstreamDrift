"""Tests for src.shared.python.upstream_drift_tools.process_calculators.analysis_utils (Issues #1949, #1744)."""

from __future__ import annotations

from typing import Any

import pytest

from src.shared.python.upstream_drift_tools.process_calculators.analysis_utils import (
    evaluate_compression_result,
    evaluate_output,
)

pytestmark = pytest.mark.unit

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

    def test_returns_tuple_of_three(self) -> None:
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


class TestEvaluateCompressionResult:
    def test_reports_expected_concerns_and_recommendations(self) -> None:
        compression_result = {
            "final_temperature": 530.0,
            "final_pressure": 120.0,
            "total_power_hp": 1200.0,
            "stages": [
                {
                    "work_isentropic": 100.0,
                    "work_actual": 60.0,
                    "water_dropout": {"water_dropout": 0.2},
                },
                {
                    "work_isentropic": None,
                    "work_actual": 80.0,
                    "water_dropout": {"water_dropout": 0.0},
                },
            ],
        }

        result = evaluate_compression_result(compression_result)

        assert (
            "High final temperature may cause material degradation"
            in result["concerns"]
        )
        assert (
            "High pressure requires special equipment and safety measures"
            in result["concerns"]
        )
        assert (
            "High power requirement - consider multiple compressors"
            in result["concerns"]
        )
        assert "Low compression efficiency detected" in result["concerns"]
        assert (
            "CRITICAL: Temperature exceeds safe operating limits" in result["warnings"]
        )
        assert "Water dropout detected: 0.20 mol%" in result["warnings"]
        assert (
            "Install water knockout drums and drainage systems"
            in result["recommendations"]
        )
        assert result["total_water_dropout"] == 0.2
        assert result["average_efficiency"] == 0.6

    def test_no_isentropic_stages_yields_none_average_efficiency(self) -> None:
        compression_result = {
            "final_temperature": 450.0,
            "final_pressure": 80.0,
            "total_power_hp": 200.0,
            "stages": [
                {
                    "work_isentropic": None,
                    "work_actual": 55.0,
                    "water_dropout": {"water_dropout": 0.0},
                }
            ],
        }

        result = evaluate_compression_result(compression_result)

        assert result["average_efficiency"] is None
        assert result["concerns"] == []
        assert result["warnings"] == []

"""Comprehensive tests for upstream_drift_tools.process_calculators.analysis_utils.

Covers evaluate_output with all branches: successful calculation, engine failure,
non-dict result, overrides, HHV injection.
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from upstream_drift_tools.process_calculators.analysis_utils import evaluate_output


class TestEvaluateOutput:
    def setup_method(self):
        """Fresh mock engine for each test."""
        self.engine = MagicMock()

    def test_successful_calculation_returns_output_value(self):
        """Happy path: engine returns dict with output variable."""
        self.engine.calculate.return_value = {
            "efficiency": 0.85,
            "state": {"temperature": 1200.0},
            "composition": {"CO2": 0.15},
        }
        value, state, composition = evaluate_output(
            self.engine, {"flow": 100.0}, 0.0, "efficiency"
        )
        assert value == pytest.approx(0.85)
        assert state == {"temperature": 1200.0}
        assert composition == {"CO2": 0.15}

    def test_hhv_injected_when_positive(self):
        """When manual_hhv > 0, it should be included in params."""
        self.engine.calculate.return_value = {"out": 1.0}
        evaluate_output(self.engine, {"a": 1.0}, 12.5, "out")
        called_kwargs = self.engine.calculate.call_args.kwargs
        assert called_kwargs.get("manual_hhv") == 12.5

    def test_hhv_not_injected_when_zero(self):
        """When manual_hhv == 0, it should NOT be injected into params."""
        self.engine.calculate.return_value = {"out": 2.0}
        evaluate_output(self.engine, {"a": 1.0}, 0.0, "out")
        called_kwargs = self.engine.calculate.call_args.kwargs
        assert "manual_hhv" not in called_kwargs

    def test_overrides_applied_to_params(self):
        """Overrides are merged on top of base_params."""
        self.engine.calculate.return_value = {"out": 3.0}
        evaluate_output(
            self.engine, {"a": 1.0, "b": 2.0}, 0.0, "out", overrides={"b": 99.0}
        )
        called_kwargs = self.engine.calculate.call_args.kwargs
        assert called_kwargs.get("b") == 99.0

    def test_no_overrides(self):
        """When overrides=None, base_params are used unchanged."""
        self.engine.calculate.return_value = {"out": 5.0}
        evaluate_output(self.engine, {"x": 10.0}, 0.0, "out", overrides=None)
        called_kwargs = self.engine.calculate.call_args.kwargs
        assert called_kwargs.get("x") == 10.0

    def test_missing_output_variable_returns_zero(self):
        """When output_variable not in result, returns 0.0."""
        self.engine.calculate.return_value = {"other": 99.0}
        value, _, _ = evaluate_output(self.engine, {}, 0.0, "nonexistent")
        assert value == pytest.approx(0.0)

    def test_missing_state_returns_empty_dict(self):
        """When 'state' key not in result, returns empty dict."""
        self.engine.calculate.return_value = {"out": 1.0}
        _, state, _ = evaluate_output(self.engine, {}, 0.0, "out")
        assert state == {}

    def test_missing_composition_returns_empty_dict(self):
        """When 'composition' key not in result, returns empty dict."""
        self.engine.calculate.return_value = {"out": 1.0}
        _, _, composition = evaluate_output(self.engine, {}, 0.0, "out")
        assert composition == {}

    def test_engine_raises_type_error_returns_zeros(self):
        """TypeError from engine is caught and returns (0.0, {}, {})."""
        self.engine.calculate.side_effect = TypeError("bad args")
        value, state, composition = evaluate_output(self.engine, {}, 0.0, "out")
        assert value == pytest.approx(0.0)
        assert state == {}
        assert composition == {}

    def test_engine_raises_value_error_returns_zeros(self):
        self.engine.calculate.side_effect = ValueError("invalid params")
        value, state, composition = evaluate_output(self.engine, {}, 0.0, "out")
        assert value == pytest.approx(0.0)

    def test_engine_raises_zero_division_returns_zeros(self):
        self.engine.calculate.side_effect = ZeroDivisionError("division by zero")
        value, state, composition = evaluate_output(self.engine, {}, 0.0, "out")
        assert value == pytest.approx(0.0)

    def test_engine_raises_overflow_returns_zeros(self):
        self.engine.calculate.side_effect = OverflowError("overflow")
        value, state, composition = evaluate_output(self.engine, {}, 0.0, "out")
        assert value == pytest.approx(0.0)

    def test_engine_returns_non_dict_returns_zeros(self):
        """When calculate returns a non-dict (e.g. None), returns (0.0, {}, {})."""
        self.engine.calculate.return_value = None
        value, state, composition = evaluate_output(self.engine, {}, 0.0, "out")
        assert value == pytest.approx(0.0)
        assert state == {}
        assert composition == {}

    def test_engine_returns_list_returns_zeros(self):
        self.engine.calculate.return_value = [1, 2, 3]
        value, _, _ = evaluate_output(self.engine, {}, 0.0, "out")
        assert value == pytest.approx(0.0)

    def test_base_params_not_mutated(self):
        """Base params dict should not be mutated by the function."""
        self.engine.calculate.return_value = {"out": 1.0}
        base = {"a": 1.0}
        evaluate_output(self.engine, base, 1.0, "out", overrides={"b": 2.0})
        assert base == {"a": 1.0}  # unchanged

"""Tests for sidekick.process_calculators.multi_param_analysis (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from sidekick.process_calculators.multi_param_analysis import (
    run_multi_parameter_analysis,
)


class _StubEngine:
    """Simple engine: output = p1 + p2."""

    def calculate(self, **params) -> dict:
        return {
            "output": params.get("p1", 0.0) + params.get("p2", 0.0),
        }


def _make_analysis_params(
    param1: str = "p1",
    param2: str = "p2",
    output: str = "output",
    base: dict | None = None,
) -> dict:
    return {
        "param1_name": param1,
        "param2_name": param2,
        "output_variable": output,
        "base_params": base or {},
    }


class TestRunMultiParameterAnalysis:
    def test_multi_param_analysis_returns_dict(self) -> None:
        p1 = np.array([1.0, 2.0])
        p2 = np.array([10.0, 20.0])
        result = run_multi_parameter_analysis(
            _StubEngine(), _make_analysis_params(), 0.0, p1, p2
        )
        assert isinstance(result, dict)

    def test_has_output_values_key(self) -> None:
        p1 = np.array([1.0, 2.0])
        p2 = np.array([10.0])
        result = run_multi_parameter_analysis(
            _StubEngine(), _make_analysis_params(), 0.0, p1, p2
        )
        assert "output_values" in result

    def test_output_shape_matches_grid(self) -> None:
        p1 = np.array([1.0, 2.0, 3.0])
        p2 = np.array([10.0, 20.0])
        result = run_multi_parameter_analysis(
            _StubEngine(), _make_analysis_params(), 0.0, p1, p2
        )
        assert result["output_values"].shape == (3, 2)

    def test_param_names_in_result(self) -> None:
        p1 = np.array([1.0])
        p2 = np.array([1.0])
        result = run_multi_parameter_analysis(
            _StubEngine(), _make_analysis_params("x", "y"), 0.0, p1, p2
        )
        assert result["param1_name"] == "x"
        assert result["param2_name"] == "y"

    def test_param_values_in_result(self) -> None:
        p1 = np.array([1.0, 2.0])
        p2 = np.array([10.0])
        result = run_multi_parameter_analysis(
            _StubEngine(), _make_analysis_params(), 0.0, p1, p2
        )
        np.testing.assert_array_equal(result["param1_values"], p1)
        np.testing.assert_array_equal(result["param2_values"], p2)

    def test_correct_output_values(self) -> None:
        p1 = np.array([1.0, 2.0])
        p2 = np.array([10.0, 20.0])
        result = run_multi_parameter_analysis(
            _StubEngine(), _make_analysis_params(), 0.0, p1, p2
        )
        # output[i,j] = p1[i] + p2[j]
        vals = result["output_values"]
        assert vals[0, 0] == pytest.approx(11.0)
        assert vals[0, 1] == pytest.approx(21.0)
        assert vals[1, 0] == pytest.approx(12.0)
        assert vals[1, 1] == pytest.approx(22.0)

    def test_single_point_grid(self) -> None:
        result = run_multi_parameter_analysis(
            _StubEngine(),
            _make_analysis_params(),
            0.0,
            np.array([5.0]),
            np.array([3.0]),
        )
        assert result["output_values"][0, 0] == pytest.approx(8.0)

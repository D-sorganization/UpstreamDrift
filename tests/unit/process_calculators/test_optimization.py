"""Tests for upstream_drift_tools.process_calculators.optimization (Issues #1949, #1744)."""

from __future__ import annotations

import numpy as np
import pytest
from src.shared.python.upstream_drift_tools.process_calculators.optimization import (
    OptimizationHistoryEntry,
    _build_override_mapping,
    find_optimal_on_surface,
    run_adam_optimization,
)


class _StubEngine:
    """Engine that computes output = Temperature - 0.5 * (O2/Feed Ratio)."""

    def calculate(self, **params) -> dict:
        temp = params.get("Temperature", 800.0)
        o2 = params.get("O2/Feed Ratio", 0.3)
        return {
            "efficiency": float(temp) - 0.5 * float(o2),
            "state": {"temp": temp},
            "composition": {"H2": 0.4},
        }


_BASE_ANALYSIS_PARAMS = {
    "base_params": {"Temperature": 800.0, "O2/Feed Ratio": 0.3},
    "output_variable": "efficiency",
}

_PARAM_CONFIGS = [
    {"name": "Temperature", "min": 700.0, "max": 900.0, "initial": 800.0},
]


class TestBuildOverrideMapping:
    def test_maps_known_keys(self) -> None:
        result = _build_override_mapping(["Temperature"], [850.0])
        assert result["Temperature"] == pytest.approx(850.0)

    def test_ignores_unknown_keys(self) -> None:
        result = _build_override_mapping(["unknown_param"], [1.0])
        assert "unknown_param" not in result

    def test_empty_inputs(self) -> None:
        result = _build_override_mapping([], [])
        assert result == {}

    def test_multiple_params(self) -> None:
        names = ["Temperature", "O2/Feed Ratio"]
        values = [850.0, 0.4]
        result = _build_override_mapping(names, values)
        assert "Temperature" in result
        assert "O2/Feed Ratio" in result


class TestOptimizationHistoryEntry:
    def test_optimization_construction(self) -> None:
        entry = OptimizationHistoryEntry(
            iteration=1, objective=0.75, parameters={"Temperature": 800.0}
        )
        assert entry.iteration == 1
        assert entry.objective == pytest.approx(0.75)


class TestRunAdamOptimization:
    def test_returns_results_dict(self) -> None:
        result = run_adam_optimization(
            _StubEngine(),
            _BASE_ANALYSIS_PARAMS,
            0.0,
            _PARAM_CONFIGS,
            maximize=True,
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=5,
            tolerance=1e-4,
        )
        assert isinstance(result, dict)

    def test_optimization_has_expected_keys(self) -> None:
        result = run_adam_optimization(
            _StubEngine(),
            _BASE_ANALYSIS_PARAMS,
            0.0,
            _PARAM_CONFIGS,
            maximize=True,
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=5,
            tolerance=1e-4,
        )
        for key in ["best_output", "best_parameters", "history", "iterations"]:
            assert key in result

    def test_history_not_empty(self) -> None:
        result = run_adam_optimization(
            _StubEngine(),
            _BASE_ANALYSIS_PARAMS,
            0.0,
            _PARAM_CONFIGS,
            maximize=True,
            learning_rate=0.01,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=3,
            tolerance=1e-4,
        )
        assert len(result["history"]) > 0

    def test_maximize_finds_higher_value(self) -> None:
        # Temperature range 700-900; higher temp → higher efficiency
        result = run_adam_optimization(
            _StubEngine(),
            _BASE_ANALYSIS_PARAMS,
            0.0,
            _PARAM_CONFIGS,
            maximize=True,
            learning_rate=10.0,
            beta1=0.9,
            beta2=0.999,
            epsilon=1e-8,
            gradient_step=1.0,
            max_iterations=10,
            tolerance=1e-6,
        )
        assert result["best_output"] >= 800.0 - 0.5 * 0.3  # At least initial value

    def test_empty_params_raises(self) -> None:
        with pytest.raises(ValueError):
            run_adam_optimization(
                _StubEngine(),
                _BASE_ANALYSIS_PARAMS,
                0.0,
                [],
                maximize=True,
                learning_rate=0.01,
                beta1=0.9,
                beta2=0.999,
                epsilon=1e-8,
                gradient_step=1.0,
                max_iterations=3,
                tolerance=1e-4,
            )


class TestFindOptimalOnSurface:
    def _make_grid(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        x = np.linspace(0.0, 10.0, 20)
        y = np.linspace(0.0, 10.0, 20)
        # Z = -(x-5)^2 - (y-5)^2 → max at (5,5)
        X, Y = np.meshgrid(x, y)
        Z = -((X - 5.0) ** 2) - (Y - 5.0) ** 2
        return x, y, Z

    def test_grid_search_returns_dict(self) -> None:
        x, y, Z = self._make_grid()
        result = find_optimal_on_surface(x, y, Z, method="Grid Search")
        assert isinstance(result, dict)

    def test_grid_search_has_optimal_xy(self) -> None:
        x, y, Z = self._make_grid()
        result = find_optimal_on_surface(x, y, Z, method="Grid Search")
        assert "optimal_x" in result
        assert "optimal_y" in result

    def test_grid_search_near_true_optimum(self) -> None:
        x, y, Z = self._make_grid()
        result = find_optimal_on_surface(x, y, Z, method="Grid Search")
        assert abs(result["optimal_x"] - 5.0) < 2.0
        assert abs(result["optimal_y"] - 5.0) < 2.0

    def test_lbfgsb_returns_dict(self) -> None:
        x, y, Z = self._make_grid()
        result = find_optimal_on_surface(x, y, Z, method="L-BFGS-B")
        assert isinstance(result, dict)
        assert "optimal_x" in result

    def test_differential_evolution_returns_dict(self) -> None:
        x, y, Z = self._make_grid()
        result = find_optimal_on_surface(x, y, Z, method="Differential Evolution")
        assert isinstance(result, dict)

    def test_optimization_unknown_method_raises(self) -> None:
        x, y, Z = self._make_grid()
        with pytest.raises(ValueError):
            find_optimal_on_surface(x, y, Z, method="Unknown Method")

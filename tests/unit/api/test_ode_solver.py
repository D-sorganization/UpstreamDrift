"""Tests for the ODE Solver API router.

This router uses a self-contained RK4 solver with safe_eval for expressions.
Tests validate solver correctness against known analytical solutions directly
without mocking (no external shared-component dependencies).
"""

from __future__ import annotations

import math

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.ode_solver import router

_app = FastAPI()
_app.include_router(router)
client = TestClient(_app)


class TestODESolverPhysics:
    """Validate RK4 integration against known analytical solutions."""

    def test_exponential_decay(self) -> None:
        """dx/dt = -k*x has analytical solution x(t) = x0 * exp(-k*t).

        With k=1, x0=1, at t=1 expected x ≈ 0.368.
        """
        payload = {
            "derivatives": {"x": "-k * x"},
            "initial_conditions": {"x": 1.0},
            "parameters": {"k": 1.0},
            "t_start": 0.0,
            "t_end": 1.0,
            "num_points": 1001,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 200
        data = response.json()
        final_x = data["solutions"]["x"][-1]
        expected = math.exp(-1.0)
        assert final_x == pytest.approx(expected, rel=1e-3)

    def test_linear_growth(self) -> None:
        """dx/dt = c has solution x(t) = x0 + c*t."""
        payload = {
            "derivatives": {"x": "c"},
            "initial_conditions": {"x": 5.0},
            "parameters": {"c": 2.0},
            "t_start": 0.0,
            "t_end": 3.0,
            "num_points": 100,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 200
        final_x = response.json()["solutions"]["x"][-1]
        assert final_x == pytest.approx(5.0 + 2.0 * 3.0, rel=1e-3)

    def test_harmonic_oscillator(self) -> None:
        """d²x/dt² = -omega²*x with x(0)=1, v(0)=0 → x(t)=cos(omega*t).

        Split as system: dx/dt = v, dv/dt = -omega^2 * x.
        At t = pi/2 with omega=1, x ≈ 0 (±small RK4 error).
        """
        half_pi = math.pi / 2
        payload = {
            "derivatives": {"x": "v", "v": "-1 * x"},
            "initial_conditions": {"x": 1.0, "v": 0.0},
            "parameters": {},
            "t_start": 0.0,
            "t_end": half_pi,
            "num_points": 1001,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 200
        final_x = response.json()["solutions"]["x"][-1]
        assert final_x == pytest.approx(0.0, abs=1e-3)

    def test_summary_statistics(self) -> None:
        """Variable summaries should reflect solution bounds correctly."""
        payload = {
            "derivatives": {"x": "-x"},
            "initial_conditions": {"x": 2.0},
            "parameters": {},
            "t_start": 0.0,
            "t_end": 5.0,
            "num_points": 100,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 200
        data = response.json()
        summary = next(s for s in data["variable_summaries"] if s["name"] == "x")
        assert summary["initial_value"] == pytest.approx(2.0, rel=1e-5)
        assert summary["final_value"] < summary["initial_value"]
        assert summary["max_value"] == pytest.approx(2.0, rel=1e-3)
        assert summary["min_value"] == summary["final_value"]

    def test_missing_initial_condition_returns_422(self) -> None:
        """Missing initial condition for a derivative variable → 422."""
        payload = {
            "derivatives": {"x": "-x", "y": "x"},
            "initial_conditions": {"x": 1.0},  # y missing
            "parameters": {},
            "t_start": 0.0,
            "t_end": 1.0,
            "num_points": 10,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 422
        assert "y" in str(response.json()["detail"])

    def test_correct_number_of_timepoints(self) -> None:
        """Solver must return exactly the requested number of time points."""
        payload = {
            "derivatives": {"x": "-x"},
            "initial_conditions": {"x": 1.0},
            "parameters": {},
            "t_start": 0.0,
            "t_end": 2.0,
            "num_points": 50,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert len(data["times"]) == 50
        assert len(data["solutions"]["x"]) == 50

    def test_multi_variable_system(self) -> None:
        """Two-variable coupled system should return solutions for both variables."""
        payload = {
            "derivatives": {"a": "-2 * a + b", "b": "a - 3 * b"},
            "initial_conditions": {"a": 1.0, "b": 0.5},
            "parameters": {},
            "t_start": 0.0,
            "t_end": 2.0,
            "num_points": 200,
        }
        response = client.post("/api/calc/ode-solver", json=payload)
        assert response.status_code == 200
        data = response.json()
        assert "a" in data["solutions"]
        assert "b" in data["solutions"]
        assert len(data["solutions"]["a"]) == 200

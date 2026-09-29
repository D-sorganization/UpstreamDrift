"""Tests for src.shared.python.calc_backend.contracts.ode_solver (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.ode_solver import (
    ODESolverRequest,
    ODESolverResponse,
    ODEVariableSummary,
)


class TestODESolverRequest:
    def _valid_request(self, **kwargs) -> ODESolverRequest:
        defaults = {
            "derivatives": {"y": "-k*y"},
            "initial_conditions": {"y": 100.0},
        }
        defaults.update(kwargs)
        return ODESolverRequest(**defaults)

    def test_contracts_ode_solver_valid_construction(self) -> None:
        req = self._valid_request()
        assert isinstance(req, ODESolverRequest)

    def test_contracts_ode_solver_default_parameters_empty(self) -> None:
        req = self._valid_request()
        assert req.parameters == {}

    def test_default_t_start(self) -> None:
        req = self._valid_request()
        assert req.t_start == pytest.approx(0.0)

    def test_default_t_end(self) -> None:
        req = self._valid_request()
        assert req.t_end == pytest.approx(20.0)

    def test_contracts_ode_solver_default_num_points(self) -> None:
        req = self._valid_request()
        assert req.num_points == 100

    def test_derivatives_stored(self) -> None:
        req = self._valid_request(derivatives={"x": "v", "v": "-omega*x"})
        assert "x" in req.derivatives
        assert "v" in req.derivatives

    def test_parameters_stored(self) -> None:
        req = self._valid_request(parameters={"k": 0.1})
        assert req.parameters["k"] == pytest.approx(0.1)

    def test_t_end_must_be_positive(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(t_end=0.0)

    def test_t_start_negative_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(t_start=-1.0)

    def test_num_points_below_10_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(num_points=5)

    def test_num_points_over_10000_rejected(self) -> None:
        with pytest.raises(ValidationError):
            self._valid_request(num_points=20000)


class TestODEVariableSummary:
    def test_contracts_ode_solver_construction(self) -> None:
        summary = ODEVariableSummary(
            name="y",
            initial_value=100.0,
            final_value=0.5,
            min_value=0.5,
            max_value=100.0,
        )
        assert isinstance(summary, ODEVariableSummary)

    def test_contracts_ode_solver_name_stored(self) -> None:
        summary = ODEVariableSummary(
            name="temperature",
            initial_value=300.0,
            final_value=350.0,
            min_value=300.0,
            max_value=350.0,
        )
        assert summary.name == "temperature"

    def test_values_stored(self) -> None:
        summary = ODEVariableSummary(
            name="y",
            initial_value=100.0,
            final_value=0.5,
            min_value=0.5,
            max_value=100.0,
        )
        assert summary.initial_value == pytest.approx(100.0)
        assert summary.final_value == pytest.approx(0.5)


class TestODESolverResponse:
    def _make_response(self) -> ODESolverResponse:
        return ODESolverResponse(
            times=[0.0, 1.0, 2.0],
            solutions={"y": [100.0, 90.5, 81.9]},
            variable_summaries=[
                ODEVariableSummary(
                    name="y",
                    initial_value=100.0,
                    final_value=81.9,
                    min_value=81.9,
                    max_value=100.0,
                )
            ],
        )

    def test_contracts_ode_solver_construction(self) -> None:
        resp = self._make_response()
        assert isinstance(resp, ODESolverResponse)

    def test_default_success_true(self) -> None:
        resp = self._make_response()
        assert resp.solver_status == "success"

    def test_default_message(self) -> None:
        resp = self._make_response()
        assert "computed" in resp.message.lower() or len(resp.message) > 0

    def test_times_stored(self) -> None:
        resp = self._make_response()
        assert len(resp.times) == 3

    def test_solutions_stored(self) -> None:
        resp = self._make_response()
        assert "y" in resp.solutions
        assert len(resp.solutions["y"]) == 3

    def test_variable_summaries_stored(self) -> None:
        resp = self._make_response()
        assert len(resp.variable_summaries) == 1
        assert resp.variable_summaries[0].name == "y"

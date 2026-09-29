# mypy: disable-error-code=no-untyped-def

from concurrent.futures import Future
from typing import Any
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from sidekick.process_calculators import multi_param_analysis as mpa
from sidekick.process_calculators.multi_param_analysis import (
    _evaluate_single_point,
    run_multi_parameter_analysis,
    run_multi_parameter_analysis_parallel,
)


@pytest.fixture
def analysis_params() -> dict:
    return {
        "base_params": {"base_1": 1.0},
        "param1_name": "p1",
        "param2_name": "p2",
        "output_variable": "out_var",
    }


def test_evaluate_single_point() -> None:
    engine = MagicMock()
    with patch(
        "upstream_drift_tools.process_calculators.multi_param_analysis.evaluate_output"
    ) as mock_eval:
        mock_eval.return_value = (42.0, None, None)

        i, j, output = _evaluate_single_point(
            0, 1, 10.0, 20.0, engine, {"base_1": 1.0}, 15.0, "p1", "p2", "out_var"
        )

        assert i == 0
        assert j == 1
        assert output == 42.0
        mock_eval.assert_called_once_with(
            engine, {"base_1": 1.0}, 15.0, "out_var", {"p1": 10.0, "p2": 20.0}
        )


def test_run_multi_parameter_analysis(analysis_params: dict) -> None:
    engine = MagicMock()
    p1_vals = np.array([1.0, 2.0])
    p2_vals = np.array([3.0, 4.0])

    with patch(
        "upstream_drift_tools.process_calculators.multi_param_analysis.evaluate_output"
    ) as mock_eval:
        mock_eval.return_value = (42.0, None, None)

        result = run_multi_parameter_analysis(
            engine, analysis_params, 15.0, p1_vals, p2_vals
        )

        assert result["param1_name"] == "p1"
        assert result["param2_name"] == "p2"
        assert result["output_name"] == "out_var"
        assert result["output_values"].shape == (2, 2)
        assert np.all(result["output_values"] == 42.0)
        assert mock_eval.call_count == 4


class DummyEngine:
    def calculate(self, **kwargs) -> Any:
        val = kwargs.get("p1", 0) + kwargs.get("p2", 0)
        return {"out_var": val}


class ImmediateExecutor:
    def __init__(self, max_workers: int | None = None) -> None:
        self.max_workers = max_workers

    def __enter__(self) -> "ImmediateExecutor":
        return self

    def __exit__(self, *exc_info: object) -> None:
        return None

    def submit(self, fn: Any, *args: object) -> Future:
        future: Future = Future()
        try:
            future.set_result(fn(*args))
        except Exception as exc:  # pragma: no cover - exercised via Future.result
            future.set_exception(exc)
        return future


def test_run_multi_parameter_analysis_parallel(
    analysis_params: dict, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(mpa, "ProcessPoolExecutor", ImmediateExecutor)
    monkeypatch.setattr(mpa, "as_completed", lambda futures: futures)
    engine = DummyEngine()
    p1_vals = np.array([1.0, 2.0])
    p2_vals = np.array([3.0, 4.0])

    result = run_multi_parameter_analysis_parallel(
        engine, analysis_params, 15.0, p1_vals, p2_vals, max_workers=2
    )

    assert result["output_values"].shape == (2, 2)
    assert result["output_values"][0, 0] == 4.0  # 1.0 + 3.0
    assert result["output_values"][1, 1] == 6.0  # 2.0 + 4.0

from typing import Any

from upstream_drift_tools.process_calculators.analysis_utils import evaluate_output


class DummyEngine:
    def calculate(self, **kwargs) -> Any:
        if kwargs.get("fail", False):
            raise ValueError("Test failure")

        return {
            "test_output": 42.0,
            "state": {"temp": 1200.0},
            "composition": {"H2": 50.0},
        }


def test_evaluate_output_success() -> None:
    engine = DummyEngine()

    val, state, comp = evaluate_output(
        engine,
        base_params={"base": 1.0},
        manual_hhv=15.0,
        output_variable="test_output",
        overrides={"override": 2.0},
    )

    assert val == 42.0
    assert state == {"temp": 1200.0}
    assert comp == {"H2": 50.0}


def test_evaluate_output_failure() -> None:
    engine = DummyEngine()

    # Trigger exception
    val, state, comp = evaluate_output(
        engine,
        base_params={"fail": True},
        manual_hhv=0.0,
        output_variable="test_output",
    )

    assert val == 0.0
    assert state == {}
    assert comp == {}


def test_evaluate_output_type_error() -> None:
    class BadEngine:
        def calculate(self, **kwargs) -> Any:
            return ["not a dict"]

    engine = BadEngine()
    val, state, comp = evaluate_output(
        engine,
        base_params={"base": 1.0},
        manual_hhv=0.0,
        output_variable="test_output",
    )

    assert val == 0.0
    assert state == {}
    assert comp == {}

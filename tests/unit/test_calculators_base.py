"""Tests for sidekick.calculators.base (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from sidekick.calculators.base import (
    BaseCalculationEngine,
)


class _ConcreteEngine(BaseCalculationEngine):
    """Minimal concrete implementation for testing the ABC."""

    def calculate(self, *args, **kwargs) -> dict:
        value = kwargs.get("x", 0.0)
        return {"result": float(value) * 2.0, "status": "ok"}


class _AddingEngine(BaseCalculationEngine):
    def calculate(self, *args, **kwargs) -> dict:
        a = float(kwargs.get("a", 0.0))
        b = float(kwargs.get("b", 0.0))
        return {"sum": a + b}


class TestBaseCalculationEngineAbstract:
    def test_cannot_instantiate_directly(self) -> None:
        with pytest.raises(TypeError):
            BaseCalculationEngine()  # type: ignore[abstract]

    def test_subclass_without_calculate_fails(self) -> None:
        class BadEngine(BaseCalculationEngine):
            pass

        with pytest.raises(TypeError):
            BadEngine()  # type: ignore[abstract]


class TestConcreteEngine:
    def test_calculators_base_construction(self) -> None:
        engine = _ConcreteEngine()
        assert engine is not None

    def test_calculate_returns_dict(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate(x=5.0)
        assert isinstance(result, dict)

    def test_calculate_doubles_input(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate(x=3.0)
        assert result["result"] == pytest.approx(6.0)

    def test_calculate_zero(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate(x=0.0)
        assert result["result"] == pytest.approx(0.0)

    def test_calculate_returns_status(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate(x=1.0)
        assert result["status"] == "ok"

    def test_calculate_no_args(self) -> None:
        engine = _ConcreteEngine()
        result = engine.calculate()
        assert "result" in result

    def test_adding_engine(self) -> None:
        engine = _AddingEngine()
        result = engine.calculate(a=3.0, b=4.0)
        assert result["sum"] == pytest.approx(7.0)

    def test_is_instance_of_base(self) -> None:
        engine = _ConcreteEngine()
        assert isinstance(engine, BaseCalculationEngine)

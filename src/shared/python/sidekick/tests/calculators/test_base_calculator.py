from typing import Any

import pytest
from upstream_drift_tools.calculators.base import BaseCalculationEngine


class DummyEngine(BaseCalculationEngine):
    def calculate(self, *args, **kwargs) -> Any:
        return {"args": args, "kwargs": kwargs}


def test_base_calculation_engine() -> None:
    engine = DummyEngine()
    result = engine.calculate(1, 2, a=3)
    assert result["args"] == (1, 2)
    assert result["kwargs"] == {"a": 3}

    # Verify that it cannot be instantiated directly
    with pytest.raises(TypeError):
        BaseCalculationEngine()

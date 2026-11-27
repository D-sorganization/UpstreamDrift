"""Oracle 6 tests: syngas non-finite typed failure (Issue #6003 Slice 5 / #14934)."""

from __future__ import annotations

import math
from typing import Any, cast
from unittest.mock import MagicMock

import pytest
from fastapi import HTTPException
from src.shared.python.calc_backend.contracts.syngas_water import (
    SyngasCalculationStatus,
    SyngasNumericalFailureError,
    SyngasWaterRequest,
)
from src.shared.python.calc_backend.routers.syngas_water import (
    _fallback_calculate,
    _require_finite,
    _sanitize,
    calculate_syngas_water,
)

pytestmark = pytest.mark.unit


def _req() -> SyngasWaterRequest:
    return SyngasWaterRequest(temperature_c=40.0, pressure_bar=30.0)


def test_sanitize_and_require_finite_valid() -> None:
    assert _require_finite(42.5, "temp") == 42.5
    assert _sanitize(0.0, "zero") == 0.0 and _sanitize(-15.2, "neg") == -15.2


@pytest.mark.parametrize("bad_val", [float("nan"), float("inf"), float("-inf")])
def test_sanitize_rejects_non_finite(bad_val: float) -> None:
    with pytest.raises(SyngasNumericalFailureError) as exc_info:
        _sanitize(bad_val, "mole_fraction_water")
    err = exc_info.value
    assert err.status == SyngasCalculationStatus.NUMERICAL_FAILURE
    assert err.field_name == "mole_fraction_water"
    assert math.isnan(err.value) if math.isnan(bad_val) else err.value == bad_val
    assert "mole_fraction_water" in str(err) and "non-finite" in str(err).lower()


def test_syngas_water_endpoint_success() -> None:
    assert calculate_syngas_water(_req()).status == SyngasCalculationStatus.SUCCEEDED


def test_syngas_water_endpoint_rejects_nan_result(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.sidekick.process_calculators.syngas_water_calculator as swc

    calc = MagicMock()
    calc.calculate_water_content.return_value = MagicMock(
        mole_fraction_water=float("nan"),
        water_content_mg_per_nm3=100.0,
        water_content_ppmv=100.0,
        water_content_g_per_m3=100.0,
        water_content_lb_per_mmscf=100.0,
        vapor_pressure_bar=1.0,
        dew_point_c=10.0,
        dew_point_margin_c=10.0,
    )
    monkeypatch.setattr(swc, "SyngasWaterCalculator", lambda: calc)
    with pytest.raises(HTTPException) as exc_info:
        calculate_syngas_water(_req())
    assert exc_info.value.status_code == 422
    d = cast(dict[str, Any], exc_info.value.detail)
    assert d["status"] == "NUMERICAL_FAILURE" and d["code"] == "NON_FINITE_OUTPUT"
    assert (
        d["field"] == "mole_fraction_water"
        and "non-finite" in str(d["message"]).lower()
    )


def test_syngas_water_endpoint_rejects_inf_risk(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.sidekick.process_calculators.syngas_water_calculator as swc

    monkeypatch.setattr(
        swc,
        "estimate_condensation_risk",
        lambda t, p: {
            "temperature_margin_c": float("inf"),
            "condensation_risk": "High",
            "recommended_temperature_c": 50.0,
        },
    )
    with pytest.raises(HTTPException) as exc_info:
        calculate_syngas_water(_req())
    assert exc_info.value.status_code == 422
    d = cast(dict[str, Any], exc_info.value.detail)
    assert d["status"] == "NUMERICAL_FAILURE" and d["code"] == "NON_FINITE_OUTPUT"
    assert (
        d["field"] == "temperature_margin_c"
        and "non-finite" in str(d["message"]).lower()
    )


def test_fallback_calculate_rejects_non_finite(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    import src.shared.python.sidekick.process_calculators.water_vapor_pressure as wvp

    monkeypatch.setattr(wvp, "antoine_pressure_pa", lambda a, b, c, t: float("nan"))
    with pytest.raises(SyngasNumericalFailureError) as exc_info:
        _fallback_calculate(_req())
    assert exc_info.value.status == SyngasCalculationStatus.NUMERICAL_FAILURE

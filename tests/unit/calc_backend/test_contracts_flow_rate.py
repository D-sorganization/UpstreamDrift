"""Tests for src.shared.python.calc_backend.contracts.flow_rate (Issues #1949, #1744)."""

from __future__ import annotations

import pytest
from pydantic import ValidationError
from src.shared.python.calc_backend.contracts.flow_rate import (
    FlowRateConvertRequest,
    FlowRateConvertResponse,
)


class TestFlowRateConvertRequest:
    def test_contracts_flow_rate_basic_construction(self) -> None:
        req = FlowRateConvertRequest(value=1.0, from_unit="kg/s", to_unit="lb/h")
        assert req.value == pytest.approx(1.0)
        assert req.from_unit == "kg/s"
        assert req.to_unit == "lb/h"

    def test_default_category_is_mass(self) -> None:
        req = FlowRateConvertRequest(value=1.0, from_unit="kg/s", to_unit="lb/h")
        assert req.category == "mass"

    def test_custom_category(self) -> None:
        req = FlowRateConvertRequest(
            value=1.0, from_unit="mol/s", to_unit="kmol/h", category="molar"
        )
        assert req.category == "molar"

    def test_volumetric_category(self) -> None:
        req = FlowRateConvertRequest(
            value=1.0, from_unit="m3/s", to_unit="L/min", category="volumetric"
        )
        assert req.category == "volumetric"

    def test_missing_value_raises(self) -> None:
        with pytest.raises(ValidationError):
            FlowRateConvertRequest(from_unit="kg/s", to_unit="lb/h")  # type: ignore[call-arg]

    def test_missing_from_unit_raises(self) -> None:
        with pytest.raises(ValidationError):
            FlowRateConvertRequest(value=1.0, to_unit="lb/h")  # type: ignore[call-arg]

    def test_missing_to_unit_raises(self) -> None:
        with pytest.raises(ValidationError):
            FlowRateConvertRequest(value=1.0, from_unit="kg/s")  # type: ignore[call-arg]

    def test_negative_value_accepted(self) -> None:
        # No constraint on sign
        req = FlowRateConvertRequest(value=-5.0, from_unit="kg/s", to_unit="lb/h")
        assert req.value == pytest.approx(-5.0)

    def test_zero_value_accepted(self) -> None:
        req = FlowRateConvertRequest(value=0.0, from_unit="kg/s", to_unit="lb/h")
        assert req.value == pytest.approx(0.0)


class TestFlowRateConvertResponse:
    def test_contracts_flow_rate_basic_construction(self) -> None:
        resp = FlowRateConvertResponse(
            result=2.2046, from_unit="kg/s", to_unit="lb/s", category="mass"
        )
        assert resp.result == pytest.approx(2.2046)

    def test_has_expected_fields(self) -> None:
        resp = FlowRateConvertResponse(
            result=3600.0, from_unit="kg/s", to_unit="kg/h", category="mass"
        )
        assert resp.from_unit == "kg/s"
        assert resp.to_unit == "kg/h"
        assert resp.category == "mass"

    def test_zero_result_accepted(self) -> None:
        resp = FlowRateConvertResponse(
            result=0.0, from_unit="kg/s", to_unit="lb/h", category="mass"
        )
        assert resp.result == pytest.approx(0.0)

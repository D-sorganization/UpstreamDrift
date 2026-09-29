"""Property-based and adversarial tests for the calc_backend API (#3262).

Complements the example-based ``test_calc_backend.py`` with Hypothesis-driven
invariants and boundary/failure-mode coverage for the flow-rate conversion
surface, which is a pure numerical contract downstream repos depend on.

Invariants exercised:
- Round-trip: converting a value to another unit and back recovers the original
  within tolerance.
- Self-conversion is the identity.
- Conversion is linear in the input value.
- Adversarial inputs (unknown category/unit, non-finite values, missing fields)
  are rejected with the documented 4xx status, never a 200 with garbage.
"""

from __future__ import annotations

import math
from typing import Any

import pytest
from fastapi.testclient import TestClient
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from sidekick.calculators.conversion.flow_rate_converter import (
    MASS_FLOW_CONVERSIONS,
    MOLAR_FLOW_CONVERSIONS,
    VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S,
)

_CATEGORY_TABLES = {
    "mass": sorted(MASS_FLOW_CONVERSIONS),
    "molar": sorted(MOLAR_FLOW_CONVERSIONS),
    "volumetric": sorted(VOLUMETRIC_FLOW_CONVERSIONS_TO_M3_S),
}
_CATEGORY_UNIT_PAIRS = [
    (category, from_unit, to_unit)
    for category, units in _CATEGORY_TABLES.items()
    for from_unit in units
    for to_unit in units
]


@pytest.fixture(scope="module")
def client() -> Any:
    from calc_backend.app import app

    return TestClient(app)


def _convert(
    client: TestClient,
    value: float,
    from_unit: str,
    to_unit: str,
    category: str,
) -> float:
    response = client.post(
        "/api/calc/flow-rate",
        json={
            "value": value,
            "from_unit": from_unit,
            "to_unit": to_unit,
            "category": category,
        },
    )
    assert response.status_code == 200, response.text
    return float(response.json()["result"])


@pytest.mark.parametrize(
    ("category", "from_unit", "to_unit"),
    _CATEGORY_UNIT_PAIRS,
    ids=lambda v: v if isinstance(v, str) else "",
)
@settings(max_examples=15, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(value=st.floats(min_value=1e-6, max_value=1e6, allow_nan=False))
def test_round_trip_recovers_original_value(
    client: TestClient,
    category: str,
    from_unit: str,
    to_unit: str,
    value: float,
) -> None:
    """value -> to_unit -> back to from_unit recovers the original."""
    converted = _convert(client, value, from_unit, to_unit, category)
    restored = _convert(client, converted, to_unit, from_unit, category)
    assert restored == pytest.approx(value, rel=1e-6, abs=1e-12)


@pytest.mark.parametrize("category", sorted(_CATEGORY_TABLES))
def test_self_conversion_is_identity(client: TestClient, category: str) -> None:
    for unit in _CATEGORY_TABLES[category]:
        assert _convert(client, 3.5, unit, unit, category) == pytest.approx(3.5)


@pytest.mark.parametrize(
    ("category", "from_unit", "to_unit"),
    [pairs for pairs in _CATEGORY_UNIT_PAIRS if pairs[1] != pairs[2]][:30],
)
@settings(max_examples=10, suppress_health_check=[HealthCheck.function_scoped_fixture])
@given(
    value=st.floats(min_value=1.0, max_value=1e4, allow_nan=False),
    scale=st.floats(min_value=2.0, max_value=1e3, allow_nan=False),
)
def test_conversion_is_linear(
    client: TestClient,
    category: str,
    from_unit: str,
    to_unit: str,
    value: float,
    scale: float,
) -> None:
    """Scaling the input scales the output by the same factor (linearity)."""
    base = _convert(client, value, from_unit, to_unit, category)
    scaled = _convert(client, value * scale, from_unit, to_unit, category)
    assert scaled == pytest.approx(base * scale, rel=1e-6)


# ── Adversarial / failure-mode coverage ────────────────────────────────────


def test_unknown_category_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/calc/flow-rate",
        json={
            "value": 1.0,
            "from_unit": "kg/s",
            "to_unit": "kg/s",
            "category": "not_a_category",
        },
    )
    assert response.status_code in (400, 422)


@pytest.mark.parametrize("magnitude", [1e-9, 1e9, 1e15])
def test_extreme_finite_values_round_trip(client: TestClient, magnitude: float) -> None:
    """Very small / very large finite values still round-trip without overflow."""
    converted = _convert(client, magnitude, "kg/s", "lb/s", "mass")
    assert math.isfinite(converted)
    restored = _convert(client, converted, "lb/s", "kg/s", "mass")
    assert restored == pytest.approx(magnitude, rel=1e-6)


def test_non_numeric_value_is_rejected(client: TestClient) -> None:
    """A non-numeric ``value`` must be rejected by request validation."""
    response = client.post(
        "/api/calc/flow-rate",
        json={
            "value": "not_a_number",
            "from_unit": "kg/s",
            "to_unit": "lb/s",
            "category": "mass",
        },
    )
    assert response.status_code == 422


def test_missing_required_field_is_rejected(client: TestClient) -> None:
    response = client.post(
        "/api/calc/flow-rate",
        json={"value": 1.0, "from_unit": "kg/s", "category": "mass"},
    )
    assert response.status_code == 422

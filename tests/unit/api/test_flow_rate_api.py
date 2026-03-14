"""Tests for the Flow Rate Converter API router.

This test file validates the dictionary table mapping conversion endpoint.
Because this endpoint relies purely on a constant mapping rather than an object
calculator Engine, we validate its proper resolution paths directly without an
engine patch.
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from src.shared.python.calc_backend.routers.flow_rate import router

client = TestClient(router)


def test_convert_flow_rate_mass_success() -> None:
    """Validate mass flow rate conversions internally using correct scaling."""
    payload = {
        "value": 1.0,
        "from_unit": "kg_s",
        "to_unit": "kg_hr",
        "category": "mass",
    }

    response = client.post("/", json=payload)

    assert response.status_code == 200
    data = response.json()
    assert data["result"] == 3600.0


def test_convert_flow_rate_invalid_category() -> None:
    """Validate fallback error protection on unknowns."""
    payload = {
        "value": 1.0,
        "from_unit": "kg_s",
        "to_unit": "kg_hr",
        "category": "UNKNOWN_SPEED_METRIC",
    }

    response = client.post("/", json=payload)

    assert response.status_code == 422
    assert "Unknown category" in response.json()["detail"]


def test_convert_flow_rate_invalid_source_unit() -> None:
    """Validate fallback error protection on bad from_unit."""
    payload = {
        "value": 1.0,
        "from_unit": "bananas_per_sec",
        "to_unit": "kg_hr",
        "category": "mass",
    }

    response = client.post("/", json=payload)

    assert response.status_code == 422
    assert "Unknown from_unit" in response.json()["detail"]


def test_convert_flow_rate_invalid_target_unit() -> None:
    """Validate fallback error protection on bad to_unit."""
    payload = {
        "value": 1.0,
        "from_unit": "kg_s",
        "to_unit": "stones_per_minute",
        "category": "mass",
    }

    response = client.post("/", json=payload)

    assert response.status_code == 422
    assert "Unknown to_unit" in response.json()["detail"]

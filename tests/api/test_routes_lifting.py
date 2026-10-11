"""API tests for the LIFT-1 baseline routes (LIFT-8, #11748)."""

from __future__ import annotations

from typing import Any

import pytest
from fastapi.testclient import TestClient

from src.api.routes.lifting import (
    _cached_baseline_receipt,
    get_lifting_baseline_receipt,
)
from src.api.server import app
from src.shared.python.lifting.pack_audit.baseline import SCHEMA

pytestmark = pytest.mark.unit


@pytest.fixture
def synthetic_receipt() -> dict[str, Any]:
    return {
        "schema": SCHEMA,
        "generated_utc": "2026-01-01T00:00:00+00:00",
        "anthropometry": {"body_mass_kg": 80.0},
        "tolerances": {"position_m": 0.02, "mass_rel": 1e-6},
        "packs": {
            "mujoco": {"repo": "MuJoCo_Models", "commit": "abc123", "licence": "MIT"},
        },
        "results": {
            "squat": {
                "mujoco": {
                    "structure": {"n_bodies": 10, "nq": 20, "nv": 18},
                    "start": {
                        "total_mass_kg": 100.0,
                        "bar_above_sole_m": 1.0,
                        "hand_mid_above_sole_m": 0.9,
                    },
                    "start_contact": {
                        "value_n": 0.0,
                        "non_ground_normal_force_n": 5.0,
                        "n_ground_contacts": 1,
                        "n_non_ground_contacts": 0,
                        "reason": None,
                    },
                    "smoke": {"loaded": True, "stepped": True, "max_abs_qvel": 0.1},
                    "phases": [],
                },
            },
        },
        "comparisons": {},
        "gaps": [
            {"key": "no_grf", "title": "x", "engines": ["mujoco"], "evidence": []}
        ],
        "deferred": [],
    }


@pytest.fixture
def client(synthetic_receipt: dict[str, Any]) -> TestClient:
    app.dependency_overrides[get_lifting_baseline_receipt] = lambda: synthetic_receipt
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.pop(get_lifting_baseline_receipt, None)


def test_get_baseline_metadata(client: TestClient) -> None:
    response = client.get("/api/lifting/baseline")
    assert response.status_code == 200
    data = response.json()
    assert data["schema"] == SCHEMA
    assert data["lifts"] == ["squat"]
    assert data["gap_count"] == 1


def test_get_baseline_lift_happy_path(client: TestClient) -> None:
    response = client.get("/api/lifting/baseline/lifts/squat")
    assert response.status_code == 200
    data = response.json()
    assert data["lift"] == "squat"
    assert data["engines"][0]["engine"] == "mujoco"
    assert data["engines"][0]["total_mass_kg"] == {"value": 100.0, "reason": None}


def test_get_baseline_lift_unknown_lift_returns_404(client: TestClient) -> None:
    response = client.get("/api/lifting/baseline/lifts/not_a_lift")
    assert response.status_code == 404
    assert "squat" in response.json()["detail"]


def test_missing_receipt_returns_503(monkeypatch: pytest.MonkeyPatch) -> None:
    import src.api.routes.lifting as lifting_routes

    def _raise() -> dict[str, Any]:
        raise FileNotFoundError("receipt not found at C:/private/receipt.json")

    monkeypatch.setattr(lifting_routes, "load_baseline", _raise)
    _cached_baseline_receipt.cache_clear()
    try:
        with TestClient(app) as test_client:
            response = test_client.get("/api/lifting/baseline")
        assert response.status_code == 503
        assert "C:/private" not in response.json()["detail"]
    finally:
        _cached_baseline_receipt.cache_clear()

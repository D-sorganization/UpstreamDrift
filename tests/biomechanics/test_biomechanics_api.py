"""HTTP scientific contracts, including JSON gap preservation."""

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.biomechanics import router

pytestmark = pytest.mark.unit


@pytest.fixture
def client():
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


def test_compute_and_display_round_trip(client):
    payload = {
        "times": [0, 1, 2],
        "source": "analytic",
        "world_frame": "Z-up",
        "segments": {
            "pelvis": {
                "positions": [[0, 0, 0]] * 3,
                "rotations": np.tile(np.eye(3), (3, 1, 1)).tolist(),
            }
        },
    }
    response = client.post("/biomechanics/compute", json=payload)
    assert response.status_code == 200, response.text
    result = response.json()
    assert result["times"] == [0, 1, 2]
    displayed = client.post(
        "/biomechanics/display", json={"result": result, "angle_unit": "deg"}
    )
    assert displayed.status_code == 200, displayed.text
    assert displayed.json()["channels"]


def test_bad_trajectory_is_client_error(client):
    response = client.post("/biomechanics/compute", json={"times": [1, 0]})
    assert response.status_code == 422


def test_conversion_explicit_sequences(client):
    response = client.post(
        "/biomechanics/convert",
        json={
            "values": [[10, 20, 30]],
            "source_representation": "euler",
            "target_representation": "matrix",
            "source_sequence": "XYZ",
            "source_degrees": True,
        },
    )
    assert response.status_code == 200, response.text
    assert np.asarray(response.json()["values"]).shape == (1, 3, 3)


def test_invalid_display_unit_is_client_error(client):
    response = client.post(
        "/biomechanics/display", json={"result": {}, "angle_unit": "radians-ish"}
    )
    assert response.status_code == 422

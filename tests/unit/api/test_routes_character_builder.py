"""Unit tests for the character builder API route."""

from __future__ import annotations

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.character_builder import router


@pytest.fixture
def app() -> FastAPI:
    """Create a FastAPI app with the character builder router."""
    test_app = FastAPI()
    test_app.include_router(router)
    return test_app


@pytest.fixture
def client(app: FastAPI) -> TestClient:
    """Create a test client."""
    return TestClient(app)


def test_generate_character_urdf_success(client: TestClient) -> None:
    """Test generating URDF successfully with valid parameters."""
    payload = {
        "height_m": 1.8,
        "mass_kg": 80.0,
        "build_type": "average",
    }
    response = client.post("/character-builder/generate", json=payload)
    assert response.status_code == 200
    assert "text/xml" in response.headers["content-type"]
    assert "attachment" in response.headers["content-disposition"]
    xml_content = response.text
    assert "<robot" in xml_content
    # Check that at least some expected humanoid links are present
    assert 'name="thorax"' in xml_content or 'name="pelvis"' in xml_content


def test_generate_character_urdf_invalid_height(client: TestClient) -> None:
    """Test height validation bounds."""
    payload = {
        "height_m": 1.2,  # Too short (min 1.5)
        "mass_kg": 80.0,
        "build_type": "average",
    }
    response = client.post("/character-builder/generate", json=payload)
    assert response.status_code == 422


def test_generate_character_urdf_invalid_weight(client: TestClient) -> None:
    """Test weight validation bounds."""
    payload = {
        "height_m": 1.8,
        "mass_kg": 300.0,  # Too heavy (max 150)
        "build_type": "average",
    }
    response = client.post("/character-builder/generate", json=payload)
    assert response.status_code == 422


def test_generate_character_urdf_invalid_build_type(client: TestClient) -> None:
    """Test build type validation."""
    payload = {
        "height_m": 1.8,
        "mass_kg": 80.0,
        "build_type": "extremely_muscular",  # Invalid enum value
    }
    response = client.post("/character-builder/generate", json=payload)
    assert response.status_code == 422

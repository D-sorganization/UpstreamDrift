"""Unit tests for the character builder API route."""

from __future__ import annotations

import sys
import builtins
from types import ModuleType

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.character_builder import router

pytestmark = pytest.mark.unit


def _install_fake_character_builder_provider(monkeypatch: pytest.MonkeyPatch) -> None:
    package = ModuleType("humanoid_character_builder")
    core = ModuleType("humanoid_character_builder.core")
    body_parameters = ModuleType("humanoid_character_builder.core.body_parameters")
    generators = ModuleType("humanoid_character_builder.generators")
    urdf_generator = ModuleType("humanoid_character_builder.generators.urdf_generator")

    class FakeBuildType:
        MESOMORPH = "mesomorph"
        AVERAGE = "average"
        ENDOMORPH = "endomorph"
        ECTOMORPH = "ectomorph"

    class FakeBodyParameters:
        def __init__(self, *, height_m: float, mass_kg: float, build_type: str) -> None:
            self.height_m = height_m
            self.mass_kg = mass_kg
            self.build_type = build_type

    class FakeHumanoidURDFGenerator:
        def generate(self, params: FakeBodyParameters) -> str:
            return (
                f'<robot name="{params.build_type}_humanoid">'
                '<link name="pelvis" />'
                "</robot>"
            )

    body_parameters.BodyParameters = FakeBodyParameters
    body_parameters.BuildType = FakeBuildType
    urdf_generator.HumanoidURDFGenerator = FakeHumanoidURDFGenerator

    monkeypatch.setitem(sys.modules, "humanoid_character_builder", package)
    monkeypatch.setitem(sys.modules, "humanoid_character_builder.core", core)
    monkeypatch.setitem(
        sys.modules,
        "humanoid_character_builder.core.body_parameters",
        body_parameters,
    )
    monkeypatch.setitem(
        sys.modules, "humanoid_character_builder.generators", generators
    )
    monkeypatch.setitem(
        sys.modules,
        "humanoid_character_builder.generators.urdf_generator",
        urdf_generator,
    )


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


def test_generate_character_urdf_success(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Test generating URDF successfully with valid parameters."""
    _install_fake_character_builder_provider(monkeypatch)
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


def test_generate_character_urdf_reports_missing_provider(
    client: TestClient,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A missing optional provider should not break API route discovery."""
    real_import = builtins.__import__

    def deny_character_builder_import(
        name: str,
        globals: dict[str, object] | None = None,
        locals: dict[str, object] | None = None,
        fromlist: tuple[str, ...] = (),
        level: int = 0,
    ) -> object:
        if name == "humanoid_character_builder" or name.startswith(
            "humanoid_character_builder."
        ):
            raise ModuleNotFoundError("No module named 'humanoid_character_builder'")
        return real_import(name, globals, locals, fromlist, level)

    for module_name in list(sys.modules):
        if module_name == "humanoid_character_builder" or module_name.startswith(
            "humanoid_character_builder."
        ):
            monkeypatch.delitem(sys.modules, module_name, raising=False)
    monkeypatch.setattr(builtins, "__import__", deny_character_builder_import)

    payload = {
        "height_m": 1.8,
        "mass_kg": 80.0,
        "build_type": "average",
    }
    response = client.post("/character-builder/generate", json=payload)
    assert response.status_code == 503
    assert "Character builder provider is unavailable" in response.text


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

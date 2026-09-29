"""Unit tests for the web settings route (issue #7457).

Covers the settings round-trip (GET defaults → PUT → GET), range
validation (DbC), atomic persistence, and corrupt-file resilience.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.routes.settings import (
    SETTINGS_PATH_ENV,
    WebSettings,
    load_settings,
    router,
    save_settings,
)

pytestmark = pytest.mark.unit


@pytest.fixture
def settings_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Point the route at a temp settings file."""
    path = tmp_path / "web_settings.json"
    monkeypatch.setenv(SETTINGS_PATH_ENV, str(path))
    return path


@pytest.fixture
def client(settings_path: Path) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    return TestClient(app)


# ---------------------------------------------------------------------------
# GET defaults
# ---------------------------------------------------------------------------


class TestGetDefaults:
    def test_get_returns_defaults_when_no_file(
        self, client: TestClient, settings_path: Path
    ) -> None:
        assert not settings_path.exists()
        response = client.get("/settings")
        assert response.status_code == 200
        data = response.json()
        assert data["appearance"]["font_scale"] == 1.0
        assert data["notifications"]["toast_duration_ms"] == 4000
        assert data["notifications"]["verbosity"] == "all"
        assert data["simulation_defaults"]["default_engine"] == "mujoco"
        assert data["simulation_defaults"]["duration"] == 3.0
        assert data["simulation_defaults"]["timestep"] == 0.002

    def test_get_returns_defaults_on_corrupt_file(
        self, client: TestClient, settings_path: Path
    ) -> None:
        settings_path.write_text("{not valid json", encoding="utf-8")
        response = client.get("/settings")
        assert response.status_code == 200
        assert response.json()["appearance"]["font_scale"] == 1.0

    def test_get_returns_defaults_on_schema_invalid_file(
        self, client: TestClient, settings_path: Path
    ) -> None:
        settings_path.write_text(
            json.dumps({"appearance": {"font_scale": 99.0}}), encoding="utf-8"
        )
        response = client.get("/settings")
        assert response.status_code == 200
        assert response.json()["appearance"]["font_scale"] == 1.0


# ---------------------------------------------------------------------------
# PUT round-trip
# ---------------------------------------------------------------------------


def _valid_payload() -> dict:
    return {
        "appearance": {"theme_id": "Light", "font_scale": 1.25},
        "notifications": {"toast_duration_ms": 8000, "verbosity": "errors"},
        "simulation_defaults": {
            "default_engine": "drake",
            "duration": 5.0,
            "timestep": 0.001,
        },
    }


class TestPutRoundTrip:
    def test_put_then_get_round_trips(
        self, client: TestClient, settings_path: Path
    ) -> None:
        payload = _valid_payload()
        put_response = client.put("/settings", json=payload)
        assert put_response.status_code == 200
        assert put_response.json() == payload

        get_response = client.get("/settings")
        assert get_response.status_code == 200
        assert get_response.json() == payload

    def test_put_persists_to_disk(
        self, client: TestClient, settings_path: Path
    ) -> None:
        client.put("/settings", json=_valid_payload())
        assert settings_path.exists()
        on_disk = json.loads(settings_path.read_text(encoding="utf-8"))
        assert on_disk["appearance"]["theme_id"] == "Light"
        assert on_disk["simulation_defaults"]["default_engine"] == "drake"

    def test_put_partial_document_fills_defaults(
        self, client: TestClient, settings_path: Path
    ) -> None:
        response = client.put("/settings", json={"appearance": {"theme_id": "Light"}})
        assert response.status_code == 200
        data = response.json()
        assert data["appearance"]["theme_id"] == "Light"
        assert data["appearance"]["font_scale"] == 1.0
        assert data["notifications"]["verbosity"] == "all"


# ---------------------------------------------------------------------------
# Validation (DbC ranges)
# ---------------------------------------------------------------------------


class TestValidation:
    @pytest.mark.parametrize("font_scale", [0.1, 2.5, -1.0])
    def test_font_scale_out_of_range_rejected(
        self, client: TestClient, font_scale: float
    ) -> None:
        response = client.put(
            "/settings", json={"appearance": {"font_scale": font_scale}}
        )
        assert response.status_code == 422

    def test_empty_theme_id_rejected(self, client: TestClient) -> None:
        response = client.put("/settings", json={"appearance": {"theme_id": ""}})
        assert response.status_code == 422

    @pytest.mark.parametrize("duration_ms", [100, 100_000])
    def test_toast_duration_out_of_range_rejected(
        self, client: TestClient, duration_ms: int
    ) -> None:
        response = client.put(
            "/settings",
            json={"notifications": {"toast_duration_ms": duration_ms}},
        )
        assert response.status_code == 422

    def test_unknown_verbosity_rejected(self, client: TestClient) -> None:
        response = client.put(
            "/settings", json={"notifications": {"verbosity": "loud"}}
        )
        assert response.status_code == 422

    @pytest.mark.parametrize("duration", [0.0, -3.0, 400.0])
    def test_duration_out_of_range_rejected(
        self, client: TestClient, duration: float
    ) -> None:
        response = client.put(
            "/settings", json={"simulation_defaults": {"duration": duration}}
        )
        assert response.status_code == 422

    def test_timestep_exceeding_duration_rejected(self, client: TestClient) -> None:
        response = client.put(
            "/settings",
            json={"simulation_defaults": {"duration": 0.5, "timestep": 0.9}},
        )
        assert response.status_code == 422

    def test_invalid_put_leaves_previous_file_untouched(
        self, client: TestClient, settings_path: Path
    ) -> None:
        client.put("/settings", json=_valid_payload())
        before = settings_path.read_text(encoding="utf-8")
        response = client.put("/settings", json={"appearance": {"font_scale": 50.0}})
        assert response.status_code == 422
        assert settings_path.read_text(encoding="utf-8") == before


# ---------------------------------------------------------------------------
# Persistence helper contracts
# ---------------------------------------------------------------------------


class TestPersistenceHelpers:
    def test_save_settings_rejects_non_model(self, tmp_path: Path) -> None:
        with pytest.raises(TypeError, match="WebSettings"):
            save_settings({"appearance": {}}, tmp_path / "x.json")  # type: ignore[arg-type]

    def test_save_creates_parent_directories(self, tmp_path: Path) -> None:
        target = tmp_path / "nested" / "dir" / "web_settings.json"
        written = save_settings(WebSettings(), target)
        assert written == target
        assert target.exists()

    def test_save_load_round_trip_via_helpers(self, tmp_path: Path) -> None:
        target = tmp_path / "web_settings.json"
        original = WebSettings.model_validate(_valid_payload())
        save_settings(original, target)
        assert load_settings(target) == original

    def test_no_temp_files_left_behind(self, tmp_path: Path) -> None:
        target = tmp_path / "web_settings.json"
        save_settings(WebSettings(), target)
        leftovers = [p for p in tmp_path.iterdir() if p.name != target.name]
        assert leftovers == []

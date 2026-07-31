"""Unit tests for the recordings API routes (issue #7451).

Covers:
- CRUD flow (persist / list / detail / delete)
- Path-traversal safety on recording ids
- Format-enumeration parity with the desktop recorder export registry
- Golden byte-parity: CSV+JSON exported via the API are byte-identical to
  the desktop export call path (``export_recording_all_formats``)
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from fastapi import FastAPI
from fastapi.testclient import TestClient

from src.api.dependencies import get_simulation_service
from src.api.routes.recordings import get_recording_store, router
from src.api.services.recording_service import (
    InvalidRecordingIdError,
    RecordingStore,
    validate_recording_id,
)
from src.shared.python.data_io.export import (
    export_recording_all_formats,
    get_available_export_formats,
)

pytestmark = pytest.mark.unit


def _make_data_dict() -> dict[str, Any]:
    """Deterministic stand-in for ``GenericPhysicsRecorder.get_data_dict()``."""
    rng = np.random.default_rng(seed=7451)
    n = 25
    times = np.linspace(0.0, 0.24, n)
    return {
        "times": times,
        "kinetic_energy": rng.random(n),
        "potential_energy": rng.random(n),
        "joint_positions": rng.random((n, 2)),
        "joint_velocities": rng.random((n, 2)),
        "joint_torques": rng.random((n, 2)),
        "induced_accelerations": {0: rng.random((n, 2))},
        "counterfactuals": {},
        "model_name": "pendulum",
        "num_frames": n,
    }


class FakeRecorder:
    """Minimal recorder double exposing the API used by the routes."""

    def __init__(self, data_dict: dict[str, Any] | None = None) -> None:
        self._data = data_dict if data_dict is not None else _make_data_dict()
        self.current_idx = int(self._data.get("num_frames", 0))

    def get_data_dict(self) -> dict[str, Any]:
        return self._data


class FakeSimulationService:
    def __init__(self, recorder: FakeRecorder | None) -> None:
        self._recorder = recorder

    def get_session_recording(self) -> tuple[FakeRecorder, dict[str, Any]] | None:
        if self._recorder is None or self._recorder.current_idx == 0:
            return None
        return self._recorder, {
            "engine": "mujoco",
            "model": "pendulum.xml",
            "duration": 0.24,
        }


@pytest.fixture
def store(tmp_path) -> RecordingStore:
    return RecordingStore(base_dir=tmp_path / "recordings")


@pytest.fixture
def client(store: RecordingStore) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_simulation_service] = lambda: FakeSimulationService(
        FakeRecorder()
    )
    app.dependency_overrides[get_recording_store] = lambda: store
    return TestClient(app)


@pytest.fixture
def empty_session_client(store: RecordingStore) -> TestClient:
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_simulation_service] = lambda: FakeSimulationService(
        None
    )
    app.dependency_overrides[get_recording_store] = lambda: store
    return TestClient(app)


# ── ID validation / path-traversal safety ───────────────────────────


@pytest.mark.parametrize(
    "bad_id",
    ["..", "../etc", "a/b", "a\\b", "a.b", "", "a b", "%2e%2e"],
)
def test_validate_recording_id_rejects_traversal(bad_id: str) -> None:
    with pytest.raises(InvalidRecordingIdError):
        validate_recording_id(bad_id)


def test_validate_recording_id_accepts_safe_ids() -> None:
    assert validate_recording_id("rec_20260612_abc-123") == "rec_20260612_abc-123"


def test_get_recording_traversal_id_rejected(client: TestClient) -> None:
    # %2E%2E decodes to ".." — must be rejected before touching the filesystem.
    response = client.get("/recordings/%2E%2E")
    assert response.status_code == 400


def test_export_traversal_id_rejected(client: TestClient) -> None:
    response = client.get("/recordings/%2E%2E%2Fsecrets/export?format=json")
    assert response.status_code in (400, 404)


# ── CRUD flow ────────────────────────────────────────────────────────


def test_create_recording_persists_session(client: TestClient) -> None:
    response = client.post("/recordings")
    assert response.status_code == 201
    body = response.json()
    assert body["engine"] == "mujoco"
    assert body["model"] == "pendulum.xml"
    assert body["frames"] == 25
    assert body["duration"] == pytest.approx(0.24)
    assert body["created"]
    validate_recording_id(body["id"])


def test_create_recording_without_session_conflicts(
    empty_session_client: TestClient,
) -> None:
    response = empty_session_client.post("/recordings")
    assert response.status_code == 409


def test_list_and_detail_and_delete(client: TestClient) -> None:
    rec_id = client.post("/recordings").json()["id"]

    listing = client.get("/recordings").json()["recordings"]
    assert [r["id"] for r in listing] == [rec_id]

    detail = client.get(f"/recordings/{rec_id}")
    assert detail.status_code == 200
    assert detail.json()["id"] == rec_id

    deleted = client.delete(f"/recordings/{rec_id}")
    assert deleted.status_code == 200
    assert client.get(f"/recordings/{rec_id}").status_code == 404
    assert client.get("/recordings").json()["recordings"] == []


def test_detail_unknown_recording_404(client: TestClient) -> None:
    assert client.get("/recordings/rec_does_not_exist").status_code == 404
    assert client.delete("/recordings/rec_does_not_exist").status_code == 404


# ── Format enumeration parity (acceptance criterion) ────────────────


def test_export_formats_parity_with_desktop_recorder(client: TestClient) -> None:
    """Formats enumerated by the API == formats the desktop recorder offers.

    Single source of truth: ``get_available_export_formats`` in
    ``src.shared.python.data_io.export`` — the same function the PyQt6
    dashboard Export tab uses to populate its format list.
    """
    response = client.get("/export/formats")
    assert response.status_code == 200
    api_formats = response.json()["formats"]
    desktop_formats = get_available_export_formats()
    assert api_formats == desktop_formats
    # Honest enumeration: availability flags must be booleans probed from
    # the optional dependencies, and the canonical five formats are listed.
    assert set(api_formats) == {"json", "csv", "mat", "hdf5", "c3d"}
    for info in api_formats.values():
        assert isinstance(info["available"], bool)


def test_export_unknown_format_rejected(client: TestClient) -> None:
    rec_id = client.post("/recordings").json()["id"]
    response = client.get(f"/recordings/{rec_id}/export?format=parquet")
    assert response.status_code == 400


# ── Golden byte-parity test (acceptance criterion) ──────────────────


def test_export_bytes_match_desktop_export_path(client: TestClient, tmp_path) -> None:
    """CSV+JSON downloaded via the API are byte-identical to a direct call
    to the desktop export implementation on the same recorder data."""
    data_dict = _make_data_dict()
    rec_id = client.post("/recordings").json()["id"]

    desktop_base = tmp_path / "desktop_export"
    results = export_recording_all_formats(
        str(desktop_base), data_dict, formats=["json", "csv"]
    )
    assert results == {"json": True, "csv": True}

    for fmt, ext in (("json", ".json"), ("csv", ".csv")):
        response = client.get(f"/recordings/{rec_id}/export?format={fmt}")
        assert response.status_code == 200, response.text
        desktop_bytes = desktop_base.with_suffix(ext).read_bytes()
        assert (
            response.content == desktop_bytes
        ), f"{fmt} export differs between API and desktop call path"


def test_export_artifact_is_cached(client: TestClient, store: RecordingStore) -> None:
    rec_id = client.post("/recordings").json()["id"]
    first = client.get(f"/recordings/{rec_id}/export?format=json")
    second = client.get(f"/recordings/{rec_id}/export?format=json")
    assert first.content == second.content
    artifact = store.base_dir / rec_id / "export.json"
    assert artifact.is_file()


# ── Store roundtrip fidelity ─────────────────────────────────────────


def test_store_roundtrip_preserves_order_types_and_values(
    store: RecordingStore,
) -> None:
    data = _make_data_dict()
    rec_id = store.persist(data, {"engine": "mujoco", "duration": 0.24})
    loaded = store.load_data(rec_id)

    assert list(loaded.keys()) == list(data.keys())
    for key, value in data.items():
        if isinstance(value, np.ndarray):
            np.testing.assert_array_equal(loaded[key], value)
        elif isinstance(value, dict):
            assert set(loaded[key].keys()) == set(value.keys())
            for sub_key, sub_value in value.items():
                assert type(sub_key) in (int, str)
                np.testing.assert_array_equal(loaded[key][sub_key], sub_value)
        else:
            assert loaded[key] == value
    # Nested int keys must stay ints (CSV column naming depends on it).
    assert 0 in loaded["induced_accelerations"]

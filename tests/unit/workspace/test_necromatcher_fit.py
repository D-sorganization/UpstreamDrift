"""Source-bound research trajectory storage and recall contracts."""

import json
from pathlib import Path
from zipfile import ZipFile

import pytest

from src.shared.python.workspace import NecromatcherLibrary, CaptureReview

pytestmark = pytest.mark.unit


@pytest.fixture
def fit_case(tmp_path):
    # Storage tests use a minimal archive; capture ingestion has independent tests.
    library = NecromatcherLibrary.create(tmp_path / "library")
    library.add_player("hogan", "Ben Hogan")
    library.add_swing("practice", "hogan", "Practice")
    model_path = tmp_path / "model.xml"
    model_path.write_text("<mujoco/>")
    model = library.add_model(
        "model-v1", "practice", model_path, engine="mujoco", dofs=("hip",)
    )
    capture_path = tmp_path / "capture.zip"
    frames = [
        {
            "frame_id": f"f{i}",
            "frame_sha256": str(i) * 64,
            "presentation_time_s": {"numerator": i, "denominator": 10},
        }
        for i in range(3)
    ]
    with ZipFile(capture_path, "w") as archive:
        archive.writestr(
            "receipt.json",
            json.dumps({"frame_count": 3, "source": {"width_px": 32, "height_px": 32}}),
        )
        archive.writestr(
            "observations.jsonl",
            "\n".join(json.dumps({"frame": f, "observation": {}}) for f in frames),
        )
    capture = library._save_asset(
        "capture-v1",
        "practice",
        capture_path,
        "image_capture",
        {"schema": "necromatcher/image-capture/1"},
    )
    payload = {
        "schema_version": "necromatcher/kinematic-fit/1",
        "qualification": "monocular_research_hypothesis",
        "physical_time_qualified": False,
        "dynamics_replayed": False,
        "model_id": model.dataset_id,
        "model_hash": model.metadata["hash"],
        "capture_id": capture.dataset_id,
        "capture_hash": capture.metadata["hash"],
        "coordinate_order": ["hip"],
        "coordinate_units": ["rad"],
        "frame_indices": [0, 2],
        "frames": [frames[0], frames[2]],
        "q": [[0.1], [0.2]],
        "provenance": {"description": "Synthetic storage test; no native validation"},
        "evidence": {"rejection_reasons": ["No physical clock"]},
    }
    source = tmp_path / "fit.json"
    source.write_text(json.dumps(payload))
    return library, source, payload


def test_fit_survives_restart_and_exports_exact_bytes(fit_case, tmp_path):
    library, source, payload = fit_case
    saved = library.add_fit("fit-v1", "practice", source)
    fresh = NecromatcherLibrary(library.root)
    assert fresh.load_fit("fit-v1") == payload
    assert saved.kind == "kinematic_fit"
    destination = tmp_path / "swing.zip"
    fresh.export_swing("practice", destination)
    with ZipFile(destination) as archive:
        assert archive.read("assets/fit-v1.json") == source.read_bytes()
    recalled = fresh.load_fit("fit-v1")
    recalled["q"][0][0] = 999
    assert fresh.load_fit("fit-v1")["q"][0][0] == 0.1


@pytest.mark.parametrize(
    "field,value",
    [
        ("model_hash", "sha256:" + "0" * 64),
        ("capture_hash", "sha256:" + "0" * 64),
        ("coordinate_order", ["knee"]),
        ("coordinate_units", ["N*m"]),
        ("physical_time_qualified", True),
        ("dynamics_replayed", True),
        ("qualification", "qualified"),
        ("q", [[float("nan")], [0.2]]),
        ("q", [["0.1"], ["0.2"]]),
        ("q", [[True], [False]]),
        ("coordinate_units", [["rad"]]),
        ("frame_indices", [2, 0]),
        ("frames", [{}, {}]),
    ],
)
def test_invalid_fit_is_not_published(fit_case, field, value):
    library, source, payload = fit_case
    payload[field] = value
    source.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        library.add_fit("bad", "practice", source)
    assert "bad" not in [x.dataset_id for x in library.assets("practice")]


def test_recall_rechecks_bound_model_and_export_refuses_stale_binding(
    fit_case, tmp_path
):
    library, source, _ = fit_case
    library.add_fit("fit-v1", "practice", source)
    model = library.load_asset("model-v1")
    Path(model.path).write_text("changed")
    with pytest.raises(ValueError, match="hash mismatch"):
        library.load_fit("fit-v1")
    with pytest.raises(ValueError):
        library.export_swing("practice", tmp_path / "bad.zip")


def test_fit_cannot_bind_another_swing(fit_case):
    library, source, _ = fit_case
    library.add_swing("other", "hogan", "Other")
    with pytest.raises(ValueError, match="same swing"):
        library.add_fit("bad", "other", source)


def test_api_import_and_source_frame_recall_use_verified_library(fit_case):
    from fastapi import FastAPI
    from fastapi.testclient import TestClient
    from src.api.routes.necromatcher import get_library, router

    library, source, payload = fit_case
    app = FastAPI()
    app.include_router(router)
    app.dependency_overrides[get_library] = lambda: library
    with TestClient(app) as client:
        saved = client.post(
            "/necromatcher/swings/practice/fits",
            json={"id": "fit-v1", "source_path": str(source)},
        )
        assert saved.status_code == 201
        assert "path" not in saved.json()
        summary = client.get("/necromatcher/fits/fit-v1")
        assert summary.status_code == 200
        assert summary.json()["frame_count"] == 2
        assert summary.json()["physical_time_qualified"] is False
        frame = client.get("/necromatcher/fits/fit-v1/frames/2")
        assert frame.status_code == 200
        assert frame.json()["q"] == [0.2]
        assert frame.json()["frame"] == payload["frames"][1]
        assert frame.json()["coordinate_units"] == ["rad"]
        assert client.get("/necromatcher/fits/fit-v1/frames/1").status_code == 404
        assert client.get("/necromatcher/fits/fit-v1/frames/-1").status_code == 404
        Path(library.load_asset("model-v1").path).write_text("changed")
        assert client.get("/necromatcher/fits/fit-v1/frames/2").status_code == 422

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
        assert (
            client.get("/necromatcher/fits/fit-v1/frames/2/projection").status_code
            == 422
        )
        Path(library.load_asset("model-v1").path).write_text("changed")
        assert client.get("/necromatcher/fits/fit-v1/frames/2").status_code == 422


def test_native_projection_verifies_rebuilt_model_and_uses_stored_camera(
    fit_case, tmp_path
):
    import numpy as np

    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.workspace.necromatcher_projection import project_fit_frame

    library, source, payload = fit_case
    definition = json.loads(
        (
            Path(__file__).resolve().parents[3]
            / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
        ).read_text()
    )
    xml, _ = export_full_body_mjcf(json.dumps(definition).encode())
    model_path = tmp_path / "native.xml"
    model_path.write_text(xml, encoding="utf-8")
    model = library.add_model(
        "native-v1",
        "practice",
        model_path,
        engine="mujoco",
        dofs=tuple(definition["coordinate_order"]),
    )
    payload.update(
        model_id="native-v1",
        model_hash=model.metadata["hash"],
        coordinate_order=definition["coordinate_order"],
        coordinate_units=["m"] * 3 + ["rad"] * 41,
        q=np.zeros((2, 44)).tolist(),
    )
    payload["provenance"]["native_definition"] = definition
    payload["q"][1][0] = 0.4
    payload["evidence"]["original_fit"] = {
        "camera": {
            "intrinsics": [[100, 0, 16], [0, 100, 16], [0, 0, 1]],
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 4],
        },
        "attachments": {"origin": [definition["joints"][0]["child"], [0, 0, 0]]},
    }
    source.write_text(json.dumps(payload))
    library.add_fit("native-fit", "practice", source)
    projected = project_fit_frame(library, "native-fit", 2)
    assert projected["points"]["origin"] == {"x": 26.0, "y": 16.0, "visibility": None}
    assert projected["coordinates"] == "image_pixels"
    assert projected["qualification"] == "monocular_research_hypothesis"
    assert projected["frame"] == payload["frames"][1]
    from src.shared.python.workspace.necromatcher_projection_process import (
        NativeFitProjectionProcess,
    )

    with NativeFitProjectionProcess(library.root) as process:
        with pytest.raises(IndexError):
            process.project("native-fit", 1)
        assert process.project("native-fit", 2)["points"]["origin"]["x"] == 26.0
        assert process.project("native-fit", 0)["points"]["origin"]["x"] == 16.0
    with pytest.raises(RuntimeError, match="closed"):
        process.project("native-fit", 0)
    # A fresh Qt parent catches Windows DLL-order failures masked by warm SDKs.
    pytest.importorskip("PyQt6.QtWidgets")
    import os
    import sys

    from src.shared.python.security import secure_run

    script = (
        "import sys; from PyQt6.QtWidgets import QApplication; "
        "app = QApplication([]); "
        "from src.shared.python.workspace import NativeFitProjectionProcess; "
        "process = NativeFitProjectionProcess(sys.argv[1]); "
        "result = process.project('native-fit', 2); process.close(); "
        "assert result['points']['origin']['x'] == 26.0"
    )
    secure_run(
        [sys.executable, "-c", script, str(library.root)],
        cwd=Path(__file__).resolve().parents[3],
        env={**os.environ, "QT_QPA_PLATFORM": "offscreen"},
        check=True,
        timeout=60,
        capture_output=True,
    )
    with pytest.raises(IndexError):
        project_fit_frame(library, "native-fit", 1)
    payload["provenance"]["native_definition"]["bodies"][1]["solids"][0]["mass_kg"] += 1
    source.write_text(json.dumps(payload))
    library.add_fit("mismatched-native-fit", "practice", source)
    with pytest.raises(ValueError, match="native model"):
        project_fit_frame(library, "mismatched-native-fit", 0)


def test_projection_timeout_terminates_owned_interpreter(tmp_path, monkeypatch):
    import sys

    from src.shared.python.workspace import necromatcher_projection_process as runtime

    launch = runtime.secure_popen
    children = []

    def stalled_worker(command, **kwargs):
        child = launch(
            [sys.executable, "-u", "-c", "import time; time.sleep(60)"], **kwargs
        )
        children.append(child)
        return child

    monkeypatch.setattr(runtime, "secure_popen", stalled_worker)
    monkeypatch.setattr(runtime, "PROJECTION_TIMEOUT_S", 0.05)
    with runtime.NativeFitProjectionProcess(tmp_path) as process:
        with pytest.raises(RuntimeError, match="timed out"):
            process.project("not-read-by-stalled-worker", 0)
    assert len(children) == 1
    assert children[0].poll() is not None
    assert children[0].stdin.closed
    assert children[0].stdout.closed

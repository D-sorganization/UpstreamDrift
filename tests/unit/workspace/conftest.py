"""Shared immutable research-fit storage specimens; no native acceptance."""

import json
from zipfile import ZipFile
import pytest
from src.shared.python.workspace import NecromatcherLibrary


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

"""Shared immutable research-fit storage specimens; no native acceptance."""

import json
from pathlib import Path
import numpy as np
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


@pytest.fixture
def native_fit_case(fit_case, tmp_path):
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )

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
        "native-model",
        "practice",
        model_path,
        engine="mujoco",
        dofs=tuple(definition["coordinate_order"]),
    )
    payload.update(
        model_id=model.dataset_id,
        model_hash=model.metadata["hash"],
        coordinate_order=definition["coordinate_order"],
        coordinate_units=["m"] * 3 + ["rad"] * 41,
        q=np.zeros((2, 44)).tolist(),
    )
    payload["provenance"]["native_definition"] = definition
    payload["evidence"]["original_fit"] = {
        "camera": {
            "intrinsics": [[100, 0, 16], [0, 100, 16], [0, 0, 1]],
            "rotation": np.eye(3).tolist(),
            "translation": [0, 0, 4],
        },
        "attachments": {"origin": [definition["joints"][0]["child"], [0, 0, 0]]},
    }
    source.write_text(json.dumps(payload))
    return library, source, payload

"""One real native compile is reused by the optional image admission boundary."""

import importlib
from dataclasses import replace
import pytest
from src.shared.python.workspace.necromatcher_shaft_evidence import BoundShaftEvidence
from tests.unit.motion_matching.test_shaft_observations import evidence

pytestmark = pytest.mark.unit


def test_admission_compiles_actual_native_once_and_calls_canonical_binder(
    native_fit_case, monkeypatch
):
    import json
    import numpy as np
    from src.shared.python.workspace import (
        load_shaft_image_residuals,
        necromatcher_native,
    )

    module = importlib.import_module(
        "src.shared.python.workspace.necromatcher_shaft_evidence"
    )
    library, source, payload = native_fit_case
    for frame in payload["frames"]:
        frame["camera_id"] = "camera"
    from zipfile import ZipFile

    old_asset = library.load_asset(payload["capture_id"])
    capture_path = source.parent / "camera-capture.zip"
    with (
        ZipFile(library.root / old_asset.path) as original,
        ZipFile(capture_path, "w") as archive,
    ):
        archive.writestr("receipt.json", original.read("receipt.json"))
        rows = [
            json.loads(line)
            for line in original.read("observations.jsonl").splitlines()
        ]
        for row in rows:
            row["frame"]["camera_id"] = "camera"
        archive.writestr(
            "observations.jsonl", "\n".join(json.dumps(row) for row in rows)
        )
    capture = library._save_asset(
        "camera-capture",
        "practice",
        capture_path,
        "image_capture",
        {"schema": "necromatcher/image-capture/1"},
    )
    payload.update(capture_id=capture.dataset_id, capture_hash=capture.metadata["hash"])
    source.write_text(json.dumps(payload), encoding="utf-8")
    library.add_fit("native-fit", "practice", source)
    value = replace(
        evidence(),
        capture_id=payload["capture_id"],
        capture_sha256=payload["capture_hash"],
    )
    calls = []
    original = necromatcher_native.get_plant

    def tracked(engine, definition):
        calls.append(engine)
        return original(engine, definition)

    monkeypatch.setattr(necromatcher_native, "get_plant", tracked)
    admitted = []

    def checked(lib, ev):
        admitted.append((lib, ev))
        return BoundShaftEvidence(ev, "sha256:" + "e" * 64, 1, 1)

    monkeypatch.setattr(module, "bind_shaft_axis_evidence", checked)
    binding, bundle = load_shaft_image_residuals(library, "native-fit", value, 0.5)
    assert calls == ["mujoco"] and admitted == [(library, value)]
    axis = bundle.terms[0].axis
    points = binding.plant.marker_positions(
        np.zeros(44),
        {"a": (axis.body, axis.point_a_m), "b": (axis.body, axis.point_b_m)},
    )
    assert points.shape == (2, 3) and np.isfinite(points).all()
    assert calls == ["mujoco"]

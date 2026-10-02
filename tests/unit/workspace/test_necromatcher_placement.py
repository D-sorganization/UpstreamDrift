"""Authored world placement preserves source pixels and immutable fit lineage."""

import json
import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def placement_case(native_fit_case):
    library, path, payload = native_fit_case
    payload["evidence"]["original_fit"]["free_coordinates"] = list(range(44))
    payload["evidence"]["original_fit"]["spline_coefficients"] = [
        "prior solver evidence"
    ]
    path.write_text(json.dumps(payload))
    library.add_fit("source-fit", "practice", path)
    return library, payload


def test_ground_revision_preserves_all_source_pixels_and_original_fit(placement_case):
    from src.shared.python.workspace.necromatcher_placement import (
        author_ground_placement,
    )
    from src.shared.python.workspace import load_native_fit_binding

    library, source = placement_case
    before = load_native_fit_binding(library, "source-fit")
    saved = author_ground_placement(
        library,
        "source-fit",
        "placed-fit",
        0,
        0.0,
        "Assumed flat ground; no observed calibration",
    )
    after = load_native_fit_binding(library, saved.dataset_id)
    record = after.fit["provenance"]["placement_revision"]
    assert record["source_fit_hash"] == before.fit_hash
    assert record["anchor_ground_clearance_m"] == pytest.approx(0, abs=1e-10)
    assert record["max_reprojection_difference_pixels"] < 1e-7
    assert record["source_frame_index"] == 0
    assert after.fit["qualification"] == "monocular_research_hypothesis"
    assert after.fit["physical_time_qualified"] is False
    assert after.fit["dynamics_replayed"] is False
    np.testing.assert_array_equal(library.load_fit("source-fit")["q"], source["q"])
    assert "spline_coefficients" not in after.fit["evidence"]["original_fit"]
    np.testing.assert_array_equal(
        np.array(after.fit["q"])[:, 3:], np.array(source["q"])[:, 3:]
    )
    for frame in source["frame_indices"]:
        first, last = before.project(frame), after.project(frame)
        assert first["frame"] == last["frame"]
        for name, point in first["points"].items():
            np.testing.assert_allclose(
                [point["x"], point["y"]],
                [last["points"][name]["x"], last["points"][name]["y"]],
                rtol=0,
                atol=1e-7,
            )
    assert library.load_asset("source-fit").metadata["hash"] == before.fit_hash


@pytest.mark.parametrize(
    "frame,clearance,description",
    [
        (True, 0.0, "assumed"),
        (1, 0.0, "assumed"),
        (0, -0.01, "assumed"),
        (0, float("nan"), "assumed"),
        (0, True, "assumed"),
        (0, 0.0, ""),
    ],
)
def test_invalid_placement_is_not_published(
    placement_case, frame, clearance, description
):
    from src.shared.python.workspace.necromatcher_placement import (
        author_ground_placement,
    )

    library, _ = placement_case
    with pytest.raises((ValueError, IndexError)):
        author_ground_placement(
            library, "source-fit", "bad-fit", frame, clearance, description
        )
    assert not any(x.dataset_id == "bad-fit" for x in library.assets("practice"))


def test_placement_recall_rejects_changed_source_parent(placement_case):
    from pathlib import Path
    from src.shared.python.workspace.necromatcher_placement import (
        author_ground_placement,
    )

    library, _ = placement_case
    author_ground_placement(
        library, "source-fit", "placed-fit", 0, 0.002, "Explicit clearance hypothesis"
    )
    source = library.load_asset("source-fit")
    Path(source.path).write_text("{}")
    with pytest.raises(ValueError, match="hash mismatch"):
        library.load_fit("placed-fit")


@pytest.mark.parametrize(
    "mutation", ["source_frame", "clearance", "joint", "camera", "source_hash"]
)
def test_revised_fit_import_rejects_inconsistent_placement_evidence(
    placement_case, mutation
):
    from pathlib import Path
    from src.shared.python.workspace import author_ground_placement

    library, _ = placement_case
    author_ground_placement(
        library, "source-fit", "placed-fit", 0, 0.0, "Explicit placement hypothesis"
    )
    payload = library.load_fit("placed-fit")
    record = payload["provenance"]["placement_revision"]
    if mutation == "source_frame":
        record["source_frame_index"] = True
    elif mutation == "clearance":
        record["requested_clearance_m"] = 1.0
    elif mutation == "joint":
        payload["q"][0][6] += 0.1
    elif mutation == "camera":
        payload["evidence"]["original_fit"]["camera"]["translation"][0] += 0.1
    else:
        record["source_fit_hash"] = "sha256:" + "0" * 64
    path = library.root / "invalid.json"
    path.write_text(json.dumps(payload))
    with pytest.raises(ValueError):
        library.add_fit("invalid-placement", "practice", path)

"""Authored model-space repairs preserve source identities and explicit limits."""

import json

import numpy as np
import pytest

pytestmark = pytest.mark.unit


@pytest.fixture
def repair_binding(native_fit_case):
    from src.shared.python.workspace import load_native_fit_binding

    library, path, payload = native_fit_case
    definition = payload["provenance"]["native_definition"]
    payload["evidence"]["original_fit"]["attachments"] = {
        label: [body, [0, 0, 0]]
        for label, body in (
            ("origin", definition["joints"][0]["child"]),
            ("hand", definition["closure"]["body_a"]),
            ("club", definition["closure"]["body_b"]),
        )
    }
    path.write_text(json.dumps(payload))
    library.add_fit("source-fit", "practice", path)
    binding = load_native_fit_binding(library, "source-fit")
    return binding


def test_repair_preserves_source_and_enforces_declared_limits(repair_binding):
    from src.shared.python.workspace import repair_native_motion

    binding = repair_binding
    original = np.asarray(binding.fit["q"]).copy()
    report = repair_native_motion(binding, iterations=8)
    assert report["source_fit_id"] == binding.fit_id
    assert report["source_fit_hash"] == binding.fit_hash
    assert report["frame_indices"] == binding.fit["frame_indices"]
    assert report["scientifically_qualified"] is False
    assert report["physical_time_qualified"] is False
    assert report["solver_options"]["solver"] == "trf"
    assert report["target_kind"] == "inferred_native_world_markers"
    assert report["interpolation"] == "none_discrete_samples_only"
    assert len(report["diagnostics"]) == len(original)
    q = np.asarray(report["q"])
    assert q.shape == original.shape
    assert np.isfinite(q).all()
    for name, (low, high) in report["bounds_rad"].items():
        column = q[:, binding.plant.coordinate_order.index(name)]
        assert np.all(column >= low)
        assert np.all(column <= high)
    np.testing.assert_array_equal(binding.fit["q"], original)


@pytest.mark.parametrize("iterations", [True, 0, 1001, 1.5])
def test_repair_rejects_invalid_budget(iterations):
    from src.shared.python.workspace.necromatcher_constraints import (
        repair_native_motion,
    )

    with pytest.raises(ValueError, match="iterations"):
        repair_native_motion(None, iterations=iterations)


@pytest.mark.parametrize(
    "ranges",
    [
        None,
        {},
        {"missing": [-1, 1]},
        {"TranslationInputX": [0, 1]},
        {"REInput": [3, 2]},
        {"REInput": [float("nan"), 2]},
    ],
)
def test_repair_rejects_invalid_or_nonangular_ranges(repair_binding, ranges):
    from src.shared.python.workspace import repair_native_motion

    repair_binding.fit["provenance"]["native_definition"]["coordinate_ranges_deg"] = (
        ranges
    )
    with pytest.raises(ValueError, match="ranges|angular|limits"):
        repair_native_motion(repair_binding, iterations=1)


def test_repair_rejects_source_change_during_execution(repair_binding, monkeypatch):
    from src.shared.python.workspace import necromatcher_constraints as module

    stamps = iter(
        [
            {"source_sha256": "before", "runtime_sha256": "same"},
            {"source_sha256": "after", "runtime_sha256": "same"},
        ]
    )
    monkeypatch.setattr(module, "fit_execution_stamp", lambda: next(stamps))
    with pytest.raises(ValueError, match="changed during"):
        module.repair_native_motion(repair_binding, iterations=1)

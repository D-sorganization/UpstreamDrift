"""Worker opt-in composition and raw diagnostics remain separate from body RMS."""

from dataclasses import replace
from types import SimpleNamespace
import json
import pytest
from src.shared.python.workspace import necromatcher_fit_worker as worker
from src.shared.python.workspace import necromatcher_fit_records as records
from tests.unit.motion_matching.test_shaft_residuals import setup
from tests.unit.workspace.test_necromatcher_fit_records import specimen

pytestmark = pytest.mark.unit


def test_compute_operation_passes_optional_bundle_without_changing_legacy_call(
    monkeypatch,
):
    calls = []
    monkeypatch.setattr(
        worker, "fit_image_trajectory", lambda *args: calls.append(args)
    )
    worker._compute_operation("fit", "native", {}, "camera", "inputs", "config")
    assert len(calls[0]) == 6
    _, _, _, bundle = setup()
    worker._compute_operation(
        "fit", "native", {}, "camera", "inputs", "config", None, bundle
    )
    assert calls[1][-1] is bundle and len(calls[1]) == 7


@pytest.fixture
def shaft_record_case(native_fit_case):
    import numpy as np
    from dataclasses import asdict
    from src.shared.python.workspace.necromatcher_native import load_native_fit_binding
    from src.shared.python.workspace.necromatcher_fit_jobs import _shaft_record
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        BoundShaftEvidence,
    )
    from src.shared.python.motion_matching.historical_fit import (
        ImageFitResult,
        ImageFitConfig,
        ShaftAxisResidualTerm,
        resolve_authored_shaft_axis,
    )

    library, path, source = native_fit_case
    library.add_fit("parent", "practice", path)
    binding = load_native_fit_binding(library, "parent")
    camera, _ = binding.review_inputs()
    _, _, term, _ = setup()
    frame = replace(
        term.evidence.frames[0],
        frame=replace(term.evidence.frames[0].frame, pts_ticks=3),
    )
    ev = replace(term.evidence, frames=(frame,))
    axis = resolve_authored_shaft_axis(
        binding.definition_bytes, binding.plant.plant_sha
    )
    term = ShaftAxisResidualTerm(ev, axis, term.source_clock_sha256, 0.5)
    result = ImageFitResult(
        np.array([0.0, 0.2]),
        np.zeros((2, 44)),
        1.0,
        1.0,
        np.zeros((2, 1)),
        2,
        binding.plant.plant_sha,
        tuple(binding.plant.coordinate_order),
        False,
        "test",
        np.array([0.0, 0.2]),
        np.zeros(4),
        (binding.plant.coordinate_order[0],),
        additional_image_assessments=(
            term.assess(binding.plant, camera, np.zeros((1, 44))),
        ),
    )
    recipe = _shaft_record(BoundShaftEvidence(ev, term.source_clock_sha256, 1, 1), 0.5)
    request = {
        "source_fit_id": "parent",
        "source_fit_hash": binding.fit_hash,
        "execution_stamp": {},
        "options": {
            "frame_indices": [0, 2],
            "operation": "fit",
            "config": asdict(ImageFitConfig()),
        },
        "shaft_images": recipe,
        "shaft_axis": axis.to_record(),
    }
    stamp = dict.fromkeys(("started_at_utc", "source_sha256", "runtime_sha256"), "test")
    return request, source, result, stamp, recipe, axis


def test_persistence_keeps_separate_raw_metrics_and_recipe(shaft_record_case):
    request, source, result, stamp, recipe, axis = shaft_record_case
    payload = records.build_native_fit_payload(
        request, source, result, ((0, 2), source["frames"], result.q), stamp, 0.0
    )
    assert payload["evidence"]["original_fit"]["rms_pixels"] == 1.0
    assert payload["evidence"]["shaft_axis"]["recipe"] == recipe
    assert payload["evidence"]["shaft_axis"]["assessments"][0]["raw_rms_pixels"] > 1
    assert payload["evidence"]["shaft_axis"]["body_observed_point_count"] == 2
    assert payload["evidence"]["shaft_axis"]["physical_time_qualified"] is False
    assert payload["provenance"]["shaft_images"]["axis"] == axis.to_record()
    json.dumps(payload, allow_nan=False)


def test_opt_in_record_rejects_missing_or_transplanted_diagnostics():
    request, source, result, dense, stamp, _ = specimen()
    request["options"]["operation"] = "fit"
    request["shaft_images"] = {"evidence_sha256": "sha256:" + "f" * 64}
    with pytest.raises(ValueError, match="shaft|Shaft"):
        records.build_native_fit_payload(request, source, result, dense, stamp, 0.0)


def test_worker_opt_in_reuses_single_bound_native_and_verifies_recipe(monkeypatch):
    from src.shared.python.workspace.necromatcher_fit_jobs import _shaft_record
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        BoundShaftEvidence,
    )

    native, _, term, bundle = setup()
    recipe = _shaft_record(
        BoundShaftEvidence(term.evidence, term.source_clock_sha256, 1, 1), 0.5
    )
    request = {"source_fit_id": "parent", "shaft_images": recipe}
    binding = SimpleNamespace(plant=native)
    calls = []
    monkeypatch.setattr(
        worker,
        "load_native_fit_binding",
        lambda *args: pytest.fail("Second native compile"),
    )

    def load(lib, identity, ev, weight):
        calls.append(identity)
        assert ev == term.evidence and weight == 0.5
        return binding, bundle

    monkeypatch.setattr(worker, "load_shaft_image_residuals", load, raising=False)
    actual, extra = worker._load_refit_binding("library", request)
    assert actual is binding and extra is bundle and calls == ["parent"]
    request["shaft_images"]["binding"]["source_clock_sha256"] = "sha256:" + "f" * 64
    with pytest.raises(ValueError, match="recipe|binding"):
        worker._load_refit_binding("library", request)


@pytest.mark.parametrize("malformed", [None, "missing"])
def test_public_builder_rejects_malformed_axis_with_boundary_error(
    shaft_record_case, malformed
):
    request, source, result, stamp, _, _ = shaft_record_case
    if malformed is None:
        request["shaft_axis"] = None
    else:
        del request["shaft_axis"]
    with pytest.raises(ValueError, match="Shaft|shaft"):
        records.build_native_fit_payload(
            request, source, result, ((0, 2), source["frames"], result.q), stamp, 0.0
        )


@pytest.mark.parametrize("field", ["reviewed_frame_count", "observed_segment_count"])
@pytest.mark.parametrize("count", [True, 1.0])
def test_worker_rejects_noninteger_binding_before_native_load(
    monkeypatch, field, count
):
    from src.shared.python.workspace.necromatcher_fit_jobs import _shaft_record
    from src.shared.python.workspace.necromatcher_shaft_evidence import (
        BoundShaftEvidence,
    )

    _, _, term, bundle = setup()
    recipe = _shaft_record(
        BoundShaftEvidence(term.evidence, term.source_clock_sha256, 1, 1), 0.5
    )
    recipe["binding"][field] = count
    monkeypatch.setattr(
        worker,
        "load_shaft_image_residuals",
        lambda *args: pytest.fail(
            "Native load reached before strict binding validation"
        ),
    )
    with pytest.raises(ValueError, match="counts|integer"):
        worker._load_refit_binding(
            "library", {"source_fit_id": "parent", "shaft_images": recipe}
        )

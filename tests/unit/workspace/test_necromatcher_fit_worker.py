"""Authored starts are separate research versions, not optimizer successes."""

from contextlib import nullcontext
from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np
import pytest

pytestmark = pytest.mark.unit


def test_author_operation_uses_public_initializer_and_never_optimizer(monkeypatch):
    from src.shared.python.motion_matching import historical_fit
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    result = SimpleNamespace(
        optimizer_ran=False, converged=False, initialization=_Initialization()
    )
    calls = []

    def initialize(*args):
        calls.append(args)
        return result

    monkeypatch.setattr(
        historical_fit, "initialize_image_trajectory", initialize, raising=False
    )
    monkeypatch.setattr(
        worker, "fit_image_trajectory", lambda *args: pytest.fail("optimizer called")
    )
    config = SimpleNamespace(initialization_policy="authored_range_project_zero_slopes")
    assert (
        worker._compute_operation(
            "author_initialization", "native", {}, "camera", "inputs", config
        )
        is result
    )
    assert len(calls) == 1 and calls[0][-1] is config


def test_author_operation_rejects_contradictory_optimizer_receipt(monkeypatch):
    from src.shared.python.motion_matching import historical_fit
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    monkeypatch.setattr(
        historical_fit,
        "initialize_image_trajectory",
        lambda *args: SimpleNamespace(
            optimizer_ran=True, converged=True, initialization=None
        ),
        raising=False,
    )
    config = SimpleNamespace(initialization_policy="authored_range_project_zero_slopes")
    with pytest.raises(ValueError, match="receipt"):
        worker._compute_operation("author_initialization", None, {}, None, None, config)


@pytest.mark.parametrize(
    "operation,policy",
    [("unknown", "strict"), (True, "strict"), ("author_initialization", "strict")],
)
def test_invalid_worker_operation_or_policy_fails(operation, policy):
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    with pytest.raises(ValueError, match="operation|policy"):
        worker._compute_operation(
            operation,
            None,
            {},
            None,
            None,
            SimpleNamespace(initialization_policy=policy),
        )


def test_authored_ranges_reject_custom_limits_without_mislabeling():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    authored = SimpleNamespace(
        named_bounds={"joint": (-1.0, 1.0)}, to_record=lambda: {"range_source": "exact"}
    )
    binding = SimpleNamespace(authored_coordinate_bounds=lambda: authored)
    config = SimpleNamespace(coordinate_bounds=(("joint", -2.0, 2.0),))
    with pytest.raises(ValueError, match="authored"):
        worker._range_provenance(binding, config, "author_initialization")
    assert worker._range_provenance(binding, config, "fit") == {
        "range_source": "custom_coordinate_bounds"
    }
    config.coordinate_bounds = (("joint", -1.0, 1.0),)
    assert worker._range_provenance(binding, config, "author_initialization") == {
        "range_source": "exact"
    }


@dataclass(frozen=True)
class _Initialization:
    policy: str = "authored_range_project_zero_slopes"
    original_coefficient_sha256: str = "sha256:old"
    initialized_coefficient_sha256: str = "sha256:new"


def test_saved_initialization_receipt_is_not_optimizer_convergence():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    result = SimpleNamespace(
        source_times=np.array([1.0, 2.0]),
        q=np.zeros((2, 1)),
        free_coordinates=("joint",),
        coordinate_order=("joint",),
        knot_times=np.array([1.0, 2.0]),
        spline_coefficients=np.zeros(4),
        initial_rms_pixels=7.0,
        rms_pixels=7.0,
        converged=False,
        optimizer_message="Authored initialization only",
        optimizer_ran=False,
        initialization=_Initialization(),
        constraint_times=np.array([]),
        constraint_row_labels=(),
        constraint_residuals=np.empty((0, 0)),
        maximum_constraint_residual=0.0,
        model_sha="model",
        initial_spline=None,
    )
    original = {"camera": {}, "attachments": {"marker": ["body", [0.0, 0.0, 0.0]]}}
    source = {
        "provenance": {"native_definition": {}},
        "evidence": {"original_fit": original},
        "q": [[9.0], [9.0]],
    }
    config = {
        "coordinate_bounds": [["joint", -1.0, 1.0]],
        "initialization_policy": "authored_range_project_zero_slopes",
    }
    request = {
        "source_fit_id": "old",
        "source_fit_hash": "sha256:old",
        "execution_stamp": {},
        "options": {
            "frame_indices": [0, 1],
            "config": config,
            "operation": "author_initialization",
        },
    }
    stamp = {
        "started_at_utc": "now",
        "source_sha256": "source",
        "runtime_sha256": "runtime",
    }
    payload = worker._build_fit_payload(
        request, source, result, ((0, 1), [{}, {}], result.q), stamp, 0.0
    )
    evidence = payload["evidence"]["original_fit"]
    assert evidence["optimizer_ran"] is False and evidence["converged"] is False
    assert evidence["initialization"]["policy"] == _Initialization().policy
    assert evidence["rms_pixels"] == 7.0 and source["q"] == [[9.0], [9.0]]
    reasons = payload["evidence"]["rejection_reasons"]
    assert "authored_initialization_only" in reasons
    assert "optimizer_not_converged" not in reasons
    assert "anatomical_ranges_not_enforced" in reasons
    assert "historical_anatomy_unqualified" in reasons
    assert evidence["constraint_assessment"]["continuous_certified"] is False


def test_exact_authored_ranges_keep_anatomy_and_unbounded_coordinates_unqualified():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    result = SimpleNamespace(optimizer_ran=True, converged=True)
    config = SimpleNamespace(coordinate_bounds=(("joint", -1.0, 1.0),))
    ranges = {
        "range_source": "bound_native_definition.coordinate_ranges_deg",
        "named_bounds": {"joint": [-1.0, 1.0]},
        "unbounded_names": ["root"],
    }
    blockers = worker._research_blockers(result, config, ranges)
    assert "historical_anatomy_unqualified" in blockers
    assert "native_coordinates_without_authored_ranges" in blockers
    assert "native_authored_ranges_not_fully_enforced" not in blockers
    custom = worker._research_blockers(
        result, config, {"range_source": "custom_coordinate_bounds"}
    )
    assert "native_authored_ranges_not_fully_enforced" in custom


def _preserved_binding():
    import hashlib
    from src.shared.python.motion_matching.historical_fit import ImageSplineStart

    model_sha = hashlib.sha256(b"{}").hexdigest()
    start = ImageSplineStart.from_coefficients(
        (1.0, 2.0), (0.2, 0.4, 0.6, -0.2), ("joint",), ("joint",), model_sha
    )
    source = {
        "coordinate_order": ["joint"],
        "provenance": {"native_definition": {}},
        "q": [[0.2], [0.4]],
        "frames": [
            {"pts_ticks": i, "timebase_numerator": 1, "timebase_denominator": 1}
            for i in (1, 2)
        ],
        "evidence": {
            "original_fit": {
                "spline_start": start.to_record(),
                "free_coordinates": ["joint"],
                "coordinate_order": ["joint"],
                "knot_times": [1.0, 2.0],
                "spline_coefficients": [0.2, 0.4, 0.6, -0.2],
            }
        },
    }
    binding = SimpleNamespace(
        fit=source,
        plant=SimpleNamespace(plant_sha=model_sha, coordinate_order=("joint",)),
    )
    return binding, start


def test_worker_reconstructs_exact_start_and_rejects_changed_parent_motion():
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    binding, start = _preserved_binding()
    options = {"knot_count": 2}
    rebuilt = worker._preserved_start(binding, options, np.array([1.0, 2.0]))
    assert rebuilt == start
    binding.fit["q"][0][0] = 0.3
    with pytest.raises(ValueError, match="samples"):
        worker._preserved_start(binding, options, np.array([1.0, 2.0]))


def test_worker_exact_input_path_never_creates_sampled_knot_grid(monkeypatch):
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    binding, start = _preserved_binding()
    evidence = SimpleNamespace(
        source_times=np.array([1.0, 2.0]),
        observed_pixels=np.zeros((2, 1, 2)),
        confidence=np.ones((2, 1)),
    )
    options = {
        "knot_count": 2,
        "coordinate_scales": [1.0],
        "config": {},
        "initialization_source": "preserved_spline",
    }
    samples = np.array([[0.2], [0.4]])
    monkeypatch.setattr(
        worker.np, "linspace", lambda *args: pytest.fail("resampled knots")
    )
    inputs, preserved = worker._worker_inputs(binding, options, evidence, samples)
    assert preserved == start and inputs.initial_samples is None
    np.testing.assert_array_equal(inputs.seed, samples[0])
    np.testing.assert_array_equal(inputs.observed_pixels, evidence.observed_pixels)
    np.testing.assert_array_equal(inputs.knot_times, start.knot_times)


@pytest.mark.parametrize("change", ["clock", "knot_count", "hash", "model", "order"])
def test_worker_rejects_incompatible_preserved_identity(change):
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    binding, start = _preserved_binding()
    options = {"knot_count": 2}
    times = np.array([1.0, 2.0])
    if change == "clock":
        times = np.array([1.0, 2.1])
    elif change == "knot_count":
        options["knot_count"] = 3
    elif change == "hash":
        binding.fit["evidence"]["original_fit"]["spline_start"]["spline_coefficients"][
            2
        ] = 0.0
    elif change == "model":
        binding.plant.plant_sha = "other"
    else:
        binding.plant.coordinate_order = ("other",)
    with pytest.raises(ValueError):
        worker._preserved_start(binding, options, times)


class _ScheduledWorkerReview:
    capture_id = "capture"
    frame_count = 3

    def __init__(self):
        self.closed = False
        self.frames = [
            {
                "asset_id": "video",
                "shot_id": "shot",
                "camera_id": "camera",
                "swing_id": "swing",
                "pts_ticks": i + 10,
                "timebase_numerator": 1,
                "timebase_denominator": 3,
                "frame_sha256": str(i) * 64,
            }
            for i in range(3)
        ]

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.closed = True

    def frame(self, index):
        return {"frame": self.frames[index]}


def _scheduled_worker_fixture(monkeypatch):
    from dataclasses import asdict
    from src.shared.python.motion_matching.constraint_kinematics import (
        ConstraintOptions,
    )
    from src.shared.python.motion_matching.contact_law import GroundPlane
    from src.shared.python.motion_matching.historical_fit import (
        ContactPinPhase,
        ContactPinSchedule,
        ImageFitConfig,
        ScheduledConstraintOptions,
    )
    from src.shared.python.workspace import necromatcher_fit_worker as worker

    review = _ScheduledWorkerReview()
    capture_hash = "sha256:" + "a" * 64
    schedule = ContactPinSchedule(
        "capture",
        capture_hash,
        (
            ContactPinPhase((10, 3), (11, 3), ("heel_r",), ("sha256:" + "0" * 64,)),
            ContactPinPhase((11, 3), (12, 3), (), ("sha256:" + "2" * 64,)),
        ),
    )
    config = ImageFitConfig(
        constraint_options=ScheduledConstraintOptions(
            ConstraintOptions(GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1),
            schedule,
        )
    )
    original = {
        "free_coordinates": ["joint"],
        "camera": {},
        "attachments": {"marker": ["body", [0, 0, 0]]},
    }
    source = {
        "capture_id": "capture",
        "capture_hash": capture_hash,
        "frame_indices": [0, 1, 2],
        "frames": review.frames,
        "q": [[0.0], [0.0], [0.0]],
        "coordinate_order": ["joint"],
        "provenance": {"native_definition": {}},
        "evidence": {"original_fit": original},
    }
    native = SimpleNamespace(closure_residuals=lambda q: np.zeros(3))
    binding = SimpleNamespace(
        fit=source,
        plant=native,
        project=lambda index: None,
        review_inputs=lambda: (object(), original["attachments"]),
    )
    stamp = {
        "source_sha256": "source",
        "runtime_sha256": "runtime",
        "started_at_utc": "now",
    }
    request = {
        "library_root": "unused",
        "source_fit_id": "parent",
        "source_fit_hash": "parent-hash",
        "execution_stamp": stamp,
        "options": {
            "frame_indices": [0, 2],
            "knot_count": 2,
            "coordinate_scales": [1.0],
            "unknown_visibility_weight": 0.5,
            "config": asdict(config),
        },
    }
    monkeypatch.setattr(worker, "fit_execution_stamp", lambda: stamp)
    monkeypatch.setattr(
        worker,
        "NecromatcherLibrary",
        lambda root: SimpleNamespace(
            authenticated_read=nullcontext,
            load_fit=lambda identity: binding.fit,
            load_asset=lambda identity: SimpleNamespace(
                metadata={"hash": "parent-hash"}
            ),
        ),
    )
    monkeypatch.setattr(
        worker, "load_native_fit_binding", lambda library, identity: binding
    )
    monkeypatch.setattr(worker, "CaptureReview", lambda library, identity: review)
    return worker, request, review


@pytest.mark.parametrize("mismatch", ["capture_id", "capture_sha256", "review_hash"])
def test_compute_worker_rejects_foreign_schedule_before_evidence_or_optimizer(
    monkeypatch, mismatch
):
    worker, request, review = _scheduled_worker_fixture(monkeypatch)
    schedule = request["options"]["config"]["constraint_options"]["schedule"]
    if mismatch == "review_hash":
        # A real source image, but outside this phase: membership alone is insufficient.
        schedule["phases"][0]["review_frame_sha256"] = ["sha256:" + "2" * 64]
    else:
        schedule[mismatch] = (
            "foreign" if mismatch == "capture_id" else "sha256:" + "b" * 64
        )
    monkeypatch.setattr(
        worker,
        "read_capture_evidence",
        lambda *args, **kwargs: pytest.fail("evidence read before binding"),
    )
    monkeypatch.setattr(
        worker, "_compute_operation", lambda *args: pytest.fail("optimizer reached")
    )
    with pytest.raises(ValueError, match="capture|review"):
        worker.compute_native_refit(request)
    assert review.closed


def test_compute_worker_retains_bound_schedule_and_original_evidence(monkeypatch):
    from copy import deepcopy
    from src.shared.python.motion_matching.historical_fit import (
        ScheduledConstraintOptions,
    )

    worker, request, review = _scheduled_worker_fixture(monkeypatch)
    before = deepcopy(request)
    calls = []

    def evidence(review, markers, indices, **kwargs):
        return SimpleNamespace(
            source_times=np.array([(i + 10) / 3 for i in indices]),
            observed_pixels=np.zeros((len(indices), 1, 2)),
            confidence=np.ones((len(indices), 1)),
            frame_indices=indices,
        )

    def compute(operation, native, attachments, camera, inputs, config, initial):
        assert isinstance(config.constraint_options, ScheduledConstraintOptions)
        np.testing.assert_array_equal(inputs.observed_pixels, np.zeros((2, 1, 2)))
        np.testing.assert_array_equal(inputs.source_times, [10 / 3, 4.0])
        calls.append(config)
        return SimpleNamespace(
            source_times=inputs.source_times,
            q=np.zeros((2, 1)),
            coordinate_order=("joint",),
            free_coordinates=("joint",),
            knot_times=inputs.knot_times,
            spline_coefficients=np.zeros(4),
            initial_rms_pixels=1.0,
            rms_pixels=1.0,
            converged=False,
            optimizer_ran=True,
            initialization=None,
            initial_spline=None,
            model_sha="c" * 64,
            optimizer_message="Fixture budget exhausted",
            constraint_times=np.array([10 / 3, 11 / 3, 4.0]),
            constraint_row_labels=("ground:heel_r",),
            constraint_residuals=np.zeros((3, 1)),
            maximum_constraint_residual=0.0,
            evaluate_source_times=lambda times: np.zeros((len(times), 1)),
        )

    monkeypatch.setattr(worker, "read_capture_evidence", evidence)
    monkeypatch.setattr(worker, "_compute_operation", compute)
    monkeypatch.setattr(
        worker, "_dense_reprojection_metrics", lambda *args: {"dense_rms_pixels": 1.0}
    )
    output = worker.compute_native_refit(request)
    assert len(calls) == 1 and review.closed
    assert request == before
    assert output["evidence"]["original_fit"]["config"] == request["options"]["config"]
    receipt = output["provenance"]["contact_schedule_binding"]
    assert receipt["capture_hash"] == "sha256:" + "a" * 64
    assert receipt["phases"][0]["boundary_frame_indices"] == [0, 1]
    assert receipt["phases"][1]["review_frame_indices"] == [2]
    assert receipt["phases"][1]["pinned_spheres"] == []
    assert receipt["continuous_certified"] is False
    assert receipt["normal_height_hypothesis_only"] is True
    assert "optimizer_not_converged" in output["evidence"]["rejection_reasons"]
    assert output["frames"] == review.frames

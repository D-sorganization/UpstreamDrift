"""Owned fit execution preserves research qualification and launch identities."""

from dataclasses import replace
import json

import pytest

pytestmark = pytest.mark.unit


def test_dense_metrics_weight_observations_and_exclude_training_frames():
    import numpy as np
    from src.shared.python.motion_matching.historical_fit import (
        CameraProjection,
        CaptureImageEvidence,
    )
    from src.shared.python.workspace.necromatcher_fit_worker import (
        _dense_reprojection_metrics,
    )

    class Geometry:
        def marker_positions(self, q, attachments):
            return np.array([[q[0], 0.0, 1.0]])

    evidence = CaptureImageEvidence(
        "synthetic",
        (0, 1, 2),
        ("a", "b", "c"),
        ("a", "b", "c"),
        np.array([0.0, 0.1, 0.2]),
        np.zeros((3, 1, 2)),
        np.array([[1.0], [0.25], [0.0]]),
        ("origin",),
        (32, 32),
        0.5,
    )
    result = _dense_reprojection_metrics(
        Geometry(),
        CameraProjection(np.eye(3), np.eye(3), np.zeros(3)),
        {"origin": ("body", [0, 0, 0])},
        evidence,
        np.array([[3], [4], [999]]),
        (0,),
    )
    assert result["dense_rms_pixels"] == pytest.approx(np.sqrt(10.4))
    assert result["held_out_rms_pixels"] == pytest.approx(4.0)


def test_refit_options_reject_invalid_sampling_and_budgets():
    from src.shared.python.workspace import NativeRefitOptions

    valid = NativeRefitOptions((0, 2), 2, (1.0,))
    for change in (
        {"frame_indices": (2, 0)},
        {"frame_indices": (False, 2)},
        {"knot_count": 1},
        {"coordinate_scales": (0.0,)},
        {"coordinate_scales": (True,)},
        {"coordinate_scales": ("1",)},
        {"config": {}},
        {"budget_wall_s": float("nan")},
        {"unknown_visibility_weight": 1.5},
    ):
        with pytest.raises(ValueError):
            replace(valid, **change)


def test_refit_operation_defaults_to_fit_and_rejects_unknown_values():
    from src.shared.python.workspace.necromatcher_fit_jobs import NativeRefitOptions

    valid = NativeRefitOptions((0, 2), 2, (1.0,))
    assert valid.operation == "fit"
    for operation in (None, True, [], "initialize", ""):
        with pytest.raises(ValueError, match="operation"):
            replace(valid, operation=operation)
    with pytest.raises(ValueError, match="policy"):
        replace(valid, operation="author_initialization")


def test_preserved_refit_options_require_fit_and_strict_policy():
    from src.shared.python.workspace.necromatcher_fit_jobs import NativeRefitOptions
    from src.shared.python.motion_matching.historical_fit import ImageFitConfig

    valid = NativeRefitOptions((0, 2), 2, (1.0,))
    assert valid.initialization_source == "sampled_parent"
    assert replace(valid, initialization_source="preserved_spline").operation == "fit"
    for value in (True, None, [], "unknown"):
        with pytest.raises(ValueError, match="initialization source"):
            replace(valid, initialization_source=value)
    authored = ImageFitConfig(
        coordinate_bounds=(("hip", -1.0, 1.0),),
        initialization_policy="authored_range_project_zero_slopes",
    )
    with pytest.raises(ValueError, match="strict"):
        replace(valid, config=authored, initialization_source="preserved_spline")
    with pytest.raises(ValueError, match="fit"):
        replace(
            valid,
            config=authored,
            operation="author_initialization",
            initialization_source="preserved_spline",
        )


def test_author_job_publishes_separate_rejected_version_and_preserves_source(
    fit_case, monkeypatch
):
    from copy import deepcopy
    from src.shared.python.motion_matching.historical_fit import ImageFitConfig
    from src.shared.python.motion_matching.jobs import (
        MatchingJobService,
        JobStatus,
        AcceptanceState,
    )
    from src.shared.python.workspace import necromatcher_fit_jobs as jobs

    library, source, original = fit_case
    library.add_fit("old-author", "practice", source)
    requests = []

    def execute(request_path, budget, cancelled):
        request = json.loads(request_path.read_text())
        requests.append(request)
        payload = deepcopy(original)
        payload["provenance"]["operation"] = "author_initialization"
        payload["evidence"] = {
            "rejection_reasons": ["authored_initialization_only"],
            "original_fit": {
                "optimizer_ran": False,
                "rms_pixels": 9.0,
                "converged": False,
            },
        }
        return {"fit": payload}

    monkeypatch.setattr(jobs, "_execute_worker", execute)
    config = ImageFitConfig(
        initialization_policy="authored_range_project_zero_slopes",
        coordinate_bounds=(("hip", -1.0, 1.0),),
    )
    options = jobs.NativeRefitOptions(
        (0, 2), 2, (1.0,), config=config, operation="author_initialization"
    )
    service = MatchingJobService()
    try:
        handle, run_root = jobs.start_native_refit(
            library, "old-author", "new-author", options, service
        )
        result = handle.join(timeout=20)
        assert (
            result.status == JobStatus.SUCCEEDED
            and result.acceptance == AcceptanceState.REJECTED
        )
        assert requests[0]["options"]["operation"] == "author_initialization"
        assert (
            json.loads((run_root / "request.json").read_text())["execution_started"]
            is True
        )
        saved = library.load_fit("new-author")
        assert saved["evidence"]["original_fit"]["optimizer_ran"] is False
        assert saved["evidence"]["original_fit"]["rms_pixels"] == 9.0
        assert library.load_fit("old-author") == original
        assert (
            library.load_asset("new-author").metadata["hash"]
            != library.load_asset("old-author").metadata["hash"]
        )
    finally:
        service.close()


@pytest.mark.requires_mujoco
def test_refit_job_persists_start_identity_and_fails_without_native_assumptions(
    fit_case,
    monkeypatch,
):
    monkeypatch.delenv("PYTHONPATH", raising=False)
    from src.shared.python.motion_matching.jobs import (
        AcceptanceState,
        JobStatus,
        MatchingJobService,
    )
    from src.shared.python.workspace import NativeRefitOptions, start_native_refit

    library, source, payload = fit_case
    library.add_fit("old-fit", "practice", source)
    service = MatchingJobService()
    try:
        handle, run_root = start_native_refit(
            library,
            "old-fit",
            "new-fit",
            NativeRefitOptions((0, 2), 2, (1.0,)),
            service,
        )
        result = handle.join(timeout=40)
        assert result.status == JobStatus.FAILED
        assert result.acceptance == AcceptanceState.REJECTED
        assert "native_definition" in result.message
        assert "ModuleNotFoundError" not in result.message
        request = json.loads((run_root / "request.json").read_text())
        assert (
            request["source_fit_hash"] == library.load_asset("old-fit").metadata["hash"]
        )
        stamp = request["execution_stamp"]
        assert stamp["started_at_utc"]
        assert stamp["source_sha256"].startswith("sha256:")
        assert stamp["runtime"]["python"]
        with pytest.raises(KeyError):
            library.load_fit("new-fit")
        assert library.load_fit("old-fit") == payload
    finally:
        service.close()

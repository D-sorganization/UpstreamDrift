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

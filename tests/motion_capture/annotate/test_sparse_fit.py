"""Sparse manual sets through reconstruction and the model fit (#9802)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.motion_capture.sparse_fit_evidence import (
    landmark_rms_mm,
    manual_set_from_detections,
)
from src.motion_capture.reconstruct.model import FitOptions
from src.motion_capture.reconstruct.model.golfer import GOLFER_LANDMARK_MAP, GOLFER_SPEC
from src.motion_capture.reconstruct.model.session import fit_session_model
from src.motion_capture.reconstruct.pipeline import (
    MatchSpec,
    reconstruct_session,
    start_cameras_from,
)
from src.motion_capture.reconstruct import __main__ as recon_cli
from src.motion_capture.variants import variant_dir

pytestmark = pytest.mark.unit


def _session_with_truth(tmp_path: Path) -> tuple[Path, Path, np.ndarray, list[str]]:
    session = tmp_path / "take"
    assert (
        recon_cli.main(
            ["synth", "--out", str(session), "--frames", "30", "--outliers", "0.0"]
        )
        == 0
    )
    truth = json.loads((session / "truth.json").read_text(encoding="utf-8"))
    cameras = session / "cameras.json"
    cameras.write_text(json.dumps(truth["cameras"]), encoding="utf-8")
    return (
        session,
        cameras,
        np.asarray(truth["joints_3d_m"]),
        list(truth["joint_names"]),
    )


@pytest.mark.timeout(600)
def test_every_fifth_frame_fits_within_twice_the_dense_error(tmp_path: Path) -> None:
    session, cameras, truth, names = _session_with_truth(tmp_path)
    results = {}
    for stride in (1, 5):
        out_set = f"observations_manual_k{stride}"
        out_dir = manual_set_from_detections(
            session, ("face_on", "down_line", "overhead"), stride, out_set
        )
        manual = json.loads((out_dir / "face_on.json").read_text(encoding="utf-8"))
        assert manual["provenance"]["estimator"] == "manual"
        assert manual["frames_with_pose"] == len(range(0, 30, stride))
        variant = f"sparse_k{stride}"
        reconstruct_session(
            session,
            start_cameras=start_cameras_from(cameras),
            scale_anchor=("neck", 0.5),
            match=MatchSpec(observation_set=out_set, variant=variant),
        )
        root = variant_dir(session, variant)
        fit, _ = fit_session_model(
            root,
            GOLFER_SPEC,
            GOLFER_LANDMARK_MAP,
            options=FitOptions(max_iterations=25),
            session_root=session,
        )
        assert fit.velocity_violations == 0
        report = json.loads((root / "model" / "fit_report.json").read_text("utf-8"))
        # Frames without clicks are still produced (continuity prior) ...
        assert len(fit.q) == 30
        results[stride] = landmark_rms_mm(fit.landmarks_m, truth, names)
        # ... and the report says how many frames actually observed each landmark.
        assert report["landmarks"]["left_shoulder"]["frames"] <= 30
    assert results[5] < 2 * results[1] + 5.0, results
    assert results[1] < 80.0, results  # synthetic motion, not a golfer swing

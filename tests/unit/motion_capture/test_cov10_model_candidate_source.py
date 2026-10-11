"""Unit tests for the engine-generic model-candidate comparison source (COV-10, #11278).

All fixtures are synthetic: they prove software contracts only and never stand in
for a real GS3DX ``capture-O`` candidate (blocked on #11165).
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

pytestmark = pytest.mark.unit

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.model_candidate_source import (
    ModelCandidateProvenance,
    ModelCandidateSource,
    compare_candidate_2d,
    source_from_overlay_dataset,
)
from src.motion_capture.reference.swing_pairing import (
    PairingConfidence,
    PairingDecisionStatus,
    SwingPairingResult,
)
from src.motion_capture.simscape_c3d_video_overlay import load_overlay_dataset

LABELS = ["synthetic_a", "synthetic_b", "synthetic_c"]
SHA = "a" * 64
HASHES = {k: f"hash_{k}" for k in ("observations", "camera", "pairing", "profile")}


def _camera() -> PinholeCamera:
    return PinholeCamera(
        camera_id="cam_synthetic",
        matrix=np.array([[1000.0, 0, 960.0], [0, 1000.0, 540.0], [0, 0, 1.0]]),
        rotation_world_from_camera=np.eye(3),
        translation_world_from_camera_m=np.array([0.0, 0.0, -4.0]),
        image_size_px=(1920, 1080),
    )


def _pairing() -> SwingPairingResult:
    return SwingPairingResult(
        video_swing_id="cov-01-s01",
        status=PairingDecisionStatus.PAIRED,
        paired_capture_swing_id="capture-O-s01",
        confidence=PairingConfidence(
            margin=0.35,
            best_normalized_distance=0.25,
            second_best_normalized_distance=0.60,
            envelope_median_distance=1.0,
            tau_pair=0.20,
            epsilon=0.05,
        ),
        normalized_distances={"capture-O-s01": 0.25},
        raw_distances={"capture-O-s01": 0.25},
        receipt_hashes={"cov-01-s01": "hash_pair"},
    )


def _prov(engine: str = "simscape-gs3dx", **kw: object) -> ModelCandidateProvenance:
    base: dict[str, object] = {
        "engine": engine,
        "candidate_id": "synthetic_cand_01",
        "model_sha256": SHA,
        "matlab_release": "R2025b" if engine.startswith("simscape") else None,
        "qualification": "qualified",
    }
    base.update(kw)
    return ModelCandidateProvenance(**base)  # type: ignore[arg-type]


def _points(t: int = 12) -> np.ndarray:
    rng = np.random.default_rng(0)
    base = np.array([[0.0, 0.0, 0.0], [0.3, 0.5, 0.1], [-0.3, 0.9, 0.0]])
    return base[None] + 0.05 * rng.standard_normal((t, 3, 3))


def _source(engine: str = "simscape-gs3dx", **kw: object) -> ModelCandidateSource:
    pts = _points()
    return ModelCandidateSource(
        provenance=_prov(engine, **kw),
        times_s=np.linspace(0.0, 1.0, len(pts)),
        labels=tuple(LABELS),
        points_world_m=pts,
    )


def _video_px(src: ModelCandidateSource, bias: float = 3.0) -> np.ndarray:
    px, _ = _camera().project(src.points_world_m.reshape(-1, 3))
    return px.reshape(len(src.times_s), len(LABELS), 2) + bias


def test_overlay_dataset_loads_as_comparison_source_with_provenance(
    tmp_path: Path,
) -> None:
    pts = _points()
    t = np.linspace(0.0, 1.0, len(pts))
    cap = tmp_path / "cap.json"
    rep = tmp_path / "rep.json"
    cap.write_text(
        json.dumps(
            {
                "labels": LABELS,
                "time_s": t.tolist(),
                "points_world_m": pts.tolist(),
                "valid": np.ones((len(t), 3), bool).tolist(),
            }
        )
    )
    rep.write_text(json.dumps({"labels": LABELS, "prediction_m": pts.tolist()}))
    ds = load_overlay_dataset(cap, rep)
    src = source_from_overlay_dataset(ds, _prov())
    assert src.labels == tuple(LABELS)
    assert src.points_world_m.shape == (len(t), 3, 3)
    assert src.provenance.model_sha256 == SHA
    assert src.provenance.engine == "simscape-gs3dx"


@pytest.mark.parametrize("release", ["R2026a", "R2024b", "", "2025a"])
def test_wrong_matlab_release_rejected_with_release_named(release: str) -> None:
    with pytest.raises(ValueError, match="R2025b") as exc:
        _prov(matlab_release=release)
    if release:
        assert release in str(exc.value)


def test_simscape_without_release_rejected() -> None:
    with pytest.raises(ValueError, match="matlab_release"):
        _prov(matlab_release=None)


def test_bad_sha_and_qualification_rejected() -> None:
    with pytest.raises(ValueError, match="sha256"):
        _prov(model_sha256="xyz")
    with pytest.raises(ValueError, match="qualification"):
        _prov(qualification="maybe")


def test_unqualified_candidate_still_compares_without_validated_wording() -> None:
    src = _source(qualification="unqualified")
    receipt = compare_candidate_2d(
        src, _camera(), _video_px(src), _pairing(), input_hashes=HASHES
    )
    assert receipt.metadata["model_qualification"] == "unqualified"
    assert receipt.l2_result is not None
    assert receipt.l2_result.unweighted_rmse_px == pytest.approx(
        np.sqrt(2) * 3.0, rel=1e-6
    )
    assert "validated" not in json.dumps(receipt.model_dump(), default=str).lower()


@pytest.mark.parametrize("engine", ["simscape-gs3dx", "mujoco", "drake"])
def test_engine_sources_share_one_code_path(engine: str) -> None:
    src = _source(engine)
    receipt = compare_candidate_2d(
        src, _camera(), _video_px(src), _pairing(), input_hashes=HASHES
    )
    assert receipt.backend == "model_candidate"
    assert receipt.metadata["provenance"]["engine"] == engine
    assert receipt.l2_result is not None
    assert receipt.l2_result.residuals_px["p50"] == pytest.approx(np.sqrt(2) * 3.0)
    payload = receipt.model_dump(exclude={"created_utc"})
    payload["metadata"] = {
        k: v for k, v in payload["metadata"].items() if k != "provenance"
    }
    assert engine not in json.dumps(payload, default=str)


def test_marker_difference_reported_when_markers_supplied() -> None:
    src = _source()
    video = _video_px(src, bias=3.0)
    marker_px = _video_px(src, bias=0.0) + 1.0  # markers sit 1 px off in each axis
    receipt = compare_candidate_2d(
        src,
        _camera(),
        video,
        _pairing(),
        input_hashes=HASHES,
        marker_projection_px=marker_px,
    )
    md = receipt.metadata
    assert md["model_vs_video_rmse_px"] == pytest.approx(np.sqrt(2) * 3.0)
    assert md["marker_vs_video_rmse_px"] == pytest.approx(np.sqrt(2) * 2.0)
    assert md["model_minus_marker_rmse_px"] == pytest.approx(np.sqrt(2))


def test_shape_mismatch_and_nonfinite_rejected() -> None:
    src = _source()
    with pytest.raises(ValueError):
        compare_candidate_2d(
            src, _camera(), _video_px(src)[:-1], _pairing(), input_hashes=HASHES
        )
    bad = _points()
    bad[0, 0, 0] = np.nan
    with pytest.raises(ValueError, match="finite"):
        ModelCandidateSource(
            provenance=_prov(),
            times_s=np.linspace(0, 1, len(bad)),
            labels=tuple(LABELS),
            points_world_m=bad,
        )


@pytest.mark.requires_matlab
@pytest.mark.live_simulation
def test_live_r2025b_candidate_export_requires_matlab() -> None:
    """Real GS3DX candidate export: runs only on an R2025b host (blocked on #11165)."""
    matlab = Path("C:/Program Files/MATLAB/R2025b/bin/matlab.exe")
    if not matlab.is_file():
        pytest.skip("MATLAB R2025b not available at its explicit path")
    pytest.skip("No stored GS3DX capture-O candidate yet (#11165)")

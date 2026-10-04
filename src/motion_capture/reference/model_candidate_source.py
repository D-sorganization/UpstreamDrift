"""Engine-generic model-candidate source for video comparison (COV-10, #11278).

Any physics engine (Simscape GS3DX, MuJoCo, Drake, Pinocchio, OpenSim) supplies
world-frame landmark trajectories plus provenance. The comparison path below
projects them through a COV-4 camera and reuses the COV-7 L2 comparison and
receipt code, so no engine-specific comparison code exists. The engine name
appears in provenance metadata only.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass
import re
from typing import Any

import numpy as np
import numpy.typing as npt

from src.motion_capture.reconstruct.cameras import PinholeCamera
from src.motion_capture.reference.comparison_2d import (
    Comparison2DReceipt,
    ComparisonLevel,
    build_2d_comparison_receipt,
    compute_l2_paired_comparison,
)
from src.motion_capture.reference.swing_pairing import SwingPairingResult
from src.motion_capture.simscape_c3d_video_overlay import OverlayDataset
from src.shared.python.motion_matching.simscape_replay_harness import (
    SimscapeContinuousReplayError,
    _validate_matlab_release,
)

CANDIDATE_BACKEND = "model_candidate"
QUALIFICATIONS = ("qualified", "unqualified")
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


@dataclass(frozen=True)
class ModelCandidateProvenance:
    """Provenance of a model candidate, as reported by its matching issue.

    Preconditions: ``model_sha256`` is a lowercase SHA-256 hex digest;
    ``qualification`` is ``qualified`` or ``unqualified``; Simscape engines
    (name starting ``simscape``) carry ``matlab_release`` equal to R2025b.
    """

    engine: str
    candidate_id: str
    model_sha256: str
    qualification: str
    matlab_release: str | None = None

    def __post_init__(self) -> None:
        if not self.engine.strip() or not self.candidate_id.strip():
            raise ValueError("engine and candidate_id must be non-empty")
        if not _SHA256_RE.match(self.model_sha256):
            raise ValueError("model_sha256 must be a lowercase 64-hex SHA-256 digest")
        if self.qualification not in QUALIFICATIONS:
            raise ValueError(
                f"qualification must be one of {QUALIFICATIONS}, "
                f"got {self.qualification!r}"
            )
        if self.engine.lower().startswith("simscape"):
            if self.matlab_release is None:
                raise ValueError("matlab_release is required for Simscape candidates")
            try:
                _validate_matlab_release(self.matlab_release)
            except SimscapeContinuousReplayError as exc:
                raise ValueError(str(exc)) from exc


@dataclass(frozen=True)
class ModelCandidateSource:
    """World-frame landmark trajectories ``(T, K, 3)`` in metres plus provenance."""

    provenance: ModelCandidateProvenance
    times_s: npt.NDArray[np.float64]
    labels: tuple[str, ...]
    points_world_m: npt.NDArray[np.float64]

    def __post_init__(self) -> None:
        pts = np.asarray(self.points_world_m, dtype=float)
        t = np.asarray(self.times_s, dtype=float)
        if pts.ndim != 3 or pts.shape[2] != 3:
            raise ValueError("points_world_m must have shape (T, K, 3)")
        if t.shape != (pts.shape[0],) or pts.shape[0] == 0:
            raise ValueError("times_s must be a non-empty 1D array matching T")
        if len(self.labels) != pts.shape[1]:
            raise ValueError("labels length must match K")
        if not (np.isfinite(pts).all() and np.isfinite(t).all()):
            raise ValueError("candidate times and points must be finite")
        object.__setattr__(self, "points_world_m", pts)
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "labels", tuple(self.labels))


def source_from_overlay_dataset(
    dataset: OverlayDataset, provenance: ModelCandidateProvenance
) -> ModelCandidateSource:
    """Wrap the model channel of a ``load_overlay_dataset`` result as a source."""
    return ModelCandidateSource(
        provenance=provenance,
        times_s=dataset.times,
        labels=tuple(dataset.labels),
        points_world_m=dataset.model_points,
    )


def project_candidate_to_camera(
    source: ModelCandidateSource, camera: PinholeCamera
) -> npt.NDArray[np.float64]:
    """Project candidate landmarks to ``(T, K, 2)`` pixels (NaN behind camera)."""
    t_len, k_len, _ = source.points_world_m.shape
    px, _ = camera.project(source.points_world_m.reshape(-1, 3))
    return px.reshape(t_len, k_len, 2)


def _rmse(res: Mapping[str, Any]) -> float:
    return float(res["unweighted_rmse_px"])


def compare_candidate_2d(
    source: ModelCandidateSource,
    camera: PinholeCamera,
    video_landmarks_px: npt.ArrayLike,
    pairing_result: SwingPairingResult,
    *,
    input_hashes: Mapping[str, str],
    marker_projection_px: npt.ArrayLike | None = None,
    visibilities: npt.ArrayLike | None = None,
    body_height_px: float | None = None,
) -> Comparison2DReceipt:
    """Compare a projected model candidate with video landmarks (L2 receipt).

    Preconditions: ``video_landmarks_px`` (and ``marker_projection_px`` when
    given) have shape ``(T, K, 2)`` matching the source and are time-aligned.
    Postconditions: the receipt backend is ``model_candidate``; metadata carries
    ``provenance`` and ``model_qualification``. An unqualified candidate still
    compares, but the receipt makes no validity claim. When marker projections
    are supplied, metadata also holds model-vs-video and marker-vs-video RMSE
    and their difference, showing where markers and the visible swing disagree.
    """
    video = np.asarray(video_landmarks_px, dtype=float)
    if video.shape != (*source.points_world_m.shape[:2], 2):
        raise ValueError(
            f"video_landmarks_px shape {video.shape} must match candidate (T, K, 2)"
        )
    common: dict[str, Any] = {
        "landmark_names": source.labels,
        "visibilities": visibilities,
        "body_height_px": body_height_px,
    }
    model_l2 = compute_l2_paired_comparison(
        video, project_candidate_to_camera(source, camera), pairing_result, **common
    )
    prov = source.provenance
    claim = (
        "qualified model candidate"
        if prov.qualification == "qualified"
        else "unqualified model candidate; descriptive residuals only"
    )
    metadata: dict[str, Any] = {
        "model_qualification": prov.qualification,
        "claim": claim,
        "provenance": asdict(prov),
        "model_vs_video_rmse_px": _rmse(model_l2.model_dump()),
    }
    if marker_projection_px is not None:
        marker_l2 = compute_l2_paired_comparison(
            video,
            np.asarray(marker_projection_px, dtype=float),
            pairing_result,
            **common,
        )
        metadata["marker_vs_video_rmse_px"] = _rmse(marker_l2.model_dump())
        metadata["model_minus_marker_rmse_px"] = (
            metadata["model_vs_video_rmse_px"] - metadata["marker_vs_video_rmse_px"]
        )
    return build_2d_comparison_receipt(
        pairing_result.video_swing_id,
        CANDIDATE_BACKEND,
        ComparisonLevel.L2,
        input_hashes,
        l2_result=model_l2,
        metadata=metadata,
    )

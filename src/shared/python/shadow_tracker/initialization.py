"""Subject shape fitting and initial-state hypothesis generation for Shadow Tracker (ST-06)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Literal


from ._validation import check_id, check_pos_float, check_str
from .contracts import Handedness, RenderRequest, SubjectModelBinding
from .mask_records import MaskFrame
from .projection import (
    PinholeCameraModel,
    SilhouetteRenderer,
    compute_silhouette_loss,
)

HypothesisLabel = Literal["observed", "inferred", "prior"]
_VALID_HYPOTHESIS_LABELS = frozenset(("observed", "inferred", "prior"))
_VALID_HANDEDNESS = frozenset(("right", "left"))


# ---------------------------------------------------------------------------
# Morphology and Parameter Separation
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class VisualMorphology:
    """Visual envelope and segment dimension morphology."""

    height_m: float
    chest_width_m: float
    depth_m: float
    segment_lengths: dict[str, float]

    def __post_init__(self) -> None:
        if self.height_m <= 0.0 or not math.isfinite(self.height_m):
            raise ValueError(f"height_m must be positive, got {self.height_m}")
        if self.chest_width_m <= 0.0 or not math.isfinite(self.chest_width_m):
            raise ValueError(
                f"chest_width_m must be positive, got {self.chest_width_m}"
            )
        if self.depth_m <= 0.0 or not math.isfinite(self.depth_m):
            raise ValueError(f"depth_m must be positive, got {self.depth_m}")

        clean_lengths: dict[str, float] = {}
        for k, v in self.segment_lengths.items():
            check_str(k, "segment_name")
            if v <= 0.0 or not math.isfinite(v):
                raise ValueError(f"segment length {k} must be positive, got {v}")
            clean_lengths[k] = float(v)
        object.__setattr__(self, "segment_lengths", MappingProxyType(clean_lengths))


@dataclass(frozen=True, slots=True, kw_only=True)
class InertialParameters:
    """Inertial dynamics assumptions separated from visual geometry."""

    mass_kg: float
    center_of_mass_body_m: tuple[float, float, float]
    moments_of_inertia_kg_m2: tuple[float, float, float]

    def __post_init__(self) -> None:
        if self.mass_kg <= 0.0 or not math.isfinite(self.mass_kg):
            raise ValueError(f"mass_kg must be positive, got {self.mass_kg}")
        for coord in self.center_of_mass_body_m:
            if not math.isfinite(coord):
                raise ValueError(
                    f"center of mass coordinate must be finite, got {coord}"
                )
        for mom in self.moments_of_inertia_kg_m2:
            if mom <= 0.0 or not math.isfinite(mom):
                raise ValueError(f"moment of inertia must be positive, got {mom}")
        object.__setattr__(
            self,
            "center_of_mass_body_m",
            tuple(float(c) for c in self.center_of_mass_body_m),
        )
        object.__setattr__(
            self,
            "moments_of_inertia_kg_m2",
            tuple(float(m) for m in self.moments_of_inertia_kg_m2),
        )


@dataclass(frozen=True, slots=True, kw_only=True)
class SubjectMorphology:
    """Bounded subject morphology binding visual and inertial parameters."""

    subject_id: str
    visual: VisualMorphology
    inertial: InertialParameters
    handedness: Handedness

    def __post_init__(self) -> None:
        check_id(self.subject_id, "subject_id")
        if self.handedness not in _VALID_HANDEDNESS:
            raise ValueError(
                f"handedness must be one of {sorted(_VALID_HANDEDNESS)}, got {self.handedness!r}"
            )


# ---------------------------------------------------------------------------
# Initial Hypotheses and Candidates
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class InitialHypothesis:
    """Hypothesis of initial kinematic state, scale, and depth."""

    hypothesis_id: str
    subject_id: str
    camera_id: str
    scale: float
    depth_m: float
    handedness: Handedness
    pose: tuple[float, ...]
    velocity: tuple[float, ...]
    score: float
    label: HypothesisLabel
    provenance: str

    def __post_init__(self) -> None:
        check_id(self.hypothesis_id, "hypothesis_id")
        check_id(self.subject_id, "subject_id")
        check_id(self.camera_id, "camera_id")
        if self.scale <= 0.0 or not math.isfinite(self.scale):
            raise ValueError(f"scale must be positive, got {self.scale}")
        if self.depth_m <= 0.0 or not math.isfinite(self.depth_m):
            raise ValueError(f"depth_m must be positive, got {self.depth_m}")
        if self.handedness not in _VALID_HANDEDNESS:
            raise ValueError(
                f"handedness must be one of {sorted(_VALID_HANDEDNESS)}, got {self.handedness!r}"
            )
        if not math.isfinite(self.score):
            raise ValueError(f"score must be finite, got {self.score}")
        if self.label not in _VALID_HYPOTHESIS_LABELS:
            raise ValueError(
                f"label must be one of {sorted(_VALID_HYPOTHESIS_LABELS)}, got {self.label!r}"
            )
        check_str(self.provenance, "provenance")

        if not self.pose:
            raise ValueError(f"pose must be a non-empty sequence, got {self.pose!r}")
        for x in self.pose:
            if not math.isfinite(float(x)):
                raise ValueError(f"pose coordinate must be finite, got {x}")

        if not self.velocity:
            raise ValueError(
                f"velocity must be a non-empty sequence, got {self.velocity!r}"
            )
        for v in self.velocity:
            if not math.isfinite(float(v)):
                raise ValueError(f"velocity coordinate must be finite, got {v}")

        object.__setattr__(self, "pose", tuple(float(x) for x in self.pose))
        object.__setattr__(self, "velocity", tuple(float(v) for v in self.velocity))


@dataclass(frozen=True, slots=True, kw_only=True)
class MultiviewFitResult:
    """Result of multi-view initial state fitting."""

    best_hypothesis: InitialHypothesis | None
    candidate_hypotheses: tuple[InitialHypothesis, ...]
    residuals_evaluated: int
    camera_ids: tuple[str, ...]


# ---------------------------------------------------------------------------
# Algorithms: Velocity and Hypothesis Generation
# ---------------------------------------------------------------------------


def estimate_short_window_velocity(
    *,
    frame_ids: Sequence[str],
    time_points_s: Sequence[float],
    positions: Sequence[Sequence[float]],
) -> tuple[float, ...]:
    """Estimate short-window initial velocity using finite differences over >= 2 frames."""
    n_frames = len(frame_ids)
    if n_frames < 2:
        raise ValueError(
            f"at least 2 frames required for velocity estimation, got {n_frames}"
        )
    if len(time_points_s) != n_frames or len(positions) != n_frames:
        raise ValueError(
            f"length mismatch: frame_ids({n_frames}), time_points_s({len(time_points_s)}), "
            f"positions({len(positions)})"
        )

    for i in range(1, n_frames):
        if time_points_s[i] <= time_points_s[i - 1]:
            raise ValueError(
                f"time_points_s must be strictly increasing: {time_points_s[i - 1]} >= {time_points_s[i]}"
            )

    dim = len(positions[0])
    dt = float(time_points_s[-1] - time_points_s[0])
    v = [(float(positions[-1][d]) - float(positions[0][d])) / dt for d in range(dim)]
    return tuple(v)


def generate_monocular_hypotheses(
    *,
    camera: PinholeCameraModel,
    observed_body_bbox: tuple[int, int, int, int],
    subject_binding: SubjectModelBinding,
    nominal_depths_m: Sequence[float],
    handedness_options: Sequence[Handedness],
) -> tuple[InitialHypothesis, ...]:
    """Generate discrete depth, scale, and handedness hypotheses for ambiguous monocular views."""
    top, left, bottom, right = observed_body_bbox
    bbox_h_px = bottom - top
    nominal_height_m = subject_binding.visual_envelope.get("height_m", 1.80)

    hypotheses: list[InitialHypothesis] = []
    idx = 0
    for handedness in handedness_options:
        for depth_m in nominal_depths_m:
            check_pos_float(depth_m, "nominal_depth_m")
            # Image projection scale: h_px = f_y * (H * scale) / depth => scale = h_px * depth / (f_y * H)
            scale = (bbox_h_px * depth_m) / (camera.fy * nominal_height_m)
            idx += 1
            hyp = InitialHypothesis(
                hypothesis_id=f"hyp_{camera.camera_id}_{idx:03d}",
                subject_id=subject_binding.subject_id,
                camera_id=camera.camera_id,
                scale=scale,
                depth_m=depth_m,
                handedness=handedness,
                pose=(0.0, 0.0, depth_m, 1.0, 0.0, 0.0, 0.0),
                velocity=(0.0, 0.0, 0.0),
                score=0.5,
                label="inferred",
                provenance="monocular_depth_grid",
            )
            hypotheses.append(hyp)
    return tuple(hypotheses)


def fit_initial_state_multiview(
    *,
    cameras: Sequence[PinholeCameraModel],
    observed_masks: Sequence[MaskFrame],
    candidate_poses: Sequence[Sequence[float]],
    subject_binding: SubjectModelBinding,
    renderer: SilhouetteRenderer,
) -> MultiviewFitResult:
    """Fit initial state across calibrated multi-view silhouettes via objective evaluation."""
    if len(cameras) != len(observed_masks):
        raise ValueError("length mismatch between cameras and observed_masks")

    evaluated_candidates: list[InitialHypothesis] = []
    best_hyp: InitialHypothesis | None = None
    best_score = -1.0

    for i, pose in enumerate(candidate_poses):
        scores: list[float] = []
        for cam, obs_mask in zip(cameras, observed_masks, strict=True):
            # Render silhouette at candidate pose using SilhouetteRenderer protocol
            req = RenderRequest(
                camera_id=cam.camera_id,
                state=tuple(float(x) for x in pose),
                image_size_px=(cam.width_px, cam.height_px),
            )
            rendered = renderer.render(req)
            loss = compute_silhouette_loss(
                rendered=rendered,
                observed=obs_mask,
            )
            scores.append(loss.body_iou)

        agg_score = float(sum(scores) / max(1, len(scores)))
        cam0 = cameras[0]
        r0 = cam0.rotation_world_to_camera
        t0 = cam0.translation_world_to_camera
        zc = (
            r0[6] * float(pose[0])
            + r0[7] * float(pose[1])
            + r0[8] * float(pose[2])
            + t0[2]
        )
        depth_m = max(0.1, zc)

        hyp = InitialHypothesis(
            hypothesis_id=f"cand_{i:03d}",
            subject_id=subject_binding.subject_id,
            camera_id=cam0.camera_id,
            scale=1.0,
            depth_m=depth_m,
            handedness=subject_binding.handedness,
            pose=tuple(float(x) for x in pose),
            velocity=(0.0, 0.0, 0.0),
            score=agg_score,
            label="inferred",
            provenance="multiview_residual_search",
        )
        evaluated_candidates.append(hyp)
        if agg_score > best_score:
            best_score = agg_score
            best_hyp = hyp

    return MultiviewFitResult(
        best_hypothesis=best_hyp,
        candidate_hypotheses=tuple(evaluated_candidates),
        residuals_evaluated=len(candidate_poses),
        camera_ids=tuple(c.camera_id for c in cameras),
    )

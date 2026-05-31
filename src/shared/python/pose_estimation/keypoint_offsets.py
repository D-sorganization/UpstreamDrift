"""Calibratable detector-keypoint to joint-center offset model.

Detector anatomical keypoints often land on visible surface landmarks, while
biomechanical fitting consumes model joint centers. This module estimates that
systematic displacement in each site's segment frame so downstream residuals
can compare keypoints against biased observation predictions instead of raw
joint-center positions.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from typing import Any

import numpy as np

from src.shared.python.core.contracts import (
    check_finite,
    ensure,
    require,
)

_IDENTITY_3 = np.eye(3)
_ROTATION_ATOL = 1e-8


@dataclass(frozen=True)
class KeypointOffsetSite:
    """Mapping from a detector keypoint to an engine-agnostic canonical site.

    Attributes:
        keypoint_name: Detector keypoint label, e.g. ``right_hip``.
        canonical_site: Canonical anatomical site / joint-center identifier.
        segment_name: Segment whose local frame expresses the offset.
        joint_center_name: Optional model-output key. Defaults to
            ``keypoint_name`` when omitted so simple keypoint-aligned callers do
            not need a second mapping.
    """

    keypoint_name: str
    canonical_site: str
    segment_name: str
    joint_center_name: str | None = None

    def center_lookup_name(self) -> str:
        """Return the joint-center dictionary key for this site."""
        return self.joint_center_name or self.keypoint_name

    def __post_init__(self) -> None:
        require(bool(self.keypoint_name), "keypoint_name must be non-empty")
        require(bool(self.canonical_site), "canonical_site must be non-empty")
        require(bool(self.segment_name), "segment_name must be non-empty")
        if self.joint_center_name is not None:
            require(bool(self.joint_center_name), "joint_center_name must be non-empty")


@dataclass(frozen=True)
class KeypointOffsetEstimate:
    """Estimated segment-frame offset and calibration uncertainty.

    ``offset_m`` is expressed in ``segment_name``'s local frame. The covariance
    and standard error are in square metres and metres respectively, computed
    from the retained calibration frames after confidence filtering.
    """

    keypoint_name: str
    canonical_site: str
    segment_name: str
    offset_m: tuple[float, float, float]
    covariance_m2: tuple[tuple[float, float, float], ...]
    standard_error_m: tuple[float, float, float]
    rms_residual_m: float
    sample_count: int
    mean_confidence: float
    joint_center_name: str | None = None

    def center_lookup_name(self) -> str:
        """Return the model joint-center key associated with this offset."""
        return self.joint_center_name or self.keypoint_name

    def predict_keypoint_m(
        self,
        joint_center_world_m: Iterable[float],
        segment_rotation_world_from_segment: Iterable[Iterable[float]],
    ) -> np.ndarray:
        """Predict detector keypoint position from a model joint center.

        Postcondition: returns a finite world-frame vector of shape ``(3,)``.
        """
        center = _as_vector("joint_center_world_m", joint_center_world_m)
        rotation = _as_rotation(
            "segment_rotation_world_from_segment",
            segment_rotation_world_from_segment,
        )
        predicted = center + rotation @ np.asarray(self.offset_m, dtype=float)
        ensure(check_finite(predicted), "predicted keypoint must be finite", predicted)
        return predicted

    def residual_m(
        self,
        observed_keypoint_world_m: Iterable[float],
        joint_center_world_m: Iterable[float],
        segment_rotation_world_from_segment: Iterable[Iterable[float]],
    ) -> np.ndarray:
        """Return observed minus predicted detector keypoint residual in metres."""
        observed = _as_vector("observed_keypoint_world_m", observed_keypoint_world_m)
        return observed - self.predict_keypoint_m(
            joint_center_world_m, segment_rotation_world_from_segment
        )

    def to_dict(self) -> dict[str, Any]:
        """Serialize this estimate for calibration reports or JSON metadata."""
        return {
            "keypoint_name": self.keypoint_name,
            "canonical_site": self.canonical_site,
            "segment_name": self.segment_name,
            "offset_m": list(self.offset_m),
            "covariance_m2": [list(row) for row in self.covariance_m2],
            "standard_error_m": list(self.standard_error_m),
            "rms_residual_m": self.rms_residual_m,
            "sample_count": self.sample_count,
            "mean_confidence": self.mean_confidence,
            "joint_center_name": self.joint_center_name,
        }

    def __post_init__(self) -> None:
        require(bool(self.keypoint_name), "keypoint_name must be non-empty")
        require(bool(self.canonical_site), "canonical_site must be non-empty")
        require(bool(self.segment_name), "segment_name must be non-empty")
        if self.joint_center_name is not None:
            require(bool(self.joint_center_name), "joint_center_name must be non-empty")
        _as_vector("offset_m", self.offset_m)
        covariance = np.asarray(self.covariance_m2, dtype=float)
        require(covariance.shape == (3, 3), "covariance_m2 must have shape (3, 3)")
        require(check_finite(covariance), "covariance_m2 must be finite", covariance)
        _as_vector("standard_error_m", self.standard_error_m)
        require(self.rms_residual_m >= 0.0, "rms_residual_m must be non-negative")
        require(self.sample_count >= 1, "sample_count must be positive")
        require(
            0.0 <= self.mean_confidence <= 1.0,
            "mean_confidence must lie in [0, 1]",
            self.mean_confidence,
        )


@dataclass(frozen=True)
class KeypointOffsetModel:
    """Collection of calibratable offsets keyed by detector keypoint name."""

    offsets: Mapping[str, KeypointOffsetEstimate]

    def offset_for(self, keypoint_name: str) -> KeypointOffsetEstimate:
        """Return the calibrated offset for a detector keypoint."""
        require(bool(keypoint_name), "keypoint_name must be non-empty")
        if keypoint_name not in self.offsets:
            raise KeyError(f"No offset estimate for keypoint '{keypoint_name}'")
        return self.offsets[keypoint_name]

    def predict_keypoint_m(
        self,
        keypoint_name: str,
        joint_center_world_m: Iterable[float],
        segment_rotation_world_from_segment: Iterable[Iterable[float]],
    ) -> np.ndarray:
        """Predict a world-frame detector keypoint for one calibrated site."""
        offset = self.offset_for(keypoint_name)
        return offset.predict_keypoint_m(
            joint_center_world_m, segment_rotation_world_from_segment
        )

    def residual_m(
        self,
        keypoint_name: str,
        observed_keypoint_world_m: Iterable[float],
        joint_center_world_m: Iterable[float],
        segment_rotation_world_from_segment: Iterable[Iterable[float]],
    ) -> np.ndarray:
        """Return observed minus predicted residual for one calibrated site."""
        offset = self.offset_for(keypoint_name)
        return offset.residual_m(
            observed_keypoint_world_m,
            joint_center_world_m,
            segment_rotation_world_from_segment,
        )

    def residuals_for_clip(
        self,
        *,
        joint_centers_world_m: Mapping[str, Iterable[Iterable[float]]],
        keypoints_world_m: Mapping[str, Iterable[Iterable[float]]],
        segment_rotations_world_from_segment: Mapping[
            str, Iterable[Iterable[Iterable[float]]]
        ],
    ) -> dict[str, np.ndarray]:
        """Return per-frame residual arrays for every calibrated keypoint.

        Postcondition: each returned array has shape ``(frames, 3)`` and finite
        metre values, suitable for later CC-18 least-squares residual assembly.
        """
        residuals: dict[str, np.ndarray] = {}
        for keypoint_name, estimate in self.offsets.items():
            centers = _lookup_series(
                joint_centers_world_m,
                estimate.center_lookup_name(),
                "joint_centers_world_m",
            )
            observed = _lookup_series(
                keypoints_world_m, keypoint_name, "keypoints_world_m"
            )
            rotations = _lookup_rotations(
                segment_rotations_world_from_segment,
                estimate.segment_name,
                "segment_rotations_world_from_segment",
            )
            _require_same_frame_count(centers, observed, rotations)
            predicted = centers + np.einsum(
                "nij,j->ni", rotations, np.asarray(estimate.offset_m, dtype=float)
            )
            site_residuals = observed - predicted
            ensure(
                check_finite(site_residuals),
                "residuals must be finite",
                site_residuals,
            )
            residuals[keypoint_name] = site_residuals
        return residuals

    def to_documentation_rows(self) -> list[dict[str, Any]]:
        """Return stable rows for a calibration report table."""
        return [self.offsets[name].to_dict() for name in sorted(self.offsets)]

    def __post_init__(self) -> None:
        require(self.offsets is not None, "offsets must be provided")
        require(len(self.offsets) > 0, "offsets must be non-empty")
        for key, estimate in self.offsets.items():
            require(key == estimate.keypoint_name, "offset key must match estimate")


def estimate_keypoint_offset(
    *,
    keypoint_name: str,
    canonical_site: str,
    segment_name: str,
    joint_centers_world_m: Iterable[Iterable[float]],
    keypoints_world_m: Iterable[Iterable[float]],
    segment_rotations_world_from_segment: Iterable[Iterable[Iterable[float]]],
    confidences: Iterable[float] | None = None,
    min_confidence: float = 0.0,
    min_samples: int = 1,
    joint_center_name: str | None = None,
) -> KeypointOffsetEstimate:
    """Estimate one keypoint-to-joint-center offset from a calibration clip.

    Each frame contributes ``R_ws.T @ (keypoint_world - joint_center_world)``.
    The returned offset is the confidence-weighted mean in the segment frame;
    uncertainty is the weighted scatter of retained per-frame estimates.
    """
    require(bool(keypoint_name), "keypoint_name must be non-empty")
    require(bool(canonical_site), "canonical_site must be non-empty")
    require(bool(segment_name), "segment_name must be non-empty")
    require(0.0 <= min_confidence <= 1.0, "min_confidence must lie in [0, 1]")
    require(min_samples >= 1, "min_samples must be positive")
    if joint_center_name is not None:
        require(bool(joint_center_name), "joint_center_name must be non-empty")

    centers = _as_vector_series("joint_centers_world_m", joint_centers_world_m)
    observed = _as_vector_series("keypoints_world_m", keypoints_world_m)
    rotations = _as_rotation_series(
        "segment_rotations_world_from_segment",
        segment_rotations_world_from_segment,
    )
    _require_same_frame_count(centers, observed, rotations)
    weights = _as_confidences(confidences, centers.shape[0])
    retained = weights >= min_confidence
    retained_count = int(np.count_nonzero(retained))
    require(
        retained_count >= min_samples,
        "not enough calibration frames meet min_confidence",
        retained_count,
    )

    centers = centers[retained]
    observed = observed[retained]
    rotations = rotations[retained]
    weights = weights[retained]
    require(float(np.sum(weights)) > 0.0, "retained confidence weights sum to zero")

    deltas_world = observed - centers
    offsets_segment = np.einsum("nji,nj->ni", rotations, deltas_world)
    normalized_weights = weights / float(np.sum(weights))
    offset = np.average(offsets_segment, axis=0, weights=normalized_weights)
    centered_offsets = offsets_segment - offset
    covariance = (centered_offsets.T * normalized_weights) @ centered_offsets
    effective_n = _effective_sample_count(weights)
    standard_error = np.sqrt(np.diag(covariance) / effective_n)

    residuals_world = observed - (centers + np.einsum("nij,j->ni", rotations, offset))
    residual_energy = np.einsum("ni,ni->n", residuals_world, residuals_world)
    rms = float(np.sqrt(np.average(residual_energy, weights=normalized_weights)))
    estimate = KeypointOffsetEstimate(
        keypoint_name=keypoint_name,
        canonical_site=canonical_site,
        segment_name=segment_name,
        offset_m=_vector_tuple(offset),
        covariance_m2=_matrix_tuple(covariance),
        standard_error_m=_vector_tuple(standard_error),
        rms_residual_m=rms,
        sample_count=retained_count,
        mean_confidence=float(np.mean(weights)),
        joint_center_name=joint_center_name,
    )
    ensure(estimate.sample_count >= min_samples, "sample_count must meet minimum")
    return estimate


def estimate_keypoint_offset_model(
    *,
    sites: Iterable[KeypointOffsetSite],
    joint_centers_world_m: Mapping[str, Iterable[Iterable[float]]],
    keypoints_world_m: Mapping[str, Iterable[Iterable[float]]],
    segment_rotations_world_from_segment: Mapping[
        str, Iterable[Iterable[Iterable[float]]]
    ],
    confidences: Mapping[str, Iterable[float]] | None = None,
    min_confidence: float = 0.0,
    min_samples: int = 1,
) -> KeypointOffsetModel:
    """Estimate a multi-site offset model from aligned calibration arrays."""
    site_list = list(sites)
    require(len(site_list) > 0, "sites must be non-empty")
    offsets: dict[str, KeypointOffsetEstimate] = {}
    for site in site_list:
        center_name = site.center_lookup_name()
        site_confidences = None
        if confidences is not None:
            site_confidences = confidences.get(site.keypoint_name)
        offsets[site.keypoint_name] = estimate_keypoint_offset(
            keypoint_name=site.keypoint_name,
            canonical_site=site.canonical_site,
            segment_name=site.segment_name,
            joint_centers_world_m=_lookup_series(
                joint_centers_world_m, center_name, "joint_centers_world_m"
            ),
            keypoints_world_m=_lookup_series(
                keypoints_world_m, site.keypoint_name, "keypoints_world_m"
            ),
            segment_rotations_world_from_segment=_lookup_rotations(
                segment_rotations_world_from_segment,
                site.segment_name,
                "segment_rotations_world_from_segment",
            ),
            confidences=site_confidences,
            min_confidence=min_confidence,
            min_samples=min_samples,
            joint_center_name=center_name,
        )
    return KeypointOffsetModel(offsets=offsets)


def _lookup_series(
    values: Mapping[str, Iterable[Iterable[float]]],
    key: str,
    label: str,
) -> np.ndarray:
    if key not in values:
        raise KeyError(f"{label} missing key '{key}'")
    return _as_vector_series(f"{label}[{key}]", values[key])


def _lookup_rotations(
    values: Mapping[str, Iterable[Iterable[Iterable[float]]]],
    key: str,
    label: str,
) -> np.ndarray:
    if key not in values:
        raise KeyError(f"{label} missing key '{key}'")
    return _as_rotation_series(f"{label}[{key}]", values[key])


def _as_vector(label: str, value: Iterable[float]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(arr.shape == (3,), f"{label} must have shape (3,)", arr.shape)
    require(check_finite(arr), f"{label} must be finite", arr)
    return arr


def _as_vector_series(label: str, values: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    require(arr.ndim == 2 and arr.shape[1] == 3, f"{label} must have shape (N, 3)")
    require(arr.shape[0] > 0, f"{label} must contain at least one frame")
    require(check_finite(arr), f"{label} must be finite", arr)
    return arr


def _as_rotation(label: str, value: Iterable[Iterable[float]]) -> np.ndarray:
    arr = np.asarray(value, dtype=float)
    require(arr.shape == (3, 3), f"{label} must have shape (3, 3)", arr.shape)
    require(check_finite(arr), f"{label} must be finite", arr)
    _require_proper_rotation(arr, label)
    return arr


def _as_rotation_series(
    label: str, values: Iterable[Iterable[Iterable[float]]]
) -> np.ndarray:
    arr = np.asarray(values, dtype=float)
    require(
        arr.ndim == 3 and arr.shape[1:] == (3, 3),
        f"{label} must have shape (N, 3, 3)",
        arr.shape,
    )
    require(arr.shape[0] > 0, f"{label} must contain at least one frame")
    require(check_finite(arr), f"{label} must be finite", arr)
    for i, rotation in enumerate(arr):
        _require_proper_rotation(rotation, f"{label}[{i}]")
    return arr


def _require_proper_rotation(rotation: np.ndarray, label: str) -> None:
    is_orthonormal = np.allclose(
        rotation.T @ rotation, _IDENTITY_3, atol=_ROTATION_ATOL
    )
    determinant = float(np.linalg.det(rotation))
    require(
        is_orthonormal and np.isclose(determinant, 1.0, atol=_ROTATION_ATOL),
        f"{label} must be a proper rotation matrix",
        rotation,
    )


def _as_confidences(values: Iterable[float] | None, frame_count: int) -> np.ndarray:
    if values is None:
        return np.ones(frame_count, dtype=float)
    arr = np.asarray(values, dtype=float)
    require(arr.shape == (frame_count,), "confidences must have shape (N,)")
    require(check_finite(arr), "confidences must be finite", arr)
    require(bool(np.all((arr >= 0.0) & (arr <= 1.0))), "confidences must be in [0, 1]")
    return arr


def _require_same_frame_count(
    centers: np.ndarray, observed: np.ndarray, rotations: np.ndarray
) -> None:
    frame_count = centers.shape[0]
    require(observed.shape[0] == frame_count, "keypoints and centers length mismatch")
    require(rotations.shape[0] == frame_count, "rotations and centers length mismatch")


def _effective_sample_count(weights: np.ndarray) -> float:
    numerator = float(np.sum(weights) ** 2)
    denominator = float(np.vdot(weights, weights))
    return numerator / denominator if denominator > 0.0 else 1.0


def _vector_tuple(values: np.ndarray) -> tuple[float, float, float]:
    vector = _as_vector("vector", values)
    return (float(vector[0]), float(vector[1]), float(vector[2]))


def _matrix_tuple(values: np.ndarray) -> tuple[tuple[float, float, float], ...]:
    matrix = np.asarray(values, dtype=float)
    require(matrix.shape == (3, 3), "matrix must have shape (3, 3)")
    require(check_finite(matrix), "matrix must be finite", matrix)
    return tuple((float(row[0]), float(row[1]), float(row[2])) for row in matrix)


__all__ = [
    "KeypointOffsetEstimate",
    "KeypointOffsetModel",
    "KeypointOffsetSite",
    "estimate_keypoint_offset",
    "estimate_keypoint_offset_model",
]

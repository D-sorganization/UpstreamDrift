"""Convert immutable capture observations into pixels without changing evidence."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import pairwise
from typing import Any, Protocol

import numpy as np

from .contracts import _array


class CaptureEvidenceSource(Protocol):
    capture_id: str

    def frame(self, index: int) -> dict[str, Any]: ...


@dataclass(frozen=True)
class CaptureImageEvidence:
    capture_id: str
    frame_indices: tuple[int, ...]
    frame_ids: tuple[str, ...]
    frame_hashes: tuple[str, ...]
    source_times: np.ndarray
    observed_pixels: np.ndarray
    confidence: np.ndarray
    marker_labels: tuple[str, ...]
    image_size: tuple[int, int]
    unknown_visibility_weight: float

    def __post_init__(self) -> None:
        for name in ("source_times", "observed_pixels", "confidence"):
            object.__setattr__(self, name, _array(getattr(self, name), name))


def read_capture_evidence(
    review: CaptureEvidenceSource,
    labels: tuple[str, ...],
    frame_indices: tuple[int, ...],
    *,
    unknown_visibility_weight: float,
) -> CaptureImageEvidence:
    """Preserve exact frame identity/PTS; unknown visibility uses an explicit prior weight."""
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Image marker labels must be nonempty and unique")
    if (
        len(frame_indices) < 2
        or any(
            isinstance(index, bool) or not isinstance(index, int) or index < 0
            for index in frame_indices
        )
        or any(b <= a for a, b in pairwise(frame_indices))
    ):
        raise ValueError("Select at least two increasing source frame indices")
    if (
        not np.isfinite(unknown_visibility_weight)
        or not 0 <= unknown_visibility_weight <= 1
    ):
        raise ValueError("Unknown visibility weight must be in [0, 1]")
    rows = [review.frame(index) for index in frame_indices]
    size = (rows[0]["image_width"], rows[0]["image_height"])
    if any((row["image_width"], row["image_height"]) != size for row in rows):
        raise ValueError("Selected frames must use one source image size")
    observed = np.zeros((len(rows), len(labels), 2))
    confidence = np.zeros((len(rows), len(labels)))
    times, identities, hashes = [], [], []
    for i, row in enumerate(rows):
        frame = row["frame"]
        times.append(
            frame["pts_ticks"]
            * frame["timebase_numerator"]
            / frame["timebase_denominator"]
        )
        identities.append(frame["frame_id"])
        hashes.append(frame["frame_sha256"])
        observation = row["observation"]
        if observation["status"] == "missing":
            continue
        for j, label in enumerate(labels):
            point = observation["landmarks"].get(label)
            if point is None:
                continue
            observed[i, j] = (point["x"] * size[0], point["y"] * size[1])
            visibility = point.get("visibility")
            confidence[i, j] = (
                unknown_visibility_weight if visibility is None else visibility
            )
    if (
        not np.isfinite(observed).all()
        or not np.isfinite(confidence).all()
        or np.any((confidence < 0) | (confidence > 1))
    ):
        raise ValueError(
            "Capture pixels and confidence must be finite; confidence is in [0, 1]"
        )
    return CaptureImageEvidence(
        review.capture_id,
        frame_indices,
        tuple(identities),
        tuple(hashes),
        np.asarray(times),
        observed,
        confidence,
        labels,
        size,
        unknown_visibility_weight,
    )

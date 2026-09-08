"""Compare two detectors on the same recordings (issue #9628).

Metrics are computed from the per-view observation files ingest writes, so a
comparison never re-runs a network. Everything is per joint and per view:

- ``coverage``: fraction of frames where the joint was reported with
  confidence at or above ``min_confidence``;
- ``mean_confidence``: mean reported confidence over those frames;
- ``jitter``: median frame-to-frame displacement of the joint, in units of
  the subject's bounding-box height in that frame, over consecutive frames
  where both are covered — a proxy for detector noise at high frame rates;
- ``agreement`` (between two detectors): median distance between their
  estimates of the same joint on frames both cover, again normalized by the
  box height. Names are matched through :data:`SHARED_JOINTS`, since the
  detectors use different skeletons.

No metric claims accuracy: without 3-D ground truth the honest statements
are coverage, self-consistency over time, and cross-detector agreement.
"""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from src.shared.python.core.contracts import require

# MediaPipe and BODY_25 share these names verbatim.
SHARED_JOINTS: tuple[str, ...] = (
    "nose",
    "left_shoulder",
    "right_shoulder",
    "left_elbow",
    "right_elbow",
    "left_wrist",
    "right_wrist",
    "left_hip",
    "right_hip",
    "left_knee",
    "right_knee",
    "left_ankle",
    "right_ankle",
)
DEFAULT_MIN_CONFIDENCE = 0.5


def _float_array(values: Any) -> np.ndarray:
    """Float array where JSON ``null`` (a joint the detector omitted) is NaN."""
    return np.array(
        [
            [np.nan if v is None else v for v in item]
            if isinstance(item, list)
            else (np.nan if item is None else item)
            for item in values
        ],
        dtype=float,
    )


class JointMetrics(BaseModel):
    model_config = ConfigDict(frozen=True)

    joint: str
    frames: int
    coverage: float
    mean_confidence: float | None
    jitter: float | None


class DetectorSeries:
    """One detector's frames of one view, indexed by frame number."""

    def __init__(self, observations: Mapping[str, Any]) -> None:
        self.names: list[str] = list(observations["detector_layout"]["keypoint_names"])
        self.fps = float(observations["fps"])
        self.frames_total = int(observations["frames_total"])
        require(self.fps > 0, "fps must be positive", self.fps)
        self._rows: dict[int, tuple[np.ndarray, np.ndarray]] = {}
        for row in observations["frames"]:
            index = int(round(float(row["time_s"]) * self.fps))
            self._rows[index] = (
                _float_array(row["keypoints_px"]),
                _float_array(row["confidence"]),
            )

    def joint(self, name: str, index: int, min_confidence: float) -> np.ndarray | None:
        """Pixel position of ``name`` at frame ``index`` when covered, else None."""
        row = self._rows.get(index)
        if row is None or name not in self.names:
            return None
        k = self.names.index(name)
        if row[1][k] < min_confidence:
            return None
        return row[0][k]

    def confidence(self, name: str, index: int) -> float | None:
        row = self._rows.get(index)
        if row is None or name not in self.names:
            return None
        return float(row[1][self.names.index(name)])

    def box_height(self, index: int, min_confidence: float) -> float | None:
        """Vertical extent of the covered joints in a frame (subject scale)."""
        row = self._rows.get(index)
        if row is None:
            return None
        ys = row[0][row[1] >= min_confidence][:, 1]
        if ys.size < 2 or ys.max() - ys.min() <= 0:
            return None
        return float(ys.max() - ys.min())


def joint_metrics(
    series: DetectorSeries,
    name: str,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> JointMetrics:
    """Coverage, confidence and jitter of one joint over the whole series."""
    require(0.0 <= min_confidence <= 1.0, "min_confidence in [0, 1]", min_confidence)
    total = series.frames_total
    covered = 0
    confidences: list[float] = []
    steps: list[float] = []
    prev: np.ndarray | None = None
    for index in range(total):
        point = series.joint(name, index, min_confidence)
        conf = series.confidence(name, index)
        if conf is not None and conf >= min_confidence:
            covered += 1
            confidences.append(conf)
        if point is not None and prev is not None:
            scale = series.box_height(index, min_confidence)
            if scale:
                steps.append(float(np.linalg.norm(point - prev)) / scale)
        prev = point
    return JointMetrics(
        joint=name,
        frames=total,
        coverage=covered / total if total else 0.0,
        mean_confidence=float(np.mean(confidences)) if confidences else None,
        jitter=float(np.median(steps)) if steps else None,
    )


def agreement(
    a: DetectorSeries,
    b: DetectorSeries,
    name: str,
    *,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> tuple[int, float | None]:
    """``(frames both cover, median normalized distance)`` for one joint."""
    distances: list[float] = []
    for index in range(min(a.frames_total, b.frames_total)):
        pa, pb = (
            a.joint(name, index, min_confidence),
            b.joint(name, index, min_confidence),
        )
        scale = a.box_height(index, min_confidence)
        if pa is None or pb is None or not scale:
            continue
        distances.append(float(np.linalg.norm(pa - pb)) / scale)
    return len(distances), (float(np.median(distances)) if distances else None)


class ComparisonReport(BaseModel):
    """``comparison.json``: per-detector joint metrics and pairwise agreement."""

    model_config = ConfigDict(frozen=True)

    schema_version: str = "detector-comparison/1.0.0"
    view: str
    detectors: tuple[str, ...]
    joints: tuple[str, ...]
    metrics: dict[str, tuple[JointMetrics, ...]]
    agreement: dict[str, dict[str, Any]] = Field(default_factory=dict)

    def markdown(self) -> str:
        """A table per joint: coverage / mean confidence / jitter per detector."""
        head = "| Joint | " + " | ".join(
            f"{d} cov | {d} conf | {d} jitter" for d in self.detectors
        )
        if self.agreement:
            head += " | agree (n) |"
        else:
            head += " |"
        lines = [head, "|" + "---|" * (head.count("|") - 1)]
        for i, joint in enumerate(self.joints):
            cells = [joint]
            for d in self.detectors:
                m = self.metrics[d][i]
                cells += [
                    f"{m.coverage:.2f}",
                    "-" if m.mean_confidence is None else f"{m.mean_confidence:.2f}",
                    "-" if m.jitter is None else f"{m.jitter:.3f}",
                ]
            if self.agreement:
                ag = self.agreement.get(joint, {})
                med = ag.get("median")
                cells.append(
                    "-" if med is None else f"{med:.3f} ({ag.get('frames', 0)})"
                )
            lines.append("| " + " | ".join(cells) + " |")
        return "\n".join(lines) + "\n"


def compare_view(
    view: str,
    series: Mapping[str, DetectorSeries],
    *,
    joints: Sequence[str] = SHARED_JOINTS,
    min_confidence: float = DEFAULT_MIN_CONFIDENCE,
) -> ComparisonReport:
    """Metrics for every detector on ``joints``; agreement when there are two."""
    require(len(series) >= 1, "at least one detector series is required")
    names = tuple(series)
    metrics = {
        d: tuple(joint_metrics(s, j, min_confidence=min_confidence) for j in joints)
        for d, s in series.items()
    }
    agree: dict[str, dict[str, Any]] = {}
    if len(names) == 2:
        a, b = series[names[0]], series[names[1]]
        for j in joints:
            n, med = agreement(a, b, j, min_confidence=min_confidence)
            agree[j] = {"frames": n, "median": med}
    return ComparisonReport(
        view=view,
        detectors=names,
        joints=tuple(joints),
        metrics=metrics,
        agreement=agree,
    )


def load_series(path: Path) -> DetectorSeries:
    return DetectorSeries(json.loads(path.read_text(encoding="utf-8")))

"""Unit tests for :mod:`anthropometrics.estimators.from_mocap`.

Covers exact-recovery on synthetic data, NaN tolerance, error paths,
method comparison (mean / median / min), and a DRY assertion that the
production code routes through ``motion_pipeline.scaling.anthropometric``
for the per-frame distance kernel.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from anthropometrics.estimators.from_mocap import (
    SegmentDef,
    estimate_segment_lengths_from_markers,
)

# --------------------------------------------------------------------------- #
# Fixtures / builders.                                                        #
# --------------------------------------------------------------------------- #
_KNOWN_LENGTH = 0.30  # metres


def _trajectory(
    n_frames: int = 100, length: float = _KNOWN_LENGTH
) -> dict[str, np.ndarray]:
    """Two markers separated by a constant Euclidean distance.

    Both markers translate together along an arbitrary path so that the
    inter-marker distance is constant at *length* on every frame.
    """
    rng = np.random.default_rng(seed=0)
    base = rng.normal(scale=1.0, size=(n_frames, 3))
    offset = np.array([length, 0.0, 0.0])
    return {
        "PROX": base.copy(),
        "DIST": base + offset,
    }


def _seg() -> list[SegmentDef]:
    return [SegmentDef(name="forearm", proximal_marker="PROX", distal_marker="DIST")]


# --------------------------------------------------------------------------- #
# Happy path: exact recovery.                                                 #
# --------------------------------------------------------------------------- #
def test_recovers_known_length_to_machine_precision() -> None:
    markers = _trajectory()
    out = estimate_segment_lengths_from_markers(markers, _seg())
    assert set(out) == {"forearm"}
    assert out["forearm"] == pytest.approx(_KNOWN_LENGTH, abs=1e-9)


def test_default_method_is_median() -> None:
    markers = _trajectory()
    default = estimate_segment_lengths_from_markers(markers, _seg())
    explicit = estimate_segment_lengths_from_markers(
        markers, _seg(), method="median_distance"
    )
    assert default == explicit


def test_preserves_segment_order() -> None:
    rng = np.random.default_rng(seed=1)
    base = rng.normal(size=(20, 3))
    markers = {
        "A": base,
        "B": base + np.array([0.4, 0.0, 0.0]),
        "C": base + np.array([0.0, 0.5, 0.0]),
    }
    segs = [
        SegmentDef("seg_ac", "A", "C"),
        SegmentDef("seg_ab", "A", "B"),
    ]
    out = estimate_segment_lengths_from_markers(markers, segs)
    assert list(out.keys()) == ["seg_ac", "seg_ab"]
    assert out["seg_ab"] == pytest.approx(0.4, abs=1e-9)
    assert out["seg_ac"] == pytest.approx(0.5, abs=1e-9)


# --------------------------------------------------------------------------- #
# NaN tolerance.                                                              #
# --------------------------------------------------------------------------- #
def test_nan_tolerance_recovers_from_remaining_frames() -> None:
    markers = _trajectory()
    # Knock out half of the proximal marker rows.
    markers["PROX"][::2] = np.nan
    out = estimate_segment_lengths_from_markers(markers, _seg())
    assert out["forearm"] == pytest.approx(_KNOWN_LENGTH, abs=1e-9)


def test_inf_marker_rows_are_treated_as_missing() -> None:
    markers = _trajectory()
    markers["DIST"][10] = np.inf
    out = estimate_segment_lengths_from_markers(markers, _seg())
    assert out["forearm"] == pytest.approx(_KNOWN_LENGTH, abs=1e-9)


def test_all_nan_for_segment_raises_value_error() -> None:
    markers = _trajectory(n_frames=10)
    markers["DIST"][:] = np.nan
    with pytest.raises(ValueError, match="No finite frames"):
        estimate_segment_lengths_from_markers(markers, _seg())


# --------------------------------------------------------------------------- #
# Error paths.                                                                #
# --------------------------------------------------------------------------- #
def test_empty_markers_raises_listing_required_markers() -> None:
    with pytest.raises(ValueError, match="Missing required marker") as exc_info:
        estimate_segment_lengths_from_markers({}, _seg())
    msg = str(exc_info.value)
    assert "PROX" in msg
    assert "DIST" in msg


def test_missing_one_marker_raises() -> None:
    markers = _trajectory()
    del markers["DIST"]
    with pytest.raises(ValueError, match="Missing required marker"):
        estimate_segment_lengths_from_markers(markers, _seg())


def test_empty_segment_definitions_raises() -> None:
    with pytest.raises(ValueError, match="must be non-empty"):
        estimate_segment_lengths_from_markers({"PROX": np.zeros((1, 3))}, [])


def test_wrong_shape_raises() -> None:
    markers = {
        "PROX": np.zeros((10, 2)),  # not (T, 3)
        "DIST": np.zeros((10, 3)),
    }
    with pytest.raises(ValueError, match=r"\(T, 3\)"):
        estimate_segment_lengths_from_markers(markers, _seg())


def test_inconsistent_frame_counts_raise() -> None:
    markers = {
        "PROX": np.zeros((10, 3)),
        "DIST": np.zeros((11, 3)),
    }
    with pytest.raises(ValueError, match="same number of frames"):
        estimate_segment_lengths_from_markers(markers, _seg())


def test_zero_frames_raises() -> None:
    markers = {
        "PROX": np.zeros((0, 3)),
        "DIST": np.zeros((0, 3)),
    }
    with pytest.raises(ValueError, match="zero frames"):
        estimate_segment_lengths_from_markers(markers, _seg())


def test_unknown_method_raises() -> None:
    with pytest.raises(ValueError, match="Unknown method"):
        estimate_segment_lengths_from_markers(
            _trajectory(),
            _seg(),
            method="bogus",  # type: ignore[arg-type]
        )


# --------------------------------------------------------------------------- #
# Method comparison.                                                          #
# --------------------------------------------------------------------------- #
def _markers_with_outliers() -> dict[str, np.ndarray]:
    """Mostly clean trajectory plus a handful of inflated frames.

    Frames 0..89 give a constant inter-marker distance of 0.30 m;
    frames 90..99 inflate the distance to 1.30 m (outliers). Result:

    * mean ~ 0.40 m  (pulled up by outliers)
    * median = 0.30 m  (robust)
    * min = 0.30 m  (conservative — drops to clean lower bound)
    """
    n = 100
    rng = np.random.default_rng(seed=2)
    base = rng.normal(size=(n, 3))
    dist = base + np.array([0.30, 0.0, 0.0])
    # Inflate the last 10 frames by an extra 1.0 m along x.
    dist[90:] += np.array([1.0, 0.0, 0.0])
    return {"PROX": base, "DIST": dist}


def test_mean_distance_is_pulled_up_by_outliers() -> None:
    out = estimate_segment_lengths_from_markers(
        _markers_with_outliers(), _seg(), method="mean_distance"
    )
    # 90 frames at 0.30 + 10 frames at 1.30 = mean of 0.40
    assert out["forearm"] == pytest.approx(0.40, abs=1e-9)


def test_median_distance_is_robust_to_outliers() -> None:
    out = estimate_segment_lengths_from_markers(
        _markers_with_outliers(), _seg(), method="median_distance"
    )
    assert out["forearm"] == pytest.approx(0.30, abs=1e-9)


def test_min_distance_is_conservative() -> None:
    out = estimate_segment_lengths_from_markers(
        _markers_with_outliers(), _seg(), method="min_distance"
    )
    # Minimum across the clean frames is exactly 0.30.
    assert out["forearm"] == pytest.approx(0.30, abs=1e-9)


def test_method_ordering_under_outliers() -> None:
    markers = _markers_with_outliers()
    mean = estimate_segment_lengths_from_markers(
        markers, _seg(), method="mean_distance"
    )["forearm"]
    median = estimate_segment_lengths_from_markers(
        markers, _seg(), method="median_distance"
    )["forearm"]
    minimum = estimate_segment_lengths_from_markers(
        markers, _seg(), method="min_distance"
    )["forearm"]
    # Mean is dragged up by outliers; median equals the clean value;
    # min sits at-or-below the clean value (it equals 0.30 here).
    assert minimum <= median < mean


# --------------------------------------------------------------------------- #
# DRY assertion: production code routes through motion_pipeline.              #
# --------------------------------------------------------------------------- #
def test_production_code_references_motion_pipeline_distance_kernel() -> None:
    """The estimator must reuse the motion-pipeline distance helper.

    Guards against silent drift where someone reimplements the
    Euclidean kernel in this module instead of delegating.
    """
    src = Path("src/shared/python/anthropometrics/estimators/from_mocap.py").read_text(
        encoding="utf-8"
    )
    assert "motion_pipeline.scaling.anthropometric" in src, (
        "from_mocap.py must import the per-frame distance helper from "
        "motion_pipeline.scaling.anthropometric (DRY)."
    )
    assert "_compute_segment_length" in src, (
        "from_mocap.py must call _compute_segment_length, not reimplement "
        "the Euclidean-distance kernel."
    )

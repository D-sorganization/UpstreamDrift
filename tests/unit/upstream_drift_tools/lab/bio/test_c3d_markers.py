"""Unit tests for ``validate_marker_positions`` heuristic + cp1252 safety.

Pins the fixes from PR #4582:

1. The min-position heuristic must NOT warn on swing-data minima around
   ``-1.97 m`` (negative values up to ~2 m are normal when the world
   origin is at the target).
2. It MUST warn on sub-millimetre POSITIVE minima (``< 1 mm``), which
   are the canonical sign of a missed mm-to-m conversion.
3. The warning message must encode under cp1252 — the Windows console
   default encoding — without raising ``UnicodeEncodeError``.
"""

from __future__ import annotations

import codecs
import logging

import numpy as np
import pytest
from src.shared.python.upstream_drift_tools.lab.bio._c3d_markers import (
    build_points_dataframe,
    validate_marker_positions,
)
from src.shared.python.upstream_drift_tools.lab.bio._c3d_models import C3DMetadata
from tests.unit.upstream_drift_tools.lab.bio._synthetic import _synthetic_c3d_dict

WARNING_LOGGER = "src.shared.python.upstream_drift_tools.lab.bio._c3d_markers"


def test_min_negative_one_point_nine_seven_does_not_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``min = -1.97 m`` (typical swing-data origin) must not trigger the warning."""
    coords = np.array(
        [
            [-1.97, 0.0, 0.0],
            [0.5, 0.5, 0.5],
            [1.2, 0.3, 0.7],
        ],
        dtype=float,
    )
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, source_units="m", target_units="m")
    suspect = [
        rec
        for rec in caplog.records
        if "Suspiciously small marker positions" in rec.getMessage()
    ]
    assert suspect == []


def test_min_sub_millimetre_positive_does_warn(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``min = +0.0005 m`` (sub-mm, positive) MUST trigger the unit-error warning."""
    coords = np.array(
        [
            [0.0005, 0.001, 0.002],
            [0.0008, 0.0009, 0.0010],
        ],
        dtype=float,
    )
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, source_units="mm", target_units="m")
    suspect = [
        rec
        for rec in caplog.records
        if "Suspiciously small marker positions" in rec.getMessage()
    ]
    assert len(suspect) == 1


def test_warning_message_encodes_under_cp1252(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """The emitted warning must round-trip through cp1252 (Windows console default)."""
    coords = np.array([[0.0005, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, source_units="mm", target_units="m")
    formatted = "\n".join(rec.getMessage() for rec in caplog.records)
    assert formatted, "expected at least one warning record"
    try:
        codecs.encode(formatted, "cp1252")
    except UnicodeEncodeError as exc:  # pragma: no cover - regression sentinel
        pytest.fail(f"warning message is not cp1252-safe: {exc}")


# ----- additional validate_marker_positions edge cases ----------------------


def test_validate_empty_array_short_circuits() -> None:
    validate_marker_positions(np.array([]).reshape(0, 3), "m", "m")


def test_validate_all_nan_warns_and_returns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    coords = np.full((3, 3), np.nan)
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, "m", None)
    assert any(
        "All marker coordinates are NaN" in r.getMessage() for r in caplog.records
    )


def test_validate_partial_nan_does_not_short_circuit(
    caplog: pytest.LogCaptureFixture,
) -> None:
    # Partial NaN: nanmin/nanmax remain finite; should NOT trigger the
    # all-NaN warning, and should not raise on healthy values.
    coords = np.array([[0.5, np.nan, 0.5], [0.5, 0.5, np.nan]], dtype=float)
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        validate_marker_positions(coords, "m", None)
    msgs = [r.getMessage() for r in caplog.records]
    assert not any("All marker coordinates are NaN" in m for m in msgs)


def test_validate_max_position_raises() -> None:
    coords = np.array([[15.0, 0.0, 0.0]], dtype=float)
    with pytest.raises(ValueError, match="exceed"):
        validate_marker_positions(coords, "m", None)


# ----- build_points_dataframe ------------------------------------------------


def _meta_for(
    c3d: dict, frame_count: int = 5, marker_labels: list[str] | None = None
) -> C3DMetadata:
    return C3DMetadata(
        marker_labels=marker_labels
        or list(c3d["parameters"]["POINT"]["LABELS"]["value"]),
        frame_count=frame_count,
        frame_rate=100.0,
        units="m",
        analog_labels=[],
        analog_units=[],
        analog_rate=None,
        events=[],
    )


def test_build_points_dataframe_basic() -> None:
    c3d = _synthetic_c3d_dict(n_frames=5, n_markers=3, marker_names=["B", "A", "C"])
    md = _meta_for(c3d, 5, ["B", "A", "C"])
    df = build_points_dataframe(c3d, md, "x.c3d", 1.0, True, None, None, None)
    # markers sorted alphabetically
    assert list(df["marker"].iloc[:3]) == ["A", "B", "C"]
    assert "time" in df.columns
    assert df["time"].iloc[0] == 0.0
    assert len(df) == 15


def test_build_points_dataframe_marker_filter() -> None:
    c3d = _synthetic_c3d_dict(n_frames=2, n_markers=3, marker_names=["A", "B", "C"])
    md = _meta_for(c3d, 2, ["A", "B", "C"])
    df = build_points_dataframe(c3d, md, "x.c3d", 1.0, False, ["A", "C"], None, None)
    assert set(df["marker"].unique()) == {"A", "C"}
    assert "time" not in df.columns


def test_build_points_dataframe_residual_threshold() -> None:
    c3d = _synthetic_c3d_dict(n_frames=2, n_markers=2, marker_names=["A", "B"])
    # set residuals: marker A frame 0 -> 5, marker B frame 1 -> 0
    c3d["data"]["points"][3, 0, 0] = 5.0
    md = _meta_for(c3d, 2, ["A", "B"])
    df = build_points_dataframe(c3d, md, "x.c3d", 1.0, False, None, 1.0, None)
    # residuals > 1.0 should drive coords to NaN
    a_frame0 = df[(df["marker"] == "A") & (df["frame"] == 0)].iloc[0]
    assert np.isnan(a_frame0["x"])


def test_build_points_dataframe_target_units_conversion() -> None:
    # Use small native-units coordinates so scale=1000x stays within the
    # biomechanical sanity range (<10 m).
    pts = np.zeros((4, 1, 1), dtype=float)
    pts[0, 0, 0] = 0.002  # 2 mm in native m
    pts[1, 0, 0] = 0.001
    pts[2, 0, 0] = 0.001
    c3d = _synthetic_c3d_dict(
        n_frames=1, n_markers=1, marker_names=["A"], point_data=pts
    )
    md = _meta_for(c3d, 1, ["A"])
    df = build_points_dataframe(c3d, md, "x.c3d", 1000.0, False, None, None, "mm")
    assert df["x"].iloc[0] == pytest.approx(2.0)


def test_build_points_dataframe_zero_frame_rate_warns(
    caplog: pytest.LogCaptureFixture,
) -> None:
    c3d = _synthetic_c3d_dict(n_frames=1, n_markers=1, marker_names=["A"])
    md = C3DMetadata(
        marker_labels=["A"],
        frame_count=1,
        frame_rate=0.0,
        units="m",
        analog_labels=[],
        analog_units=[],
        analog_rate=None,
        events=[],
    )
    with caplog.at_level(logging.WARNING, logger=WARNING_LOGGER):
        df = build_points_dataframe(c3d, md, "x.c3d", 1.0, True, None, None, None)
    assert "time" not in df.columns
    assert any("Frame rate is 0" in r.getMessage() for r in caplog.records)

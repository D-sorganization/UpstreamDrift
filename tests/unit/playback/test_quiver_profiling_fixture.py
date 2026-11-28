"""Unit tests for Slice 0 playback responsibility map and 60 FPS quiver profiling fixture."""

from __future__ import annotations

import math
import time

import numpy as np
import pytest

from src.shared.python.playback.quiver_profiling_fixture import (
    FRAME_BUDGET_MS,
    FRAME_DT_S,
    REQUIRED_STAGE_NAMES,
    FrameBudgetExceededError,
    PlaybackResponsibilityMap,
    QuiverFrameSample,
    QuiverProfileReport,
    QuiverProfilingFixture,
    ResponsibilityStage,
    build_60fps_quiver_fixture,
    default_playback_responsibility_map,
    profile_quiver_frame_budget,
)

pytestmark = pytest.mark.unit


def test_default_responsibility_map_covers_required_acyclic_stages() -> None:
    """Default responsibility map defines an acyclic DAG across all five stages."""
    rmap = default_playback_responsibility_map()
    assert isinstance(rmap, PlaybackResponsibilityMap)
    order = rmap.topological_order()
    assert rmap.has_required_stages()
    for prev, nxt in zip(REQUIRED_STAGE_NAMES, REQUIRED_STAGE_NAMES[1:], strict=False):
        assert order.index(prev) < order.index(nxt)


def test_responsibility_map_rejects_duplicate_owners_and_cycles() -> None:
    """Responsibility map rejects duplicate stage owners, duplicate names, and cycles."""
    with pytest.raises(ValueError, match="owner"):
        PlaybackResponsibilityMap(
            (
                ResponsibilityStage("source", "shared.owner", "Read frames."),
                ResponsibilityStage(
                    "interpolator", "shared.owner", "Interp.", ("source",)
                ),
            )
        )

    with pytest.raises(ValueError, match="[Cc]ircular|[Cc]ycle"):
        PlaybackResponsibilityMap(
            (
                ResponsibilityStage("source", "owner.a", "Stage A.", ("interpolator",)),
                ResponsibilityStage("interpolator", "owner.b", "Stage B.", ("source",)),
            )
        )

    with pytest.raises(ValueError, match="Duplicate stage name"):
        PlaybackResponsibilityMap(
            (
                ResponsibilityStage("source", "owner.a", "Stage A."),
                ResponsibilityStage("source", "owner.b", "Stage B."),
            )
        )

    with pytest.raises(ValueError, match="Unknown dependency"):
        PlaybackResponsibilityMap(
            (ResponsibilityStage("source", "owner.a", "Stage A.", ("missing_stage",)),)
        )


def test_build_60fps_quiver_fixture_is_deterministic_and_finite() -> None:
    """Whole-swing 60 FPS fixture produces monotonic, finite, deterministic frames."""
    fixture_a = build_60fps_quiver_fixture(duration_s=1.0, n_quivers=6, seed=42)
    fixture_b = build_60fps_quiver_fixture(duration_s=1.0, n_quivers=6, seed=42)
    fixture_c = build_60fps_quiver_fixture(duration_s=1.0, n_quivers=6, seed=99)

    assert isinstance(fixture_a, QuiverProfilingFixture)
    assert fixture_a.fps == pytest.approx(60.0)
    assert fixture_a.dt_s == pytest.approx(FRAME_DT_S)
    assert math.isclose(fixture_a.dt_s, 1.0 / 60.0, rel_tol=0.0, abs_tol=1e-12)
    assert fixture_a.fixture_sha256 == fixture_b.fixture_sha256
    assert fixture_a.fixture_sha256 != fixture_c.fixture_sha256
    assert len(fixture_a.frames) == 61

    times = [frame.time_s for frame in fixture_a.frames]
    assert all(t1 < t2 for t1, t2 in zip(times, times[1:], strict=False))
    assert np.allclose(np.diff(times), 1.0 / 60.0, atol=1e-12)

    for idx, frame in enumerate(fixture_a.frames):
        assert isinstance(frame, QuiverFrameSample)
        assert frame.frame_index == idx
        assert frame.origins_m.shape == (6, 3) and frame.vectors.shape == (6, 3)
        assert np.all(np.isfinite(frame.origins_m)) and np.all(
            np.isfinite(frame.vectors)
        )


def test_profile_quiver_frame_budget_passes_default_and_fails_closed_when_slow() -> (
    None
):
    """Profiler passes fast synthetic builder and fails closed when budget is exceeded."""
    fixture = build_60fps_quiver_fixture(duration_s=0.1, n_quivers=6, seed=7)
    assert math.isclose(FRAME_BUDGET_MS, 1000.0 / 60.0, rel_tol=1e-9)

    report = profile_quiver_frame_budget(fixture)
    assert isinstance(report, QuiverProfileReport)
    assert report.frame_count == len(fixture.frames)
    assert report.within_budget is True
    assert report.over_budget_frames == 0
    assert report.max_ms <= report.budget_ms

    def slow_builder(sample: QuiverFrameSample) -> np.ndarray:
        """Simulate an artificially slow quiver builder exceeding the 16.67 ms budget."""
        time.sleep(0.022)
        return sample.vectors * 1.0

    with pytest.raises(FrameBudgetExceededError, match="budget"):
        profile_quiver_frame_budget(fixture, slow_builder)

    failed_report = profile_quiver_frame_budget(
        fixture, slow_builder, fail_on_exceeded=False
    )
    assert failed_report.within_budget is False
    assert failed_report.over_budget_frames > 0
    assert failed_report.max_ms > FRAME_BUDGET_MS

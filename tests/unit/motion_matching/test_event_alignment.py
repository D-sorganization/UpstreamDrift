"""Event-aware impact alignment for the C3D loaders.

Issue #4709: when a C3D file carries an ``EVENT`` group with labels such as
``"Impact"``, the matcher's loaders should be able to use that frame for
impact alignment instead of the kinematic-peak heuristic. These tests cover
both :func:`load_club_target_c3d` and :func:`load_body_target_c3d` plus the
fallback behaviour and the unknown-label error path.

Synthetic C3D data is built with :func:`_synthetic_c3d_dict` and patched into
the canonical I/O entry point so the loaders see real-shaped ezc3d
dictionaries without any C3D file having to live in the repo.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import (
    AlignOptions,
    load_body_target_c3d,
    load_club_target_c3d,
)
from src.shared.python.motion_matching.loaders.c3d_body import (
    default_anatomical_marker_set,
)
from tests.unit.upstream_drift_tools.lab.bio._synthetic import _synthetic_c3d_dict

# Frame counts and rates picked so a uniform timeline cleanly separates an
# "Impact" event (frame 50, t=0.5 s) from the kinematic-peak frame (80) of a
# sinusoid clubhead trajectory.
_N_FRAMES = 100
_FRAME_RATE = 100.0  # Hz
_EVENT_FRAME = 50
_EVENT_TIME = _EVENT_FRAME / _FRAME_RATE  # 0.5 s
_PEAK_FRAME = 80
_PEAK_TIME = _PEAK_FRAME / _FRAME_RATE  # 0.8 s


def _opts() -> AlignOptions:
    return AlignOptions(
        sample_rate_hz=1000.0,
        simulation_time_s=0.6,
        time_alignment="impact",
        impact_target_t_s=0.25,
    )


# --------------------------------------------------------------------------
# Synthetic-C3D builders
# --------------------------------------------------------------------------


def _club_point_data() -> np.ndarray:
    """Two markers (BUTT, CH); CH x = sin(pi*(i-_PEAK_FRAME)/40) so peak at 80.

    The butt marker stays put. ``sin`` peaks at frame 80 because its phase
    derivative is largest there, and the position at the *event* frame
    differs sharply from the position at the *peak* frame, which lets the
    test distinguish event-aligned vs heuristic-aligned outputs by inspecting
    ``clubhead[impact_idx]``.
    """
    n_markers = 2
    points = np.zeros((4, n_markers, _N_FRAMES), dtype=float)
    # BUTT at (0.5, 0.4, 0.3) for all frames.
    points[0, 0, :] = 0.5
    points[1, 0, :] = 0.4
    points[2, 0, :] = 0.3
    # CH at (sin(pi*(i-80)/40), 0.4, 1.4) — speed peaks at i=80, distinct
    # from the EVENT at frame 50.
    i = np.arange(_N_FRAMES)
    points[0, 1, :] = np.sin(np.pi * (i - _PEAK_FRAME) / 40.0)
    points[1, 1, :] = 0.4
    points[2, 1, :] = 1.4
    return points


def _build_club_dict(
    *,
    with_events: bool,
    event_labels: list[str] | None = None,
    event_times: list[float] | None = None,
    event_times_2d: bool = False,
) -> dict[str, Any]:
    return _synthetic_c3d_dict(
        n_frames=_N_FRAMES,
        n_markers=2,
        marker_names=["BUTT", "CH"],
        frame_rate=_FRAME_RATE,
        units="m",
        point_data=_club_point_data(),
        with_events=with_events,
        event_labels=event_labels,
        event_times=event_times,
        event_times_2d=event_times_2d,
    )


def _body_marker_names() -> list[str]:
    """Default anatomical set plus the excluded ``RShoulderTop`` so the
    loader's default marker set is fully resolvable on synthetic data."""
    base = list(default_anatomical_marker_set())
    if "RShoulderTop" not in base:
        base.append("RShoulderTop")
    return base


def _body_point_data(marker_names: list[str]) -> np.ndarray:
    """Simulate a wrist-speed peak at frame 80 distinct from event at 50."""
    n_markers = len(marker_names)
    points = np.zeros((4, n_markers, _N_FRAMES), dtype=float)
    i = np.arange(_N_FRAMES)
    for m, name in enumerate(marker_names):
        # Static skeleton, varied per marker, biomechanical range.
        points[0, m, :] = 0.5 + 0.01 * m
        points[1, m, :] = 0.4 + 0.01 * m
        points[2, m, :] = 0.3 + 0.01 * m
        if name in ("RWristTop", "LWristTop"):
            # Make wrist marker speed peak at frame 80.
            points[0, m, :] = 0.5 + np.sin(np.pi * (i - _PEAK_FRAME) / 40.0)
    return points


def _build_body_dict(
    *,
    with_events: bool,
    event_labels: list[str] | None = None,
    event_times: list[float] | None = None,
) -> dict[str, Any]:
    names = _body_marker_names()
    return _synthetic_c3d_dict(
        n_frames=_N_FRAMES,
        n_markers=len(names),
        marker_names=names,
        frame_rate=_FRAME_RATE,
        units="m",
        point_data=_body_point_data(names),
        with_events=with_events,
        event_labels=event_labels,
        event_times=event_times,
    )


@pytest.fixture
def fake_c3d_path(tmp_path: Path) -> Path:
    """Return a real, on-disk path satisfying ``Path.exists()`` preconditions.

    ezc3d itself is bypassed via the ``patch_load_c3d`` fixture below.
    """
    p = tmp_path / "synthetic.c3d"
    p.write_bytes(b"")  # contents irrelevant; load_c3d is monkeypatched
    return p


def _patch_load_c3d(monkeypatch: pytest.MonkeyPatch, payload: dict[str, Any]) -> None:
    """Replace the canonical ``load_c3d`` so the loader sees ``payload``."""
    from sidekick.lab.bio import _c3d_io, c3d_reader

    monkeypatch.setattr(_c3d_io, "load_c3d", lambda _p: payload)
    monkeypatch.setattr(c3d_reader, "load_c3d", lambda _p: payload)


# --------------------------------------------------------------------------
# Club-target tests
# --------------------------------------------------------------------------


def test_club_event_alignment_uses_event_frame(
    monkeypatch: pytest.MonkeyPatch,
    fake_c3d_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``event_label_for_alignment="Impact"`` pins impact to the event frame."""
    payload = _build_club_dict(
        with_events=True,
        event_labels=["Impact", "Top"],
        event_times=[_EVENT_TIME, 0.7],
    )
    _patch_load_c3d(monkeypatch, payload)
    opts = _opts()
    with caplog.at_level(logging.INFO, logger="src.shared.python.motion_matching.loaders.c3d"):
        target = load_club_target_c3d(fake_c3d_path, opts, event_label_for_alignment="Impact")
    # impact_target_t_s=0.25 corresponds to raw t=0.5 (the event time).
    # CH x at frame 50 is sin(pi*(50-80)/40) = sin(-3pi/4) ~= -0.7071.
    sim_idx = int(target.impact_idx) - 1  # impact_idx is 1-based per contract
    expected = float(np.sin(np.pi * (_EVENT_FRAME - _PEAK_FRAME) / 40.0))
    assert target.clubhead[sim_idx, 0] == pytest.approx(expected, abs=1e-3)
    # Log must mention event-driven alignment.
    assert any("EVENT" in rec.message and "Impact" in rec.message for rec in caplog.records)


def test_club_no_event_falls_back_to_heuristic(
    monkeypatch: pytest.MonkeyPatch,
    fake_c3d_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    """No events present + ``event_label_for_alignment=None`` -> kinematic peak."""
    payload = _build_club_dict(with_events=False)
    _patch_load_c3d(monkeypatch, payload)
    opts = _opts()
    with caplog.at_level(logging.INFO, logger="src.shared.python.motion_matching.loaders.c3d"):
        target = load_club_target_c3d(fake_c3d_path, opts)
    # Heuristic locks raw frame 80 onto sim t=0.25; CH x at frame 80 is sin(0)=0.
    sim_idx = int(target.impact_idx) - 1
    assert target.clubhead[sim_idx, 0] == pytest.approx(0.0, abs=1e-3)
    # Fallback INFO log must be emitted.
    assert any("falling back" in rec.message.lower() for rec in caplog.records)


def test_club_unknown_event_label_lists_available(
    monkeypatch: pytest.MonkeyPatch, fake_c3d_path: Path
) -> None:
    """Requesting a missing event label raises ValueError listing options."""
    payload = _build_club_dict(
        with_events=True,
        event_labels=["Impact", "Top"],
        event_times=[_EVENT_TIME, 0.7],
    )
    _patch_load_c3d(monkeypatch, payload)
    with pytest.raises(ValueError, match=r"DoesNotExist.*Impact.*Top"):
        load_club_target_c3d(fake_c3d_path, _opts(), event_label_for_alignment="DoesNotExist")


def test_club_event_label_but_file_has_no_events_raises(
    monkeypatch: pytest.MonkeyPatch, fake_c3d_path: Path
) -> None:
    """Requesting a label on an event-less file is a hard error."""
    payload = _build_club_dict(with_events=False)
    _patch_load_c3d(monkeypatch, payload)
    with pytest.raises(ValueError, match="no EVENT annotations"):
        load_club_target_c3d(fake_c3d_path, _opts(), event_label_for_alignment="Impact")


# --------------------------------------------------------------------------
# Body-target tests (mirror of the club-target tests)
# --------------------------------------------------------------------------


def test_body_event_alignment_uses_event_frame(
    monkeypatch: pytest.MonkeyPatch,
    fake_c3d_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _build_body_dict(
        with_events=True,
        event_labels=["Impact", "Top"],
        event_times=[_EVENT_TIME, 0.7],
    )
    _patch_load_c3d(monkeypatch, payload)
    opts = _opts()
    with caplog.at_level(
        logging.INFO,
        logger="src.shared.python.motion_matching.loaders.c3d_body",
    ):
        bt = load_body_target_c3d(fake_c3d_path, opts, event_label_for_alignment="Impact")
    # impact_idx is 0-based on the resampled grid for BodyTarget.
    sim_t_at_impact = float(bt.time[int(bt.impact_idx)])
    assert sim_t_at_impact == pytest.approx(opts.impact_target_t_s, abs=1.5e-3)
    # The wrist marker x at sim impact_target_t_s should equal raw value at
    # event frame 50: 0.5 + sin(-3pi/4) ~= 0.5 - 0.7071.
    wrist_idx = bt.marker_names.index("RWristTop")
    expected = 0.5 + float(np.sin(np.pi * (_EVENT_FRAME - _PEAK_FRAME) / 40.0))
    assert bt.marker_xyz[int(bt.impact_idx), wrist_idx, 0] == pytest.approx(expected, abs=5e-3)
    assert any("EVENT" in rec.message and "Impact" in rec.message for rec in caplog.records)


def test_body_no_event_falls_back_to_wrist_heuristic(
    monkeypatch: pytest.MonkeyPatch,
    fake_c3d_path: Path,
    caplog: pytest.LogCaptureFixture,
) -> None:
    payload = _build_body_dict(with_events=False)
    _patch_load_c3d(monkeypatch, payload)
    opts = _opts()
    with caplog.at_level(
        logging.INFO,
        logger="src.shared.python.motion_matching.loaders.c3d_body",
    ):
        bt = load_body_target_c3d(fake_c3d_path, opts)
    # Heuristic pins wrist peak (frame 80) to sim t=0.25; wrist x there is 0.5.
    wrist_idx = bt.marker_names.index("RWristTop")
    assert bt.marker_xyz[int(bt.impact_idx), wrist_idx, 0] == pytest.approx(0.5, abs=5e-3)
    assert any("falling back" in rec.message.lower() for rec in caplog.records)


def test_body_unknown_event_label_lists_available(
    monkeypatch: pytest.MonkeyPatch, fake_c3d_path: Path
) -> None:
    payload = _build_body_dict(
        with_events=True,
        event_labels=["Impact", "Top"],
        event_times=[_EVENT_TIME, 0.7],
    )
    _patch_load_c3d(monkeypatch, payload)
    with pytest.raises(ValueError, match=r"DoesNotExist.*Impact.*Top"):
        load_body_target_c3d(fake_c3d_path, _opts(), event_label_for_alignment="DoesNotExist")


def test_body_event_label_but_file_has_no_events_raises(
    monkeypatch: pytest.MonkeyPatch, fake_c3d_path: Path
) -> None:
    payload = _build_body_dict(with_events=False)
    _patch_load_c3d(monkeypatch, payload)
    with pytest.raises(ValueError, match="no EVENT annotations"):
        load_body_target_c3d(fake_c3d_path, _opts(), event_label_for_alignment="Impact")

"""Debounce, memoization and axes-reuse tests for the analysis tabs (#8932)."""

from __future__ import annotations

from typing import Any

import numpy as np
import pytest
from src.shared.python.dashboard import advanced_analysis
from src.shared.python.dashboard._analysis_refresh import (
    BoundedResultCache,
    DebouncedRefresh,
    analysis_cache_key,
)
from src.shared.python.dashboard.advanced_analysis import (
    CoherenceTab,
    CorrelationTab,
    PhasePlaneTab,
    SpectrogramTab,
    SwingPlaneTab,
    WaveletTab,
)

pytestmark = pytest.mark.unit


class MutableRecorder:
    """Recorder whose data can be replaced between refreshes."""

    engine: Any = None

    def __init__(self) -> None:
        self.t = np.linspace(0, 1, 200)
        self.signal = np.column_stack(
            [np.sin(2 * np.pi * 10 * self.t), np.cos(2 * np.pi * 5 * self.t)]
        )
        self.position = np.column_stack(
            [np.cos(2 * np.pi * self.t), np.sin(2 * np.pi * self.t), self.t]
        )

    def get_time_series(self, key: str) -> tuple:
        """Return the 3D club path or the 2-column test signal."""
        if key == "club_head_position":
            return self.t, self.position
        return self.t, self.signal

    def get_induced_acceleration_series(
        self, source_name: str | int
    ) -> tuple[np.ndarray, np.ndarray]:
        """Stub for induced acceleration time series."""
        return self.t, np.zeros_like(self.signal)

    def set_analysis_config(self, config: dict[str, Any]) -> None:
        """Stub for analysis config."""


class CallCounter:
    """Wrap a function and count its invocations."""

    def __init__(self, fn) -> None:
        self.fn = fn
        self.calls = 0

    def __call__(self, *args, **kwargs):
        self.calls += 1
        return self.fn(*args, **kwargs)


# --- DebouncedRefresh ------------------------------------------------------


def test_debounce_coalesces_burst_into_one_call(qtbot) -> None:
    calls: list[int] = []
    refresh = DebouncedRefresh(lambda: calls.append(1), interval_ms=50)
    for value in range(10):
        refresh.trigger(value)
    assert calls == []
    assert refresh.pending
    qtbot.waitUntil(lambda: len(calls) == 1, timeout=1000)
    qtbot.wait(150)
    assert calls == [1]
    assert not refresh.pending


def test_debounce_default_interval_is_150ms(qtbot) -> None:
    assert DebouncedRefresh(lambda: None).interval_ms == 150


@pytest.mark.parametrize(
    ("callback", "interval", "exc"),
    [(None, 150, TypeError), (lambda: None, -1, ValueError)],
)
def test_debounce_rejects_invalid_arguments(callback, interval, exc) -> None:
    with pytest.raises(exc):
        DebouncedRefresh(callback, interval_ms=interval)


# --- BoundedResultCache / analysis_cache_key -------------------------------


def test_cache_hits_for_identical_key_and_misses_otherwise() -> None:
    cache = BoundedResultCache(maxsize=4)
    compute = CallCounter(lambda: object())
    first = cache.get_or_compute("a", compute)
    assert cache.get_or_compute("a", compute) is first
    assert compute.calls == 1
    cache.get_or_compute("b", compute)
    assert compute.calls == 2


def test_cache_is_bounded_and_evicts_least_recently_used() -> None:
    cache = BoundedResultCache(maxsize=2)
    cache.get_or_compute("a", lambda: 1)
    cache.get_or_compute("b", lambda: 2)
    cache.get_or_compute("a", lambda: 1)  # refresh "a"
    cache.get_or_compute("c", lambda: 3)
    assert len(cache) == 2
    assert "a" in cache
    assert "b" not in cache


def test_cache_rejects_non_positive_size() -> None:
    with pytest.raises(ValueError, match="maxsize"):
        BoundedResultCache(maxsize=0)


def test_cache_key_tracks_every_input() -> None:
    sig = np.arange(10, dtype=float)
    base = analysis_cache_key("k", 0, 100.0, sig, w0=6.0)
    assert analysis_cache_key("k", 0, 100.0, sig.copy(), w0=6.0) == base
    changed = sig.copy()
    changed[3] = -1.0
    variants = [
        analysis_cache_key("other", 0, 100.0, sig, w0=6.0),
        analysis_cache_key("k", 1, 100.0, sig, w0=6.0),
        analysis_cache_key("k", 0, 50.0, sig, w0=6.0),
        analysis_cache_key("k", 0, 100.0, sig, w0=7.0),
        analysis_cache_key("k", 0, 100.0, sig[:-1], w0=6.0),
        analysis_cache_key("k", 0, 100.0, changed, w0=6.0),
    ]
    assert all(v != base for v in variants)


@pytest.mark.parametrize(
    ("dim", "fs", "signal"),
    [
        (-1, 100.0, np.ones(4)),
        (0, 0.0, np.ones(4)),
        (0, float("nan"), np.ones(4)),
        (0, 100.0, np.ones((4, 2))),
        (0, 100.0, np.array([])),
    ],
)
def test_cache_key_rejects_invalid_inputs(dim, fs, signal) -> None:
    with pytest.raises(ValueError):
        analysis_cache_key("k", dim, fs, signal)


# --- Tabs ------------------------------------------------------------------


@pytest.fixture
def cwt_spy(monkeypatch) -> CallCounter:
    spy = CallCounter(advanced_analysis.compute_cwt)
    monkeypatch.setattr(advanced_analysis, "compute_cwt", spy)
    return spy


@pytest.fixture
def spectrogram_spy(monkeypatch) -> CallCounter:
    spy = CallCounter(advanced_analysis.compute_spectrogram)
    monkeypatch.setattr(advanced_analysis, "compute_spectrogram", spy)
    return spy


@pytest.fixture
def coherence_spy(monkeypatch) -> CallCounter:
    spy = CallCounter(advanced_analysis.compute_coherence)
    monkeypatch.setattr(advanced_analysis, "compute_coherence", spy)
    return spy


def test_wavelet_spinbox_burst_recomputes_once(qtbot, cwt_spy) -> None:
    tab = WaveletTab(MutableRecorder())
    qtbot.addWidget(tab)
    assert cwt_spy.calls == 1  # initial synchronous plot

    for w0 in (6.5, 7.0, 7.5, 8.0, 8.5):
        tab.spin_w0.setValue(w0)
    assert cwt_spy.calls == 1  # nothing recomputed mid-burst

    qtbot.waitUntil(lambda: cwt_spy.calls == 2, timeout=2000)
    qtbot.wait(250)
    assert cwt_spy.calls == 2


def test_spectrogram_dim_burst_recomputes_once(qtbot, spectrogram_spy) -> None:
    tab = SpectrogramTab(MutableRecorder())
    qtbot.addWidget(tab)
    assert spectrogram_spy.calls == 1

    tab.spin_dim.setValue(1)
    tab.spin_dim.setValue(0)
    tab.spin_dim.setValue(1)
    assert spectrogram_spy.calls == 1

    qtbot.waitUntil(lambda: spectrogram_spy.calls == 2, timeout=2000)
    qtbot.wait(250)
    assert spectrogram_spy.calls == 2


def test_wavelet_memoizes_and_invalidates_on_data_change(qtbot, cwt_spy) -> None:
    recorder = MutableRecorder()
    tab = WaveletTab(recorder)
    qtbot.addWidget(tab)
    assert cwt_spy.calls == 1

    tab.update_plot()  # identical params and data -> cache hit
    assert cwt_spy.calls == 1
    assert len(tab.ax.collections) > 0

    recorder.signal = recorder.signal * 2.0  # same shape, new data -> miss
    tab.update_plot()
    assert cwt_spy.calls == 2


def test_spectrogram_memoizes_and_invalidates_on_data_change(
    qtbot, spectrogram_spy
) -> None:
    recorder = MutableRecorder()
    tab = SpectrogramTab(recorder)
    qtbot.addWidget(tab)
    tab.update_plot()
    assert spectrogram_spy.calls == 1

    recorder.signal = recorder.signal[:150]  # fewer samples -> miss
    recorder.t = recorder.t[:150]
    tab.update_plot()
    assert spectrogram_spy.calls == 2


def test_swing_plane_reuses_axes_across_refreshes(qtbot) -> None:
    recorder = MutableRecorder()
    tab = SwingPlaneTab(recorder)
    qtbot.addWidget(tab)
    axes_before = list(tab.canvas.fig.axes)
    assert len(axes_before) == 2

    recorder.position = recorder.position * 1.5
    tab.update_plot()
    tab.update_plot()

    assert tab.canvas.fig.axes == axes_before
    assert all(a is b for a, b in zip(tab.canvas.fig.axes, axes_before, strict=True))
    ax3d = axes_before[0]
    # exactly one trajectory scatter + one fitted plane, no accumulation
    assert len(ax3d.collections) == 2


def test_swing_plane_no_data_then_data_keeps_axes(qtbot) -> None:
    recorder = MutableRecorder()
    good_position = recorder.position
    recorder.position = np.zeros((0, 3))
    tab = SwingPlaneTab(recorder)
    qtbot.addWidget(tab)
    axes_before = list(tab.canvas.fig.axes)

    recorder.position = good_position
    tab.update_plot()
    assert tab.canvas.fig.axes == axes_before
    assert len(axes_before[0].collections) == 2


def test_phase_plane_dim_burst_recomputes_once(qtbot, monkeypatch) -> None:
    calls: list[int] = []
    monkeypatch.setattr(PhasePlaneTab, "update_plot", lambda self: calls.append(1))
    recorder = MutableRecorder()
    tab = PhasePlaneTab(recorder)
    qtbot.addWidget(tab)
    assert len(calls) == 1  # initial synchronous plot

    for dim in (1, 0, 1):
        tab.spin_dim.setValue(dim)
    assert len(calls) == 1  # debounced, not called immediately

    qtbot.waitUntil(lambda: len(calls) == 2, timeout=2000)
    qtbot.wait(250)
    assert len(calls) == 2


def test_coherence_dim_burst_recomputes_once(qtbot, coherence_spy) -> None:
    recorder = MutableRecorder()
    tab = CoherenceTab(recorder)
    qtbot.addWidget(tab)
    assert coherence_spy.calls == 1

    for dim in (1, 0, 1):
        tab.spin_dim.setValue(dim)
    assert coherence_spy.calls == 1  # not recomputed mid-burst

    qtbot.waitUntil(lambda: coherence_spy.calls == 2, timeout=2000)
    qtbot.wait(250)
    assert coherence_spy.calls == 2


def test_coherence_memoizes_and_invalidates_on_data_change(
    qtbot, coherence_spy
) -> None:
    recorder = MutableRecorder()
    tab = CoherenceTab(recorder)
    qtbot.addWidget(tab)
    assert coherence_spy.calls == 1

    tab.update_plot()  # identical inputs -> cache hit
    assert coherence_spy.calls == 1

    recorder.signal = recorder.signal * 2.0  # modified data -> miss
    tab.update_plot()
    assert coherence_spy.calls == 2


@pytest.mark.parametrize("tab_cls", [PhasePlaneTab, CoherenceTab, CorrelationTab])
def test_analysis_tabs_use_draw_idle_not_blocking_draw(
    qtbot, monkeypatch, tab_cls
) -> None:
    recorder = MutableRecorder()
    tab = tab_cls(recorder)
    qtbot.addWidget(tab)

    draw_spy = CallCounter(tab.canvas.draw)
    idle_spy = CallCounter(tab.canvas.draw_idle)
    monkeypatch.setattr(tab.canvas, "draw", draw_spy)
    monkeypatch.setattr(tab.canvas, "draw_idle", idle_spy)

    tab.update_plot()
    assert draw_spy.calls == 0, f"{tab_cls.__name__} called blocking canvas.draw()"
    assert idle_spy.calls >= 1, f"{tab_cls.__name__} did not call canvas.draw_idle()"

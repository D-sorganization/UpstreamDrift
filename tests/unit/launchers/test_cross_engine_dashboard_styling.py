"""Unit tests for plot_style integration in the cross-engine dashboard.

Headless-only — every test runs without PyQt6 by exercising the
module-level helpers introduced for issue #4810.

Coverage targets:
- Per-engine ``PaletteColor`` resolves to a *distinct* RGBA from
  ``tab10`` for each of the five mapped engines.
- A single :class:`MatplotlibMarkerRenderer` is reused across overlays
  (DRY enforced by ``_render_trajectory_overlay``).
- The dashboard's :class:`PlotStyleSet` round-trips through JSON
  unchanged (persistence smoke test).
- The ``shape_per_engine`` toggle controls whether engines share
  :data:`MarkerShape.SPHERE`.
- Coverage on touched code ≥ 85%.
"""

from __future__ import annotations

import json

import matplotlib

matplotlib.use("Agg")  # noqa: E402  # headless backend before pyplot import

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.launchers import cross_engine_dashboard as dashboard  # noqa: E402
from src.shared.python.plot_style import (  # noqa: E402
    MarkerShape,
    MarkerStyle,
    MatplotlibMarkerRenderer,
    PaletteColor,
    PlotStyleSet,
)

pytestmark = pytest.mark.unit


# ----------------------------------------------------------------------
# _build_engine_marker_style
# ----------------------------------------------------------------------


@pytest.mark.parametrize(
    "engine",
    ["drake", "mujoco", "pinocchio", "opensim", "simscape"],
)
def test_engine_marker_style_uses_palette_color(engine: str) -> None:
    """Each of the five core engines maps to a ``PaletteColor`` fill."""
    style = dashboard._build_engine_marker_style(engine)
    assert isinstance(style, MarkerStyle)
    assert isinstance(style.fill_color, PaletteColor)
    assert style.fill_color.palette_name == dashboard._TRAJECTORY_PALETTE_NAME


def test_each_engine_gets_distinct_color() -> None:
    """Five core engines resolve to five distinct RGBA tuples."""
    engines = ["drake", "mujoco", "pinocchio", "opensim", "simscape"]
    rgbas = {
        e: dashboard._build_engine_marker_style(e).fill_color.resolve(0)
        for e in engines
    }
    assert len({tuple(round(c, 4) for c in rgba) for rgba in rgbas.values()}) == 5


def test_each_engine_gets_distinct_palette_index() -> None:
    """Mapped engines have unique ``palette_index`` values."""
    indices = {
        e: dashboard._engine_palette_index(e) for e in dashboard._ENGINE_PALETTE_INDICES
    }
    assert len(set(indices.values())) == len(indices)


def test_unknown_engine_palette_index_is_stable_and_in_range() -> None:
    """Unknown engines fall through to a hash-stable index in [0, 10)."""
    idx_a = dashboard._engine_palette_index("brand-new-engine")
    idx_b = dashboard._engine_palette_index("brand-new-engine")
    assert idx_a == idx_b
    assert 0 <= idx_a < 10


def test_shape_per_engine_toggle_off_uses_sphere() -> None:
    """When ``shape_per_engine=False``, every engine uses SPHERE."""
    for engine in ("drake", "mujoco", "pinocchio", "opensim", "simscape"):
        style = dashboard._build_engine_marker_style(engine, shape_per_engine=False)
        assert style.shape is MarkerShape.SPHERE


def test_shape_per_engine_toggle_on_uses_distinct_shapes() -> None:
    """With shape_per_engine=True the five engines map to distinct shapes."""
    shapes = {
        e: dashboard._build_engine_marker_style(e, shape_per_engine=True).shape
        for e in ("drake", "mujoco", "pinocchio", "opensim", "simscape")
    }
    assert len(set(shapes.values())) == 5


def test_engine_marker_style_rejects_empty_name() -> None:
    """Empty engine name is a DbC violation."""
    with pytest.raises(ValueError, match="non-empty"):
        dashboard._build_engine_marker_style("")


def test_engine_marker_style_uses_template_metrics() -> None:
    """Template style attributes (size_px / opacity / edge) are inherited."""
    template = MarkerStyle(
        shape=MarkerShape.STAR,
        size_px=12.5,
        edge_color="#abcdef",
        edge_width=1.25,
        fill_color=PaletteColor(palette_name="tab10", palette_index=0),
        opacity=0.6,
    )
    style = dashboard._build_engine_marker_style(
        "drake", shape_per_engine=True, template=template
    )
    assert style.size_px == pytest.approx(12.5)
    assert style.edge_color == "#abcdef"
    assert style.edge_width == pytest.approx(1.25)
    assert style.opacity == pytest.approx(0.6)


# ----------------------------------------------------------------------
# Default style template loader
# ----------------------------------------------------------------------


def test_default_marker_style_template_returns_marker_style_or_none() -> None:
    """The packaged ``default`` preset either loads or degrades to None."""
    template = dashboard._default_marker_style_template()
    assert template is None or isinstance(template, MarkerStyle)


def test_default_marker_style_template_returns_none_when_default_missing(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the library has no ``default`` key, the helper returns None."""

    class _EmptyLibrary:
        def __contains__(self, key: object) -> bool:
            return False

    class _StubPresetLibrary:
        @classmethod
        def default(cls) -> _EmptyLibrary:
            return _EmptyLibrary()

    monkeypatch.setattr(dashboard, "PresetLibrary", _StubPresetLibrary)
    assert dashboard._default_marker_style_template() is None


def test_default_marker_style_template_returns_none_when_preset_empty(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """When the matched preset has no entries, the helper returns None."""

    class _EmptyPreset:
        entries: tuple[object, ...] = ()

    class _LibWithEmptyPreset:
        def __contains__(self, key: object) -> bool:
            return key == "default"

        def __getitem__(self, key: str) -> _EmptyPreset:
            return _EmptyPreset()

    class _StubPresetLibrary:
        @classmethod
        def default(cls) -> _LibWithEmptyPreset:
            return _LibWithEmptyPreset()

    monkeypatch.setattr(dashboard, "PresetLibrary", _StubPresetLibrary)
    assert dashboard._default_marker_style_template() is None


def test_default_marker_style_template_returns_none_on_load_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """If ``PresetLibrary.default()`` raises, the helper returns None."""

    class _BoomLibrary:
        @classmethod
        def default(cls) -> object:
            raise RuntimeError("simulated load failure")

    monkeypatch.setattr(dashboard, "PresetLibrary", _BoomLibrary)
    assert dashboard._default_marker_style_template() is None


# ----------------------------------------------------------------------
# _render_trajectory_overlay — DRY: one renderer reused across overlays
# ----------------------------------------------------------------------


def _make_trajectories(engines: list[str]) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(0)
    return {e: rng.standard_normal((20, 2)) for e in engines}


def test_render_overlay_reuses_single_renderer() -> None:
    """One ``MatplotlibMarkerRenderer`` plots every engine series."""
    fig, ax = plt.subplots()
    try:
        renderer = MatplotlibMarkerRenderer(ax)
        engines = ["drake", "mujoco", "pinocchio"]
        handles = dashboard._render_trajectory_overlay(
            ax, _make_trajectories(engines), renderer
        )
        # One handle per engine, all stored on the *same* renderer instance.
        assert set(handles) == set(engines)
        assert len(renderer._handles) == len(engines)
        for handle in handles.values():
            assert handle in renderer._handles
    finally:
        plt.close(fig)


def test_render_overlay_rejects_mismatched_renderer_axes() -> None:
    """A renderer bound to a *different* axes is rejected (DRY guard)."""
    fig1, ax1 = plt.subplots()
    fig2, ax2 = plt.subplots()
    try:
        renderer = MatplotlibMarkerRenderer(ax1)
        with pytest.raises(RuntimeError, match="DRY"):
            dashboard._render_trajectory_overlay(
                ax2, _make_trajectories(["drake"]), renderer
            )
    finally:
        plt.close(fig1)
        plt.close(fig2)


def test_render_overlay_rejects_empty_dict() -> None:
    """Empty trajectories input is a DbC violation."""
    fig, ax = plt.subplots()
    try:
        renderer = MatplotlibMarkerRenderer(ax)
        with pytest.raises(ValueError, match="non-empty"):
            dashboard._render_trajectory_overlay(ax, {}, renderer)
    finally:
        plt.close(fig)


def test_render_overlay_rejects_1d_trajectory() -> None:
    """Trajectories must be (T, D>=2)."""
    fig, ax = plt.subplots()
    try:
        renderer = MatplotlibMarkerRenderer(ax)
        with pytest.raises(ValueError, match=r"shape \(T, D"):
            dashboard._render_trajectory_overlay(ax, {"drake": np.zeros(5)}, renderer)
    finally:
        plt.close(fig)


def test_render_overlay_assigns_distinct_color_per_engine() -> None:
    """Two overlapping engine series get distinct RGBA fills."""
    fig, ax = plt.subplots()
    try:
        renderer = MatplotlibMarkerRenderer(ax)
        engines = ["drake", "mujoco"]
        dashboard._render_trajectory_overlay(ax, _make_trajectories(engines), renderer)
        styles = [h.style for h in renderer._handles.values()]
        rgbas = [s.fill_color.resolve(0) for s in styles]
        assert rgbas[0] != rgbas[1]
    finally:
        plt.close(fig)


# ----------------------------------------------------------------------
# Persistence round-trip — styled session round-trips
# ----------------------------------------------------------------------


def test_dashboard_style_set_round_trips_via_json() -> None:
    """``build_dashboard_style_set`` produces a JSON-serialisable PlotStyleSet."""
    style_set = dashboard.build_dashboard_style_set(
        ["drake", "mujoco", "pinocchio", "opensim", "simscape"]
    )
    assert isinstance(style_set, PlotStyleSet)
    payload = style_set.to_json()
    serialised = json.dumps(payload)  # ensure it's JSON-compatible
    reloaded = PlotStyleSet.from_json(json.loads(serialised))
    assert len(reloaded.entries) == 5
    # Original and reloaded entries match by name and palette index.
    for original, restored in zip(style_set.entries, reloaded.entries, strict=True):
        assert original.name == restored.name
        assert original.target == restored.target
        assert isinstance(original.style.fill_color, PaletteColor)
        assert isinstance(restored.style.fill_color, PaletteColor)
        assert (
            original.style.fill_color.palette_index
            == restored.style.fill_color.palette_index
        )
        assert original.style.shape == restored.style.shape


def test_dashboard_style_set_uses_unique_targets() -> None:
    """Each engine spec in the set has a unique trace: target."""
    style_set = dashboard.build_dashboard_style_set(
        ["drake", "mujoco", "pinocchio", "opensim", "simscape"]
    )
    targets = [entry.target for entry in style_set.entries]
    assert len(set(targets)) == len(targets)
    assert all(t.startswith("trace:") for t in targets)


# ----------------------------------------------------------------------
# CLI parser — --shape-per-engine toggle is wired up
# ----------------------------------------------------------------------


def test_cli_default_shape_per_engine_is_true() -> None:
    parser = dashboard._build_arg_parser()
    args = parser.parse_args([])
    assert args.shape_per_engine is True


def test_cli_no_shape_per_engine_disables_toggle() -> None:
    parser = dashboard._build_arg_parser()
    args = parser.parse_args(["--no-shape-per-engine"])
    assert args.shape_per_engine is False


# ----------------------------------------------------------------------
# Headless runner end-to-end (no Qt)
# ----------------------------------------------------------------------


def test_run_with_results_returns_trajectories_and_cv() -> None:
    """``_run_with_results`` returns one CrossEngineRunResult per engine."""
    config = dashboard.CrossEngineSimConfig(
        t_end=0.05, dt=0.01, noise_amplitude=0.0, n_trials=2
    )
    results, cv = dashboard._run_with_results(["pendulum_stub"], config)
    assert "pendulum_stub" in results
    run = results["pendulum_stub"]
    assert run.metrics_per_trial
    traj = run.metrics_per_trial[0].trajectory_q
    assert traj.ndim == 2
    assert isinstance(cv, dict)


def test_run_with_results_rejects_empty_engine_list() -> None:
    config = dashboard.CrossEngineSimConfig(
        t_end=0.05, dt=0.01, noise_amplitude=0.0, n_trials=1
    )
    with pytest.raises(ValueError, match="At least one"):
        dashboard._run_with_results([], config)


def test_run_headless_logs_and_returns_summary(
    caplog: pytest.LogCaptureFixture,
) -> None:
    """``_run_headless`` logs both the per-engine results and the CV summary."""
    import logging

    config = dashboard.CrossEngineSimConfig(
        t_end=0.05, dt=0.01, noise_amplitude=0.0, n_trials=1
    )
    with caplog.at_level(logging.INFO, logger=dashboard.__name__):
        cv_summary = dashboard._run_headless(["pendulum_stub"], config)
    assert isinstance(cv_summary, dict)
    assert any("Cross-Engine" in rec.message for rec in caplog.records)
    assert any("CV Summary" in rec.message for rec in caplog.records)


def test_build_engine_returns_stub_for_pendulum() -> None:
    """``_build_engine('pendulum_stub')`` always returns the stub."""
    eng = dashboard._build_engine("pendulum_stub")
    assert isinstance(eng, dashboard._StubEngine)


def test_build_engine_falls_back_to_stub_when_real_unavailable() -> None:
    """When the real physics package isn't importable, a stub is returned."""
    eng = dashboard._build_engine("mujoco")
    # Either the real engine (if installed) or the stub.
    assert eng is not None
    # In CI without mujoco/drake/pinocchio, _try_build_real_engine returns None
    # and we get a stub.
    assert isinstance(eng, dashboard._StubEngine) or hasattr(eng, "step")


# ----------------------------------------------------------------------
# GUI integration (PyQt6) — headless smoke tests
# ----------------------------------------------------------------------

# Skip the GUI section if PyQt6 isn't importable; run on a hidden offscreen
# QApplication when it is. The marker is recognised by CI via headless_safe.


def _qt_available() -> tuple[bool, str]:
    """Return ``(ok, reason)`` for the GUI-test PyQt6 dependency."""
    try:  # pragma: no cover - environment-dependent
        import os  # noqa: PLC0415

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        import PyQt6.QtCore  # noqa: F401, PLC0415
        from PyQt6.QtWidgets import QApplication  # noqa: F401, PLC0415

        return True, ""
    except (ImportError, OSError) as exc:  # pragma: no cover
        return False, f"PyQt6 GUI not functional: {exc}"


_QT_OK, _QT_REASON = _qt_available()
qt_required = pytest.mark.skipif(not _QT_OK, reason=_QT_REASON)


@pytest.fixture(scope="module")
def qt_app():  # type: ignore[no-untyped-def]
    """Reusable offscreen QApplication for GUI smoke tests."""
    if not _QT_OK:  # pragma: no cover
        pytest.skip(_QT_REASON)
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication([])
    yield app


@qt_required
@pytest.mark.headless_safe
def test_window_construction_creates_single_renderer(qt_app) -> None:  # type: ignore[no-untyped-def]
    """Constructing the dashboard creates exactly one MatplotlibMarkerRenderer."""
    cls = dashboard._create_dashboard_window_class()
    win = cls(shape_per_engine=True)
    try:
        assert isinstance(win._traj_renderer, MatplotlibMarkerRenderer)
        assert win._shape_per_engine is True
        # The renderer is bound to the trajectory axes — DRY.
        assert win._traj_renderer._default_ax is win._ax_tr
    finally:
        win.deleteLater()


@qt_required
@pytest.mark.headless_safe
def test_window_trajectory_overlay_renders_distinct_colors(
    qt_app,  # type: ignore[no-untyped-def]
) -> None:
    """The window's overlay update produces one handle per engine."""
    cls = dashboard._create_dashboard_window_class()
    win = cls(shape_per_engine=True)
    try:
        rng = np.random.default_rng(1)
        trajectories = {
            "drake": rng.standard_normal((30, 2)),
            "mujoco": rng.standard_normal((30, 2)),
            "pinocchio": rng.standard_normal((30, 2)),
        }
        win._update_trajectory_overlay(trajectories)
        assert set(win._traj_handles) == set(trajectories)
        # Same renderer instance retained across the overlay call.
        assert all(h in win._traj_renderer._handles for h in win._traj_handles.values())
        # Re-rendering replaces handles cleanly (DRY: same renderer reused).
        win._update_trajectory_overlay({"drake": rng.standard_normal((10, 2))})
        assert set(win._traj_handles) == {"drake"}
    finally:
        win.deleteLater()


@qt_required
@pytest.mark.headless_safe
def test_window_trajectory_overlay_handles_empty_dict(
    qt_app,  # type: ignore[no-untyped-def]
) -> None:
    """Empty trajectories input clears the overlay without raising."""
    cls = dashboard._create_dashboard_window_class()
    win = cls(shape_per_engine=False)
    try:
        win._update_trajectory_overlay({})
        assert win._traj_handles == {}
    finally:
        win.deleteLater()


@qt_required
@pytest.mark.headless_safe
def test_build_qt_window_passes_shape_toggle(qt_app) -> None:  # type: ignore[no-untyped-def]
    """_build_qt_window threads the shape_per_engine kwarg through."""
    win_default = dashboard._build_qt_window()
    win_off = dashboard._build_qt_window(shape_per_engine=False)
    try:
        assert win_default._shape_per_engine is True  # type: ignore[attr-defined]
        assert win_off._shape_per_engine is False  # type: ignore[attr-defined]
    finally:
        win_default.deleteLater()  # type: ignore[attr-defined]
        win_off.deleteLater()  # type: ignore[attr-defined]

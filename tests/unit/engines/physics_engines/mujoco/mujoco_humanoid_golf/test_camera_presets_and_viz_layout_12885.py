"""Quick camera buttons and the Visualization tab layout (UD #12885, #12889).

#12885: ``ControlsTab`` called ``main_window.update_visualization_camera_sliders``
-- a method that no longer exists -- so clicking *Side* (or any quick camera
button) raised ``AttributeError`` inside a Qt slot and the launcher's crash
dialog closed the whole app. The tab now only *announces* the request through
``camera_preset_requested``; the main window routes it to the Visualization
tab, which owns the preset combo and the camera sliders.

#12889: the Visualization tab stacked ~10 group boxes in a plain layout, so in
the launcher's ~700 px side panel every box was squeezed below its size hint
and the controls overlapped. The content now lives in a ``QScrollArea``.
"""

from __future__ import annotations

from types import SimpleNamespace
from typing import Any

import pytest

pytestmark = [pytest.mark.unit]

pytest.importorskip("mujoco")
pytest.importorskip("cv2")
try:
    from PyQt6 import QtWidgets
except (ImportError, OSError) as exc:  # pragma: no cover - optional GUI dependency
    pytest.skip(f"PyQt6 not loadable: {exc}", allow_module_level=True)

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf._camera_presets import (  # noqa: E402
    CAMERA_PRESETS,
    CameraPreset,
    camera_preset,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.core.main_window import (  # noqa: E402
    AdvancedGolfAnalysisWindow,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.controls_tab import (  # noqa: E402
    ControlsTab,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.visualization_tab import (  # noqa: E402
    VisualizationTab,
)


class _SimWidget:
    """Records camera calls; every other sim-widget call is a no-op."""

    def __init__(self) -> None:
        self.cameras: list[str] = []
        self.camera = SimpleNamespace()

    def set_camera(self, name: str) -> None:
        self.cameras.append(name)

    def get_camera_azimuth(self) -> float:
        return 90.0

    def get_camera_elevation(self) -> float:
        return -20.0

    def get_camera_distance(self) -> float:
        return 3.0

    def get_camera_lookat(self) -> tuple[float, float, float]:
        return (0.0, 0.0, 1.0)

    def has_model(self) -> bool:
        return False

    def __getattr__(self, name: str) -> Any:
        return lambda *args, **kwargs: None


class _NoAttributeMainWindow:
    """A main window that fails loudly if a tab reaches into it."""

    def __getattr__(self, name: str) -> Any:
        raise AttributeError(f"ControlsTab must not touch main_window.{name}")


@pytest.fixture
def qapp() -> Any:
    return QtWidgets.QApplication.instance() or QtWidgets.QApplication([])


@pytest.fixture
def viz_tab(qapp: Any) -> Any:
    sim = _SimWidget()
    tab = VisualizationTab(sim)  # type: ignore[arg-type]
    yield tab, sim
    tab.deleteLater()
    qapp.processEvents()


@pytest.fixture
def controls_tab(qapp: Any) -> Any:
    tab = ControlsTab(_SimWidget(), _NoAttributeMainWindow())  # type: ignore[arg-type]
    yield tab
    tab.deleteLater()
    qapp.processEvents()


# ---- camera preset table (pure) ------------------------------------------


@pytest.mark.unit
def test_camera_presets_cover_the_quick_buttons_and_combo() -> None:
    assert {"side", "front", "top", "follow", "down-the-line"} <= set(CAMERA_PRESETS)
    assert all(isinstance(p, CameraPreset) for p in CAMERA_PRESETS.values())


@pytest.mark.unit
def test_side_preset_values_are_unchanged() -> None:
    assert camera_preset("side") == CameraPreset(90.0, -20.0, 3.0)


@pytest.mark.unit
def test_unknown_camera_preset_is_rejected() -> None:
    with pytest.raises(ValueError, match="nope"):
        camera_preset("nope")


# ---- #12885: ControlsTab emits, never reaches into the main window -------


@pytest.mark.unit
@pytest.mark.parametrize("preset", ["front", "side", "top", "follow"])
def test_quick_camera_button_emits_request_without_touching_main_window(
    controls_tab: Any, preset: str
) -> None:
    requested: list[str] = []
    controls_tab.camera_preset_requested.connect(requested.append)

    controls_tab.quick_camera_buttons[preset].click()

    assert requested == [preset]


@pytest.mark.unit
def test_visualization_tab_applies_preset_and_syncs_combo(viz_tab: Any) -> None:
    tab, sim = viz_tab
    tab.apply_camera_preset("top")
    assert tab.camera_combo.currentText() == "top"
    assert sim.cameras == ["top"]


@pytest.mark.unit
def test_reapplying_the_current_preset_still_moves_the_camera(viz_tab: Any) -> None:
    """The combo already shows 'side'; the Side button must still reset the view."""
    tab, sim = viz_tab
    assert tab.camera_combo.currentText() == "side"
    tab.apply_camera_preset("side")
    assert sim.cameras == ["side"]


@pytest.mark.unit
def test_visualization_tab_rejects_unknown_preset(viz_tab: Any) -> None:
    tab, sim = viz_tab
    with pytest.raises(ValueError):
        tab.apply_camera_preset("nope")
    assert sim.cameras == []


@pytest.mark.unit
def test_main_window_routes_quick_camera_to_visualization_tab() -> None:
    applied: list[str] = []
    fake = SimpleNamespace(
        visualization_tab=SimpleNamespace(apply_camera_preset=applied.append)
    )
    AdvancedGolfAnalysisWindow._on_quick_camera_clicked(fake, "side")  # type: ignore[arg-type]
    assert applied == ["side"]


@pytest.mark.unit
def test_quick_camera_button_end_to_end(
    qapp: Any, controls_tab: Any, viz_tab: Any
) -> None:
    """Button -> signal -> main-window router -> Visualization tab -> camera."""
    tab, sim = viz_tab
    router = SimpleNamespace(visualization_tab=tab)
    controls_tab.camera_preset_requested.connect(
        lambda name: AdvancedGolfAnalysisWindow._on_quick_camera_clicked(router, name)  # type: ignore[arg-type]
    )
    for preset in ("front", "side", "top", "follow"):
        controls_tab.quick_camera_buttons[preset].click()
    assert sim.cameras == ["front", "side", "top", "follow"]
    assert tab.camera_combo.currentText() == "follow"


# ---- #12889: Visualization tab content scrolls instead of overlapping ----


@pytest.mark.unit
def test_visualization_content_is_in_a_scroll_area(viz_tab: Any) -> None:
    tab, _ = viz_tab
    scroll = tab.findChild(QtWidgets.QScrollArea)
    assert scroll is not None
    assert scroll.widgetResizable()
    groups = scroll.widget().findChildren(QtWidgets.QGroupBox)
    assert len(groups) >= 8


@pytest.mark.unit
def test_visualization_group_boxes_are_not_squeezed_in_a_short_panel(
    viz_tab: Any, qapp: Any
) -> None:
    """At the launcher's side-panel size no group box may fall below its hint.

    The launcher hosts the tab in a fixed-height side panel, so the host -- not
    the tab -- decides the height; a top-level tab would just grow instead.
    """
    tab, _ = viz_tab
    host = QtWidgets.QTabWidget()
    host.addTab(tab, "Visualization")
    host.setFixedSize(480, 400)
    host.show()
    qapp.processEvents()
    try:
        top_level = [
            g
            for g in tab.findChildren(QtWidgets.QGroupBox)
            if not isinstance(g.parentWidget(), QtWidgets.QGroupBox)
        ]
        assert top_level
        squeezed = [
            (g.title(), g.height(), g.minimumSizeHint().height())
            for g in top_level
            if g.height() < g.minimumSizeHint().height()
        ]
        assert squeezed == []
    finally:
        host.hide()
        tab.setParent(None)
        host.deleteLater()

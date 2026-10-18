"""MuJoCo GUI force/torque overlays go through the shared renderers (FTO-10, #11295)."""

from __future__ import annotations

import subprocess
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

mujoco = pytest.importorskip("mujoco")
pytest.importorskip("cv2")  # renderers package imports OpenCV eagerly

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (  # noqa: E402
    meshcat_adapter as meshcat_adapter_mod,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyph_overlay import (  # noqa: E402
    build_overlay_glyphs,
    overlay_style,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_rendering_mixin import (  # noqa: E402
    SimRenderingMixin,
)
from src.shared.python.force_overlay.contracts import (  # noqa: E402
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (  # noqa: E402
    MeshcatGlyphRenderer,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[6]

_XML = """<mujoco><worldbody><body name="b"><joint type="hinge" axis="0 1 0"/>
<geom type="sphere" size="0.05"/></body></worldbody></mujoco>"""


def _frame() -> ForceTorqueFrame:
    return ForceTorqueFrame(
        time_s=0.0,
        engine="synthetic_test",
        wrenches=(
            OverlayWrench(
                kind=WrenchKind.CONTACT,
                label="contact:foot",
                body="b",
                point_m=(0.0, 0.0, 0.0),
                force_n=(0.0, 0.0, 100.0),
                source="synthetic_test",
            ),
            OverlayWrench(
                kind=WrenchKind.JOINT_ACTUATOR,
                label="actuator:hinge",
                body="b",
                point_m=(0.0, 0.0, 0.5),
                torque_nm=(0.0, 20.0, 0.0),
                source="synthetic_test",
            ),
        ),
    )


class _FakeEngine:
    def __init__(self) -> None:
        self.calls = 0

    def get_force_torque_frame(self) -> ForceTorqueFrame:
        self.calls += 1
        return _frame()


class _Widget(SimRenderingMixin):
    def __init__(self, force: bool, torque: bool, scale: float = 1e-3) -> None:
        self.engine = _FakeEngine()
        self.show_force_vectors = force
        self.show_torque_vectors = torque
        self.force_scale = scale
        self.torque_scale = scale
        self.isolate_forces_visualization = False
        self.manipulator = None
        self.force_legend_text = ""


def _scene() -> Any:
    model = mujoco.MjModel.from_xml_string(_XML)
    return mujoco.MjvScene(model, maxgeom=100)


def test_toggle_on_calls_provider_once_and_adds_geoms() -> None:
    widget = _Widget(force=True, torque=True)
    scene = _scene()
    before = scene.ngeom
    glyphs = widget._render_force_glyphs(scene)
    assert widget.engine.calls == 1
    assert glyphs is not None
    assert scene.ngeom - before >= len(glyphs.arrows) + len(glyphs.torque_arcs) > 0


def test_toggles_off_add_no_geoms_and_skip_provider() -> None:
    widget = _Widget(force=False, torque=False)
    scene = _scene()
    assert widget._render_force_glyphs(scene) is None
    assert scene.ngeom == 0
    assert widget.engine.calls == 0


def test_force_only_draws_no_torque_arcs() -> None:
    glyphs = build_overlay_glyphs(
        _frame(),
        show_force=True,
        show_torque=False,
        force_scale=1e-3,
        torque_scale=1e-3,
    )
    assert glyphs is not None
    assert len(glyphs.arrows) == 1 and not glyphs.torque_arcs


def test_scale_changes_arrow_length_proportionally() -> None:
    def length(scale: float) -> float:
        g = build_overlay_glyphs(
            _frame(),
            show_force=True,
            show_torque=False,
            force_scale=scale,
            torque_scale=1e-3,
        )
        assert g is not None
        a = g.arrows[0]
        return float(np.linalg.norm(np.subtract(a.tip_m, a.tail_m)))

    assert length(2e-3) == pytest.approx(2.0 * length(1e-3), rel=1e-6)


def test_isolate_filters_to_selected_body() -> None:
    glyphs = build_overlay_glyphs(
        _frame(),
        show_force=True,
        show_torque=True,
        force_scale=1e-3,
        torque_scale=1e-3,
        body_name="other",
    )
    assert glyphs is not None and not glyphs.arrows and not glyphs.torque_arcs


@pytest.mark.parametrize("bad", [0.0, -1.0, float("nan"), float("inf")])
def test_scale_validation(bad: float) -> None:
    with pytest.raises(ValueError):
        overlay_style(bad, 1e-3)
    with pytest.raises(TypeError):
        overlay_style("x", 1e-3)  # type: ignore[arg-type]


def test_legend_emitted_when_overlays_on() -> None:
    widget = _Widget(force=True, torque=False)
    widget._build_force_glyphs()
    assert "Engine: synthetic_test" in widget.force_legend_text
    widget.show_force_vectors = False
    widget._build_force_glyphs()
    assert widget.force_legend_text == ""


class _Sink:
    def __init__(self) -> None:
        self.cylinders: list[str] = []
        self.deleted: list[str] = []

    def set_cylinder(self, path: str, *args: Any) -> None:
        self.cylinders.append(path)

    def set_transform(self, path: str, matrix4x4: Any) -> None:
        pass

    def delete(self, path: str) -> None:
        self.deleted.append(path)


def test_meshcat_adapter_draws_shaft_head_and_deletes_removed_labels() -> None:
    adapter = meshcat_adapter_mod.MuJoCoMeshcatAdapter.__new__(
        meshcat_adapter_mod.MuJoCoMeshcatAdapter
    )
    adapter.vis = SimpleNamespace()
    sink = _Sink()
    adapter._glyph_renderer = MeshcatGlyphRenderer(sink)
    both = build_overlay_glyphs(
        _frame(), show_force=True, show_torque=True, force_scale=1e-3, torque_scale=1e-3
    )
    adapter.draw_glyphs(both)
    assert any(p.endswith("/shaft") for p in sink.cylinders)
    assert any(p.endswith("/head") for p in sink.cylinders)
    force_only = build_overlay_glyphs(
        _frame(),
        show_force=True,
        show_torque=False,
        force_scale=1e-3,
        torque_scale=1e-3,
    )
    adapter.draw_glyphs(force_only)
    assert any("actuator:hinge" in p for p in sink.deleted)
    adapter.draw_glyphs(None)
    assert "/force_overlay" in sink.deleted


def test_no_flat_xaxis_slices_remain() -> None:
    src = REPO_ROOT / "src"
    out = subprocess.run(
        ["grep", "-rnE", r"xaxis\[3 ?\* ?", "--include=*.py", str(src)],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.stdout == ""


def test_style_options_body_weight_and_groups() -> None:
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.force_glyph_overlay import (
        style_options_from_controls,
    )

    options = style_options_from_controls(
        scale_mode="body_weight",
        body_mass_kg=80.0,
        peak_force_n=2000.0,
        reference_length_m=0.5,
        groups=["per_foot", "net"],
    )
    assert options["scale_mode"] == "body_weight"
    assert options["reference_force_n"] == pytest.approx(80.0 * 9.80665)
    assert options["groups"] == frozenset({"per_foot", "net"})
    peak = style_options_from_controls(
        scale_mode="peak",
        body_mass_kg=80.0,
        peak_force_n=2000.0,
        reference_length_m=0.7,
        groups=["net"],
    )
    assert peak["reference_force_n"] == 2000.0
    fixed = style_options_from_controls(
        scale_mode="fixed",
        body_mass_kg=80.0,
        peak_force_n=2000.0,
        reference_length_m=0.5,
        groups=["net"],
    )
    assert "reference_force_n" not in fixed
    with pytest.raises(ValueError):
        style_options_from_controls(
            scale_mode="body_weight",
            body_mass_kg=0.0,
            peak_force_n=1.0,
            reference_length_m=0.5,
            groups=[],
        )


def test_overlay_glyphs_follow_body_weight_options() -> None:
    options = {
        "scale_mode": "body_weight",
        "reference_force_n": 100.0,
        "reference_length_m": 0.5,
    }
    glyphs = build_overlay_glyphs(
        _frame(),
        show_force=True,
        show_torque=False,
        force_scale=1e-3,
        torque_scale=1e-3,
        style_options=options,
    )
    assert glyphs is not None
    tip = glyphs.arrows[0].tip_m
    assert tip[2] == pytest.approx(0.5)  # 100 N == 1 reference force == 0.5 m
    with pytest.raises(ValueError):
        overlay_style(1e-3, 1e-3, {"scale_mode": "body_weight"})
    with pytest.raises(ValueError):
        overlay_style(1e-3, 1e-3, {"bogus": 1})


def test_widget_style_options_reach_the_renderer() -> None:
    widget = _Widget(force=True, torque=False)
    widget.force_style_options = {
        "scale_mode": "peak",
        "reference_force_n": 50.0,
        "reference_length_m": 0.2,
    }
    glyphs = widget._render_force_glyphs(_scene())
    assert glyphs is not None
    assert glyphs.arrows[0].tip_m[2] == pytest.approx(0.4)  # 100 N / 50 N * 0.2 m


# --- Inline tension/compression controls and persistence (#13146) -------------


class _FakeSettings:
    def __init__(self, data: dict[str, Any] | None = None) -> None:
        self.data: dict[str, Any] = dict(data or {})

    def value(self, key: str, default: Any = None) -> Any:
        return self.data.get(key, default)

    def setValue(self, key: str, value: Any) -> None:  # noqa: N802
        self.data[key] = value


@pytest.fixture
def gui(monkeypatch: pytest.MonkeyPatch) -> Any:
    try:
        from PyQt6 import QtWidgets
    except (ImportError, OSError) as exc:  # pragma: no cover
        pytest.skip(f"PyQt6 not loadable: {exc}")
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf import (
        sim_force_color_mixin as mixin_mod,
        sim_widget as sim_widget_mod,
    )
    from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.gui.tabs.visualization_tab import (
        VisualizationTab,
    )

    app = QtWidgets.QApplication.instance() or QtWidgets.QApplication([])
    store = _FakeSettings()
    monkeypatch.setattr(
        mixin_mod, "open_force_color_settings", lambda: store, raising=True
    )

    def no_meshcat() -> Any:
        raise RuntimeError("meshcat disabled in tests")

    monkeypatch.setattr(sim_widget_mod, "MuJoCoMeshcatAdapter", no_meshcat)

    def build() -> tuple[Any, Any]:
        sim = sim_widget_mod.MuJoCoSimWidget(None, 320, 240, 1)
        sim.timer.stop()
        return sim, VisualizationTab(sim)

    sim, tab = build()
    yield SimpleNamespace(sim=sim, tab=tab, store=store, build=build, qt=QtWidgets)
    tab.deleteLater()
    sim.deleteLater()
    app.processEvents()


def _inline(tab: Any, qt: Any) -> tuple[Any, Any]:
    return (
        tab.findChild(qt.QCheckBox, "force_color_inline_enabled"),
        tab.findChild(qt.QSlider, "force_color_inline_opacity"),
    )


def _open_dialog(sim: Any, qt: Any) -> Any:
    sim.open_force_color_dialog()
    dialog = sim.findChild(qt.QDialog, "segment_force_colors_dialog")
    assert dialog is not None
    return dialog


def test_inline_checkbox_toggles_sim_scale(gui: Any) -> None:
    check, _ = _inline(gui.tab, gui.qt)
    assert check is not None and not check.isChecked()
    check.setChecked(True)
    assert gui.sim.axial_color_scale.enabled
    check.setChecked(False)
    assert not gui.sim.axial_color_scale.enabled


def test_inline_opacity_slider_updates_sim_scale(gui: Any) -> None:
    _, slider = _inline(gui.tab, gui.qt)
    assert slider is not None and slider.value() == 100
    slider.setValue(30)
    assert gui.sim.axial_color_scale.opacity == pytest.approx(0.3)


def test_dialog_changes_update_inline_controls(gui: Any) -> None:
    from src.shared.python.body_part_viz.force_color_controls import (
        ForceColorControls,
    )

    check, slider = _inline(gui.tab, gui.qt)
    dialog = _open_dialog(gui.sim, gui.qt)
    controls = dialog.findChild(ForceColorControls)
    controls.findChild(gui.qt.QCheckBox, "force_color_enabled").setChecked(True)
    controls.findChild(gui.qt.QSlider, "force_color_opacity").setValue(45)
    assert check.isChecked() and slider.value() == 45
    assert gui.sim.axial_color_scale.enabled
    # and the other direction: inline edits reach the open dialog
    slider.setValue(80)
    assert controls.findChild(gui.qt.QSlider, "force_color_opacity").value() == 80
    check.setChecked(False)
    assert not controls.findChild(gui.qt.QCheckBox, "force_color_enabled").isChecked()


def test_programmatic_scale_updates_inline_controls(gui: Any) -> None:
    from src.shared.python.body_part_viz import ForceColorScale

    check, slider = _inline(gui.tab, gui.qt)
    gui.sim.set_axial_color_scale(ForceColorScale(enabled=True, opacity=0.5))
    assert check.isChecked() and slider.value() == 50


def test_dialog_button_opens_full_dialog(gui: Any) -> None:
    button = gui.tab.findChild(gui.qt.QPushButton, "open_segment_force_colors")
    assert button is not None
    button.click()
    assert gui.sim.findChild(gui.qt.QDialog, "segment_force_colors_dialog")


def test_vector_labels_are_accurate_and_names_kept(gui: Any) -> None:
    assert gui.tab.show_torques_cb.text() == "Show Joint Torque Vectors (curl arrows)"
    assert gui.tab.show_forces_cb.text() == "Show Joint Force Vectors"


def test_state_persists_across_instances(gui: Any) -> None:
    check, slider = _inline(gui.tab, gui.qt)
    check.setChecked(True)
    slider.setValue(40)
    gui.tab.show_torques_cb.setChecked(True)
    gui.tab.show_forces_cb.setChecked(True)
    sim2, tab2 = gui.build()
    try:
        assert sim2.axial_color_scale.enabled
        assert sim2.axial_color_scale.opacity == pytest.approx(0.4)
        assert sim2.show_torque_vectors and sim2.show_force_vectors
        check2, slider2 = _inline(tab2, gui.qt)
        assert check2.isChecked() and slider2.value() == 40
        assert tab2.show_torques_cb.isChecked() and tab2.show_forces_cb.isChecked()
    finally:
        tab2.deleteLater()
        sim2.deleteLater()


def test_corrupt_stored_value_falls_back_to_defaults(
    gui: Any, caplog: pytest.LogCaptureFixture
) -> None:
    gui.store.data["force_color/mujoco"] = "{not json"
    sim2, tab2 = gui.build()
    try:
        assert not sim2.axial_color_scale.enabled
        assert sim2.axial_color_scale.opacity == 1.0
        assert not sim2.show_force_vectors and not sim2.show_torque_vectors
    finally:
        tab2.deleteLater()
        sim2.deleteLater()

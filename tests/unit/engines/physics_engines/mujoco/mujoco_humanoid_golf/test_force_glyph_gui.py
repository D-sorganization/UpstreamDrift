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

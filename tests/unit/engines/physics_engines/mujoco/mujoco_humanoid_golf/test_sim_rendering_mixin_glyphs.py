"""Unit tests for MuJoCo GUI force/torque overlay rendering (FTO-10, #11295).

Verifies that SimRenderingMixin and MuJoCoMeshcatAdapter render force/torque
overlays through the shared force_overlay renderers (native MjvScene geoms and
MeshcatGlyphRenderer), that user toggles and scale sliders work correctly,
and that the buggy xaxis[3*j:3*j+3] slicing is completely eliminated.
"""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

pytestmark = [pytest.mark.unit]

mujoco = pytest.importorskip("mujoco")

from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.meshcat_adapter import (
    MuJoCoMeshcatAdapter,
)
from src.engines.physics_engines.mujoco.python.mujoco_humanoid_golf.sim_rendering_mixin import (
    SimRenderingMixin,
)
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from src.shared.python.force_overlay.renderers.meshcat_glyphs import MeshcatSink


def _make_synthetic_frame() -> ForceTorqueFrame:
    """Create a synthetic ForceTorqueFrame with a force and a torque wrench."""
    w1 = OverlayWrench(
        kind=WrenchKind.CONTACT,
        label="contact:ground:0",
        body="foot",
        point_m=(0.0, 0.0, 0.0),
        force_n=(0.0, 0.0, 500.0),
        torque_nm=None,
        source="mujoco",
    )
    w2 = OverlayWrench(
        kind=WrenchKind.JOINT_ACTUATOR,
        label="actuator:knee",
        body="shank",
        point_m=(0.0, 0.0, 0.5),
        force_n=None,
        torque_nm=(10.0, 0.0, 0.0),
        source="mujoco",
    )
    return ForceTorqueFrame(
        time_s=1.0,
        engine="mujoco",
        wrenches=(w1, w2),
        world_frame="world_Zup",
    )


class _DummySimWidget(SimRenderingMixin):
    """Test harness inheriting SimRenderingMixin without full Qt/GUI dependencies."""

    def __init__(self, frame: ForceTorqueFrame | None = None) -> None:
        self.show_force_vectors = False
        self.show_torque_vectors = False
        self.force_scale = 0.001
        self.torque_scale = 0.005
        self.force_legend_text = ""
        self.status_bar = MagicMock()

        self.engine = SimpleNamespace()
        self._frame = frame or _make_synthetic_frame()
        self.engine.get_force_torque_frame = MagicMock(return_value=self._frame)


class _RecordingMeshcatSink:
    """Fake MeshcatSink recording calls for test assertion."""

    def __init__(self) -> None:
        self.cylinders: dict[str, dict[str, Any]] = {}
        self.transforms: dict[str, np.ndarray] = {}
        self.deleted_paths: list[str] = []

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        self.cylinders[path] = {
            "length_m": length_m,
            "radius_top_m": radius_top_m,
            "radius_bottom_m": radius_bottom_m,
            "rgba": rgba,
        }

    def set_transform(self, path: str, matrix4x4: np.ndarray) -> None:
        self.transforms[path] = np.asarray(matrix4x4, dtype=np.float64)

    def delete(self, path: str) -> None:
        self.deleted_paths.append(path)
        to_del = [p for p in self.cylinders if p == path or p.startswith(path + "/")]
        for p in to_del:
            self.cylinders.pop(p, None)
            self.transforms.pop(p, None)


def test_show_force_vectors_adds_geoms_to_scene() -> None:
    """With show_force_vectors=True, mixin calls engine provider and adds geoms."""
    widget = _DummySimWidget()
    widget.show_force_vectors = True
    widget.show_torque_vectors = False

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, 100)
    initial_ngeom = scene.ngeom

    widget._render_force_glyphs(scene)

    widget.engine.get_force_torque_frame.assert_called_once()
    assert scene.ngeom > initial_ngeom
    assert widget.force_legend_text != ""


def test_turning_toggles_off_adds_no_geoms() -> None:
    """Turning toggles off adds 0 geoms and does not call the provider."""
    widget = _DummySimWidget()
    widget.show_force_vectors = False
    widget.show_torque_vectors = False

    model = mujoco.MjModel.from_xml_string("<mujoco><worldbody/></mujoco>")
    scene = mujoco.MjvScene(model, 100)
    initial_ngeom = scene.ngeom

    widget._render_force_glyphs(scene)

    widget.engine.get_force_torque_frame.assert_not_called()
    assert scene.ngeom == initial_ngeom
    assert widget.force_legend_text == ""


def test_scale_slider_changes_arrow_length_proportionally() -> None:
    """Scale slider changes arrow length proportionally in generated glyphs."""
    frame = _make_synthetic_frame()
    style_base = ForceGlyphStyle(
        force_scale_m_per_n=0.001,
        min_length_m=0.001,
        max_length_m=10.0,
    )
    style_double = ForceGlyphStyle(
        force_scale_m_per_n=0.002,
        min_length_m=0.001,
        max_length_m=10.0,
    )

    glyphs_base = build_glyphs(frame, style_base)
    glyphs_double = build_glyphs(frame, style_double)

    arrow_base = next(a for a in glyphs_base.arrows if a.label == "contact:ground:0")
    arrow_double = next(
        a for a in glyphs_double.arrows if a.label == "contact:ground:0"
    )

    len_base = np.linalg.norm(np.array(arrow_base.tip_m) - np.array(arrow_base.tail_m))
    len_double = np.linalg.norm(
        np.array(arrow_double.tip_m) - np.array(arrow_double.tail_m)
    )

    assert len_double == pytest.approx(2.0 * len_base, rel=1e-3)


def test_meshcat_adapter_fake_sink_records_and_deletes() -> None:
    """MeshCat adapter records shaft/head cylinder calls and cleans up paths on toggle off."""
    sink = _RecordingMeshcatSink()
    adapter = MuJoCoMeshcatAdapter.__new__(MuJoCoMeshcatAdapter)
    adapter.vis = SimpleNamespace()
    adapter.model = None

    from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
        MeshcatGlyphRenderer,
    )

    adapter._glyph_renderer = MeshcatGlyphRenderer(sink)

    frame = _make_synthetic_frame()
    mock_data = MagicMock()

    # Draw force only
    adapter.draw_vectors(
        mock_data,
        show_force=True,
        show_torque=False,
        force_scale=0.001,
        torque_scale=0.005,
        frame=frame,
    )

    # Verify shaft and head were recorded
    active_paths = list(sink.cylinders.keys())
    assert any("shaft" in p for p in active_paths)
    assert any("head" in p for p in active_paths)

    # Now turn off both toggles
    adapter.draw_vectors(
        mock_data,
        show_force=False,
        show_torque=False,
        frame=frame,
    )

    # Sink should have deleted the root overlay
    assert any(
        "/force_overlay" in p or "force_overlay" in p for p in sink.deleted_paths
    )


def test_regression_no_xaxis_slice_in_src() -> None:
    """Regression test ensuring `xaxis[3*...` is completely absent from src/."""
    src_dir = Path(__file__).resolve().parents[6] / "src"
    assert src_dir.is_dir(), f"src directory not found at {src_dir}"

    matches: list[str] = []
    for py_file in src_dir.rglob("*.py"):
        try:
            content = py_file.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        for line_no, line in enumerate(content.splitlines(), start=1):
            if "xaxis[3" in line or "xaxis[ 3" in line or "xaxis[3 *" in line:
                matches.append(f"{py_file}:{line_no}: {line.strip()}")

    assert not matches, "Found buggy xaxis slicing in src/:\n" + "\n".join(matches)

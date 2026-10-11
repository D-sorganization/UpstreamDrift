"""Tests for MeshCat 3D glyph renderer (FTO-5, #11290)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.force_overlay.glyphs import (
    ArrowGlyph,
    ForceGlyphStyle,
    GlyphSet,
    LegendSpec,
    TorqueArcGlyph,
    build_glyphs,
)
from src.shared.python.force_overlay.renderers.meshcat_glyphs import (
    MeshcatGlyphRenderer,
    MeshcatPythonSink,
    MeshcatSink,
    align_y_to,
    legend_text,
)

pytestmark = pytest.mark.unit


@dataclass
class CylinderRecord:
    path: str
    length_m: float
    radius_top_m: float
    radius_bottom_m: float
    rgba: tuple[float, float, float, float]


@dataclass
class TransformRecord:
    path: str
    matrix4x4: np.ndarray


class RecordingFakeSink:
    """In-memory recording sink implementing MeshcatSink protocol."""

    def __init__(self) -> None:
        self.cylinders: list[CylinderRecord] = []
        self.transforms: list[TransformRecord] = []
        self.deletes: list[str] = []

    def set_cylinder(
        self,
        path: str,
        length_m: float,
        radius_top_m: float,
        radius_bottom_m: float,
        rgba: tuple[float, float, float, float],
    ) -> None:
        self.cylinders.append(
            CylinderRecord(path, length_m, radius_top_m, radius_bottom_m, rgba)
        )

    def set_transform(self, path: str, matrix4x4: np.ndarray) -> None:
        self.transforms.append(TransformRecord(path, np.copy(matrix4x4)))

    def delete(self, path: str) -> None:
        self.deletes.append(path)


def test_align_y_to() -> None:
    """Verify align_y_to maps +y onto target direction with proper rotation det=1."""
    y = np.array([0.0, 1.0, 0.0], dtype=np.float64)

    # 1. Test +x direction
    d_x = np.array([1.0, 0.0, 0.0], dtype=np.float64)
    r_x = align_y_to(d_x)
    assert np.linalg.det(r_x) == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(r_x @ y, d_x, atol=1e-7)

    # 2. Test -y direction (antiparallel singularity)
    d_minus_y = np.array([0.0, -1.0, 0.0], dtype=np.float64)
    r_minus_y = align_y_to(d_minus_y)
    assert np.linalg.det(r_minus_y) == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(r_minus_y @ y, d_minus_y, atol=1e-7)

    # 3. Test +y direction (identity)
    r_y = align_y_to(y)
    assert np.linalg.det(r_y) == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(r_y @ y, y, atol=1e-7)

    # 4. Test arbitrary unit vector
    vec = np.array([1.0, 2.0, 3.0], dtype=np.float64)
    d_arb = vec / np.linalg.norm(vec)
    r_arb = align_y_to(d_arb)
    assert np.linalg.det(r_arb) == pytest.approx(1.0, abs=1e-7)
    np.testing.assert_allclose(r_arb @ y, d_arb, atol=1e-7)


def test_single_force_arrow_draws_shaft_and_head() -> None:
    """One force arrow produces a cylinder shaft and cone head with radius_top=0."""
    sink = RecordingFakeSink()
    renderer = MeshcatGlyphRenderer(sink, root="/force_overlay")

    arrow = ArrowGlyph(
        label="contact:ground",
        kind="contact",
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(0.0, 0.0, 0.5),
        head_base_m=(0.0, 0.0, 0.4),
        shaft_radius_m=0.01,
        head_radius_m=0.025,
        rgba=(0.0, 1.0, 0.0, 1.0),
        magnitude=500.0,
        units="N",
        clamped=False,
    )
    glyphs = GlyphSet(
        schema_version="glyph-set-v1",
        time_s=0.0,
        arrows=(arrow,),
        torque_arcs=(),
        legend=LegendSpec(
            force_reference_n=500.0,
            force_reference_length_m=0.5,
            kinds_present=("contact",),
            unavailable_labels=(),
            engine="mujoco",
        ),
    )

    renderer.update(glyphs)

    # Expected paths
    shaft_path = "/force_overlay/contact:ground/shaft"
    head_path = "/force_overlay/contact:ground/head"

    cylinder_paths = [c.path for c in sink.cylinders]
    assert shaft_path in cylinder_paths
    assert head_path in cylinder_paths

    shaft_rec = next(c for c in sink.cylinders if c.path == shaft_path)
    assert shaft_rec.length_m == pytest.approx(0.4)
    assert shaft_rec.radius_top_m == pytest.approx(0.01)
    assert shaft_rec.radius_bottom_m == pytest.approx(0.01)
    assert shaft_rec.rgba == (0.0, 1.0, 0.0, 1.0)

    head_rec = next(c for c in sink.cylinders if c.path == head_path)
    assert head_rec.length_m == pytest.approx(0.1)
    assert head_rec.radius_top_m == pytest.approx(0.0)  # cone tip
    assert head_rec.radius_bottom_m == pytest.approx(0.025)

    # Transforms placed at segment centers
    transform_paths = [t.path for t in sink.transforms]
    assert shaft_path in transform_paths
    assert head_path in transform_paths


def test_update_caching_and_deletion() -> None:
    """Second identical update sends no set_cylinder calls; disappearing label calls delete."""
    sink = RecordingFakeSink()
    renderer = MeshcatGlyphRenderer(sink, root="/force_overlay")

    arrow_a = ArrowGlyph(
        label="arrow_a",
        kind="contact",
        tail_m=(0.0, 0.0, 0.0),
        tip_m=(0.0, 1.0, 0.0),
        head_base_m=(0.0, 0.8, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(1.0, 0.0, 0.0, 1.0),
        magnitude=100.0,
        units="N",
        clamped=False,
    )
    arrow_b = ArrowGlyph(
        label="arrow_b",
        kind="external",
        tail_m=(1.0, 0.0, 0.0),
        tip_m=(1.0, 1.0, 0.0),
        head_base_m=(1.0, 0.8, 0.0),
        shaft_radius_m=0.01,
        head_radius_m=0.02,
        rgba=(0.0, 0.0, 1.0, 1.0),
        magnitude=100.0,
        units="N",
        clamped=False,
    )

    set_1 = GlyphSet(
        schema_version="glyph-set-v1",
        time_s=0.0,
        arrows=(arrow_a, arrow_b),
        torque_arcs=(),
        legend=LegendSpec(kinds_present=("contact", "external"), engine="mujoco"),
    )

    renderer.update(set_1)
    initial_cyl_count = len(sink.cylinders)
    assert initial_cyl_count == 4  # 2 shafts + 2 heads

    # Identical update: no new cylinder creation calls
    renderer.update(set_1)
    assert len(sink.cylinders) == initial_cyl_count

    # Update removing arrow_b: should delete arrow_b paths only
    set_2 = GlyphSet(
        schema_version="glyph-set-v1",
        time_s=0.1,
        arrows=(arrow_a,),
        torque_arcs=(),
        legend=LegendSpec(kinds_present=("contact",), engine="mujoco"),
    )
    renderer.update(set_2)

    deleted_paths = sink.deletes
    assert "/force_overlay/arrow_b/shaft" in deleted_paths
    assert "/force_overlay/arrow_b/head" in deleted_paths
    assert "/force_overlay/arrow_a/shaft" not in deleted_paths


def test_torque_arc_segments_and_head() -> None:
    """Torque arc with 32 segments creates 32 segment cylinders plus a cone head."""
    sink = RecordingFakeSink()
    renderer = MeshcatGlyphRenderer(sink, root="/force_overlay")

    # Generate 33 points (32 segments) along a circle in XY plane
    thetas = np.linspace(0.0, np.pi, 33)
    poly = tuple((float(np.cos(th)), float(np.sin(th)), 0.0) for th in thetas)

    arc = TorqueArcGlyph(
        label="joint:wrist",
        kind="joint_actuator",
        center_m=(0.0, 0.0, 0.0),
        axis_unit=(0.0, 0.0, 1.0),
        radius_m=1.0,
        polyline_m=poly,
        head_tip_m=poly[-1],
        head_base_m=(poly[-1][0], poly[-1][1] - 0.1, 0.0),
        rgba=(0.5, 0.5, 0.0, 1.0),
        magnitude=25.0,
        units="N*m",
        clamped=False,
    )
    glyphs = GlyphSet(
        schema_version="glyph-set-v1",
        time_s=0.0,
        arrows=(),
        torque_arcs=(arc,),
        legend=LegendSpec(kinds_present=("joint_actuator",), engine="mujoco"),
    )

    renderer.update(glyphs)

    arc_paths = [c.path for c in sink.cylinders if "/joint:wrist/arc/" in c.path]
    assert len(arc_paths) == 32
    head_paths = [
        c.path for c in sink.cylinders if c.path == "/force_overlay/joint:wrist/head"
    ]
    assert len(head_paths) == 1
    assert (
        next(c for c in sink.cylinders if c.path == head_paths[0]).radius_top_m == 0.0
    )


def test_meshcat_python_sink_records_calls() -> None:
    """MeshcatPythonSink delegates to visualizer node methods."""
    mock_vis = MagicMock()
    mock_node = MagicMock()
    mock_vis.__getitem__.return_value = mock_node

    sink = MeshcatPythonSink(mock_vis)
    sink.set_transform("root/test", np.eye(4))
    mock_node.set_transform.assert_called_once()

    sink.delete("root/test")
    mock_node.delete.assert_called_once()


def test_drake_meshcat_sink_protocol() -> None:
    """Verify DrakeMeshcatSink conforms to MeshcatSink protocol when pydrake is present."""
    pydrake = pytest.importorskip("pydrake")
    from src.engines.physics_engines.drake.python.src.drake_meshcat_sink import (
        DrakeMeshcatSink,
    )

    mock_drake_meshcat = MagicMock()
    sink = DrakeMeshcatSink(mock_drake_meshcat)
    assert isinstance(sink, MeshcatSink)


def test_legend_text() -> None:
    """Verify legend_text formats summary string."""
    glyphs = GlyphSet(
        schema_version="glyph-set-v1",
        time_s=0.0,
        arrows=(),
        torque_arcs=(),
        legend=LegendSpec(
            force_reference_n=250.0,
            force_reference_length_m=0.35,
            torque_reference_nm=45.0,
            torque_reference_radius_m=0.12,
            kinds_present=("contact", "joint_reaction"),
            unavailable_labels=(),
            engine="mujoco",
        ),
    )
    text = legend_text(glyphs)
    assert "250" in text
    assert "mujoco" in text
    assert "contact" in text

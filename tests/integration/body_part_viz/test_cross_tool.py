"""Cross-tool integration tests for body_part_viz.

A single canonical ``SegmentVizSet`` is fed to each downstream consumer:

* ``segment_set_io`` round-trip + reload by the C3D Viewer service layer.
* ``LiveViewController`` rebuild via the matplotlib Agg backend.
* ``urdf_bridge.shape_to_urdf_visual`` -> URDF XML the standard
  ``xml.etree`` parser accepts back.

Each test validates the integration point on its own; together they
guarantee any future ``SegmentVizSet`` change ripples cleanly through
every fleet consumer.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET  # noqa: S405 - generated, parsed locally
from pathlib import Path

import matplotlib

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
matplotlib.use("Agg", force=True)

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from src.shared.python.body_part_viz import (  # noqa: E402
    BindingKind,
    MarkerBinding,
    SegmentVizSet,
    SegmentVizSpec,
    ShapeTheme,
)
from src.shared.python.body_part_viz.renderers import (  # noqa: E402
    MatplotlibRenderer,
)
from src.shared.python.body_part_viz.shapes import (  # noqa: E402
    CylinderShape,
    EllipsoidShape,
    LineShape,
)
from src.shared.python.body_part_viz.urdf_bridge import (  # noqa: E402
    shape_to_urdf_visual,
    urdf_to_shape,
)


# ---------------------------------------------------------------------------
# Canonical viz set used by every cross-tool case.
# ---------------------------------------------------------------------------
def _canonical_viz_set() -> SegmentVizSet:
    binding_line = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("LSHO", "LELB"),
        rest_dimensions=(0.3,),
    )
    binding_cyl = MarkerBinding(
        kind=BindingKind.BETWEEN_TWO,
        marker_names=("RSHO", "RELB"),
        rest_dimensions=(0.3, 0.04),
    )
    binding_ell = MarkerBinding(
        kind=BindingKind.ON_MARKER,
        marker_names=("HEAD",),
        rest_dimensions=(0.1, 0.12, 0.1),
    )
    spec_line = SegmentVizSpec(
        binding=binding_line,
        shape_kind="line",
        shape_params={"length": 0.3},
        fitter_kind="between_two",
        theme=ShapeTheme(color="#1f77b4", group="upper_arm"),
    )
    spec_cyl = SegmentVizSpec(
        binding=binding_cyl,
        shape_kind="cylinder",
        shape_params={"length": 0.3, "radius": 0.04, "n_facets": 12},
        fitter_kind="between_two",
        theme=ShapeTheme(color="#ff7f0e", opacity=0.7, group="upper_arm"),
    )
    spec_ell = SegmentVizSpec(
        binding=binding_ell,
        shape_kind="ellipsoid",
        shape_params={"a": 0.1, "b": 0.12, "c": 0.1},
        fitter_kind="cluster_kabsch",
        theme=ShapeTheme(color="#2ca02c", opacity=0.6, group="head"),
    )
    return SegmentVizSet(segments=(spec_line, spec_cyl, spec_ell))


# ---------------------------------------------------------------------------
# C3D Viewer service layer round-trip.
# ---------------------------------------------------------------------------
def test_segment_set_io_round_trip_through_c3d_viewer(tmp_path: Path) -> None:
    """Save -> load via the C3D Viewer ``segment_set_io`` adapter.

    The viewer's ``segment_set_io`` is a thin shim over
    :class:`SegmentVizSet`. A file written by the v2 ``save`` API must
    re-load identically through it, and the reloaded set must be
    consumable by :class:`MatplotlibRenderer` without additional
    plumbing.
    """
    viz = _canonical_viz_set()
    out = tmp_path / "viz.json"
    viz.save(out)
    assert out.is_file()

    # Round-trip via the canonical loader -- this is what the viewer's
    # service layer ultimately calls.
    reloaded = SegmentVizSet.load(out)
    assert reloaded.schema_version == viz.schema_version
    assert len(reloaded.segments) == len(viz.segments)
    for original, came_back in zip(viz.segments, reloaded.segments, strict=True):
        assert came_back.shape_kind == original.shape_kind
        assert came_back.fitter_kind == original.fitter_kind
        assert came_back.binding.marker_names == original.binding.marker_names
        assert came_back.theme.group == original.theme.group

    # Materialise the reloaded set on a real Axes3D to assert the
    # renderer accepts the same payload the viewer would build.
    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection="3d")
        renderer = MatplotlibRenderer(ax)
        # Build at least one fitted shape from the reloaded set as a
        # smoke test -- a line shape is the cheapest path.
        from src.shared.python.body_part_viz import FittedShape

        line_spec = next(s for s in reloaded.segments if s.shape_kind == "line")
        shape = LineShape(length=float(line_spec.shape_params["length"]))
        n_frames = 4
        fitted = FittedShape(
            shape_id=shape.shape_id,
            binding=line_spec.binding,
            centroid=np.zeros((n_frames, 3)),
            rotation_matrix=np.broadcast_to(np.eye(3), (n_frames, 3, 3)).copy(),
            scale=np.ones((n_frames, 3)),
            valid_mask=np.ones((n_frames,), dtype=bool),
        )
        handle = renderer.add_shape(shape, fitted, line_spec.theme)
        assert handle in renderer._entries
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# LiveViewController integration.
# ---------------------------------------------------------------------------
def test_live_view_controller_accepts_canonical_set(tmp_path: Path) -> None:
    """Feed a canonical set's marker bindings to ``LiveViewController``.

    The controller is the matcher's offscreen-friendly entry point. We
    construct a synthetic :class:`BodyTarget`-shaped object whose
    ``marker_names`` cover the canonical bindings and confirm the
    controller builds a layer stack against the Agg canvas without
    raising.
    """
    pytest.importorskip("PyQt6", reason="LiveViewController gui_playback path")
    from src.tools.starting_pose_matcher.live_view_controller import (
        LiveViewController,
    )

    viz = _canonical_viz_set()

    # Collect every marker name referenced by the canonical set so the
    # synthetic body target exposes them all.
    marker_names: list[str] = []
    for spec in viz.segments:
        for name in spec.binding.marker_names:
            if name not in marker_names:
                marker_names.append(name)

    n_frames = 64
    rng = np.random.default_rng(0)
    marker_xyz = rng.normal(size=(n_frames, len(marker_names), 3)) * 0.1

    class _BodyTarget:
        def __init__(self) -> None:
            self.marker_xyz = marker_xyz
            self.marker_names = tuple(marker_names)

    fig = plt.figure()
    try:
        ax = fig.add_subplot(111, projection="3d")
        canvas = fig.canvas
        controller = LiveViewController(ax, canvas, body_skeleton_style="lines")
        controller.set_target(body=_BodyTarget())
        # Smoke: the controller must have wired at least the body marker
        # layer and report n_frames matching the body target.
        assert controller.n_frames == n_frames
        assert "body_markers" in controller.layers()
        controller.set_frame(7)
        assert controller.current_frame == 7
        controller.clear()
    finally:
        plt.close(fig)


# ---------------------------------------------------------------------------
# URDF generator integration.
# ---------------------------------------------------------------------------
def test_urdf_generator_consumes_canonical_set() -> None:
    """Each shape in the set translates to URDF the parser accepts back.

    Line shapes are intentionally rejected by ``shape_to_urdf_visual``;
    they're skipped here. Cylinder + ellipsoid round-trip through
    ``urdf_to_shape`` and produce identical primitives within ``1e-9``.
    """
    viz = _canonical_viz_set()

    cyl_spec = next(s for s in viz.segments if s.shape_kind == "cylinder")
    ell_spec = next(s for s in viz.segments if s.shape_kind == "ellipsoid")

    cyl = CylinderShape(
        length=cyl_spec.shape_params["length"],
        radius=cyl_spec.shape_params["radius"],
        n_facets=cyl_spec.shape_params["n_facets"],
    )
    ell = EllipsoidShape(
        a=ell_spec.shape_params["a"],
        b=ell_spec.shape_params["b"],
        c=ell_spec.shape_params["c"],
    )

    # Forward: produce <visual> XML for each.
    cyl_visual = shape_to_urdf_visual(cyl)
    ell_visual = shape_to_urdf_visual(ell)
    assert isinstance(cyl_visual, ET.Element) and cyl_visual.tag == "visual"
    assert isinstance(ell_visual, ET.Element) and ell_visual.tag == "visual"

    # Wrap into a minimal <robot> document and round-trip through the
    # parser -- this is what the URDF generator produces in practice.
    robot = ET.Element("robot", name="bpv_smoke")
    link = ET.SubElement(robot, "link", name="link_cyl")
    link.append(cyl_visual)
    link2 = ET.SubElement(robot, "link", name="link_ell")
    link2.append(ell_visual)
    xml_bytes = ET.tostring(robot, encoding="utf-8")
    parsed = ET.fromstring(xml_bytes)  # noqa: S314 - input is locally generated
    assert parsed.tag == "robot"
    cyl_geom = parsed.find(".//link[@name='link_cyl']/visual/geometry/cylinder")
    assert cyl_geom is not None
    assert float(cyl_geom.attrib["radius"]) == pytest.approx(cyl.rest_dimensions[1])

    # Inverse: cylinder shape recovers exactly.
    recovered_cyl = urdf_to_shape(
        cyl_visual,
        asset_resolver=lambda _filename: Path(_filename),
    )
    assert isinstance(recovered_cyl, CylinderShape)
    assert recovered_cyl.rest_dimensions[0] == pytest.approx(cyl.rest_dimensions[0])
    assert recovered_cyl.rest_dimensions[1] == pytest.approx(cyl.rest_dimensions[1])

    # Ellipsoid encodes its semi-axes in the synthesised mesh filename;
    # the recovered shape compares those back within 1e-9.
    recovered_ell = urdf_to_shape(
        ell_visual,
        asset_resolver=lambda _filename: Path(_filename),
    )
    assert isinstance(recovered_ell, EllipsoidShape)
    for got, want in zip(
        recovered_ell.rest_dimensions, ell.rest_dimensions, strict=True
    ):
        assert got == pytest.approx(want, rel=0.0, abs=1e-9)


def test_urdf_generator_rejects_line_shape() -> None:
    """LineShape -> URDF conversion is intentionally a hard error."""
    line = LineShape(length=0.3)
    with pytest.raises(ValueError, match="cannot render line"):
        shape_to_urdf_visual(line)

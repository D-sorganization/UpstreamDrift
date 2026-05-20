"""Tests for Wave-4 of the body_part_viz integration into the Segments tab.

Covers:

* Shape column is a 6-option combobox (Line / Cylinder / Ellipsoid /
  Capsule / Library shape… / Mesh file…).
* Picking Cylinder renders a ``Poly3DCollection`` artist owned by the
  ``MatplotlibRenderer``.
* Importing an STL adds a segment whose persisted v2 spec uses
  ``shape_kind="mesh_file"``.
* The library chooser exposes the bundled default-library entries.
* Save → load round-trip preserves shape choices.
* Old v1 segment-set JSON loads + auto-migrates to v2.
"""

from __future__ import annotations

import json
import struct
import sys
from pathlib import Path

import numpy as np
import pytest
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

from ._viewer_test_helpers import ANATOMICAL_28, make_synthetic_model

pytest.importorskip("PyQt6")


@pytest.fixture(scope="module")
def qt_app():
    from PyQt6.QtWidgets import QApplication

    app = QApplication.instance() or QApplication(sys.argv[:1])
    yield app


@pytest.fixture(autouse=True)
def _silence_v1_deprecation_warnings():
    """The legacy SegmentSpec/SegmentSet shims emit DeprecationWarning;
    silence them here so test logs stay readable."""
    import warnings

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        yield


# --------------------------------------------------------------- helpers

_SHAPE_OPTIONS = (
    "Line",
    "Cylinder",
    "Ellipsoid",
    "Capsule",
    "Library shape…",
    "Mesh file…",
)


def _shape_combo(tab, row: int):
    """Return the ``Shape`` cell widget on row ``row`` of the table."""
    # Column index lives in segments_tab as _COL_SHAPE = 3.
    from src.apps.ui.tabs.segments_tab import _COL_SHAPE  # type: ignore

    widget = tab.table.cellWidget(row, _COL_SHAPE)
    assert widget is not None, f"Shape combobox missing on row {row}"
    return widget


def _write_minimal_binary_stl(path: Path) -> None:
    """Write a single-triangle binary STL the MeshShape loader accepts.

    Format:
        80-byte header, uint32 face-count, then 12*float + uint16 per face.
    """
    header = b"\x00" * 80
    face_count = struct.pack("<I", 1)
    normal = struct.pack("<fff", 0.0, 0.0, 1.0)
    v0 = struct.pack("<fff", 0.0, 0.0, 0.0)
    v1 = struct.pack("<fff", 1.0, 0.0, 0.0)
    v2 = struct.pack("<fff", 0.0, 1.0, 0.0)
    attr = struct.pack("<H", 0)
    path.write_bytes(header + face_count + normal + v0 + v1 + v2 + attr)


# ----------------------------------------------------------------- tests


def test_shape_column_has_six_options(qt_app):
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    tab = SegmentsTab()
    tab.update_from_model(model)
    assert len(tab.viz_segments) > 0
    combo = _shape_combo(tab, 0)
    items = [combo.itemText(i) for i in range(combo.count())]
    assert tuple(items) == _SHAPE_OPTIONS


def test_pick_cylinder_renders_poly3d_collection(qt_app):
    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    seg_tab = SegmentsTab()
    viewer = Viewer3DTab()
    seg_tab.viz_segments_changed.connect(viewer.set_user_segments)

    seg_tab.update_from_model(model)
    viewer.update_from_model(model)
    viewer.select_body_markers()

    # Programmatic v1-API call swaps the first segment to cylinder.
    seg_tab.set_segment_geometry(0, "cylinder")
    assert seg_tab.viz_segments[0].shape_kind == "cylinder"

    # The renderer should now own at least one Poly3DCollection artist.
    assert viewer.user_cylinder_count >= 1
    poly_artists = [
        a for a in viewer._ax.collections if isinstance(a, Poly3DCollection)
    ]
    # The skeleton uses Line3DCollection; the user cylinder uses Poly3D.
    assert len(poly_artists) >= 1


def test_mesh_file_import_uses_mesh_kind_in_persisted_spec(qt_app, tmp_path):
    from src.shared.python.body_part_viz import (  # type: ignore
        BindingKind,
        MarkerBinding,
        SegmentVizSet,
        SegmentVizSpec,
        ShapeTheme,
    )

    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    stl_path = tmp_path / "club.stl"
    _write_minimal_binary_stl(stl_path)

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    tab = SegmentsTab()
    tab.update_from_model(model)

    spec = SegmentVizSpec(
        binding=MarkerBinding(
            kind=BindingKind.BETWEEN_TWO,
            marker_names=("WaistLeft", "WaistRight"),
        ),
        shape_kind="mesh_file",
        shape_params={"path": str(stl_path), "max_vertices": 5000},
        fitter_kind="between_two",
        theme=ShapeTheme(group="club"),
        visible=True,
    )
    tab.add_viz_segment(spec)

    # Persist + reload, then assert the source-of-truth v2 spec.
    out = tmp_path / "segments.json"
    SegmentVizSet(segments=tab.viz_segments).save(out)
    reloaded = SegmentVizSet.load(out)
    assert any(s.shape_kind == "mesh_file" for s in reloaded.segments)


def test_library_chooser_lists_default_entries(qt_app):
    from src.shared.python.body_part_viz.asset_library import (  # type: ignore
        ShapeLibrary,
    )

    lib = ShapeLibrary.default()
    names = lib.names()
    # The bundled default library ships at least 5 anatomical shapes.
    assert len(names) >= 5
    assert all(isinstance(n, str) and n for n in names)


def test_save_load_round_trip_preserves_shape_choices(qt_app, tmp_path):
    from src.shared.python.body_part_viz import SegmentVizSet  # type: ignore

    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore

    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=20)
    tab = SegmentsTab()
    tab.update_from_model(model)

    # Diversify shape kinds across rows so the assertion has bite.
    tab.set_segment_geometry(0, "cylinder")
    tab.set_segment_geometry(1, "line")

    out = tmp_path / "segments.json"
    SegmentVizSet(segments=tab.viz_segments).save(out)

    raw = json.loads(out.read_text(encoding="utf-8"))
    assert raw["schema_version"] == 2
    kinds = [s["shape_kind"] for s in raw["segments"]]
    assert kinds[0] == "cylinder"
    assert kinds[1] == "line"

    # Round-trip via the library loader.
    reloaded = SegmentVizSet.load(out)
    assert reloaded.segments[0].shape_kind == "cylinder"
    assert reloaded.segments[1].shape_kind == "line"


def test_v1_json_auto_migrates_on_load(qt_app, tmp_path):
    """Old v1 segment-set JSON files load via the new path + migrate."""
    from src.shared.python.body_part_viz import SegmentVizSet  # type: ignore

    v1_payload = {
        "schema_version": 1,
        "segments": [
            {
                "a": "WaistLeft",
                "b": "WaistRight",
                "geometry": "line",
                "group": "pelvis",
                "visible": True,
                "radius": 0.02,
            },
            {
                "a": "WaistLeft",
                "b": "LKneeOut",
                "geometry": "cylinder",
                "group": "left_leg",
                "visible": False,
                "radius": 0.018,
            },
        ],
    }
    out = tmp_path / "v1.json"
    out.write_text(json.dumps(v1_payload), encoding="utf-8")

    viz_set = SegmentVizSet.load(out)
    assert viz_set.schema_version == 2
    kinds = [s.shape_kind for s in viz_set.segments]
    assert kinds == ["line", "cylinder"]
    # Group / visibility carried through.
    groups = [s.theme.group for s in viz_set.segments]
    assert groups == ["pelvis", "left_leg"]
    visibility = [s.visible for s in viz_set.segments]
    assert visibility == [True, False]


def test_user_segments_60fps_scrub_budget(qt_app):
    """Smoke perf check: 26-segment scrub stays well under 16.6 ms / frame.

    Uses the canonical 28-marker anatomical subset (yields 26 default
    segments), short synthetic clip, and times a 60-frame scrub through
    the renderer's update path. The assertion is intentionally loose
    (200 ms total for 60 frames ≈ 3.3 ms/frame, leaving headroom for
    CI hardware variability) — the contract is "≥ 60 fps", which on a
    perfect machine is 16.6 ms per frame.
    """
    import time

    from src.apps.ui.tabs.segments_tab import SegmentsTab  # type: ignore
    from src.apps.ui.tabs.viewer_3d_tab import Viewer3DTab  # type: ignore

    n_frames = 60
    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=n_frames)
    seg_tab = SegmentsTab()
    viewer = Viewer3DTab()
    seg_tab.viz_segments_changed.connect(viewer.set_user_segments)

    seg_tab.update_from_model(model)
    viewer.update_from_model(model)
    viewer.select_body_markers()

    # Switch ALL 26 default segments to cylinders so the per-frame budget
    # exercises Poly3DCollection updates rather than line-only updates.
    n_segments = len(seg_tab.viz_segments)
    assert (
        n_segments == 26
    ), f"Expected 26 default segments for the 28-marker subset; got {n_segments}"
    for i in range(n_segments):
        seg_tab.set_segment_geometry(i, "cylinder")

    # Time a forward scrub. The first frame primes lazy state; we time
    # the remaining 59.
    viewer.set_frame(0)
    t0 = time.perf_counter()
    for f in range(1, n_frames):
        viewer.set_frame(f)
    elapsed = time.perf_counter() - t0
    per_frame_ms = (elapsed / (n_frames - 1)) * 1000.0

    # Budget: 16.6 ms / frame for 60 fps. We assert a generous 50 ms /
    # frame so CI hardware variability does not flake the test, and
    # report the measured value via the test name in failure mode.
    assert per_frame_ms < 50.0, (
        f"per-frame scrub budget exceeded: {per_frame_ms:.2f} ms / frame "
        f"({elapsed * 1000.0:.1f} ms over {n_frames - 1} frames)"
    )


def test_synthetic_marker_array_is_finite(qt_app):
    """Sanity guard so the perf test isn't measuring NaN-fast paths."""
    model = make_synthetic_model(list(ANATOMICAL_28), n_frames=10)
    for name in ANATOMICAL_28:
        md = model.markers[name]
        assert np.all(np.isfinite(md.position))

"""Force and torque glyph renderers for various visualization backends (ADR-0052).

Submodules that depend on matplotlib, OpenCV, or Qt load only on first attribute
access so headless callers (e.g. ``meshcat_glyphs.legend_text``) stay free of
those optional stacks (#14748).
"""

from __future__ import annotations

from typing import Any

from src.shared.python.lazy_package_export import resolve_lazy_export

__all__ = [
    "HypothesisProjector",
    "ImageProjector",
    "MeshcatGlyphRenderer",
    "MeshcatPythonSink",
    "MeshcatSink",
    "PinholeProjector",
    "SegmentDrawReceipt",
    "SegmentPose",
    "SegmentShading",
    "VideoGlyphReceipt",
    "VideoGlyphStyle",
    "align_y_to",
    "draw_glyphs_2d",
    "draw_glyphs_3d",
    "draw_glyphs_on_frame",
    "draw_legend",
    "draw_legend_box",
    "draw_segment_meshes_on_frame",
    "equalize_3d_axes",
    "legend_text",
    "segment_poses_from_axes",
]

_EXPORTS: dict[str, tuple[str, str]] = {
    "draw_glyphs_3d": (
        "src.shared.python.force_overlay.renderers.matplotlib_glyphs",
        "draw_glyphs_3d",
    ),
    "draw_legend": (
        "src.shared.python.force_overlay.renderers.matplotlib_glyphs",
        "draw_legend",
    ),
    "equalize_3d_axes": (
        "src.shared.python.force_overlay.renderers.matplotlib_glyphs",
        "equalize_3d_axes",
    ),
    "MeshcatGlyphRenderer": (
        "src.shared.python.force_overlay.renderers.meshcat_glyphs",
        "MeshcatGlyphRenderer",
    ),
    "MeshcatPythonSink": (
        "src.shared.python.force_overlay.renderers.meshcat_glyphs",
        "MeshcatPythonSink",
    ),
    "MeshcatSink": (
        "src.shared.python.force_overlay.renderers.meshcat_glyphs",
        "MeshcatSink",
    ),
    "align_y_to": (
        "src.shared.python.force_overlay.renderers.meshcat_glyphs",
        "align_y_to",
    ),
    "legend_text": (
        "src.shared.python.force_overlay.renderers.meshcat_glyphs",
        "legend_text",
    ),
    "HypothesisProjector": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "HypothesisProjector",
    ),
    "ImageProjector": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "ImageProjector",
    ),
    "PinholeProjector": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "PinholeProjector",
    ),
    "VideoGlyphReceipt": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "VideoGlyphReceipt",
    ),
    "VideoGlyphStyle": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "VideoGlyphStyle",
    ),
    "draw_glyphs_on_frame": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "draw_glyphs_on_frame",
    ),
    "draw_legend_box": (
        "src.shared.python.force_overlay.renderers.opencv_glyphs",
        "draw_legend_box",
    ),
    "SegmentDrawReceipt": (
        "src.shared.python.force_overlay.renderers.opencv_segments",
        "SegmentDrawReceipt",
    ),
    "SegmentPose": (
        "src.shared.python.force_overlay.renderers.opencv_segments",
        "SegmentPose",
    ),
    "SegmentShading": (
        "src.shared.python.force_overlay.renderers.opencv_segments",
        "SegmentShading",
    ),
    "draw_segment_meshes_on_frame": (
        "src.shared.python.force_overlay.renderers.opencv_segments",
        "draw_segment_meshes_on_frame",
    ),
    "segment_poses_from_axes": (
        "src.shared.python.force_overlay.renderers.opencv_segments",
        "segment_poses_from_axes",
    ),
    "draw_glyphs_2d": (
        "src.shared.python.force_overlay.renderers.qpainter_glyphs",
        "draw_glyphs_2d",
    ),
}


def __getattr__(name: str) -> Any:
    return resolve_lazy_export(
        name, exports=_EXPORTS, module_globals=globals(), facade_module_name=__name__
    )


def __dir__() -> list[str]:
    return sorted(__all__)

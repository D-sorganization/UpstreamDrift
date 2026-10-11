"""Force and torque glyph renderers for various visualization backends (ADR-0052)."""

from __future__ import annotations

from .matplotlib_glyphs import draw_glyphs_3d, draw_legend, equalize_3d_axes
from .meshcat_glyphs import (
    MeshcatGlyphRenderer,
    MeshcatPythonSink,
    MeshcatSink,
    align_y_to,
    legend_text,
)
from .opencv_glyphs import (
    HypothesisProjector,
    ImageProjector,
    PinholeProjector,
    VideoGlyphReceipt,
    VideoGlyphStyle,
    draw_glyphs_on_frame,
    draw_legend_box,
)
from .opencv_segments import (
    SegmentDrawReceipt,
    SegmentPose,
    SegmentShading,
    draw_segment_meshes_on_frame,
    segment_poses_from_axes,
)
from .qpainter_glyphs import draw_glyphs_2d

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

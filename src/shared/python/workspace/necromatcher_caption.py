"""Opt-in measured captions without changing source geometry or qualification."""

from collections.abc import Mapping
from dataclasses import asdict, dataclass
from fractions import Fraction
import math
from numbers import Real
from typing import Any

import numpy as np


@dataclass(frozen=True)
class CaptionOverlayOptions:
    """A versioned display recipe; None at the caller retains legacy captions."""

    style: str = "compact_research_v1"

    def __post_init__(self) -> None:
        if self.style != "compact_research_v1":
            raise ValueError("Unsupported caption style")

    def to_record(self) -> dict[str, str]:
        return {"style": self.style}

    @classmethod
    def from_record(cls, record: Mapping[str, Any]) -> "CaptionOverlayOptions":
        if not isinstance(record, Mapping) or set(record) != {"style"}:
            raise ValueError("Caption options require exactly style")
        return cls(record["style"])


@dataclass(frozen=True)
class CaptionFrame:
    """Exact source identity and display diagnostics, never a physical clock."""

    index: int
    pts: Fraction
    rms: float | None
    count: int
    shaft: bool = False
    shape_opacity: float | None = None

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or type(self.count) is not int
            or self.count < 0
        ):
            raise ValueError("Caption identities require nonnegative integer counts")
        if not isinstance(self.pts, Fraction) or type(self.shaft) is not bool:
            raise ValueError("Caption requires exact rational PTS and shaft flag")
        for value in (self.rms, self.shape_opacity):
            if value is not None and (
                isinstance(value, bool)
                or not isinstance(value, Real)
                or not math.isfinite(value)
                or value < 0
            ):
                raise ValueError(
                    "Caption diagnostics must be finite nonnegative numbers"
                )
        if self.shape_opacity is not None and self.shape_opacity > 1:
            raise ValueError("Caption opacity exceeds one")


@dataclass(frozen=True)
class CaptionLine:
    text: str
    position: tuple[int, int]
    bounds: tuple[int, int, int, int]


@dataclass(frozen=True)
class CaptionLayout:
    """Measured glyph/stroke bounds and one auditable occlusion rectangle."""

    lines: tuple[CaptionLine, ...]
    scale: float
    rectangle: tuple[int, int, int, int]

    def to_record(self) -> dict[str, Any]:
        import json

        return json.loads(json.dumps(asdict(self)))


def caption_provenance(options: CaptionOverlayOptions) -> dict[str, Any]:
    return {
        "options": options.to_record(),
        "maximum_strip_fraction": 0.2,
        "qualification": "monocular_research_hypothesis",
        "physical_time_qualified": False,
        "camera_qualified": False,
        "anatomy_qualified": False,
        "legend": {
            "G": "Green observed landmarks",
            "B": "Blue native rigid skeleton and attachment seeds, not surfaces",
            "Y": "Yellow matched-marker residuals",
            "M": "Magenta observed interior shaft fragments",
            "C": "Cyan infinite authored shaft axis, not physical endpoints",
            "surfaces": "Multiple model material colors; uncalibrated model visual proxies",
        },
        "matched_rms_definition": "Unweighted Euclidean RMS of common observed/native marker identities in pixels",
        "source_time": "Exact rational video presentation seconds; physical time unknown",
        "font": "OpenCV Hershey Simplex",
        "white_thickness": 1,
        "black_stroke_thickness": 2,
    }


def _texts(frame: CaptionFrame) -> tuple[str, ...]:
    clock = f"{frame.pts.numerator}/{frame.pts.denominator}s"
    metric = f"RMS {frame.rms:.2f}px" if frame.rms is not None else "RMS unavailable"
    lines = [
        "RESEARCH | Camera/Anatomy Unqualified",
        f"Physical Time Unknown | F{frame.index} | PTS {clock}",
        f"{metric} | G:Obs B:Rig Y:Residuals",
    ]
    extra = "M:Fragment C:Axis" if frame.shaft else ""
    if frame.shape_opacity is not None and frame.shape_opacity > 0:
        extra += (
            f" | Surfaces:Proxy a{frame.shape_opacity:.2f}"
            if extra
            else f"Surfaces:Proxy a{frame.shape_opacity:.2f}"
        )
    if extra:
        lines.append(extra + " | Uncalibrated")
    return tuple(lines)


def caption_layout(
    size: tuple[int, int], frame: CaptionFrame, options: CaptionOverlayOptions
) -> CaptionLayout:
    """Fit complete text without cropping, font shrinking below legacy minimum or letterboxing."""
    import cv2

    if not isinstance(options, CaptionOverlayOptions):
        raise TypeError("Caption layout requires typed options")
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("Caption dimensions require positive integers")
    width, height = size
    texts = _texts(frame)
    for hundredths in range(min(52, max(25, width * 100 // 1300)), 24, -1):
        scale = hundredths / 100
        metrics = [
            cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, 2) for text in texts
        ]
        row_height = max(h + baseline + 2 for (_, h), baseline in metrics)
        strip = row_height * len(texts) + 2
        if strip > height // 5 or any(w + 10 > width for (w, _), _ in metrics):
            continue
        top = height - strip
        lines = []
        for i, (text, ((w, h), baseline)) in enumerate(
            zip(texts, metrics, strict=True)
        ):
            y = top + 1 + i * row_height
            lines.append(
                CaptionLine(text, (5, y + h + 1), (4, y, w + 2, h + baseline + 2))
            )
        return CaptionLayout(tuple(lines), scale, (0, top, width, strip))
    raise ValueError("Complete research caption cannot fit source dimensions")


def draw_caption(image: np.ndarray, layout: CaptionLayout) -> None:
    import cv2

    _, top, width, strip = layout.rectangle
    if image.shape != (top + strip, width, 3) or image.dtype != np.uint8:
        raise ValueError(
            "Caption raster must match measured source dimensions and BGR8"
        )
    for line in layout.lines:
        for color, thickness in (((0, 0, 0), 2), ((255, 255, 255), 1)):
            cv2.putText(
                image,
                line.text,
                line.position,
                cv2.FONT_HERSHEY_SIMPLEX,
                layout.scale,
                color,
                thickness,
                cv2.LINE_AA,
            )


def validate_caption_manifest(
    manifest: dict[str, Any], options: CaptionOverlayOptions
) -> None:
    """Recompute complete semantic/layout metadata before owned publication/download."""
    if manifest.get("caption_overlay") != caption_provenance(options):
        raise ValueError("Caption qualification or options differ from request")
    shape = manifest.get("shape_overlay", {}).get("options", {})
    for row in manifest["frames"]:
        identity = row["frame"]
        pts = Fraction(
            identity["pts_ticks"] * identity["timebase_numerator"],
            identity["timebase_denominator"],
        )
        frame = CaptionFrame(
            row["frame_index"],
            pts,
            row["matched_rms_pixels"],
            row["matched_marker_count"],
            "shaft_overlay" in manifest,
            shape.get("opacity"),
        )
        expected = caption_layout(
            tuple(manifest["image_size"]), frame, options
        ).to_record()
        if row.get("caption_overlay") != expected:
            raise ValueError(
                "Caption layout, identity or visible qualification differs"
            )

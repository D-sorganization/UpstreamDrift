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
    authored_seed: bool = False
    restricted_seed: bool = False

    def __post_init__(self) -> None:
        if (
            type(self.index) is not int
            or self.index < 0
            or type(self.count) is not int
            or self.count < 0
        ):
            raise ValueError("Caption identities require nonnegative integer counts")
        if (
            not isinstance(self.pts, Fraction)
            or type(self.shaft) is not bool
            or type(self.authored_seed) is not bool
            or type(self.restricted_seed) is not bool
            or (self.authored_seed and self.restricted_seed)
        ):
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
            "surfaces": "Display-colored uncalibrated model visual proxies; colors are not measured materials",
        },
        "matched_rms_definition": "Unweighted Euclidean RMS of common observed/native marker identities in pixels",
        "source_time": "Exact rational video presentation seconds; physical time unknown",
        "font": "OpenCV Hershey Simplex",
        "white_thickness": 1,
        "black_stroke_thickness": 2,
    }


def authored_seed_status(fit: Mapping[str, Any]) -> bool:
    """Derive an unoptimized authored seed from canonical fit metadata only.

    Missing legacy provenance is not relabelled. Contradictory authored-operation
    records reject; callers cannot supply a display option that claims seed status.
    """
    if not isinstance(fit, Mapping):
        raise ValueError("Caption requires a canonical fit record")
    provenance = fit.get("provenance", {})
    if not isinstance(provenance, Mapping):
        raise ValueError("Caption requires canonical fit provenance")
    if provenance.get("operation") != "author_initialization":
        return False
    evidence = fit.get("evidence", {})
    if not isinstance(evidence, Mapping):
        raise ValueError("Caption requires canonical fit evidence")
    original = evidence.get("original_fit", {})
    request = provenance.get("request_options", {})
    policy = "authored_range_project_zero_slopes"
    if (
        not isinstance(original, Mapping)
        or original.get("optimizer_ran") is not False
        or original.get("converged") is not False
        or not isinstance(original.get("initialization"), Mapping)
        or original["initialization"].get("policy") != policy
        or not isinstance(request, Mapping)
        or request.get("operation") != "author_initialization"
        or not isinstance(request.get("config"), Mapping)
        or request["config"].get("initialization_policy") != policy
    ):
        raise ValueError("Authored seed caption contradicts canonical fit metadata")
    return True


def restricted_seed_status(fit: Mapping[str, Any]) -> bool:
    """Classify freshly Library-authenticated metadata, not arbitrary receipt bytes.

    The restriction payload owner alone rederives hashes, parent and scope.
    Missing legacy provenance is unchanged; contradictory mode metadata rejects.
    """
    if not isinstance(fit, Mapping):
        raise ValueError("Caption requires a canonical fit record")
    provenance = fit.get("provenance", {})
    evidence = fit.get("evidence", {})
    if not isinstance(provenance, Mapping) or not isinstance(evidence, Mapping):
        raise ValueError("Caption requires canonical fit provenance and evidence")
    request = provenance.get("request_options", {})
    original = evidence.get("original_fit", {})
    if not isinstance(request, Mapping) or not isinstance(original, Mapping):
        raise ValueError("Caption requires canonical request and original fit")
    declared = (
        provenance.get("operation") == "restrict_initialization"
        or request.get("operation") == "restrict_initialization"
        or request.get("initialization_source") == "restricted_spline"
        or original.get("initialization_source") == "restricted_spline"
        or "spline_interval_restriction" in provenance
        or "spline_restriction_prior" in provenance
    )
    if not declared:
        return False
    config = request.get("config", {})
    if (
        provenance.get("operation") != "restrict_initialization"
        or request.get("operation") != "restrict_initialization"
        or request.get("initialization_source") != "restricted_spline"
        or not isinstance(config, Mapping)
        or config.get("initialization_policy") != "strict"
        or not isinstance(provenance.get("spline_interval_restriction"), Mapping)
        or not isinstance(provenance.get("spline_restriction_prior"), Mapping)
        or original.get("optimizer_ran") is not False
        or original.get("converged") is not False
        or "initialization" not in original
        or original["initialization"] is not None
    ):
        raise ValueError("Restricted seed caption contradicts canonical fit metadata")
    return True


def _texts(frame: CaptionFrame) -> tuple[str, ...]:
    clock = f"{frame.pts.numerator}/{frame.pts.denominator}s"
    metric = f"RMS {frame.rms:.2f}px" if frame.rms is not None else "RMS unavailable"
    lines = [
        "RESEARCH | Camera/Anatomy Unqualified",
        f"Physical Time Unknown | F{frame.index} | PTS {clock}",
        f"{metric} | G:Obs B:Rig Y:Residuals",
    ]
    if frame.authored_seed:
        lines.insert(0, "UNOPTIMIZED AUTHORED RESEARCH SEED")
    elif frame.restricted_seed:
        lines.insert(0, "UNOPTIMIZED RESTRICTED RESEARCH SEED")
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
    manifest: dict[str, Any],
    options: CaptionOverlayOptions,
    fit: Mapping[str, Any] | None = None,
) -> None:
    """Recompute complete semantic/layout metadata before owned publication/download."""
    if manifest.get("caption_overlay") != caption_provenance(options):
        raise ValueError("Caption qualification or options differ from request")
    seed = authored_seed_status(fit) if fit is not None else False
    if "authored_initialization_seed" in manifest:
        if manifest["authored_initialization_seed"] is not True or not seed:
            raise ValueError("Seed caption requires authenticated canonical fit")
    elif seed:
        raise ValueError("Authored seed manifest is missing its visible status")
    restricted = restricted_seed_status(fit) if fit is not None else False
    if "restricted_initialization_seed" in manifest:
        if manifest["restricted_initialization_seed"] is not True or not restricted:
            raise ValueError("Restricted seed requires authenticated canonical fit")
    elif restricted:
        raise ValueError("Restricted seed manifest is missing its visible status")
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
            seed,
            restricted,
        )
        expected = caption_layout(
            tuple(manifest["image_size"]), frame, options
        ).to_record()
        if row.get("caption_overlay") != expected:
            raise ValueError(
                "Caption layout, identity or visible qualification differs"
            )

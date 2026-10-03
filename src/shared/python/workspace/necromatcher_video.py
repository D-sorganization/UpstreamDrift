"""Source-sized native research overlays with immutable source-clock provenance.

Exported motion follows source presentation timestamps, never an inferred physical
clock. Anatomical seed markers are model-conditioned and uncalibrated; missing
attachment offsets and observations are omitted rather than reconstructed here.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence, cast

import numpy as np

from src.shared.python.motion_matching import default_body_segments
from src.shared.python.motion_matching.historical_fit import (
    AuthoredShaftAxis,
    ShaftAxisAssessment,
    ShaftAxisEvidence,
    ShaftAxisResidualTerm,
    resolve_authored_shaft_axis,
)
from src.shared.python.body_part_viz.overlay_options import ShapeOverlayOptions
from .necromatcher_shape_overlay import NativeShapeOverlay
from .necromatcher_caption import (
    CaptionFrame,
    CaptionOverlayOptions,
    caption_layout,
    caption_provenance,
    draw_caption,
)
from .necromatcher import NecromatcherLibrary
from .necromatcher_native import NativeFitBinding, load_native_fit_binding
from .necromatcher_review import CaptureReview
from .necromatcher_shaft_evidence import BoundShaftEvidence, bind_fit_shaft_evidence

_OBSERVED = (90, 230, 90)
_NATIVE = (255, 170, 30)
_RESIDUAL = (40, 210, 255)
_SHAFT_OBSERVED = (230, 60, 230)
_SHAFT_NATIVE = (230, 230, 40)


@dataclass(frozen=True)
class ShaftVideoOverlay:
    """Prepared sparse overlay; export always freshly binds caller evidence."""

    bound: BoundShaftEvidence
    axis: AuthoredShaftAxis
    assessment: ShaftAxisAssessment

    def to_record(self) -> dict[str, Any]:
        record = {
            "schema": "necromatcher/shaft-video-overlay/1",
            "evidence": self.bound.evidence.to_record(),
            "evidence_sha256": self.bound.evidence.sha256,
            "source_clock_sha256": self.bound.source_clock_sha256,
            "axis": self.axis.to_record(),
            "assessment": asdict(self.assessment),
            "uncertainty_calibrated": False,
            "physical_geometry_qualified": False,
            "legend": "Magenta: observed interior fragment; cyan: infinite authored axis",
        }
        return cast(dict[str, Any], json.loads(json.dumps(record, allow_nan=False)))


@dataclass(frozen=True)
class VideoOverlayLayers:
    """Optional display layers share one existing binding and saved camera."""

    shaft: ShaftVideoOverlay | None = None
    shapes: NativeShapeOverlay | None = None
    captions: CaptionOverlayOptions | None = None


def prepare_shaft_overlay(
    library: NecromatcherLibrary,
    binding: NativeFitBinding,
    evidence: ShaftAxisEvidence,
) -> ShaftVideoOverlay:
    """Rebind PNG/PTS/camera and resolve geometry using one existing native plant."""
    bound = bind_fit_shaft_evidence(library, binding.fit, evidence)
    indices = binding.fit["frame_indices"]
    if any(item.frame_index not in indices for item in evidence.frames):
        raise ValueError("Shaft review frames must belong to the bound exported fit")
    axis = resolve_authored_shaft_axis(
        binding.definition_bytes, binding.plant.plant_sha
    )
    poses = np.asarray(
        [binding.fit["q"][indices.index(item.frame_index)] for item in evidence.frames]
    )
    camera, _ = binding.review_inputs()
    # assess is raw/unweighted; this visibility option is not applied to diagnostics.
    term = ShaftAxisResidualTerm(evidence, axis, bound.source_clock_sha256, 0.5)
    return ShaftVideoOverlay(bound, axis, term.assess(binding.plant, camera, poses))


def clip_infinite_axis(
    points: np.ndarray, size: tuple[int, int]
) -> tuple[np.ndarray, np.ndarray] | None:
    """Clip an infinite projected axis, never its authored reference segment."""
    points = np.asarray(points, dtype=float)
    if points.shape != (2, 2) or not np.isfinite(points).all():
        raise ValueError("Shaft projection requires two finite image points")
    if len(size) != 2 or any(type(value) is not int or value <= 0 for value in size):
        raise ValueError("Shaft raster dimensions must be positive integers")
    direction = points[1] - points[0]
    length = float(np.linalg.norm(direction))
    if not np.isfinite(length) or length <= 1e-10:
        raise ValueError("Projected shaft axis is degenerate")
    normal = np.array([-direction[1], direction[0]]) / length
    offset = float(normal @ points[0])
    if not np.isfinite(offset):
        raise ValueError("Shaft line coefficients must remain finite")
    width, height = size
    candidates: list[np.ndarray] = []
    if abs(normal[1]) > 1e-12:
        candidates.extend(
            np.array([x, (offset - normal[0] * x) / normal[1]]) for x in (0, width - 1)
        )
    if abs(normal[0]) > 1e-12:
        candidates.extend(
            np.array([(offset - normal[1] * y) / normal[0], y]) for y in (0, height - 1)
        )
    valid: list[np.ndarray] = []
    for point in candidates:
        if (
            np.isfinite(point).all()
            and -1e-9 <= point[0] <= width - 1 + 1e-9
            and -1e-9 <= point[1] <= height - 1 + 1e-9
            and not any(np.linalg.norm(point - previous) <= 1e-9 for previous in valid)
        ):
            valid.append(np.clip(point, [0, 0], [width - 1, height - 1]))
    return (valid[0], valid[1]) if len(valid) >= 2 else None


def _draw_shaft(
    binding: NativeFitBinding,
    overlay: ShaftVideoOverlay,
    index: int,
    pose: np.ndarray,
    image: np.ndarray,
) -> tuple[dict[str, Any], list[str]]:
    camera, _ = binding.review_inputs()
    axis = overlay.axis
    projected = camera.project(
        binding.plant.marker_positions(
            pose,
            {
                "shaft_a": (axis.body, axis.point_a_m),
                "shaft_b": (axis.body, axis.point_b_m),
            },
        )
    )
    clipped = clip_infinite_axis(projected, (image.shape[1], image.shape[0]))
    if clipped is not None:
        _line(image, clipped[0], clipped[1], _SHAFT_NATIVE)
    positions = [item.frame_index for item in overlay.bound.evidence.frames]
    record: dict[str, Any] = {
        "status": "unreviewed",
        "raw_rms_pixels": None,
        "angular_error_deg": None,
        "native_axis_visible": clipped is not None,
        "projected_axis_reference_pixels": projected.tolist(),
        "clipped_infinite_axis_pixels": [point.tolist() for point in clipped]
        if clipped is not None
        else None,
    }
    if index in positions:
        position = positions.index(index)
        segment = overlay.bound.evidence.frames[position].segment
        record["status"] = segment.status
        errors = overlay.assessment.perpendicular_errors_pixels[position]
        record["angular_error_deg"] = overlay.assessment.angular_errors_deg[position]
        if errors is not None:
            record["perpendicular_errors_pixels"] = list(errors)
            record["raw_rms_pixels"] = float(np.sqrt(np.mean(np.square(errors))))
        if segment.points_px is not None:
            a, b = np.asarray(segment.points_px)
            _line(image, a, b, _SHAFT_OBSERVED, 2)
            _draw_points(image, {"a": a, "b": b}, _SHAFT_OBSERVED)
    metric = (
        f"Shaft raw RMS {record['raw_rms_pixels']:.2f} px | Axis angle {record['angular_error_deg']:.2f} deg"
        if record["raw_rms_pixels"] is not None
        else f"Shaft observation {record['status']} | Raw line RMS unavailable"
    )
    return record, [
        metric,
        "Magenta: fragment | Cyan: infinite authored axis (uncalibrated)",
    ]


def source_frame_rate(
    indices: Sequence[int], frames: Sequence[dict[str, Any]]
) -> Fraction:
    """Return uniform source PTS rate; reject gaps, irregular time and invalid types."""
    if (
        len(indices) < 2
        or len(indices) != len(frames)
        or any(type(i) is not int or i < 0 for i in indices)
    ):
        raise ValueError("Video requires at least two contiguous source indices")
    if any(b != a + 1 for a, b in zip(indices, indices[1:], strict=False)):
        raise ValueError("Video source indices must be contiguous")
    times = []
    for frame in frames:
        values = [
            frame.get(name)
            for name in ("pts_ticks", "timebase_numerator", "timebase_denominator")
        ]
        if (
            any(type(value) is not int for value in values)
            or not isinstance(values[1], int)
            or values[1] <= 0
            or not isinstance(values[2], int)
            or values[2] <= 0
        ):
            raise ValueError("Video requires exact positive source timebase integers")
        ticks, numerator, denominator = cast(list[int], values)
        times.append(Fraction(ticks * numerator, denominator))
    steps = [b - a for a, b in zip(times, times[1:], strict=False)]
    if steps[0] <= 0 or any(step != steps[0] for step in steps):
        raise ValueError("Video source PTS must be strictly increasing and uniform")
    return 1 / steps[0]


def _anatomical_attachments(
    binding: NativeFitBinding,
) -> tuple[dict[str, Any], list[str]]:
    records = binding.fit["provenance"]["native_definition"].get(
        "marker_attachments", {}
    )
    available, missing = {}, []
    for name, record in records.items():
        if record.get("offset_m") is None:
            missing.append(name)
            continue
        offset = np.asarray(record["offset_m"], dtype=float)
        if (
            offset.shape != (3,)
            or not np.isfinite(offset).all()
            or not isinstance(record.get("body"), str)
        ):
            raise ValueError(
                "Anatomical attachments require declared bodies and finite offsets"
            )
        available[name] = (record["body"], offset.tolist())
    return available, sorted(missing)


def _pixels(points: np.ndarray, labels: Sequence[str]) -> dict[str, np.ndarray]:
    if points.shape != (len(labels), 2) or not np.isfinite(points).all():
        raise ValueError("Projected native points must be finite image XY")
    return dict(zip(labels, points, strict=True))


def _draw_points(
    image: np.ndarray, points: dict[str, np.ndarray], color: tuple[int, int, int]
) -> None:
    import cv2

    height, width = image.shape[:2]
    for point in points.values():
        if 0 <= point[0] < width and 0 <= point[1] < height:
            cv2.circle(
                image, tuple(np.rint(point).astype(int)), 3, color, -1, cv2.LINE_AA
            )


def _caption(image: np.ndarray, lines: Sequence[str]) -> None:
    import cv2

    width = image.shape[1]
    scale = min(0.52, max(0.25, width / 1300))
    line_height = max(11, int(25 * scale))
    for index, line in enumerate(lines):
        position = (4, image.shape[0] - (len(lines) - index - 1) * line_height - 5)
        cv2.putText(
            image,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (0, 0, 0),
            3,
            cv2.LINE_AA,
        )
        cv2.putText(
            image,
            line,
            position,
            cv2.FONT_HERSHEY_SIMPLEX,
            scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )


def _line(
    image: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    color: tuple[int, int, int],
    thickness: int = 1,
) -> None:
    import cv2

    if (
        not np.isfinite(np.concatenate([a, b])).all()
        or np.max(np.abs(np.concatenate([a, b]))) > 2**30
    ):
        raise ValueError("Projection exceeds finite raster coordinate bounds")
    valid, start, end = cv2.clipLine(
        (0, 0, image.shape[1], image.shape[0]),
        tuple(np.rint(a).astype(int)),
        tuple(np.rint(b).astype(int)),
    )
    if valid:
        cv2.line(image, start, end, color, thickness, cv2.LINE_AA)


def _rigid_segments(binding: NativeFitBinding) -> list[dict[str, str]]:
    definition = binding.fit["provenance"]["native_definition"]
    bodies = {body["name"] for body in definition["bodies"] if body["name"] != "world"}
    return [
        {"a": joint["parent"], "b": joint["child"]}
        for joint in definition["joints"]
        if joint["parent"] in bodies
        and joint["child"] in bodies
        and joint["parent"] != joint["child"]
    ]


def _draw_rigid_skeleton(
    binding: NativeFitBinding, pose: np.ndarray, image: np.ndarray
) -> None:
    edges = _rigid_segments(binding)
    names = sorted({name for edge in edges for name in edge.values()})
    poses = binding.plant.frame_poses({name: (name, (0, 0, 0)) for name in names}, pose)
    if set(poses) != set(names):
        raise ValueError(
            "Native body origins differ from declared joint-tree identities"
        )
    camera, _ = binding.review_inputs()
    points = _pixels(
        camera.project(np.array([poses[name][1] for name in names])), names
    )
    for edge in edges:
        a, b = points[edge["a"]], points[edge["b"]]
        if np.linalg.norm(a - b) > 1e-9:
            _line(image, a, b, _NATIVE, 2)


def _observations(row: dict[str, Any]) -> dict[str, np.ndarray]:
    observed = {}
    if row["observation"]["status"] != "missing":
        for name, point in row["observation"]["landmarks"].items():
            xy = np.asarray(
                [point["x"] * row["image_width"], point["y"] * row["image_height"]],
                dtype=float,
            )
            if not np.isfinite(xy).all():
                raise ValueError("Observed image landmarks must be finite")
            observed[name] = xy
    return observed


def _render(
    binding: NativeFitBinding,
    review: CaptureReview,
    index: int,
    anatomy: dict[str, Any],
    shaft: ShaftVideoOverlay | None = None,
    shapes: NativeShapeOverlay | None = None,
    captions: CaptionOverlayOptions | None = None,
) -> tuple[np.ndarray, dict[str, Any]]:
    import cv2

    row = review.frame(index)
    original_png = review.image(index)
    image = cv2.imdecode(np.frombuffer(original_png, np.uint8), cv2.IMREAD_COLOR)
    if image is None or image.shape != (row["image_height"], row["image_width"], 3):
        raise ValueError("Source PNG must decode at its declared original dimensions")
    position = binding.fit["frame_indices"].index(index)
    if row["frame"] != binding.fit["frames"][position]:
        raise ValueError("Overlay source frame differs from bound fit identity")
    camera, attachments = binding.review_inputs()
    pose = np.asarray(binding.fit["q"][position])
    native = _pixels(
        camera.project(binding.plant.marker_positions(pose, attachments)),
        tuple(attachments),
    )
    anatomical = (
        _pixels(
            camera.project(binding.plant.marker_positions(pose, anatomy)),
            tuple(anatomy),
        )
        if anatomy
        else {}
    )
    if shapes is not None:
        image = shapes.composite(binding, pose, image)
    _draw_rigid_skeleton(binding, pose, image)
    for segment in default_body_segments(tuple(anatomical)):
        _line(image, anatomical[segment.a], anatomical[segment.b], _NATIVE, 2)
    observed = _observations(row)
    shared = native.keys() & observed.keys()
    errors = [float(np.linalg.norm(native[name] - observed[name])) for name in shared]
    for name in shared:
        _line(image, native[name], observed[name], _RESIDUAL)
    _draw_points(image, anatomical | native, _NATIVE)
    _draw_points(image, observed, _OBSERVED)
    rms = float(np.sqrt(np.mean(np.square(errors)))) if errors else None
    shaft_record, shaft_lines = _shaft_frame(
        binding, shaft, index, pose, image, original_png
    )
    if shapes is not None and shapes.options.opacity > 0:
        shaft_lines.append(
            f"Model visual proxies | Opacity {shapes.options.opacity:.2f} | Anatomy unqualified"
        )
    identity = row["frame"]
    pts = Fraction(
        identity["pts_ticks"] * identity["timebase_numerator"],
        identity["timebase_denominator"],
    )
    compact = None
    if captions is None:
        _render_caption(image, index, pts, rms, len(errors), shaft_lines)
    else:
        compact = caption_layout(
            (image.shape[1], image.shape[0]),
            CaptionFrame(
                index,
                pts,
                rms,
                len(errors),
                shaft is not None,
                shapes.options.opacity if shapes else None,
            ),
            captions,
        )
        draw_caption(image, compact)
    record: dict[str, Any] = {
        "frame_index": index,
        "frame": identity,
        "original_png_sha256": hashlib.sha256(original_png).hexdigest(),
        "matched_marker_count": len(errors),
        "matched_rms_pixels": rms,
    }
    if shaft is not None:
        record["shaft_overlay"] = shaft_record
    if compact is not None:
        record["caption_overlay"] = compact.to_record()
    return image, record


def _shaft_frame(
    binding: NativeFitBinding,
    shaft: ShaftVideoOverlay | None,
    index: int,
    pose: np.ndarray,
    image: np.ndarray,
    original_png: bytes,
) -> tuple[dict[str, Any], list[str]]:
    if shaft is None:
        return {}, []
    for item in shaft.bound.evidence.frames:
        if item.frame_index == index and (
            item.png_sha256 != "sha256:" + hashlib.sha256(original_png).hexdigest()
        ):
            raise ValueError("Shaft original PNG changed during overlay render")
    return _draw_shaft(binding, shaft, index, pose, image)


def _render_caption(
    image: np.ndarray,
    index: int,
    pts: Fraction,
    rms: float | None,
    count: int,
    shaft_lines: list[str],
) -> None:
    metric = (
        f"Matched RMS {rms:.2f} px ({count} markers)"
        if rms is not None
        else "Matched RMS unavailable (no common observed markers)"
    )
    _caption(
        image,
        [
            f"MONOCULAR RESEARCH | Source frame {index} | PTS {float(pts):.3f}s",
            "Camera/anatomy unqualified | Physical time unknown",
            metric,
            "Green: observations | Blue: native rig/seeds | Yellow: residuals",
            *shaft_lines,
        ],
    )


def _verify_video(path: Path, count: int, size: tuple[int, int]) -> None:
    import cv2

    reader = cv2.VideoCapture(str(path))
    decoded = 0
    try:
        if not reader.isOpened():
            raise ValueError("Exported video codec cannot be reopened")
        while True:
            ok, image = reader.read()
            if not ok:
                break
            if (image.shape[1], image.shape[0]) != size:
                raise ValueError("Exported video changed original source dimensions")
            decoded += 1
    finally:
        reader.release()
    if decoded != count:
        raise ValueError("Exported video decoded frame count differs from source")


def _sha(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def export_fit_video(
    library: NecromatcherLibrary,
    fit_id: str,
    destination: Path,
    *,
    selected_frames: Sequence[int] = (),
    shaft_evidence: ShaftAxisEvidence | None = None,
    shape_overlay: ShapeOverlayOptions | None = None,
    caption_overlay: CaptionOverlayOptions | None = None,
) -> dict[str, Any]:
    """Publish a new MP4/PNG/manifest directory only after complete codec verification.

    Preconditions: hash-bound native fit, contiguous uniform source PTS and selected
    PNG indices inside that fit. Postcondition: source bytes remain untouched and
    output hashes, frame identities and unqualified scientific status are recorded.
    """
    _validate_overlay_options(shape_overlay, caption_overlay)
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    binding = load_native_fit_binding(library, fit_id)
    indices, frames = binding.fit["frame_indices"], binding.fit["frames"]
    rate = source_frame_rate(indices, frames)
    if any(type(index) is not int or index not in indices for index in selected_frames):
        raise ValueError("Selected PNG frames must belong to the bound fit")
    anatomy, missing, overlays = _prepare_export_layers(
        library, binding, shaft_evidence, shape_overlay, caption_overlay
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="necromatcher-video-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        manifest = _write_export(
            binding,
            library,
            staging,
            rate,
            set(selected_frames),
            anatomy,
            missing,
            overlays,
        )
        _verify_export_binding(library, binding, overlays.shaft)
        _publish(staging, destination)
    return manifest


def export_fit_stills(
    library: NecromatcherLibrary,
    fit_id: str,
    destination: Path,
    *,
    selected_frames: Sequence[int],
    shaft_evidence: ShaftAxisEvidence | None = None,
    shape_overlay: ShapeOverlayOptions | None = None,
    caption_overlay: CaptionOverlayOptions | None = None,
) -> dict[str, Any]:
    """Publish selected source-sized PNGs using canonical video composition.

    Selection is nonempty, unique and source-bound. Only selected poses are
    rendered; no codec, frame-rate inference or physical-clock assumption occurs.
    Exact source identities/PTS and lossless output hashes accompany the manifest.
    """
    _validate_overlay_options(shape_overlay, caption_overlay)
    selected = tuple(selected_frames)
    if (
        not selected
        or any(type(index) is not int for index in selected)
        or len(set(selected)) != len(selected)
    ):
        raise ValueError(
            "Selected still frames must be nonempty unique integer indices"
        )
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    binding = load_native_fit_binding(library, fit_id)
    if any(index not in binding.fit["frame_indices"] for index in selected):
        raise ValueError("Selected still frames must belong to the bound fit")
    anatomy, missing, overlays = _prepare_export_layers(
        library, binding, shaft_evidence, shape_overlay, caption_overlay
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    with TemporaryDirectory(
        prefix="necromatcher-stills-", dir=destination.parent
    ) as temporary:
        staging = Path(temporary) / "export"
        staging.mkdir()
        manifest = _write_stills(
            binding, library, staging, selected, anatomy, missing, overlays
        )
        _verify_export_binding(library, binding, overlays.shaft)
        _publish(staging, destination)
    return manifest


def _write_stills(
    binding: NativeFitBinding,
    library: NecromatcherLibrary,
    staging: Path,
    selected: tuple[int, ...],
    anatomy: dict[str, Any],
    missing: list[str],
    overlays: VideoOverlayLayers,
) -> dict[str, Any]:
    records, pngs, times = [], [], []
    with CaptureReview(library, binding.fit["capture_id"]) as review:
        first = review.frame(selected[0])
        size = [first["image_width"], first["image_height"]]
        for index in selected:
            image, record = _render_layers(binding, review, index, anatomy, overlays)
            if list(image.shape[:2][::-1]) != size:
                raise ValueError("Selected still changed original source dimensions")
            frame = record["frame"]
            pts = Fraction(
                frame["pts_ticks"] * frame["timebase_numerator"],
                frame["timebase_denominator"],
            )
            records.append(record)
            times.append(
                {
                    "frame_index": index,
                    "numerator": pts.numerator,
                    "denominator": pts.denominator,
                }
            )
            pngs.append(_save_png(staging / f"frame-{index:06d}.png", image))
    manifest = _export_manifest(
        binding,
        library,
        records,
        pngs,
        anatomy,
        missing,
        overlays,
        {"schema": "necromatcher/source-overlay-stills/1", "image_size": size},
    )
    manifest["frame_pts"] = times
    _save_manifest(staging, manifest)
    return manifest


def _save_png(path: Path, image: np.ndarray) -> dict[str, str]:
    import cv2

    if not cv2.imwrite(str(path), image):
        raise ValueError("Selected overlay PNG could not be encoded")
    restored = cv2.imread(str(path), cv2.IMREAD_COLOR)
    if restored is None or not np.array_equal(restored, image):
        raise ValueError("Selected overlay PNG failed lossless readback verification")
    return {"path": path.name, "sha256": _sha(path)}


def _publish(staging: Path, destination: Path) -> None:
    """Reserve output exclusively; link immutable bytes and completion manifest last."""
    destination.mkdir()
    owned = []
    try:
        files = sorted(staging.iterdir(), key=lambda path: path.name == "manifest.json")
        for source in files:
            target = destination / source.name
            os.link(source, target)
            owned.append(target)
    except OSError:
        for target in owned:
            target.unlink()
        if not any(destination.iterdir()):
            destination.rmdir()
        raise


def _render_layers(
    binding: NativeFitBinding,
    review: CaptureReview,
    index: int,
    anatomy: dict[str, Any],
    overlays: VideoOverlayLayers | None,
) -> tuple[np.ndarray, dict[str, Any]]:
    shaft = overlays.shaft if overlays else None
    shapes = overlays.shapes if overlays else None
    if overlays and overlays.captions:
        return _render(
            binding, review, index, anatomy, shaft, shapes, overlays.captions
        )
    return _render(binding, review, index, anatomy, shaft, shapes)


def _write_export(
    binding: NativeFitBinding,
    library: NecromatcherLibrary,
    staging: Path,
    rate: Fraction,
    selected: set[int],
    anatomy: dict[str, Any],
    missing: list[str],
    overlays: VideoOverlayLayers | None = None,
) -> dict[str, Any]:
    import cv2

    records, pngs = [], []
    with CaptureReview(library, binding.fit["capture_id"]) as review:
        first = review.frame(binding.fit["frame_indices"][0])
        size = (first["image_width"], first["image_height"])
        writer = cv2.VideoWriter(
            str(staging / "overlay.mp4"),
            cv2.VideoWriter.fourcc(*"mp4v"),
            float(rate),
            size,
        )
        try:
            if not writer.isOpened():
                raise ValueError("MP4 encoder could not open")
            for index in binding.fit["frame_indices"]:
                image, record = _render_layers(
                    binding, review, index, anatomy, overlays
                )
                writer.write(image)
                records.append(record)
                if index in selected:
                    name = f"frame-{index:06d}.png"
                    pngs.append(_save_png(staging / name, image))
        finally:
            writer.release()
    _verify_video(staging / "overlay.mp4", len(records), size)
    media = {
        "schema": "necromatcher/source-overlay-video/1",
        "source_frame_rate": {
            "numerator": rate.numerator,
            "denominator": rate.denominator,
        },
        "image_size": list(size),
        "video": {
            "path": "overlay.mp4",
            "codec": "mp4v",
            "sha256": _sha(staging / "overlay.mp4"),
            "bytes": (staging / "overlay.mp4").stat().st_size,
        },
    }
    manifest = _export_manifest(
        binding, library, records, pngs, anatomy, missing, overlays, media
    )
    _save_manifest(staging, manifest)
    return manifest


def _export_manifest(
    binding: NativeFitBinding,
    library: NecromatcherLibrary,
    records: list[dict[str, Any]],
    pngs: list[dict[str, str]],
    anatomy: dict[str, Any],
    missing: list[str],
    overlays: VideoOverlayLayers | None,
    media: dict[str, Any],
) -> dict[str, Any]:
    import cv2

    shaft = overlays.shaft if overlays else None
    shapes = overlays.shapes if overlays else None
    manifest = {
        "schema": media["schema"],
        "fit_id": binding.fit_id,
        "fit_hash": binding.fit_hash,
        "model_id": binding.model_id,
        "model_hash": binding.model_hash,
        "capture_id": binding.fit["capture_id"],
        "capture_hash": binding.fit["capture_hash"],
        "source_video_sha256": library.load_asset(
            binding.fit["capture_id"]
        ).metadata.get("source_sha256"),
        "qualification": "monocular_research_hypothesis",
        "camera_qualified": False,
        "anatomy_qualified": False,
        "export_implementation_sha256": _sha(Path(__file__)),
        "opencv_version": cv2.__version__,
        "physical_time_qualified": False,
        **(
            {"source_frame_rate": media["source_frame_rate"]}
            if "source_frame_rate" in media
            else {}
        ),
        "image_size": media["image_size"],
        "frames": records,
        "anatomical_marker_names": list(anatomy),
        "missing_anatomical_offsets": missing,
        "rigid_body_segments": _rigid_segments(binding),
        "rigid_skeleton_semantics": "declared joint-tree body origins; not skin markers or a surface mesh",
        "body_segments": [
            {"a": s.a, "b": s.b, "group": s.group}
            for s in default_body_segments(tuple(anatomy))
        ],
        "club_representation": "declared attachment points only; no mesh or inferred shaft",
        "pngs": pngs,
        **({"video": media["video"]} if "video" in media else {}),
    }
    if shapes is not None:
        manifest["shape_overlay"] = shapes.provenance
        manifest["club_representation"] = (
            "declared attachment points and model-conditioned visual proxies; not measured shaft anatomy"
        )
    if shaft is not None:
        manifest["shaft_overlay"] = shaft.to_record()
        manifest["club_representation"] = (
            "declared attachments and projected infinite authored axis; not physical endpoints or shaft length"
        )
    if overlays and overlays.captions:
        manifest["caption_overlay"] = caption_provenance(overlays.captions)
    return manifest


def _save_manifest(staging: Path, manifest: dict[str, Any]) -> None:
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )


def _verify_export_binding(
    library: NecromatcherLibrary,
    binding: NativeFitBinding,
    shaft: ShaftVideoOverlay | None,
) -> None:
    """Recheck immutable fit/parents and optional source evidence before publication."""
    if library.load_asset(binding.fit_id).metadata["hash"] != binding.fit_hash:
        raise ValueError("Fit bytes changed during overlay export")
    library.load_fit(binding.fit_id)
    if shaft is not None:
        if (
            bind_fit_shaft_evidence(library, binding.fit, shaft.bound.evidence)
            != shaft.bound
        ):
            raise ValueError("Shaft source clock changed during overlay export")


def _validate_overlay_options(
    shape_overlay: ShapeOverlayOptions | None,
    caption_overlay: CaptionOverlayOptions | None,
) -> None:
    if shape_overlay is not None and not isinstance(shape_overlay, ShapeOverlayOptions):
        raise TypeError("Shape export requires typed options")
    if caption_overlay is not None and not isinstance(
        caption_overlay, CaptionOverlayOptions
    ):
        raise TypeError("Caption export requires typed options")


def _prepare_export_layers(
    library: NecromatcherLibrary,
    binding: NativeFitBinding,
    shaft_evidence: ShaftAxisEvidence | None,
    shape_overlay: ShapeOverlayOptions | None,
    caption_overlay: CaptionOverlayOptions | None,
) -> tuple[dict[str, Any], list[str], VideoOverlayLayers]:
    anatomy, missing = _anatomical_attachments(binding)
    shaft = (
        prepare_shaft_overlay(library, binding, shaft_evidence)
        if shaft_evidence is not None
        else None
    )
    shapes = (
        NativeShapeOverlay.prepare(binding, shape_overlay)
        if shape_overlay is not None
        else None
    )
    return anatomy, missing, VideoOverlayLayers(shaft, shapes, caption_overlay)

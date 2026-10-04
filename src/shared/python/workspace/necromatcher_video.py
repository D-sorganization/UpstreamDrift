"""Source-sized native research overlays with immutable source-clock provenance.

Exported motion follows source presentation timestamps, never an inferred physical
clock. Anatomical seed markers are model-conditioned and uncalibrated; missing
attachment offsets and observations are omitted rather than reconstructed here.
"""

from __future__ import annotations

from fractions import Fraction
import hashlib
import json
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import Any, Sequence, cast

import numpy as np

from src.shared.python.motion_matching import default_body_segments
from src.shared.python.shadow_tracker.source_records import VariableFrameRateError
from .necromatcher import NecromatcherLibrary
from .necromatcher_native import NativeFitBinding, load_native_fit_binding
from .necromatcher_review import CaptureReview
from .necromatcher_video_forces import (
    ForceLayer,
    ForceLayerRenderer,
    ForceSamplerFactory,
)

_OBSERVED = (90, 230, 90)
_NATIVE = (255, 170, 30)
_RESIDUAL = (40, 210, 255)


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
    if steps[0] <= 0:
        raise ValueError("Video source PTS must be strictly increasing")
    if any(step != steps[0] for step in steps):
        raise VariableFrameRateError(
            "Video source PTS must be strictly increasing and uniform"
        )
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
) -> dict[str, np.ndarray]:
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
    return {name: np.asarray(poses[name][1], dtype=float) for name in names}


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
    force: ForceLayerRenderer | None = None,
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
    origins = _draw_rigid_skeleton(binding, pose, image)
    layer_record = force.draw(image, position, pose, origins) if force else {}
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
    identity = row["frame"]
    pts = Fraction(
        identity["pts_ticks"] * identity["timebase_numerator"],
        identity["timebase_denominator"],
    )
    metric = (
        f"Matched RMS {rms:.2f} px ({len(errors)} markers)"
        if rms is not None
        else "Matched RMS unavailable (no common observed markers)"
    )
    unavailable = (
        [f"Force layer unavailable: {force.reason}"] if force and force.reason else []
    )
    _caption(
        image,
        [
            f"MONOCULAR RESEARCH | Source frame {index} | PTS {float(pts):.3f}s",
            "Camera/anatomy unqualified | Physical time unknown",
            metric,
            "Green: observations | Blue: native rig/seeds | Yellow: residuals",
            *unavailable,
        ],
    )
    return image, {
        "frame_index": index,
        "frame": identity,
        "original_png_sha256": hashlib.sha256(original_png).hexdigest(),
        "matched_marker_count": len(errors),
        "matched_rms_pixels": rms,
        **layer_record,
    }


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
    force_layer: ForceLayer | None = None,
    force_sampler_factory: ForceSamplerFactory | None = None,
) -> dict[str, Any]:
    """Publish a new MP4/PNG/manifest directory only after complete codec verification.

    Preconditions: hash-bound native fit, contiguous uniform source PTS and selected
    PNG indices inside that fit. Postcondition: source bytes remain untouched and
    output hashes, frame identities and unqualified scientific status are recorded.
    An enabled ``force_layer`` additionally requires ``force_sampler_factory``.
    """
    if (
        force_layer is not None
        and force_layer.enabled
        and force_sampler_factory is None
    ):
        raise ValueError("An enabled force layer requires a force sampler factory")
    destination = Path(destination)
    if destination.exists():
        raise FileExistsError(destination)
    binding = load_native_fit_binding(library, fit_id)
    indices, frames = binding.fit["frame_indices"], binding.fit["frames"]
    rate = source_frame_rate(indices, frames)
    if any(type(index) is not int or index not in indices for index in selected_frames):
        raise ValueError("Selected PNG frames must belong to the bound fit")
    anatomy, missing = _anatomical_attachments(binding)
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
            ForceLayerRenderer(
                binding, force_layer, force_sampler_factory, _rigid_segments(binding)
            )
            if force_layer is not None
            and force_layer.enabled
            and force_sampler_factory is not None
            else None,
        )
        # Existing library authority rechecks parent and fit hashes before publication.
        if library.load_asset(fit_id).metadata["hash"] != binding.fit_hash:
            raise ValueError("Fit bytes changed during overlay export")
        library.load_fit(fit_id)
        _publish(staging, destination)
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


def _write_export(
    binding: NativeFitBinding,
    library: NecromatcherLibrary,
    staging: Path,
    rate: Fraction,
    selected: set[int],
    anatomy: dict[str, Any],
    missing: list[str],
    force: ForceLayerRenderer | None = None,
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
                image, record = _render(binding, review, index, anatomy, force)
                writer.write(image)
                records.append(record)
                if index in selected:
                    name = f"frame-{index:06d}.png"
                    pngs.append(_save_png(staging / name, image))
        finally:
            writer.release()
    _verify_video(staging / "overlay.mp4", len(records), size)
    manifest = {
        "schema": "necromatcher/source-overlay-video/1",
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
        "source_frame_rate": {
            "numerator": rate.numerator,
            "denominator": rate.denominator,
        },
        "image_size": list(size),
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
        "video": {
            "path": "overlay.mp4",
            "codec": "mp4v",
            "sha256": _sha(staging / "overlay.mp4"),
            "bytes": (staging / "overlay.mp4").stat().st_size,
        },
    }
    if force is not None:
        manifest["force_layer"] = force.manifest()
    (staging / "manifest.json").write_text(
        json.dumps(manifest, indent=2, allow_nan=False) + "\n", encoding="utf-8"
    )
    return manifest

"""Preserve source identities when exact consecutive images are deduplicated."""

from collections.abc import Iterable
from fractions import Fraction
import hashlib
from pathlib import Path
from typing import Any

from .ingestion import compute_frame_hash


def map_frames(frames: Iterable[tuple[int, int, str]]) -> list[dict[str, int | str]]:
    """Map each source index and PTS to an output index without inventing timing.

    Preconditions: source indices are contiguous from zero, PTS strictly increases,
    and each digest is nonempty. Only consecutive identical digests are collapsed.
    """
    rows: list[dict[str, int | str]] = []
    previous_pts: int | None = None
    previous_hash: str | None = None
    output_index = -1
    for index, pts, digest in frames:
        if type(index) is not int or index != len(rows):
            raise ValueError("Source indices must be contiguous from zero")
        if type(pts) is not int or (previous_pts is not None and pts <= previous_pts):
            raise ValueError("Source PTS must increase strictly")
        if not isinstance(digest, str) or not digest:
            raise ValueError("Frame digest is required")
        if digest != previous_hash:
            output_index += 1
        rows.append(
            {
                "source_index": index,
                "pts": pts,
                "output_index": output_index,
                "frame_sha256": digest,
            }
        )
        previous_pts, previous_hash = pts, digest
    return rows


def pin_source(source: Path, *, output: Path | None = None) -> dict[str, Any]:
    """Inspect a progressive source and map every decoded PTS to a unique image.

    Interlaced input is rejected until its field order and deinterlacing are
    explicitly reviewed. Presentation timestamps are not physical swing time.
    """
    import av

    if output is not None:
        import cv2

        output.mkdir(parents=True, exist_ok=False)

    with source.open("rb") as handle:
        source_hash = hashlib.file_digest(handle, "sha256").hexdigest()
    identities: list[tuple[int, int, str]] = []
    timebase: Fraction | None = None
    last_hash: str | None = None
    output_index = -1
    with av.open(str(source)) as container:
        stream = container.streams.video[0]
        width, height = stream.width, stream.height
        for index, frame in enumerate(container.decode(stream)):
            if frame.interlaced_frame:
                raise ValueError("Interlaced frames require reviewed deinterlacing")
            if frame.pts is None or frame.time_base is None:
                raise ValueError("Decoded frame lacks PTS or timebase")
            if timebase is not None and frame.time_base != timebase:
                raise ValueError("Frame timebase changed within source")
            timebase = frame.time_base
            image = frame.to_ndarray(format="bgr24")
            digest = compute_frame_hash(image.tobytes(), decoder_name="pyav")
            identities.append((index, frame.pts, digest))
            if digest != last_hash:
                output_index += 1
                if output is not None and not cv2.imwrite(
                    str(output / f"{output_index:06d}.png"), image
                ):
                    raise OSError("Could not write deduplicated frame")
            last_hash = digest
    if timebase is None:
        raise ValueError("Source contains no decoded frames")
    return {
        "source_sha256": source_hash,
        "dimensions_px": [width, height],
        "decoder": {"name": "pyav", "version": av.__version__, "pixel_format": "bgr24"},
        "field_order": "progressive_decoder_reported",
        "timebase": [timebase.numerator, timebase.denominator],
        "physical_time_verified": False,
        "qualification": "image_observations_only",
        "frames": map_frames(identities),
    }

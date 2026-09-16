"""Video ingestion, shot partitioning, timing mappings, and capture evidence (ST-03).

This module provides the ingestion pipeline for modern and archive footage:
- Ingesting media files into validated immutable `SourceAsset` records with SHA-256 content hashes.
- In-memory `SourceCatalog` for asset registration and lookup.
- Shot partitioning with frame cuts, image transform provenance (mirror, crop, rotate, slow-mo).
- Strict rejection of cross-swing fusion and inconsistent camera/asset identities.
- Detection of telecine duplicate frames while preserving full auditability.
- Exact rational PTS preservation, affine timing mapping, piecewise timing mapping,
  and multi-camera synchronization with offset and drift uncertainty.
- Adapting frame records to `FrameObservation`.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass, field
from fractions import Fraction
import hashlib
import math
from pathlib import Path
from typing import Any, Protocol, runtime_checkable

from ._validation import (
    FRAME_OBSERVATION_SCHEMA_VERSION,
    SHOT_SCHEMA_VERSION,
    SOURCE_SCHEMA_VERSION,
    check_id,
    check_int,
    check_pos_int,
    check_str,
)
from .contracts import (
    FrameObservation,
    Shot,
)
from .source_records import (
    FrameIdentity,
    RightsStatus,
    SourceAsset,
)

_CHUNK_SIZE = 64 * 1024


# ---------------------------------------------------------------------------
# 1. Source Asset Ingestion & Catalog
# ---------------------------------------------------------------------------


def ingest_source_asset(
    path: Path | str,
    *,
    asset_id: str,
    width_px: int,
    height_px: int,
    rights_status: RightsStatus = "unknown",
    rights_note: str = "",
    source_uri: str | None = None,
) -> SourceAsset:
    """Ingest a media file into an immutable, checksum-verified SourceAsset record.

    Preconditions:
        - `path` must be an existing regular file and non-empty.
        - `asset_id` must be a valid identifier.
        - `width_px` and `height_px` must be positive integers.
        - `rights_status` and `rights_note` must conform to `SourceAsset` constraints.
    """
    file_path = Path(path)
    if not file_path.is_file():
        raise FileNotFoundError(f"Source media file not found: {file_path}")

    size = file_path.stat().st_size
    if size == 0:
        raise ValueError(f"Source media file is corrupt or empty: {file_path}")

    hasher = hashlib.sha256()
    with file_path.open("rb") as stream:
        while chunk := stream.read(_CHUNK_SIZE):
            hasher.update(chunk)
    content_sha256 = hasher.hexdigest()

    uri = source_uri if source_uri is not None else f"urn:asset:{asset_id}"

    return SourceAsset(
        schema_version=SOURCE_SCHEMA_VERSION,
        asset_id=asset_id,
        source_uri=uri,
        content_sha256=content_sha256,
        width_px=width_px,
        height_px=height_px,
        rights_status=rights_status,
        rights_note=rights_note,
    )


class SourceCatalog:
    """Catalog of registered source assets with content hash integrity checking."""

    __slots__ = ("_assets",)

    def __init__(self) -> None:
        self._assets: dict[str, SourceAsset] = {}

    def register(self, asset: SourceAsset) -> None:
        """Register a source asset in the catalog.

        Idempotent if registered with identical content hash; raises ValueError
        if an asset with the same asset_id exists with different content.
        """
        if not isinstance(asset, SourceAsset):
            raise TypeError(f"Expected SourceAsset, got {type(asset).__name__}")
        existing = self._assets.get(asset.asset_id)
        if existing is not None:
            if existing.content_sha256 != asset.content_sha256:
                raise ValueError(
                    f"Asset {asset.asset_id!r} is already registered with different content "
                    f"({existing.content_sha256[:12]}... vs {asset.content_sha256[:12]}...)"
                )
            return
        self._assets[asset.asset_id] = asset

    def get(self, asset_id: str) -> SourceAsset:
        """Retrieve asset by asset_id or raise KeyError."""
        if asset_id not in self._assets:
            raise KeyError(f"Asset ID {asset_id!r} not found in catalog")
        return self._assets[asset_id]

    def __len__(self) -> int:
        return len(self._assets)

    def __contains__(self, asset_id: str) -> bool:
        return asset_id in self._assets


# ---------------------------------------------------------------------------
# 2. Shot Partitioning, Cuts & Cross-Swing Fusion Rejection
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True, kw_only=True)
class ShotDefinition:
    """Specification for defining a bounded continuous shot within an asset."""

    shot_id: str
    start_pts: int
    end_pts: int
    start_frame_id: str
    end_frame_id: str
    subject_id: str
    swing_id: str
    camera_id: str
    cuts: tuple[tuple[int, int], ...] = ()
    transforms: tuple[str, ...] = ()


def create_shot(
    asset: SourceAsset,
    definition: ShotDefinition,
) -> Shot:
    """Create a validated continuous Shot record tied to a parent SourceAsset."""
    if not isinstance(asset, SourceAsset):
        raise TypeError(f"Expected SourceAsset, got {type(asset).__name__}")
    if not isinstance(definition, ShotDefinition):
        raise TypeError(f"Expected ShotDefinition, got {type(definition).__name__}")
    return Shot(
        schema_version=SHOT_SCHEMA_VERSION,
        asset_id=asset.asset_id,
        shot_id=definition.shot_id,
        start_pts=definition.start_pts,
        end_pts=definition.end_pts,
        start_frame_id=definition.start_frame_id,
        end_frame_id=definition.end_frame_id,
        subject_id=definition.subject_id,
        swing_id=definition.swing_id,
        camera_id=definition.camera_id,
        cuts=definition.cuts,
        transforms=definition.transforms,
    )


def validate_shot_frames(shot: Shot, frames: Sequence[FrameIdentity]) -> None:
    """Validate that every frame in the sequence strictly belongs to the given shot.

    Preconditions:
        - Frames cannot mix swing_id (rejects cross-swing fusion).
        - Frames cannot mix asset_id or camera_id.
        - Frame PTS must fall within [shot.start_pts, shot.end_pts].
    """
    if not isinstance(shot, Shot):
        raise TypeError(f"Expected Shot, got {type(shot).__name__}")

    for idx, frame in enumerate(frames):
        if not isinstance(frame, FrameIdentity):
            raise TypeError(
                f"Element at index {idx} must be a FrameIdentity, got {type(frame).__name__}"
            )
        if frame.asset_id != shot.asset_id:
            raise ValueError(
                f"asset_id mismatch at frame index {idx}: frame has {frame.asset_id!r}, "
                f"shot has {shot.asset_id!r}"
            )
        if frame.swing_id != shot.swing_id:
            raise ValueError(
                f"swing_id mismatch (cross-swing fusion rejected) at frame index {idx}: "
                f"frame has {frame.swing_id!r}, shot has {shot.swing_id!r}"
            )
        if frame.camera_id != shot.camera_id:
            raise ValueError(
                f"camera_id mismatch at frame index {idx}: frame has {frame.camera_id!r}, "
                f"shot has {shot.camera_id!r}"
            )
        if frame.pts_ticks < shot.start_pts:
            raise ValueError(
                f"frame pts_ticks ({frame.pts_ticks}) precedes shot start_pts ({shot.start_pts}) "
                f"at frame index {idx}"
            )
        if frame.pts_ticks > shot.end_pts:
            raise ValueError(
                f"frame pts_ticks ({frame.pts_ticks}) exceeds shot end_pts ({shot.end_pts}) "
                f"at frame index {idx}"
            )


def filter_shot_frames(
    shot: Shot, frames: Sequence[FrameIdentity]
) -> list[FrameIdentity]:
    """Filter frames to those belonging to the shot and excluding cut intervals."""
    validate_shot_frames(shot, frames)
    valid_frames: list[FrameIdentity] = []

    for frame in frames:
        is_cut = False
        for cut_start, cut_end in shot.cuts:
            if cut_start <= frame.pts_ticks <= cut_end:
                is_cut = True
                break
        if not is_cut:
            valid_frames.append(frame)

    return valid_frames


# ---------------------------------------------------------------------------
# 3. Telecine Duplicate Detection
# ---------------------------------------------------------------------------


def detect_telecine_duplicates(
    frames: Sequence[FrameIdentity],
) -> tuple[list[FrameIdentity], list[FrameIdentity]]:
    """Detect consecutive duplicate frames (e.g. from 3:2 pulldown / telecine conversion).

    Returns:
        `(deduplicated_frames, duplicate_frames)`
        - `deduplicated_frames`: First occurrence of each unique consecutive frame.
        - `duplicate_frames`: Identical consecutive frames that were repeated.
    """
    if not frames:
        return [], []

    deduplicated: list[FrameIdentity] = []
    duplicates: list[FrameIdentity] = []
    last_hash: str | None = None

    for frame in frames:
        if last_hash is not None and frame.frame_sha256 == last_hash:
            duplicates.append(frame)
        else:
            deduplicated.append(frame)
            last_hash = frame.frame_sha256

    return deduplicated, duplicates


# ---------------------------------------------------------------------------
# 4. Timing Mappings & Multi-Camera Synchronization
# ---------------------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AffineTimingMapping:
    """Affine mapping from presentation time to physical swing time.

    ``physical_time_s = scale * presentation_time_s + offset_seconds``
    """

    scale: Fraction
    offset_seconds: float = 0.0

    def __post_init__(self) -> None:
        if not isinstance(self.scale, (Fraction, int)):
            raise TypeError(
                f"scale must be Fraction or int, got {type(self.scale).__name__}"
            )
        if self.scale <= 0:
            raise ValueError(f"scale must be positive, got {self.scale}")
        if not math.isfinite(self.offset_seconds):
            raise ValueError(
                f"offset_seconds must be finite, got {self.offset_seconds}"
            )

    def to_physical_time(self, presentation_time_s: Fraction | float) -> float:
        """Convert presentation time in seconds to physical swing time."""
        return float(presentation_time_s) * float(self.scale) + self.offset_seconds

    def to_presentation_time(self, physical_time_s: float) -> Fraction:
        """Invert physical swing time back to rational presentation time."""
        if not math.isfinite(physical_time_s):
            raise ValueError("physical_time_s must be finite")
        diff = Fraction(physical_time_s - self.offset_seconds)
        return diff / self.scale


@dataclass(frozen=True, slots=True)
class PiecewiseTimingMapping:
    """Piecewise timing mapping across intervals for variable speed / archive edits."""

    intervals: Sequence[tuple[Fraction, Fraction, AffineTimingMapping]]

    def __post_init__(self) -> None:
        if not self.intervals:
            raise ValueError("Piecewise intervals sequence cannot be empty")
        for idx, item in enumerate(self.intervals):
            if len(item) != 3:
                raise ValueError(
                    f"Interval {idx} must be a 3-tuple (start, end, mapping)"
                )
            start, end, mapping = item
            if start >= end:
                raise ValueError(
                    f"Interval {idx} start ({start}) must precede end ({end})"
                )
            if not isinstance(mapping, AffineTimingMapping):
                raise TypeError(f"Interval {idx} mapping must be AffineTimingMapping")

    def to_physical_time(self, presentation_time_s: Fraction | float) -> float:
        """Map presentation time to physical time within the piece intervals."""
        t_rat = (
            Fraction(presentation_time_s)
            if not isinstance(presentation_time_s, Fraction)
            else presentation_time_s
        )
        for start, end, mapping in self.intervals:
            if start <= t_rat <= end:
                return mapping.to_physical_time(t_rat)
        raise ValueError(
            f"Presentation time {t_rat} is outside piecewise timing intervals"
        )


@dataclass(frozen=True, slots=True)
class CameraSynchronization:
    """Multi-camera synchronization model with offset and linear drift."""

    camera_id: str
    reference_camera_id: str
    offset_seconds: float
    drift_rate: float = 0.0
    uncertainty_seconds: float = 0.0

    def __post_init__(self) -> None:
        check_id(self.camera_id, "camera_id")
        check_id(self.reference_camera_id, "reference_camera_id")
        if not math.isfinite(self.offset_seconds):
            raise ValueError("offset_seconds must be finite")
        if not math.isfinite(self.drift_rate):
            raise ValueError("drift_rate must be finite")
        if not math.isfinite(self.uncertainty_seconds) or self.uncertainty_seconds < 0:
            raise ValueError("uncertainty_seconds must be non-negative and finite")

    def reference_to_camera_time(self, t_ref: float) -> float:
        """Map reference camera timestamp to this camera's timestamp."""
        return t_ref + self.offset_seconds + self.drift_rate * t_ref

    def camera_to_reference_time(self, t_cam: float) -> float:
        """Invert camera timestamp to reference camera timestamp."""
        denom = 1.0 + self.drift_rate
        if denom == 0.0:
            raise ZeroDivisionError(
                "Cannot invert synchronization with drift_rate == -1.0"
            )
        return (t_cam - self.offset_seconds) / denom


# ---------------------------------------------------------------------------
# 5. Observation Mapping
# ---------------------------------------------------------------------------


def map_frame_to_observation(
    frame: FrameIdentity,
    *,
    body_mask_ref: str,
    club_mask_ref: str,
    valid_mask_ref: str,
    confidence_provenance: str,
) -> FrameObservation:
    """Map an ingested FrameIdentity and mask references into a FrameObservation record."""
    if not isinstance(frame, FrameIdentity):
        raise TypeError(f"Expected FrameIdentity, got {type(frame).__name__}")
    reason = frame.physical_time_reason or "nominal_clock"
    return FrameObservation(
        schema_version=FRAME_OBSERVATION_SCHEMA_VERSION,
        shot_id=frame.shot_id,
        camera_id=frame.camera_id,
        frame_id=frame.frame_id,
        pts_ticks=frame.pts_ticks,
        timebase_numerator=frame.timebase_numerator,
        timebase_denominator=frame.timebase_denominator,
        physical_time_s=frame.physical_time_s,
        physical_time_reason=reason,
        body_mask_ref=body_mask_ref,
        club_mask_ref=club_mask_ref,
        valid_mask_ref=valid_mask_ref,
        confidence_provenance=confidence_provenance,
    )


def compute_frame_hash(
    frame_bytes: bytes | bytearray | memoryview,
    *,
    decoder_name: str = "opencv",
    pixel_format: str = "bgr24",
) -> str:
    """Compute deterministic SHA-256 hash bound to decoder and pixel-format provenance."""
    if not isinstance(frame_bytes, (bytes, bytearray, memoryview)):
        raise TypeError(
            f"frame_bytes must be bytes-like, got {type(frame_bytes).__name__}"
        )
    if not decoder_name or not decoder_name.strip():
        raise ValueError("decoder_name must be non-empty")
    if not pixel_format or not pixel_format.strip():
        raise ValueError("pixel_format must be non-empty")
    prefix = f"{decoder_name.strip()}:{pixel_format.strip()}:".encode("ascii")
    return hashlib.sha256(prefix + bytes(frame_bytes)).hexdigest()


@runtime_checkable
class VideoDecoderAdapter(Protocol):
    """Protocol for frame-accurate video decoders providing rational presentation timestamps."""

    @property
    def frame_count(self) -> int: ...

    @property
    def width(self) -> int: ...

    @property
    def height(self) -> int: ...

    @property
    def timebase_numerator(self) -> int: ...

    @property
    def timebase_denominator(self) -> int: ...

    @property
    def is_timing_exact(self) -> bool: ...

    @property
    def timing_mode(self) -> str: ...

    @property
    def decoder_name(self) -> str: ...

    @property
    def pixel_format(self) -> str: ...

    def pts_ticks(self, frame_index: int) -> int: ...

    def read_frame_hash(self, frame_index: int) -> str: ...


def _reduce_timebase(timebase: tuple[int, int]) -> tuple[int, int]:
    num, den = timebase
    check_pos_int(num, "timebase_numerator")
    check_pos_int(den, "timebase_denominator")
    gcd = math.gcd(num, den)
    return num // gcd, den // gcd


class SyntheticVideoDecoder:
    """Deterministic synthetic video decoder for testing and offline ingestion."""

    __slots__ = (
        "_frame_count",
        "_width",
        "_height",
        "_pts_ticks",
        "_timebase_numerator",
        "_timebase_denominator",
        "_is_timing_exact",
        "_timing_mode",
        "_decoder_name",
        "_pixel_format",
    )

    def __init__(
        self,
        frame_count: int,
        *,
        width: int = 1920,
        height: int = 1080,
        pts_ticks: Sequence[int] | None = None,
        timebase: tuple[int, int] = (1, 1),
    ) -> None:
        check_pos_int(frame_count, "frame_count")
        check_pos_int(width, "width")
        check_pos_int(height, "height")
        self._frame_count = frame_count
        self._width = width
        self._height = height
        if pts_ticks is not None:
            if len(pts_ticks) != frame_count:
                raise ValueError(
                    f"pts_ticks length ({len(pts_ticks)}) must match frame_count ({frame_count})"
                )
            self._pts_ticks = tuple(pts_ticks)
        else:
            self._pts_ticks = tuple(range(frame_count))

        (
            self._timebase_numerator,
            self._timebase_denominator,
        ) = _reduce_timebase(timebase)
        self._is_timing_exact = True
        self._timing_mode = "authoritative"
        self._decoder_name = "synthetic"
        self._pixel_format = "synthetic"

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def timebase_numerator(self) -> int:
        return self._timebase_numerator

    @property
    def timebase_denominator(self) -> int:
        return self._timebase_denominator

    @property
    def is_timing_exact(self) -> bool:
        return self._is_timing_exact

    @property
    def timing_mode(self) -> str:
        return self._timing_mode

    @property
    def decoder_name(self) -> str:
        return self._decoder_name

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    def pts_ticks(self, frame_index: int) -> int:
        if not 0 <= frame_index < self._frame_count:
            raise IndexError(
                f"frame_index {frame_index} out of bounds [0, {self._frame_count})"
            )
        return self._pts_ticks[frame_index]

    def read_frame_hash(self, frame_index: int) -> str:
        if not 0 <= frame_index < self._frame_count:
            raise IndexError(
                f"frame_index {frame_index} out of bounds [0, {self._frame_count})"
            )
        raw = f"synth_frame:{frame_index}:{self._pts_ticks[frame_index]}".encode(
            "ascii"
        )
        return compute_frame_hash(
            raw,
            decoder_name=self._decoder_name,
            pixel_format=self._pixel_format,
        )


@runtime_checkable
class CaptureRigView(Protocol):
    """Protocol for Capture Rig view media records."""

    @property
    def view(self) -> str: ...

    @property
    def identity(self) -> str: ...

    @property
    def playable(self) -> Path | None: ...


def ingest_capture_rig_view(
    view: CaptureRigView,
    *,
    asset_id: str,
    width_px: int,
    height_px: int,
    rights_status: RightsStatus = "unknown",
    rights_note: str = "",
) -> SourceAsset:
    """Ingest a Capture Rig view's playable media into a SourceAsset record."""
    playable = view.playable
    if playable is None:
        raise ValueError(
            f"Capture Rig view {view.view!r} has no playable recording or proxy"
        )
    return ingest_source_asset(
        playable,
        asset_id=asset_id,
        width_px=width_px,
        height_px=height_px,
        rights_status=rights_status,
        rights_note=rights_note,
        source_uri=f"urn:capture-rig:view:{view.view}:{asset_id}",
    )


@dataclass(frozen=True, slots=True, kw_only=True)
class DecodeLimits:
    """Execution bounds and cooperative cancellation controls for bounded video decoding."""

    max_frames: int | None = None
    is_cancelled: Callable[[], bool] | None = None

    def __post_init__(self) -> None:
        if self.max_frames is not None and self.max_frames < 0:
            raise ValueError(f"max_frames cannot be negative, got {self.max_frames}")


class OpenCvVideoDecoder:
    """Frame-accurate video decoder backed by OpenCV VideoCapture with bounded memory.

    Preconditions:
        - `path` must exist and be an openable, non-empty media file.
        - Container metadata must report positive frame dimensions and at least 1 frame.
    """

    __slots__ = (
        "_path",
        "_frame_count",
        "_width",
        "_height",
        "_fps",
        "_timebase_numerator",
        "_timebase_denominator",
        "_is_timing_exact",
        "_timing_mode",
        "_decoder_name",
        "_pixel_format",
        "_explicit_pts",
        "_cache",
    )

    def __init__(
        self,
        path: Path | str,
        *,
        pts_ticks: Sequence[int] | None = None,
        timebase: tuple[int, int] | None = None,
    ) -> None:
        import cv2

        file_path = Path(path)
        if not file_path.is_file():
            raise FileNotFoundError(f"Video file not found: {file_path}")
        if file_path.stat().st_size == 0:
            raise ValueError(f"Video file is empty: {file_path}")

        cap = cv2.VideoCapture(str(file_path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {file_path}")

        try:
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            fps = float(cap.get(cv2.CAP_PROP_FPS))
            raw_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            if width <= 0 or height <= 0:
                raise ValueError(
                    f"Video has invalid dimensions: {width}x{height} in {file_path}"
                )

            # Probe exactly one frame to verify media is decodable and non-corrupt
            ret, frame0 = cap.read()
            if not ret or frame0 is None:
                raise ValueError(f"Could not decode initial frame from {file_path}")
        finally:
            cap.release()

        self._path = file_path
        self._width = width
        self._height = height
        self._fps = fps
        self._decoder_name = "opencv"
        self._pixel_format = "bgr24"
        self._cache: dict[int, str] = {
            0: compute_frame_hash(
                frame0.tobytes(), decoder_name="opencv", pixel_format="bgr24"
            )
        }

        if pts_ticks is not None:
            if len(pts_ticks) == 0:
                raise ValueError("pts_ticks must not be empty")
            self._explicit_pts: tuple[int, ...] | None = tuple(pts_ticks)
            self._frame_count = len(pts_ticks)
            self._is_timing_exact = True
            self._timing_mode = "authoritative"
        else:
            self._explicit_pts = None
            self._frame_count = max(raw_count, 1)
            if fps > 0 and math.isfinite(fps):
                self._is_timing_exact = True
                self._timing_mode = "estimated_cfr"
            else:
                self._is_timing_exact = False
                self._timing_mode = "estimated_cfr"

        if timebase is not None:
            (
                self._timebase_numerator,
                self._timebase_denominator,
            ) = _reduce_timebase(timebase)
        elif fps > 0 and math.isfinite(fps):
            frac = Fraction(fps).limit_denominator(100000)
            (
                self._timebase_numerator,
                self._timebase_denominator,
            ) = _reduce_timebase((frac.denominator, frac.numerator))
        else:
            self._timebase_numerator = 1
            self._timebase_denominator = 1

    @property
    def frame_count(self) -> int:
        return self._frame_count

    @property
    def width(self) -> int:
        return self._width

    @property
    def height(self) -> int:
        return self._height

    @property
    def timebase_numerator(self) -> int:
        return self._timebase_numerator

    @property
    def timebase_denominator(self) -> int:
        return self._timebase_denominator

    @property
    def is_timing_exact(self) -> bool:
        return self._is_timing_exact

    @property
    def timing_mode(self) -> str:
        return self._timing_mode

    @property
    def decoder_name(self) -> str:
        return self._decoder_name

    @property
    def pixel_format(self) -> str:
        return self._pixel_format

    def pts_ticks(self, frame_index: int) -> int:
        """Presentation timestamp in ticks for the specified frame index."""
        if not 0 <= frame_index < self._frame_count:
            raise IndexError(
                f"frame_index {frame_index} out of bounds [0, {self._frame_count})"
            )
        if self._explicit_pts is not None:
            return self._explicit_pts[frame_index]
        return frame_index

    def read_frame_hash(self, frame_index: int) -> str:
        """Deterministic SHA-256 hash of decoded raw frame pixels with provenance."""
        if not 0 <= frame_index < self._frame_count:
            raise IndexError(
                f"frame_index {frame_index} out of bounds [0, {self._frame_count})"
            )
        if frame_index in self._cache:
            return self._cache[frame_index]

        import cv2

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {self._path}")
        try:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ret, frame = cap.read()
            if not ret or frame is None:
                raise ValueError(
                    f"Could not decode frame {frame_index} from {self._path}"
                )
            h = compute_frame_hash(
                frame.tobytes(),
                decoder_name=self._decoder_name,
                pixel_format=self._pixel_format,
            )
            if len(self._cache) > 128:
                self._cache.clear()
            self._cache[frame_index] = h
            return h
        finally:
            cap.release()

    def stream_frames(
        self,
        limits: DecodeLimits | None = None,
    ) -> Iterator[tuple[int, int, str]]:
        """Incrementally decode frames with bounded memory, cooperative cancellation, and guaranteed resource cleanup."""
        if limits is None:
            limits = DecodeLimits()
        import cv2

        cap = cv2.VideoCapture(str(self._path))
        if not cap.isOpened():
            raise ValueError(f"Could not open video file {self._path}")

        limit = (
            self._frame_count
            if limits.max_frames is None
            else min(self._frame_count, limits.max_frames)
        )
        try:
            for i in range(limit):
                if limits.is_cancelled is not None and limits.is_cancelled():
                    break
                ret, frame = cap.read()
                if not ret or frame is None:
                    break
                pts = self.pts_ticks(i)
                h = compute_frame_hash(
                    frame.tobytes(),
                    decoder_name=self._decoder_name,
                    pixel_format=self._pixel_format,
                )
                yield (i, pts, h)
        finally:
            cap.release()


def decode_video_frames(
    decoder: VideoDecoderAdapter,
    *,
    asset: SourceAsset,
    shot_id: str,
    swing_id: str,
    camera_id: str,
    physical_time_s_fn: Callable[[int, Fraction], float | None] | None = None,
    physical_time_reason: str = "unknown physical time without evidenced clock mapping",
    limits: DecodeLimits | None = None,
) -> Iterator[FrameIdentity]:
    """Bounded iterator decoding a video into validated immutable FrameIdentity records.

    Preconditions:
        - `decoder` conforms to VideoDecoderAdapter.
        - `asset` is a valid SourceAsset.
        - `shot_id`, `swing_id`, `camera_id` are valid non-empty identifiers.
        - `limits` specifies optional max_frames and cooperative cancellation.
        - `physical_time_s_fn` is an optional explicit mapping from (frame_idx, presentation_time)
          to physical time in SI seconds. If None, physical_time_s is unknown (None).
    """
    from ._validation import FRAME_SCHEMA_VERSION, check_id

    check_id(shot_id, "shot_id")
    check_id(swing_id, "swing_id")
    check_id(camera_id, "camera_id")

    tb_num = decoder.timebase_numerator
    tb_den = decoder.timebase_denominator
    active_limits = limits if limits is not None else DecodeLimits()

    if hasattr(decoder, "stream_frames"):
        frame_source = decoder.stream_frames(limits=active_limits)
    else:

        def _generic_stream() -> Iterator[tuple[int, int, str]]:
            limit = (
                decoder.frame_count
                if active_limits.max_frames is None
                else min(decoder.frame_count, active_limits.max_frames)
            )
            for idx in range(limit):
                if (
                    active_limits.is_cancelled is not None
                    and active_limits.is_cancelled()
                ):
                    break
                yield (idx, decoder.pts_ticks(idx), decoder.read_frame_hash(idx))

        frame_source = _generic_stream()

    for i, pts, frame_hash in frame_source:
        pres_time = Fraction(pts * tb_num, tb_den)

        if physical_time_s_fn is not None:
            phys_time = physical_time_s_fn(i, pres_time)
            reason = (
                physical_time_reason
                if phys_time is not None
                else (physical_time_reason or "unknown physical time")
            )
        else:
            phys_time = None
            reason = (
                physical_time_reason
                or "unknown physical time without evidenced clock mapping"
            )

        yield FrameIdentity(
            schema_version=FRAME_SCHEMA_VERSION,
            asset_id=asset.asset_id,
            shot_id=shot_id,
            swing_id=swing_id,
            camera_id=camera_id,
            frame_id=f"frame-{i:06d}",
            pts_ticks=pts,
            timebase_numerator=tb_num,
            timebase_denominator=tb_den,
            physical_time_s=phys_time,
            physical_time_reason=reason,
            frame_sha256=frame_hash,
        )

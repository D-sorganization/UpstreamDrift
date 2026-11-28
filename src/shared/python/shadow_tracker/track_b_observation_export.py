"""Track B Shadow Tracker observation export (T1-WP4, shadow-tracker/source/1.0.0)."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass
import hashlib
import json
import math
from typing import Any, Final

import numpy as np
from scipy.ndimage import label as _nd_label

from . import _validation as _v
from .source_records import FrameIdentity, SourceAsset, validate_frame_sequence

try:
    import cv2
except ImportError:  # pragma: no cover
    cv2 = None  # type: ignore[assignment]

TRACK_B_SCHEMA_VERSION: Final[str] = _v.SOURCE_SCHEMA_VERSION
TRACK_B_QUALIFICATION: Final[str] = "image_observations_only"
MAX_ANGULAR_UNCERTAINTY_RAD: Final[float] = math.pi / 2.0


def _xy(pt: object, name: str) -> tuple[float, float]:
    """Validate and convert a 2-tuple into finite (x, y) floats."""
    if not isinstance(pt, tuple) or len(pt) != 2:
        raise TypeError(f"{name} must be a 2-tuple")
    return (_v.check_float(pt[0], f"{name}[0]"), _v.check_float(pt[1], f"{name}[1]"))


def _rng(val: object, name: str, lo: float, hi: float, open_lo: bool = False) -> float:
    """Validate a finite float within [lo, hi] (or (lo, hi] when open_lo)."""
    x = _v.check_float(val, name)
    if not ((x > lo if open_lo else x >= lo) and x <= hi):
        raise ValueError(f"{name} out of range [{lo}, {hi}]: {x}")
    return x


def _check_prov(rec: Any) -> None:
    """Validate Track B non-physical provenance and schema fields."""
    _v.check_schema_version(rec.schema_version, TRACK_B_SCHEMA_VERSION)
    if _v.check_bool(rec.physical_time_verified, "physical_time_verified") is not False:
        raise ValueError("physical_time_verified must be False")
    if _v.check_str(rec.qualification, "qualification") != TRACK_B_QUALIFICATION:
        raise ValueError(f"qualification must be {TRACK_B_QUALIFICATION!r}")


def _check_mask(mask: object, name: str) -> np.ndarray:
    """Validate a non-empty 2D boolean mask array and return a contiguous copy."""
    if not isinstance(mask, np.ndarray):
        raise TypeError(f"{name} must be numpy.ndarray")
    if mask.ndim != 2 or min(mask.shape) <= 0:
        raise ValueError(f"{name} must be non-empty 2D")
    if mask.dtype != np.bool_:
        raise TypeError(f"{name} must have bool dtype")
    return np.ascontiguousarray(mask, dtype=np.bool_)


@dataclass(frozen=True, slots=True)
class ContourBoundary:
    """Ordered 2D outer contour boundary extracted from a binary body mask."""

    points_xy: tuple[tuple[float, float], ...]
    area_px: int
    perimeter_px: float
    centroid_xy: tuple[float, float] | None
    bbox_xywh: tuple[int, int, int, int] | None

    def __post_init__(self) -> None:
        """Validate contour geometry, bounds, and empty-mask consistency."""
        if not isinstance(self.points_xy, tuple):
            raise TypeError("points_xy must be tuple")
        for p in self.points_xy:
            _xy(p, "points_xy")
        if _v.check_int(self.area_px, "area_px") < 0:
            raise ValueError("area_px must be >= 0")
        _rng(self.perimeter_px, "perimeter_px", 0.0, math.inf)
        if self.centroid_xy is not None:
            _xy(self.centroid_xy, "centroid_xy")
        if self.bbox_xywh is not None:
            if not isinstance(self.bbox_xywh, tuple) or len(self.bbox_xywh) != 4:
                raise TypeError("bbox_xywh must be 4-tuple")
            for i, v in enumerate(self.bbox_xywh):
                if _v.check_int(v, "bbox") < (1 if i >= 2 else 0):
                    raise ValueError("invalid bbox_xywh")
        if self.area_px == 0 and (self.points_xy or self.centroid_xy or self.bbox_xywh):
            raise ValueError("Empty ContourBoundary cannot carry geometry")


@dataclass(frozen=True, slots=True)
class ShaftSegmentObservation:
    """Extracted visible club-shaft line segment in 2D image coordinates."""

    start_xy: tuple[float, float]
    end_xy: tuple[float, float]
    center_xy: tuple[float, float]
    angle_rad: float
    length_px: float
    pixel_count: int
    angular_uncertainty_rad: float

    def __post_init__(self) -> None:
        """Validate shaft endpoints, orientation angle, length, and uncertainty."""
        for k in ("start_xy", "end_xy", "center_xy"):
            _xy(getattr(self, k), k)
        _rng(self.angle_rad, "angle_rad", 0.0, math.nextafter(math.pi, 0.0))
        _rng(self.length_px, "length_px", 0.0, math.inf, True)
        if _v.check_pos_int(self.pixel_count, "pixel_count") < 2:
            raise ValueError("pixel_count must be >= 2")
        _rng(self.angular_uncertainty_rad, "unc", 0.0, math.pi / 2.0, True)


@dataclass(frozen=True, slots=True)
class TrackBFrameExport:
    """Single-frame Track B observation export bound to FrameIdentity."""

    frame: FrameIdentity
    width_px: int
    height_px: int
    body_contour: ContourBoundary
    visible_shaft_segments: tuple[ShaftSegmentObservation, ...]
    occluded: bool
    occlusion_fraction: float
    motion_blur_score: float
    angular_uncertainty_rad: float
    body_mask_sha256: str
    shaft_mask_sha256: str
    mask_digest_sha256: str
    schema_version: str = TRACK_B_SCHEMA_VERSION
    physical_time_verified: bool = False
    qualification: str = TRACK_B_QUALIFICATION

    def __post_init__(self) -> None:
        """Validate frame binding, observation invariants, and SHA-256 digests."""
        _check_prov(self)
        if not isinstance(self.frame, FrameIdentity):
            raise TypeError("frame must be FrameIdentity")
        if not isinstance(self.body_contour, ContourBoundary):
            raise TypeError("body_contour must be ContourBoundary")
        _v.check_pos_int(self.width_px, "width_px")
        _v.check_pos_int(self.height_px, "height_px")
        segs = self.visible_shaft_segments
        if not isinstance(segs, tuple):
            raise TypeError("visible_shaft_segments must be tuple")
        for s in segs:
            if not isinstance(s, ShaftSegmentObservation):
                raise TypeError("segment must be ShaftSegmentObservation")
        _v.check_bool(self.occluded, "occluded")
        _rng(self.occlusion_fraction, "occlusion_fraction", 0.0, 1.0)
        _rng(self.motion_blur_score, "motion_blur_score", 0.0, 1.0)
        unc = _rng(self.angular_uncertainty_rad, "unc", 0.0, math.pi / 2.0, True)
        if not segs and (not self.occluded or not math.isclose(unc, math.pi / 2.0)):
            raise ValueError("Empty shaft requires occluded=True and unc=pi/2")
        for k in ("body_mask_sha256", "shaft_mask_sha256", "mask_digest_sha256"):
            _v.check_sha256(getattr(self, k), k)

    def to_dict(self) -> dict[str, Any]:
        """Serialize frame observation export to JSON-compatible dict."""
        data = asdict(self)
        data["frame"] = self.frame.to_dict()
        return data


@dataclass(frozen=True, slots=True)
class TrackBSequenceExport:
    """Validated Track B observation sequence export bound to SourceAsset."""

    source_asset: SourceAsset
    frames: tuple[TrackBFrameExport, ...]
    manifest_json: str
    manifest_sha256: str
    schema_version: str = TRACK_B_SCHEMA_VERSION
    physical_time_verified: bool = False
    qualification: str = TRACK_B_QUALIFICATION

    def __post_init__(self) -> None:
        """Validate sequence frames, asset binding, ordering, and manifest hash."""
        _check_prov(self)
        if not isinstance(self.source_asset, SourceAsset):
            raise TypeError("source_asset must be SourceAsset")
        if not isinstance(self.frames, tuple) or not self.frames:
            raise ValueError("frames must be non-empty tuple")
        sa = self.source_asset
        exp_meta = (sa.asset_id, sa.width_px, sa.height_px)
        for i, fx in enumerate(self.frames):
            if not isinstance(fx, TrackBFrameExport):
                raise TypeError("frames items must be TrackBFrameExport")
            if (fx.frame.asset_id, fx.width_px, fx.height_px) != exp_meta:
                raise ValueError(f"asset/dimension mismatch at {i}")
        validate_frame_sequence([f.frame for f in self.frames])
        _v.check_str(self.manifest_json, "manifest_json")
        _v.check_sha256(self.manifest_sha256, "manifest_sha256")
        exp = hashlib.sha256(self.manifest_json.encode()).hexdigest()
        if self.manifest_sha256 != exp:
            raise ValueError("manifest_sha256 mismatch")


def _extract_body_contour(mask: np.ndarray) -> ContourBoundary:
    """Extract the outer contour, area, perimeter, centroid, and bbox from a mask."""
    area = int(np.sum(mask))
    if area == 0:
        return ContourBoundary((), 0, 0.0, None, None)
    ys, xs = np.nonzero(mask)
    cx, cy = float(np.mean(xs)), float(np.mean(ys))
    pts: tuple[tuple[float, float], ...] = ()
    perim = 0.0
    if cv2 is not None:
        u8 = mask.astype(np.uint8)
        cnts, _ = cv2.findContours(u8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
        if cnts:
            top = max(cnts, key=lambda c: (abs(cv2.contourArea(c)), len(c)))
            pts = tuple((float(q[0]), float(q[1])) for q in top.reshape(-1, 2))
            perim = float(cv2.arcLength(top, True))
    if not pts:  # pragma: no cover
        p = np.pad(mask, 1)
        bys, bxs = np.nonzero(mask & ~(p[1:-1, 1:-1] & p[:-2, 1:-1] & p[2:, 1:-1]))
        idx = np.argsort(np.arctan2(bys - cy, bxs - cx))
        pts = tuple((float(bxs[i]), float(bys[i])) for i in idx)
        perim = float(len(pts))
    x0, y0 = int(np.min(xs)), int(np.min(ys))
    bbox = (x0, y0, int(np.max(xs)) - x0 + 1, int(np.max(ys)) - y0 + 1)
    return ContourBoundary(pts, area, perim, (cx, cy), bbox)


def _fit_segment(pts: np.ndarray, b: float, o: float) -> ShaftSegmentObservation | None:
    """Fit a 2D PCA line segment and angular uncertainty to a shaft component."""
    n = int(pts.shape[0])
    mu = np.mean(pts, axis=0)
    ctr = pts - mu
    vals, vecs = np.linalg.eigh((ctr.T @ ctr) / float(n))
    vx, vy = float(vecs[0, 1]), float(vecs[1, 1])
    if abs(vy) <= 1e-12:
        vy, vx = 0.0, abs(vx)
    elif vy < 0.0:
        vx, vy = -vx, -vy
    proj = ctr[:, 0] * vx + ctr[:, 1] * vy
    t0, t1 = float(np.min(proj)), float(np.max(proj))
    length = t1 - t0
    if length <= 0.0:
        return None
    sp = math.sqrt(max(float(vals[0]), 0.0))
    comb = math.atan2(1.0 + 2.0 * sp, max(length, 1.0)) + 0.5 * (b + o)
    raw = MAX_ANGULAR_UNCERTAINTY_RAD * (1.0 - math.exp(-comb))
    unc = min(max(raw, 1e-9), math.nextafter(MAX_ANGULAR_UNCERTAINTY_RAD, 0.0))
    s_xy = (float(mu[0] + t0 * vx), float(mu[1] + t0 * vy))
    e_xy = (float(mu[0] + t1 * vx), float(mu[1] + t1 * vy))
    c_xy = (float(mu[0]), float(mu[1]))
    ang = math.atan2(vy, vx) % math.pi
    return ShaftSegmentObservation(s_xy, e_xy, c_xy, ang, length, n, unc)


def extract_track_b_frame(
    frame: FrameIdentity,
    body_mask: np.ndarray,
    shaft_mask: np.ndarray,
    *,
    occlusion_fraction: float = 0.0,
    motion_blur_score: float = 0.0,
) -> TrackBFrameExport:
    """Extract Track B image-space contour and shaft observations from binary masks."""
    if not isinstance(frame, FrameIdentity):
        raise TypeError("frame must be FrameIdentity")
    body = _check_mask(body_mask, "body_mask")
    shaft = _check_mask(shaft_mask, "shaft_mask")
    if body.shape != shaft.shape:
        raise ValueError(f"Mask shape mismatch: {body.shape} vs {shaft.shape}")
    occ = _rng(occlusion_fraction, "occlusion_fraction", 0.0, 1.0)
    blur = _rng(motion_blur_score, "motion_blur_score", 0.0, 1.0)
    h, w = int(body.shape[0]), int(body.shape[1])
    b_sha = hashlib.sha256(body.tobytes()).hexdigest()
    s_sha = hashlib.sha256(shaft.tobytes()).hexdigest()
    digest = hashlib.sha256(f"{w}x{h}:{b_sha}:{s_sha}".encode()).hexdigest()
    segs: list[ShaftSegmentObservation] = []
    if occ < 1.0 and int(np.sum(shaft)) >= 2:
        labels, n_lbl = _nd_label(shaft, structure=np.ones((3, 3), dtype=int))
        for lid in range(1, n_lbl + 1):
            ys, xs = np.nonzero(labels == lid)
            if len(xs) >= 2:
                seg = _fit_segment(np.column_stack((xs, ys)).astype(float), blur, occ)
                if seg is not None:
                    segs.append(seg)
    segs.sort(key=lambda s: (-s.pixel_count, s.center_xy[1], s.center_xy[0]))
    vis = tuple(segs)
    unc = min((s.angular_uncertainty_rad for s in vis), default=math.pi / 2.0)
    occ_flag = (not vis) or (occ > 0.0)
    bc = _extract_body_contour(body)
    return TrackBFrameExport(
        frame, w, h, bc, vis, occ_flag, occ, blur, unc, b_sha, s_sha, digest
    )


def export_track_b_sequence(
    source_asset: SourceAsset, frames: Sequence[TrackBFrameExport]
) -> TrackBSequenceExport:
    """Export a validated Track B observation sequence bound to a SourceAsset."""
    if not isinstance(source_asset, SourceAsset):
        raise TypeError("source_asset must be SourceAsset")
    if (
        isinstance(frames, (str, bytes))
        or not isinstance(frames, Sequence)
        or not frames
    ):
        raise ValueError("frames must be a non-empty Sequence")
    ft = tuple(frames)
    payload: dict[str, Any] = {
        "schema_version": TRACK_B_SCHEMA_VERSION,
        "qualification": TRACK_B_QUALIFICATION,
        "physical_time_verified": False,
        "source_asset": source_asset.to_dict(),
        "frame_count": len(ft),
        "frames": [i.to_dict() for i in ft],
    }
    manifest = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), allow_nan=False
    )
    sha = hashlib.sha256(manifest.encode()).hexdigest()
    return TrackBSequenceExport(source_asset, ft, manifest, sha)

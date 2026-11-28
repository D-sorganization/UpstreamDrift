"""Unit tests for Track B Shadow Tracker observation export (T1-WP4, Refs #15837)."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
import math
from typing import Any

import numpy as np
import pytest

from src.shared.python.shadow_tracker import track_b_observation_export as tb
from src.shared.python.shadow_tracker.source_records import FrameIdentity, SourceAsset

pytestmark = pytest.mark.unit


def _asset(w: int = 128, h: int = 128, aid: str = "a-1") -> SourceAsset:
    """Create a synthetic SourceAsset fixture."""
    return SourceAsset(
        schema_version="shadow-tracker/source/1.0.0",
        asset_id=aid,
        source_uri="urn:a:1",
        content_sha256="a" * 64,
        width_px=w,
        height_px=h,
        rights_status="permitted",
        rights_note="ok",
    )


def _frame(aid: str = "a-1", fid: str = "f-1", pts: int = 0) -> FrameIdentity:
    """Create a synthetic FrameIdentity fixture."""
    return FrameIdentity(
        schema_version="shadow-tracker/frame/1.1.0",
        asset_id=aid,
        shot_id="s1",
        swing_id="w1",
        camera_id="c1",
        frame_id=fid,
        pts_ticks=pts,
        timebase_numerator=1,
        timebase_denominator=30,
        physical_time_s=None,
        physical_time_reason="x",
        frame_sha256="b" * 64,
    )


def _ellipse(h: int, w: int, cx: float, cy: float, rx: float, ry: float) -> np.ndarray:
    """Render a filled boolean ellipse mask."""
    yy, xx = np.ogrid[:h, :w]
    return np.asarray(((xx - cx) / rx) ** 2 + ((yy - cy) / ry) ** 2 <= 1.0, bool)


def _line(h: int, w: int, x0: float, y0: float, x1: float, y1: float) -> np.ndarray:
    """Render a 1-pixel boolean line segment mask."""
    m = np.zeros((h, w), dtype=bool)
    for t in np.linspace(0.0, 1.0, 256):
        m[round(y0 + t * (y1 - y0)), round(x0 + t * (x1 - x0))] = True
    return m


@pytest.mark.parametrize("ang", [0.0, math.pi / 6, math.pi / 4, math.pi / 2, 2.0])
def test_clean_silhouette_and_shaft_recovery(ang: float) -> None:
    """Clean synthetic ellipse + shaft line recovers contour and shaft angle."""
    body = _ellipse(128, 128, 40.0, 64.0, 16.0, 30.0)
    dx, dy = 30.0 * math.cos(ang), 30.0 * math.sin(ang)
    shaft = _line(128, 128, 80 - dx, 64 - dy, 80 + dx, 64 + dy)
    res = tb.extract_track_b_frame(_frame(), body, shaft)
    bc = res.body_contour
    assert isinstance(bc, tb.ContourBoundary) and bc.area_px == int(np.sum(body))
    assert len(bc.points_xy) >= 20 and bc.perimeter_px > 50.0
    assert bc.bbox_xywh == (24, 34, 33, 61) and bc.centroid_xy is not None
    assert abs(bc.centroid_xy[0] - 40.0) < 1.0 and not res.occluded
    seg = res.visible_shaft_segments[0]
    assert len(res.visible_shaft_segments) == 1
    assert isinstance(seg, tb.ShaftSegmentObservation)
    diff = abs(seg.angle_rad - ang)
    assert min(diff, abs(math.pi - diff)) < 0.05 and abs(seg.length_px - 60.0) < 4.0
    assert 0.0 < res.angular_uncertainty_rad < math.pi / 2.0


def test_occlusion_blur_and_short_segment_uncertainty() -> None:
    """Full occlusion gives pi/2; partial/short/blur inflate uncertainty."""
    ex, body = tb.extract_track_b_frame, _ellipse(128, 128, 30.0, 64.0, 12.0, 24.0)
    dot = np.zeros((128, 128), dtype=bool)
    dot[64, 90] = True
    full = _line(128, 128, 50.0, 20.0, 110.0, 100.0)
    for shaft, occ in ((np.zeros((128, 128), bool), 0.0), (dot, 0.0), (full, 1.0)):
        r = ex(_frame(), body, shaft, occlusion_fraction=occ)
        assert r.occluded and r.visible_shaft_segments == ()
        assert r.angular_uncertainty_rad == pytest.approx(math.pi / 2.0)
    split = full.copy()
    split[55:65, :] = False
    clean = ex(_frame(), body, full)
    part = ex(_frame(), body, split, occlusion_fraction=0.45)
    short = ex(_frame(), body, _line(128, 128, 70, 40, 80, 50))
    assert part.occluded and len(part.visible_shaft_segments) == 2
    assert clean.angular_uncertainty_rad < part.angular_uncertainty_rad < math.pi / 2
    assert short.angular_uncertainty_rad > clean.angular_uncertainty_rad
    rs = [ex(_frame(), body, full, motion_blur_score=v) for v in (0.0, 0.3, 0.7, 1.0)]
    u = [r.angular_uncertainty_rad for r in rs]
    assert all(0.0 < p < c < math.pi / 2 for p, c in zip(u[:-1], u[1:], strict=True))


def test_provenance_binding_manifest_determinism_and_immutability() -> None:
    """Export binds SourceAsset, FrameIdentity, SHA-256 digests, and non-physical."""
    asset = _asset(w=64, h=64)
    body, shaft = _ellipse(64, 64, 20, 32, 8, 14), _line(64, 64, 32, 12, 56, 48)
    f0 = tb.extract_track_b_frame(_frame(fid="f-0", pts=0), body, shaft)
    flip = body.copy()
    flip[0, 0] = not flip[0, 0]
    f1 = tb.extract_track_b_frame(_frame(fid="f-1", pts=1), flip, shaft)
    assert f0.body_mask_sha256 != f1.body_mask_sha256
    assert f0.mask_digest_sha256 != f1.mask_digest_sha256
    exp = tb.export_track_b_sequence
    sa = exp(asset, [f0, f1])
    assert isinstance(sa, tb.TrackBSequenceExport)
    assert sa.manifest_json == exp(asset, [f0, f1]).manifest_json
    assert sa.manifest_sha256 == hashlib.sha256(sa.manifest_json.encode()).hexdigest()
    for rec in (f0, sa):
        assert rec.schema_version == "shadow-tracker/source/1.0.0"
        assert rec.physical_time_verified is False
        assert rec.qualification == "image_observations_only"
    doc: dict[str, Any] = json.loads(sa.manifest_json)
    assert doc["schema_version"] == "shadow-tracker/source/1.0.0"
    assert doc["frame_count"] == 2 and doc["physical_time_verified"] is False
    assert doc["qualification"] == "image_observations_only"
    assert doc["source_asset"] == asset.to_dict()
    with pytest.raises((AttributeError, TypeError)):
        f0.occluded = True  # type: ignore[misc]


def test_fail_closed_masks_scores_sequence_and_provenance() -> None:
    """Rejects bad dimensions, non-bool/corrupt masks, bad scores, and mismatches."""
    ex, ok = tb.extract_track_b_frame, np.zeros((8, 8), dtype=bool)
    z = np.zeros
    b1: list[Any] = [z(8, bool), z((4, 4, 1), bool), [[True]], b"x", z((0, 8), bool)]
    b2: list[Any] = [z((8, 8), np.uint8), np.full((8, 8), float("nan"))]
    for bad in (*b1, *b2, np.array([[None]], object)):
        for bm, sm in ((bad, ok), (ok, bad)):
            with pytest.raises((TypeError, ValueError)):
                ex(_frame(), bm, sm)
    for bs in (-0.01, 1.01, float("nan"), float("inf"), True, "0.5"):
        with pytest.raises((TypeError, ValueError)):
            ex(_frame(), ok, ok, occlusion_fraction=bs)  # type: ignore[arg-type]
        with pytest.raises((TypeError, ValueError)):
            ex(_frame(), ok, ok, motion_blur_score=bs)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="mismatch"):
        ex(_frame(), ok, np.zeros((8, 9), bool))
    asset = _asset(8, 8, "a-a")
    with pytest.raises(ValueError):
        tb.export_track_b_sequence(asset, [])
    with pytest.raises(ValueError, match="mismatch"):
        tb.export_track_b_sequence(asset, [ex(_frame("a-b", "f", 0), ok, ok)])
    f1, f0 = ex(_frame("a-a", "f1", 1), ok, ok), ex(_frame("a-a", "f0", 0), ok, ok)
    with pytest.raises(ValueError, match="mismatch"):
        tb.export_track_b_sequence(_asset(16, 16, "a-a"), [f1])
    with pytest.raises(ValueError, match="Presentation time"):
        tb.export_track_b_sequence(asset, [f1, f0])
    with pytest.raises(ValueError, match="physical_time_verified"):
        replace(f1, physical_time_verified=True)
    with pytest.raises(ValueError, match="qualification"):
        replace(f1, qualification="validated_profile")
    assert isinstance(f1, tb.TrackBFrameExport)

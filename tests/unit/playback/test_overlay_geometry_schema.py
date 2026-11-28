"""Tests for versioned overlay and geometry schema referencing canonical-v2 (#15662)."""

from __future__ import annotations

import copy
import json
import math
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.model_appearance.canonical_golfer_gate import (
    NonCanonicalGolferError,
    plant_hash,
)
from src.shared.python.model_appearance.golfer_model_versions import (
    current_golfer_model,
)
from src.shared.python.motion_matching.execution.spec_revisions import (
    with_current_shaft,
)
from src.shared.python.playback.overlay_geometry_schema import (
    PLANE_KINDS,
    SCHEMA_VERSION,
    OverlayGeometryBundle,
    OverlayGeometryHeader,
    PlaneGeometrySpec,
    QuiverFieldSpec,
    build_overlay_geometry_bundle,
)

pytestmark = pytest.mark.unit
REPO = Path(__file__).resolve().parents[3]
BASE_SPEC: dict[str, Any] = json.loads(
    (REPO / "tests/fixtures/club_face/spec_driver.json").read_text(encoding="utf-8")
)


def _canonical_spec() -> dict[str, Any]:
    """Return a canonical golfer spec aligned with the current model registry."""
    spec = copy.deepcopy(BASE_SPEC)
    spec["subject"]["hand_visual_backend"] = "skinned_human"
    spec["subject"]["body_visual_backend"] = "skinned_cc0"
    spec["spec_revision"] = {"version": 1, "configuration": "skinned_hand_wrists"}
    return with_current_shaft(spec)


def _planes() -> tuple[PlaneGeometrySpec, ...]:
    """Return sample plane specs covering all four biomechanical plane kinds."""
    s, u, p = 1.0 / math.sqrt(2.0), (1.0, 0.0, 0.0), PlaneGeometrySpec
    return (
        p("kwon", "kwon_fsp", (0.0, 0.2, 0.9), (0.0, -s, s), u),
        p("mack", "mackenzie_net_force", (0.1, 0.2, 1.0), (0.0, 0.0, 1.0), u),
        p("tm", "trackman", (0.0, 0.0, 0.05), (0.0, -0.5, math.sqrt(0.75)), u),
        p("hogan", "hogan", (0.0, 0.3, 1.15), (0.0, -0.6, 0.8), u),
    )


def _quivers() -> tuple[QuiverFieldSpec, ...]:
    """Return sample quiver field specs for WSCG force and velocity overlays."""
    q = QuiverFieldSpec
    return (
        q("q_force", "force", "N", "lead_hand", 0.002, 0.8, 2, "#FF5522"),
        q("q_vel", "velocity", "m/s", "clubhead", 0.02, 1.2, 1, "#22AAFF"),
    )


@pytest.mark.parametrize("part", ["capsule", "loft", "spec_revision", "shaft"])
def test_non_canonical_golfer_spec_rejected(part: str) -> None:
    """Non-canonical golfer specs are rejected via require_canonical_golfer."""
    spec = _canonical_spec()
    if part == "capsule":
        spec["subject"]["hand_visual_backend"] = "capsule"
    elif part == "loft":
        spec["subject"]["body_visual_backend"] = "loft"
    elif part == "spec_revision":
        spec["spec_revision"] = {"configuration": "old"}
    else:
        spec["club"]["shaft"] = {"model": "rigid"}
    with pytest.raises(NonCanonicalGolferError) as exc_info:
        build_overlay_geometry_bundle(spec, planes=_planes(), quivers=_quivers())
    assert exc_info.value.part == part


def test_plane_rejects_non_unit_or_non_finite_normal() -> None:
    """PlaneGeometrySpec rejects non-unit and non-finite normals fail-closed."""
    bad = [
        (0.0, 0.0, 0.0),
        (1.0, 1.0, 0.0),
        (0.0, 0.0, 2.0),
        (math.nan, 0.0, 1.0),
        (0.0, math.inf, 0.0),
    ]
    for normal in bad:
        with pytest.raises(ValueError):
            PlaneGeometrySpec("bad", "kwon_fsp", (0.0, 0.0, 0.0), normal)


def test_plane_and_quiver_reject_invalid_fields() -> None:
    """PlaneGeometrySpec and QuiverFieldSpec validate fields fail-closed."""
    p_base: dict[str, Any] = {
        "plane_id": "p",
        "kind": "hogan",
        "origin_m": (0.0, 0.0, 0.0),
        "normal": (0.0, 0.0, 1.0),
    }
    p_cases: list[tuple[dict[str, Any], str]] = [
        ({"kind": "unknown"}, "kind"),
        ({"origin_m": (0.0, math.nan, 0.0)}, "origin_m"),
        ({"tangent_u": (0.0, 0.0, 1.0)}, "orthogonal"),
        ({"extent_m": (0.0, 1.0)}, "extent_m"),
        ({"frame": "bad_frame"}, "frame"),
    ]
    for p_kw, match in p_cases:
        with pytest.raises(ValueError, match=match):
            PlaneGeometrySpec(**{**p_base, **p_kw})
    q_base: dict[str, Any] = {
        "field_id": "q1",
        "quantity": "force",
        "units": "N",
        "anchor_body": "lead_hand",
    }
    q_cases: list[dict[str, Any]] = [
        {"field_id": ""},
        {"scale_m_per_unit": 0.0},
        {"sample_stride": 0},
        {"color_hex": "red"},
    ]
    for q_kw in q_cases:
        with pytest.raises(ValueError):
            QuiverFieldSpec(**{**q_base, **q_kw})


def test_bundle_round_trip_and_sha256_across_all_planes_and_quivers() -> None:
    """Bundle serializes deterministically and verifies SHA-256 across all planes."""
    spec, planes, quivers = _canonical_spec(), _planes(), _quivers()
    assert tuple(p.kind for p in planes) == PLANE_KINDS
    a = build_overlay_geometry_bundle(spec, planes=planes, quivers=quivers)
    b = build_overlay_geometry_bundle(spec, planes=planes, quivers=quivers)
    cur = current_golfer_model()
    assert a.header.schema_version == SCHEMA_VERSION == "overlay-geometry/1.0.0"
    assert a.header.golfer_model_version == cur.version
    assert a.header.plant_hash == plant_hash(spec)
    assert len(a.content_sha256) == 64 and a.content_sha256 == b.content_sha256
    assert a.verify_content_sha256()
    raw = json.loads(json.dumps(a.to_dict(), sort_keys=True))
    assert OverlayGeometryBundle.from_dict(raw) == a
    alt = build_overlay_geometry_bundle(spec, planes=planes[:-1], quivers=quivers)
    assert alt.content_sha256 != a.content_sha256


def test_from_dict_rejects_unknown_fields_and_mismatched_schema_version() -> None:
    """OverlayGeometryBundle.from_dict rejects unknown keys, versions, and hashes."""
    b = build_overlay_geometry_bundle(_canonical_spec(), _planes(), _quivers())
    cases: list[tuple[tuple[Any, ...], Any, str]] = [
        (("unexpected",), True, "Unknown"),
        (("planes", 0, "extra"), 1, "Unknown"),
        (("quivers", 0, "extra"), 1, "Unknown"),
        (("header", "extra"), "x", "Unknown"),
        (("schema_version",), "overlay-geometry/9.9", "schema_version"),
        (("header", "schema_version"), "bad", "schema_version"),
        (("content_sha256",), "0" * 64, "content_sha256"),
    ]
    for path, val, match in cases:
        d = copy.deepcopy(b.to_dict())
        cur: Any = d
        for key in path[:-1]:
            cur = cur[key]
        cur[path[-1]] = val
        with pytest.raises(ValueError, match=match):
            OverlayGeometryBundle.from_dict(d)


def test_header_and_bundle_reject_invalid_metadata_and_duplicates() -> None:
    """Header and bundle reject invalid hashes, frames, and duplicate IDs."""
    spec = _canonical_spec()
    hd = build_overlay_geometry_bundle(spec).header.to_dict()
    with pytest.raises(ValueError, match="plant_hash"):
        OverlayGeometryHeader.from_dict(dict(hd, plant_hash="bad"))
    with pytest.raises(ValueError, match="world_frame"):
        OverlayGeometryHeader.from_dict(dict(hd, world_frame="bad"))
    with pytest.raises(ValueError, match="Duplicate plane_id"):
        build_overlay_geometry_bundle(spec, planes=(_planes()[0], _planes()[0]))
    with pytest.raises(ValueError, match="Duplicate field_id"):
        build_overlay_geometry_bundle(spec, quivers=(_quivers()[0], _quivers()[0]))

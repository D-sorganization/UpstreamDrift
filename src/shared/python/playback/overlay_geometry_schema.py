"""Versioned overlay and geometry schema referencing canonical-v2 (#15660, #15662)."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, fields
import hashlib
import json
import math
import re
from typing import Any, Self

from src.shared.python.core.canonical_floats import canonical_floats
from src.shared.python.model_appearance.canonical_golfer_gate import (
    plant_hash,
    require_canonical_golfer,
)

SCHEMA_VERSION = "overlay-geometry/1.0.0"
PLANE_KINDS: tuple[str, ...] = ("kwon_fsp", "mackenzie_net_force", "trackman", "hogan")
ALLOWED_WORLD_FRAMES: tuple[str, ...] = ("world_Zup", "adr0041_world")
_UNIT_TOL, _ORTHO_TOL = 1e-6, 1e-5
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_HEX_RE = re.compile(r"^#[0-9A-Fa-f]{6}(?:[0-9A-Fa-f]{2})?$")
_BUNDLE_KEYS = frozenset(
    {"schema_version", "header", "planes", "quivers", "content_sha256"}
)


def _req_strs(obj: Any, *attrs: str) -> None:
    """Validate that each named attribute on ``obj`` is a non-empty string."""
    for attr in attrs:
        val = getattr(obj, attr)
        if not isinstance(val, str) or not val.strip():
            raise ValueError(f"{attr} must be a non-empty string")


def _req_real(value: Any, name: str, *, pos: bool = False) -> float:
    """Validate a finite (and optionally strictly positive) real number."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a finite real number")
    num = float(value)
    if not math.isfinite(num) or (pos and num <= 0.0):
        raise ValueError(f"{name} must be finite" + (" and positive" if pos else ""))
    return num


def _nums(value: Any, name: str, n: int, *, pos: bool = False) -> tuple[float, ...]:
    """Validate a fixed-length sequence of finite (or positive) real numbers."""
    if not isinstance(value, (tuple, list)) or len(value) != n:
        raise ValueError(f"{name} must be a {n}-element sequence")
    return tuple(_req_real(value[i], f"{name}[{i}]", pos=pos) for i in range(n))


def _unit_vec3(value: Any, name: str) -> tuple[float, float, float]:
    """Validate a 3D unit vector within ``_UNIT_TOL``."""
    x, y, z = _nums(value, name, 3)
    norm = math.sqrt(x * x + y * y + z * z)
    if not math.isclose(norm, 1.0, rel_tol=0.0, abs_tol=_UNIT_TOL):
        raise ValueError(f"{name} must be a unit vector (norm={norm})")
    return (x, y, z)


def _exact_keys(data: Any, expected: frozenset[str], label: str) -> Mapping[str, Any]:
    """Validate that ``data`` is a mapping with the exact expected key set."""
    if not isinstance(data, Mapping):
        raise ValueError(f"{label} must be a mapping")
    keys = set(data.keys())
    if extra := keys - expected:
        raise ValueError(f"Unknown keys in {label}: {sorted(extra)}")
    if missing := expected - keys:
        raise ValueError(f"Missing keys in {label}: {sorted(missing)}")
    return data


def _sha256_hex(value: Any, name: str) -> str:
    """Validate a 64-character lowercase hex SHA-256 digest."""
    if not isinstance(value, str) or not _SHA256_RE.match(value):
        raise ValueError(f"{name} must be a 64-char lowercase hex SHA-256")
    return value


def _unique(
    items: Sequence[Any], cls: type[Any], attr: str, label: str
) -> tuple[Any, ...]:
    """Validate element types and uniqueness of ``attr`` in a sequence."""
    seq = tuple(items)
    if any(not isinstance(x, cls) for x in seq):
        raise ValueError(f"{label} entries must be {cls.__name__}")
    if len({getattr(x, attr) for x in seq}) != len(seq):
        raise ValueError(f"Duplicate {attr} in {label}")
    return seq


class _DictCodec:
    """Shared fail-closed dict serialization for flat frozen dataclasses."""

    def to_dict(self) -> dict[str, Any]:
        """Serialize this dataclass to a JSON-safe dict with list sequences."""
        raw = asdict(self)  # type: ignore[call-overload]
        return {k: list(v) if isinstance(v, tuple) else v for k, v in raw.items()}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> Self:
        """Deserialize from a mapping, rejecting unknown or missing keys."""
        expected = frozenset(f.name for f in fields(cls))  # type: ignore[arg-type]
        m = _exact_keys(data, expected, cls.__name__)
        return cls(**{k: m[k] for k in expected})


_HDR_STRS = (
    "golfer_model_version",
    "spec_revision",
    "hand_visual_backend",
    "body_visual_backend",
    "shaft_model",
)
_QVR_STRS = ("field_id", "quantity", "units", "anchor_body")


@dataclass(frozen=True)
class OverlayGeometryHeader(_DictCodec):
    """Versioned canonical-v2 identity header for an overlay geometry bundle."""

    golfer_model_version: str
    spec_revision: str
    hand_visual_backend: str
    body_visual_backend: str
    shaft_model: str
    plant_hash: str
    world_frame: str = "world_Zup"
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate canonical golfer header metadata and schema version."""
        if self.schema_version != SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {SCHEMA_VERSION!r}")
        _req_strs(self, *_HDR_STRS)
        _sha256_hex(self.plant_hash, "plant_hash")
        if self.world_frame not in ALLOWED_WORLD_FRAMES:
            raise ValueError(f"Invalid world_frame: {self.world_frame!r}")


@dataclass(frozen=True)
class PlaneGeometrySpec(_DictCodec):
    """Biomechanical reference plane geometry specification."""

    plane_id: str
    kind: str
    origin_m: tuple[float, float, float]
    normal: tuple[float, float, float]
    tangent_u: tuple[float, float, float] | None = None
    extent_m: tuple[float, float] = (1.5, 1.5)
    frame: str = "world_Zup"
    label: str = ""

    def __post_init__(self) -> None:
        """Validate plane kind, unit normal, optional orthogonal tangent, and extent."""
        _req_strs(self, "plane_id")
        if not isinstance(self.kind, str) or self.kind not in PLANE_KINDS:
            raise ValueError(f"Invalid kind: {self.kind!r}")
        ox, oy, oz = _nums(self.origin_m, "origin_m", 3)
        object.__setattr__(self, "origin_m", (ox, oy, oz))
        un = _unit_vec3(self.normal, "normal")
        object.__setattr__(self, "normal", un)
        if self.tangent_u is not None:
            ut = _unit_vec3(self.tangent_u, "tangent_u")
            dot = un[0] * ut[0] + un[1] * ut[1] + un[2] * ut[2]
            if abs(dot) > _ORTHO_TOL:
                raise ValueError(f"tangent_u must be orthogonal to normal (dot={dot})")
            object.__setattr__(self, "tangent_u", ut)
        ex, ey = _nums(self.extent_m, "extent_m", 2, pos=True)
        object.__setattr__(self, "extent_m", (ex, ey))
        if self.frame not in ALLOWED_WORLD_FRAMES:
            raise ValueError(f"Invalid frame: {self.frame!r}")
        if not isinstance(self.label, str):
            raise ValueError("label must be a string")


@dataclass(frozen=True)
class QuiverFieldSpec(_DictCodec):
    """WSCG or biomechanical vector-field quiver overlay specification."""

    field_id: str
    quantity: str
    units: str
    anchor_body: str
    scale_m_per_unit: float = 0.001
    max_arrow_length_m: float = 1.0
    sample_stride: int = 1
    color_hex: str = "#FF8844"

    def __post_init__(self) -> None:
        """Validate quiver field identifiers, scaling bounds, stride, and color."""
        _req_strs(self, "field_id", "quantity", "units", "anchor_body")
        s = _req_real(self.scale_m_per_unit, "scale_m_per_unit", pos=True)
        m_len = _req_real(self.max_arrow_length_m, "max_arrow_length_m", pos=True)
        object.__setattr__(self, "scale_m_per_unit", s)
        object.__setattr__(self, "max_arrow_length_m", m_len)
        st = self.sample_stride
        if isinstance(st, bool) or not isinstance(st, int) or st < 1:
            raise ValueError("sample_stride must be an integer >= 1")
        if not isinstance(self.color_hex, str) or not _HEX_RE.match(self.color_hex):
            raise ValueError(f"Invalid color_hex: {self.color_hex!r}")


@dataclass(frozen=True)
class OverlayGeometryBundle:
    """Versioned overlay geometry bundle sealed with a canonical SHA-256 digest."""

    header: OverlayGeometryHeader
    planes: tuple[PlaneGeometrySpec, ...] = ()
    quivers: tuple[QuiverFieldSpec, ...] = ()
    content_sha256: str = ""

    def __post_init__(self) -> None:
        """Validate component types, unique IDs, and SHA-256 content integrity."""
        if not isinstance(self.header, OverlayGeometryHeader):
            raise ValueError("header must be an OverlayGeometryHeader")
        p_t = _unique(self.planes, PlaneGeometrySpec, "plane_id", "planes")
        q_t = _unique(self.quivers, QuiverFieldSpec, "field_id", "quivers")
        object.__setattr__(self, "planes", p_t)
        object.__setattr__(self, "quivers", q_t)
        expected = self.compute_content_sha256()
        if self.content_sha256 == "":
            object.__setattr__(self, "content_sha256", expected)
            return
        _sha256_hex(self.content_sha256, "content_sha256")
        if self.content_sha256 != expected:
            raise ValueError("content_sha256 mismatch")

    def _payload_dict(self) -> dict[str, Any]:
        """Build the unhashed canonical payload dictionary."""
        return {
            "schema_version": self.header.schema_version,
            "header": self.header.to_dict(),
            "planes": [p.to_dict() for p in self.planes],
            "quivers": [q.to_dict() for q in self.quivers],
        }

    def compute_content_sha256(self) -> str:
        """Compute the canonical-float SHA-256 hex digest of the bundle payload."""
        raw = canonical_floats(self._payload_dict())
        text = json.dumps(raw, sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def verify_content_sha256(self) -> bool:
        """Return whether ``content_sha256`` matches the recomputed payload digest."""
        return self.content_sha256 == self.compute_content_sha256()

    def to_dict(self) -> dict[str, Any]:
        """Serialize the bundle and its SHA-256 digest to a JSON-safe dictionary."""
        return {**self._payload_dict(), "content_sha256": self.content_sha256}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> OverlayGeometryBundle:
        """Deserialize a bundle fail-closed, checking schema version and SHA-256."""
        m = _exact_keys(data, _BUNDLE_KEYS, "bundle")
        if m["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Invalid schema_version: {m['schema_version']!r}")
        p_raw, q_raw = m["planes"], m["quivers"]
        if not isinstance(p_raw, (list, tuple)) or not isinstance(q_raw, (list, tuple)):
            raise ValueError("planes and quivers must be sequences")
        return cls(
            header=OverlayGeometryHeader.from_dict(m["header"]),
            planes=tuple(PlaneGeometrySpec.from_dict(p) for p in p_raw),
            quivers=tuple(QuiverFieldSpec.from_dict(q) for q in q_raw),
            content_sha256=_sha256_hex(m["content_sha256"], "content_sha256"),
        )


def build_overlay_geometry_bundle(
    spec: Mapping[str, Any],
    planes: Sequence[PlaneGeometrySpec] = (),
    quivers: Sequence[QuiverFieldSpec] = (),
    world_frame: str = "world_Zup",
) -> OverlayGeometryBundle:
    """Build a canonical-v2 gated overlay geometry bundle from a golfer spec."""
    cur = require_canonical_golfer(spec)
    rev_raw = spec.get("spec_revision")
    rev = (
        str(rev_raw.get("configuration", ""))
        if isinstance(rev_raw, Mapping)
        else cur.spec_revision_configuration
    )
    hdr = OverlayGeometryHeader(
        cur.version,
        rev,
        cur.hand_visual_backend,
        cur.body_visual_backend,
        cur.shaft_model,
        plant_hash(spec),
        world_frame,
    )
    return OverlayGeometryBundle(hdr, tuple(planes), tuple(quivers))

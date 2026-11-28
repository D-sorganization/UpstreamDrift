"""Immutable ModelDefinition, SubjectProfile, and MatchArtifact contracts (#15672)."""

from __future__ import annotations

import copy
import hashlib
import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass, field, fields, is_dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any

from src.shared.python.core.canonical_floats import canonical_floats
from src.shared.python.core.handedness import validate_handedness
from src.shared.python.model_appearance.canonical_golfer_gate import (
    plant_hash as compute_plant_hash,
    require_canonical_golfer,
)

SCHEMA_VERSION = "model-library/1.0.0"
MANIFEST_FILENAME = "manifest.json"
MUTABLE_REFS = frozenset({"latest", "main", "master", "head", "current", "default"})
_SHA256_RE = re.compile(r"^[0-9a-f]{64}$")
_DIRS = {
    "model_definition": "models",
    "subject_profile": "subjects",
    "match_artifact": "matches",
}


class ModelLibraryError(ValueError):
    """Base error for model library schema, reference, and integrity checks."""


class MutableReferenceError(ModelLibraryError):
    """Raised when a mutable tag or unpinned relative path is used."""


class ArtifactIntegrityError(ModelLibraryError):
    """Raised when an artifact fails SHA-256 verification or JSON parsing."""


class MissingArtifactError(ModelLibraryError):
    """Raised when a required SHA-256-pinned artifact is absent."""


def _reject_non_finite(token: str) -> Any:
    """Reject non-finite JSON numeric constants fail-closed."""
    raise ArtifactIntegrityError(f"non-finite JSON constant {token!r}")


def _to_plain(obj: Any) -> Any:
    """Convert dataclasses, mappings, and sequences to plain JSON types."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return {f.name: _to_plain(getattr(obj, f.name)) for f in fields(obj)}
    if isinstance(obj, Mapping):
        return {str(k): _to_plain(v) for k, v in obj.items()}
    if isinstance(obj, (tuple, list)):
        return [_to_plain(v) for v in obj]
    return obj


def _canonical_sha256(payload: Mapping[str, Any]) -> str:
    """Compute canonical-float deterministic SHA-256 hex digest of a mapping."""
    norm = canonical_floats(_to_plain(payload))
    raw = json.dumps(norm, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _read_json_strict(path: Path) -> tuple[bytes, dict[str, Any]]:
    """Read raw bytes and strictly parse a JSON object from disk."""
    try:
        raw = path.read_bytes()
        parsed = json.loads(raw.decode("utf-8"), parse_constant=_reject_non_finite)
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ArtifactIntegrityError(f"failed reading {path}: {exc}") from exc
    if not isinstance(parsed, dict):
        raise ArtifactIntegrityError(f"{path} must contain a JSON object")
    return raw, parsed


def validate_immutable_ref(ref: object, *, label: str = "ref") -> str:
    """Validate and normalize an immutable SHA-256 ref; reject mutable refs."""
    if not isinstance(ref, str) or not ref.strip() or ref != ref.strip():
        raise MutableReferenceError(f"{label} must be a non-empty SHA-256")
    low = ref.lower()
    if low in MUTABLE_REFS or "/" in ref or "\\" in ref or "." in ref:
        raise MutableReferenceError(f"{label} {ref!r} is mutable or unpinned")
    cand = ref[7:] if low.startswith("sha256:") else ref
    if not _SHA256_RE.fullmatch(cand):
        raise MutableReferenceError(f"{label} {ref!r} is not 64-char SHA-256")
    return cand


def _validate_id(value: object, *, label: str) -> str:
    """Validate a non-empty identifier that cannot masquerade as a mutable ref."""
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{label} must be non-empty")
    low = value.lower()
    if low in MUTABLE_REFS or "/" in value or "\\" in value or ".json" in low:
        raise MutableReferenceError(f"{label} {value!r} cannot be mutable")
    return value


def _empty_metrics() -> Mapping[str, float]:
    """Return an empty immutable metrics mapping."""
    return MappingProxyType({})


def _freeze_metrics(metrics: Mapping[str, float]) -> Mapping[str, float]:
    """Validate finite metric scalars and return an immutable mapping."""
    clean: dict[str, float] = {}
    for k, v in metrics.items():
        if isinstance(v, bool) or not isinstance(v, (int, float)):
            raise ValueError(f"metric {k!r} must be finite")
        num = float(v)
        if not math.isfinite(num):
            raise ValueError(f"metric {k!r} must be finite")
        clean[str(k)] = num
    return MappingProxyType(clean)


def _finalize_digest(rec: Any, field: str, kind: str | None = None) -> None:
    """Compute or verify the SHA-256 digest field on a frozen record."""
    if rec.schema_version != SCHEMA_VERSION:
        raise ArtifactIntegrityError(f"unsupported schema {rec.schema_version!r}")
    payload = _to_plain(rec)
    payload.pop(field, None)
    if kind is not None:
        payload["record_kind"] = kind
    expected = _canonical_sha256(payload)
    given = getattr(rec, field)
    if given == "":
        object.__setattr__(rec, field, expected)
        return
    if validate_immutable_ref(given, label=field) != expected:
        raise ArtifactIntegrityError(f"{type(rec).__name__} {field} mismatch")


@dataclass(frozen=True)
class ModelDefinition:
    """Immutable canonical golfer + club model definition pinned by SHA-256."""

    model_id: str
    spec_sha256: str
    plant_hash: str
    registry_version: str
    spec: Mapping[str, Any]
    content_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Enforce canonical golfer gate, immutable spec, and content SHA-256."""
        _validate_id(self.model_id, label="model_id")
        validate_immutable_ref(self.spec_sha256, label="spec_sha256")
        validate_immutable_ref(self.plant_hash, label="plant_hash")
        if self.registry_version != require_canonical_golfer(self.spec).version:
            raise ArtifactIntegrityError("registry_version mismatch")
        frozen = MappingProxyType(copy.deepcopy(dict(self.spec)))
        object.__setattr__(self, "spec", frozen)
        _finalize_digest(self, "content_sha256", "model_definition")

    @classmethod
    def from_spec(
        cls, spec: Mapping[str, Any], *, model_id: str, source_ref: str | None = None
    ) -> ModelDefinition:
        """Construct a canonical ModelDefinition from a spec mapping."""
        if source_ref is not None:
            validate_immutable_ref(source_ref, label="source_ref")
        ver = require_canonical_golfer(spec).version
        cp = copy.deepcopy(dict(spec))
        return cls(model_id, _canonical_sha256(cp), compute_plant_hash(cp), ver, cp)


@dataclass(frozen=True)
class SubjectProfile:
    """Immutable golfer subject profile pinned by SHA-256."""

    subject_id: str
    handedness: str
    height_m: float
    mass_kg: float
    source_ref: str | None = None
    content_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Validate subject biometrics, immutable source_ref, and digest."""
        _validate_id(self.subject_id, label="subject_id")
        object.__setattr__(self, "handedness", validate_handedness(self.handedness))
        h, m = self.height_m, self.mass_kg
        if h <= 0.0 or m <= 0.0 or not math.isfinite(h + m):
            raise ValueError("height_m and mass_kg must be finite and > 0")
        if self.source_ref is not None:
            ref = validate_immutable_ref(self.source_ref, label="source_ref")
            object.__setattr__(self, "source_ref", ref)
        _finalize_digest(self, "content_sha256", "subject_profile")


@dataclass(frozen=True)
class MatchArtifact:
    """Immutable match record binding ModelDefinition, SubjectProfile, trajectory."""

    match_id: str
    model_sha256: str
    subject_sha256: str
    trajectory_ref: str
    metrics: Mapping[str, float] = field(default_factory=_empty_metrics)
    content_sha256: str = ""
    schema_version: str = SCHEMA_VERSION

    def __post_init__(self) -> None:
        """Enforce immutable SHA-256 cross-references, finite metrics, and digest."""
        _validate_id(self.match_id, label="match_id")
        for f_name in ("model_sha256", "subject_sha256", "trajectory_ref"):
            val = validate_immutable_ref(getattr(self, f_name), label=f_name)
            object.__setattr__(self, f_name, val)
        object.__setattr__(self, "metrics", _freeze_metrics(self.metrics))
        _finalize_digest(self, "content_sha256", "match_artifact")


LibraryArtifact = ModelDefinition | SubjectProfile | MatchArtifact
_KIND_OF: dict[type[Any], str] = {
    ModelDefinition: "model_definition",
    SubjectProfile: "subject_profile",
    MatchArtifact: "match_artifact",
}


def _pick(data: Mapping[str, Any], *keys: str) -> dict[str, Any]:
    """Extract required string fields from a JSON mapping."""
    return {k: str(data[k]) for k in keys}


def _parse_artifact(d: Mapping[str, Any]) -> tuple[str, LibraryArtifact]:
    """Parse and integrity-verify a JSON object into a typed library artifact."""
    kind = str(d.get("record_kind", ""))
    sha = validate_immutable_ref(d.get("content_sha256"), label="content_sha256")
    ver = str(d.get("schema_version", ""))
    if kind == "model_definition":
        kw = _pick(d, "model_id", "spec_sha256", "plant_hash", "registry_version")
        return kind, ModelDefinition(
            **kw, spec=dict(d["spec"]), content_sha256=sha, schema_version=ver
        )
    if kind == "subject_profile":
        kw = _pick(d, "subject_id", "handedness")
        ref = str(d["source_ref"]) if d.get("source_ref") is not None else None
        kw.update(
            height_m=float(d["height_m"]), mass_kg=float(d["mass_kg"]), source_ref=ref
        )
        return kind, SubjectProfile(**kw, content_sha256=sha, schema_version=ver)
    if kind == "match_artifact":
        kw = _pick(d, "match_id", "model_sha256", "subject_sha256", "trajectory_ref")
        kw.update(
            metrics=dict(d.get("metrics", {})), content_sha256=sha, schema_version=ver
        )
        return kind, MatchArtifact(**kw)
    raise ArtifactIntegrityError(f"unknown artifact record_kind {kind!r}")


def _write_json(path: Path, payload: Mapping[str, Any]) -> Path:
    """Write canonical JSON to disk and return the path."""
    path.parent.mkdir(parents=True, exist_ok=True)
    text = json.dumps(_to_plain(payload), sort_keys=True, indent=2, allow_nan=False)
    path.write_text(text + "\n", encoding="utf-8")
    return path


def save_artifact(root: Path, artifact: LibraryArtifact) -> Path:
    """Write an immutable artifact under its content-addressed path in root."""
    kind = _KIND_OF[type(artifact)]
    payload = {**_to_plain(artifact), "record_kind": kind}
    return _write_json(root / _DIRS[kind] / f"{artifact.content_sha256}.json", payload)


@dataclass(frozen=True)
class ModelLibraryManifest:
    """Deterministic, rebuildable index of content-addressed library artifacts."""

    schema_version: str
    entries: Mapping[str, Mapping[str, Any]]
    manifest_sha256: str = ""

    def __post_init__(self) -> None:
        """Verify manifest schema, sorted entries, cross-references, and digest."""
        ordered = {
            k: MappingProxyType(dict(v)) for k, v in sorted(self.entries.items())
        }
        for sha, e in ordered.items():
            validate_immutable_ref(sha, label="content_sha256")
            validate_immutable_ref(e.get("file_sha256"), label="file_sha256")
            if e.get("kind") not in _DIRS:
                raise ArtifactIntegrityError(f"unknown entry kind {e.get('kind')!r}")
            if e["kind"] == "match_artifact":
                deps = e.get("deps", ())
                if len(deps) != 2:
                    raise ArtifactIntegrityError("match_artifact needs 2 deps")
                m_e, s_e = ordered.get(deps[0]), ordered.get(deps[1])
                if not (m_e and m_e["kind"] == "model_definition"):
                    raise MissingArtifactError(f"match missing model {deps[0]}")
                if not (s_e and s_e["kind"] == "subject_profile"):
                    raise MissingArtifactError(f"match missing subject {deps[1]}")
        object.__setattr__(self, "entries", MappingProxyType(ordered))
        _finalize_digest(self, "manifest_sha256")

    def save(self, root: Path) -> Path:
        """Write manifest.json under root."""
        return _write_json(root / MANIFEST_FILENAME, _to_plain(self))

    @classmethod
    def load(cls, root: Path) -> ModelLibraryManifest:
        """Load and integrity-verify manifest.json from root."""
        _, parsed = _read_json_strict(root / MANIFEST_FILENAME)
        raw = parsed.get("entries")
        if not isinstance(raw, dict):
            raise ArtifactIntegrityError("manifest entries must be a JSON object")
        sha = validate_immutable_ref(parsed.get("manifest_sha256"), label="manifest")
        return cls(str(parsed["schema_version"]), raw, sha)

    @classmethod
    def rebuild_from_directory(cls, root: Path) -> ModelLibraryManifest:
        """Deterministically scan root, verify all artifacts, and build manifest."""
        entries: dict[str, dict[str, Any]] = {}
        for path in sorted(root.rglob("*.json")) if root.exists() else ():
            if path.name == MANIFEST_FILENAME:
                continue
            raw, parsed = _read_json_strict(path)
            kind, art = _parse_artifact(parsed)
            sha = art.content_sha256
            if sha in entries:
                raise ArtifactIntegrityError(f"duplicate digest {sha}")
            deps: tuple[str, ...] = ()
            if isinstance(art, MatchArtifact):
                deps = (art.model_sha256, art.subject_sha256)
            entries[sha] = {
                "kind": kind,
                "file_sha256": hashlib.sha256(raw).hexdigest(),
                "path": path.relative_to(root).as_posix(),
                "deps": deps,
            }
        return cls(SCHEMA_VERSION, entries)


@dataclass(frozen=True)
class ModelLibraryResolver:
    """Fail-closed resolver verifying SHA-256 integrity on every artifact read."""

    root: Path
    manifest: ModelLibraryManifest

    @classmethod
    def from_directory(
        cls, root: Path, *, rebuild: bool = False
    ) -> ModelLibraryResolver:
        """Create a resolver by loading or rebuilding the manifest under root."""
        if rebuild or not (root / MANIFEST_FILENAME).is_file():
            return cls(root, ModelLibraryManifest.rebuild_from_directory(root))
        return cls(root, ModelLibraryManifest.load(root))

    def resolve(self, ref: str, *, expected_kind: str | None = None) -> LibraryArtifact:
        """Resolve an immutable SHA-256 ref from disk with strict verification."""
        digest = validate_immutable_ref(ref, label="ref")
        entries = self.manifest.entries
        entry = entries.get(digest)
        if entry is None:
            raise MissingArtifactError(f"artifact {digest} not found")
        if expected_kind is not None and entry["kind"] != expected_kind:
            raise ArtifactIntegrityError(f"artifact {digest} kind mismatch")
        rel = str(entry["path"])
        raw, parsed = _read_json_strict(self.root / rel)
        if hashlib.sha256(raw).hexdigest() != entry["file_sha256"]:
            raise ArtifactIntegrityError(f"tampering in {rel}")
        kind, art = _parse_artifact(parsed)
        if kind != entry["kind"] or art.content_sha256 != digest:
            raise ArtifactIntegrityError(f"digest mismatch for {rel}")
        if isinstance(art, MatchArtifact):
            self.resolve(art.model_sha256, expected_kind="model_definition")
            self.resolve(art.subject_sha256, expected_kind="subject_profile")
        return art

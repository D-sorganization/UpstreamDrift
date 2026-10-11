"""Source-bound, visual-only resource closure for native OpenSim replay.

The shared replay bundle identifies its source XML. OpenSim visual meshes are
an adapter-specific extension, so their relative paths and byte digests live in
this separately versioned envelope rather than in an unrelated shared field.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path, PurePosixPath, PureWindowsPath
from tempfile import TemporaryDirectory
from typing import Any, Iterator
from urllib.parse import urlsplit

from defusedxml import ElementTree as SafeET
from defusedxml.common import DefusedXmlException
from src.engines.native_replay_contracts import require_sha256_digest

_SCHEMA_VERSION = "opensim-native-visual-resource-closure/1"
_MAX_SOURCE_BYTES = 128 * 1024 * 1024
_MAX_RESOURCE_COUNT = 128
_MAX_CLOSURE_BYTES = 512 * 1024 * 1024
_VISUAL_PURPOSE = "attached-geometry-visual-mesh"
_SUPPORTED_MESH_SUFFIXES = frozenset({".vtp"})


@dataclass(frozen=True, slots=True)
class VisualMeshReference:
    """One mesh path owned by a body frame's visual attached_geometry."""

    relative_path: str
    component_path: str


@dataclass(frozen=True, slots=True)
class SourceVisualResource:
    """Exact source-relative bytes and visual owner admitted by this policy."""

    relative_path: str
    component_path: str
    sha256: str
    purpose: str = _VISUAL_PURPOSE

    def __post_init__(self) -> None:
        _validate_relative_path(self.relative_path)
        if not isinstance(
            self.component_path, str
        ) or not self.component_path.startswith("/bodyset/"):
            raise ValueError("visual mesh must belong to an attached body geometry")
        require_sha256_digest(self.sha256, "visual resource sha256")
        if self.purpose != _VISUAL_PURPOSE:
            raise ValueError("resource purpose must be attached-geometry visual mesh")


@dataclass(frozen=True, slots=True)
class NativeMuscleResourceEnvelope:
    """Versioned adapter envelope around the unchanged shared replay contract."""

    replay_bundle: Any
    source_model_sha256: str
    resources: tuple[SourceVisualResource, ...]
    closure_sha256: str
    schema_version: str = _SCHEMA_VERSION

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
        require_sha256_digest(self.source_model_sha256, "source model sha256")
        if (
            getattr(
                getattr(self.replay_bundle, "model", None), "source_model_sha256", None
            )
            != self.source_model_sha256
        ):
            raise ValueError(
                "visual closure source identity differs from replay bundle"
            )
        resources = tuple(self.resources)
        if not resources or len(resources) > _MAX_RESOURCE_COUNT:
            raise ValueError(
                "visual resource closure must contain a bounded non-empty set"
            )
        if any(not isinstance(item, SourceVisualResource) for item in resources):
            raise TypeError("resources must contain SourceVisualResource values")
        keys = tuple(item.relative_path for item in resources)
        if len(set(keys)) != len(keys):
            raise ValueError("visual resource paths must be unique")
        if tuple(sorted(resources, key=lambda item: item.relative_path)) != resources:
            raise ValueError("visual resources must be in canonical path order")
        owners = tuple(item.component_path for item in resources)
        if len(set(owners)) != len(owners):
            raise ValueError("visual mesh component paths must be unique")
        expected = _closure_digest(self.source_model_sha256, resources)
        if self.closure_sha256 != expected:
            raise ValueError("visual resource closure digest differs")
        object.__setattr__(self, "resources", resources)


def _validate_relative_path(value: str) -> PurePosixPath:
    if (
        not isinstance(value, str)
        or not value
        or "\\" in value
        or urlsplit(value).scheme
        or PureWindowsPath(value).is_absolute()
        or PureWindowsPath(value).drive
    ):
        raise ValueError("visual resource path must be a contained relative path")
    path = PurePosixPath(value)
    if (
        path.is_absolute()
        or value != path.as_posix()
        or any(part in {"", ".", ".."} for part in path.parts)
    ):
        raise ValueError("visual resource path must be a contained relative path")
    return path


def _body_component_path(element: Any, parents: Mapping[Any, Any]) -> str:
    mesh = parents[element]
    attached = parents.get(mesh)
    body = parents.get(attached)
    if (
        mesh.tag != "Mesh"
        or attached is None
        or attached.tag != "attached_geometry"
        or body is None
        or body.tag != "Body"
        or parents.get(body) is None
        or parents[body].tag != "objects"
    ):
        raise ValueError("external resource is not body-owned attached visual geometry")
    body_set = parents[parents[body]]
    if body_set.tag != "BodySet":
        raise ValueError("visual mesh owner is not a native body")
    body_name = body.get("name")
    mesh_name = mesh.get("name")
    if not body_name or not mesh_name or "/" in body_name or "/" in mesh_name:
        raise ValueError("visual mesh and body need safe native component names")
    return f"/bodyset/{body_name}/{mesh_name}"


def discover_visual_meshes(raw: bytes) -> tuple[VisualMeshReference, ...]:
    """Parse exact external references and admit only body attached Mesh assets."""
    if not isinstance(raw, bytes) or not raw or len(raw) > _MAX_SOURCE_BYTES:
        raise ValueError("source XML must be non-empty and bounded")
    try:
        root = SafeET.fromstring(raw, forbid_dtd=True, forbid_entities=True)
    except (SafeET.ParseError, DefusedXmlException) as error:
        raise ValueError("native source XML must be self-contained") from error
    if root.tag != "OpenSimDocument" or len(root.findall("Model")) != 1:
        raise ValueError("native source must contain one inline OpenSim Model")
    parents = {child: parent for parent in root.iter() for child in parent}
    found: list[VisualMeshReference] = []
    seen_components: set[str] = set()
    for element in root.iter():
        tag = element.tag.lower()
        has_file_attribute = any(
            name.lower() in {"file", "filename", "href"} for name in element.attrib
        )
        is_file_property = tag in {"file", "filename", "file_name"} or tag.endswith(
            "_file"
        )
        if has_file_attribute:
            raise ValueError(
                "external source attributes need a separate identity policy"
            )
        if not is_file_property:
            continue
        if tag != "mesh_file" or not (relative := element.text or "").strip():
            raise ValueError(
                "external resources are limited to attached visual Mesh files"
            )
        relative = relative.strip()
        mesh_path = _validate_relative_path(relative)
        if mesh_path.suffix.lower() not in _SUPPORTED_MESH_SUFFIXES:
            raise ValueError("attached visual Mesh resource format is unsupported")
        component_path = _body_component_path(element, parents)
        if component_path in seen_components:
            raise ValueError("visual mesh component paths must be unique")
        seen_components.add(component_path)
        found.append(VisualMeshReference(relative, component_path))
    found.sort(key=lambda item: item.relative_path)
    if len(found) > _MAX_RESOURCE_COUNT:
        raise ValueError("visual resource count exceeds the admission bound")
    if len({item.relative_path for item in found}) != len(found):
        raise ValueError("visual resource paths must be unique")
    return tuple(found)


def capture_visual_resources(
    model_path: Path, raw: bytes
) -> tuple[SourceVisualResource, ...]:
    """Hash every exact visual reference under the model directory, fail closed."""
    model_path = Path(model_path)
    if model_path.is_symlink() or not model_path.is_file():
        raise ValueError("native source model must be a regular non-symlink file")
    root = model_path.parent.resolve(strict=True)
    if hashlib.sha256(model_path.read_bytes()).digest() != hashlib.sha256(raw).digest():
        raise ValueError("native source model changed during resource discovery")
    references = discover_visual_meshes(raw)
    resources: list[SourceVisualResource] = []
    total_bytes = 0
    for reference in references:
        relative = _validate_relative_path(reference.relative_path)
        path = root
        for part in relative.parts:
            path = path / part
            if path.is_symlink():
                raise ValueError("visual resources must not traverse symlinks")
        resolved = path.resolve(strict=True)
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError("visual resource is missing or escapes its source root")
        data = resolved.read_bytes()
        total_bytes += len(data)
        if total_bytes > _MAX_CLOSURE_BYTES:
            raise ValueError("visual resource closure exceeds the byte bound")
        resources.append(
            SourceVisualResource(
                reference.relative_path,
                reference.component_path,
                hashlib.sha256(data).hexdigest(),
            )
        )
    return tuple(sorted(resources, key=lambda item: item.relative_path))


def visual_mesh_component_paths(
    resources: tuple[SourceVisualResource, ...],
) -> frozenset[str]:
    """Return the exact native Mesh component paths proven by source XML."""
    return frozenset(item.component_path for item in resources)


def _closure_digest(
    source_sha256: str, resources: tuple[SourceVisualResource, ...]
) -> str:
    payload = {
        "schema_version": _SCHEMA_VERSION,
        "source_model_sha256": source_sha256,
        "resources": [
            {
                "relative_path": item.relative_path,
                "component_path": item.component_path,
                "sha256": item.sha256,
                "purpose": item.purpose,
            }
            for item in resources
        ],
    }
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def make_resource_envelope(
    replay_bundle: Any,
    source_model_sha256: str,
    resources: tuple[SourceVisualResource, ...],
) -> NativeMuscleResourceEnvelope:
    ordered = tuple(sorted(resources, key=lambda item: item.relative_path))
    return NativeMuscleResourceEnvelope(
        replay_bundle,
        source_model_sha256,
        ordered,
        _closure_digest(source_model_sha256, ordered),
    )


def dumps_resource_envelope(
    envelope: NativeMuscleResourceEnvelope, contracts: Any
) -> str:
    """Serialize an adapter-specific resource envelope around standard T01 JSON."""
    if not isinstance(envelope, NativeMuscleResourceEnvelope):
        raise TypeError("envelope must be a NativeMuscleResourceEnvelope")
    replay_json = json.loads(
        contracts.dumps_experiment_replay_bundle(envelope.replay_bundle)
    )
    payload = {
        "schema_version": envelope.schema_version,
        "replay_bundle": replay_json,
        "source_model_sha256": envelope.source_model_sha256,
        "resources": [
            {
                "relative_path": item.relative_path,
                "component_path": item.component_path,
                "sha256": item.sha256,
                "purpose": item.purpose,
            }
            for item in envelope.resources
        ],
        "closure_sha256": envelope.closure_sha256,
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":")) + "\n"


def loads_resource_envelope(text: str, contracts: Any) -> NativeMuscleResourceEnvelope:
    """Load strict versioned envelope and its unchanged shared T01 payload."""
    try:
        payload = json.loads(text)
    except (TypeError, json.JSONDecodeError) as error:
        raise ValueError("native resource envelope must contain valid JSON") from error
    expected = {
        "schema_version",
        "replay_bundle",
        "source_model_sha256",
        "resources",
        "closure_sha256",
    }
    if not isinstance(payload, dict) or set(payload) != expected:
        raise ValueError("native resource envelope fields differ")
    if payload["schema_version"] != _SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {_SCHEMA_VERSION!r}")
    if not isinstance(payload["resources"], list):
        raise ValueError("resources must be a JSON array")
    resources: list[SourceVisualResource] = []
    resource_fields = {"relative_path", "component_path", "sha256", "purpose"}
    for item in payload["resources"]:
        if not isinstance(item, dict) or set(item) != resource_fields:
            raise ValueError("visual resource fields differ")
        resources.append(SourceVisualResource(**item))
    shared_bundle = contracts.load_experiment_replay_bundle(
        json.dumps(payload["replay_bundle"], sort_keys=True, separators=(",", ":"))
    )
    return NativeMuscleResourceEnvelope(
        shared_bundle,
        payload["source_model_sha256"],
        tuple(resources),
        payload["closure_sha256"],
        payload["schema_version"],
    )


@contextmanager
def freeze_source_and_resources(
    model_path: Path,
    raw: bytes,
    resources: tuple[SourceVisualResource, ...],
) -> Iterator[Path]:
    """Materialize exact bounded source closure in a temporary owned directory."""
    total_bytes = len(raw)
    if total_bytes > _MAX_SOURCE_BYTES:
        raise ValueError("source XML exceeds the byte bound")
    root_source = model_path.parent.resolve(strict=True)
    resolved_resources: list[tuple[SourceVisualResource, bytes]] = []
    for resource in resources:
        candidate = root_source
        for part in _validate_relative_path(resource.relative_path).parts:
            candidate = candidate / part
            if candidate.is_symlink():
                raise ValueError("visual resources must not traverse symlinks")
        resolved = candidate.resolve(strict=True)
        data = resolved.read_bytes()
        if not resolved.is_relative_to(root_source) or not resolved.is_file():
            raise ValueError("visual resource is missing or escapes its source root")
        if hashlib.sha256(data).hexdigest() != resource.sha256:
            raise ValueError(
                f"visual resource digest differs: {resource.relative_path}"
            )
        total_bytes += len(data)
        if total_bytes > _MAX_SOURCE_BYTES + _MAX_CLOSURE_BYTES:
            raise ValueError("source and visual snapshot exceed the byte bound")
        resolved_resources.append((resource, data))
    with TemporaryDirectory(prefix="opensim-visual-source-") as directory:
        owned_root = Path(directory)
        owned_model = owned_root / model_path.name
        owned_model.write_bytes(raw)
        for resource, data in resolved_resources:
            destination = owned_root.joinpath(
                *_validate_relative_path(resource.relative_path).parts
            )
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(data)
        yield owned_model


__all__ = [
    "NativeMuscleResourceEnvelope",
    "SourceVisualResource",
    "capture_visual_resources",
    "discover_visual_meshes",
    "dumps_resource_envelope",
    "freeze_source_and_resources",
    "loads_resource_envelope",
    "make_resource_envelope",
    "visual_mesh_component_paths",
]

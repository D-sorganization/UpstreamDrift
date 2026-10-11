"""Exact-source contact declarations and the shared native wrench sampler."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import Any

from src.shared.python.force_overlay import WrenchKind

_SCHEMA = "native-sphere-halfspace-contact-v1"
_CONTACT_CLASSES = {"HuntCrossleyForce", "SmoothSphereHalfSpaceForce"}
_GEOMETRY_CLASSES = {"ContactSphere", "ContactHalfSpace"}


def _sha(value: bytes) -> str:
    return hashlib.sha256(value).hexdigest()


def _is_sha(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _force_identities(model: Any) -> tuple[NativeContactForceIdentity, ...]:
    rows = []
    forces = model.getForceSet()
    for index in range(forces.getSize()):
        force = forces.get(index)
        concrete_class = str(force.getConcreteClassName())
        if concrete_class not in _CONTACT_CLASSES:
            continue
        rows.append(
            NativeContactForceIdentity(
                str(force.getAbsolutePathString()),
                concrete_class,
                _sha(force.dump().encode("utf-8")),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.absolute_path))


def _geometry_identities(model: Any) -> tuple[NativeContactGeometryIdentity, ...]:
    rows = []
    geometries = model.getContactGeometrySet()
    for index in range(geometries.getSize()):
        geometry = geometries.get(index)
        concrete_class = str(geometry.getConcreteClassName())
        if concrete_class not in _GEOMETRY_CLASSES:
            raise ValueError("contact geometry class is outside the native policy")
        rows.append(
            NativeContactGeometryIdentity(
                str(geometry.getAbsolutePathString()),
                concrete_class,
                _sha(geometry.dump().encode("utf-8")),
            )
        )
    return tuple(sorted(rows, key=lambda item: item.absolute_path))


@dataclass(frozen=True)
class NativeContactForceIdentity:
    """Identity of one supported native ground-contact force."""

    absolute_path: str
    concrete_class: str
    serialized_sha256: str

    def __post_init__(self) -> None:
        if not self.absolute_path.startswith("/"):
            raise ValueError("contact force path must be absolute")
        if self.concrete_class not in _CONTACT_CLASSES:
            raise ValueError("contact force class is outside the native policy")
        if not _is_sha(self.serialized_sha256):
            raise ValueError("contact force needs a serialized SHA-256")


@dataclass(frozen=True)
class NativeContactGeometryIdentity:
    """Identity of one exact sphere or half-space in the source model."""

    absolute_path: str
    concrete_class: str
    serialized_sha256: str

    def __post_init__(self) -> None:
        if not self.absolute_path.startswith("/"):
            raise ValueError("contact geometry path must be absolute")
        if self.concrete_class not in _GEOMETRY_CLASSES:
            raise ValueError("contact geometry class is outside the native policy")
        if not _is_sha(self.serialized_sha256):
            raise ValueError("contact geometry needs a serialized SHA-256")


@dataclass(frozen=True)
class NativeContactExecutionDeclaration:
    """Source-bound allowlist for complete supported ground-contact sampling."""

    schema_version: str
    source_sha256: str
    forces: tuple[NativeContactForceIdentity, ...]
    geometries: tuple[NativeContactGeometryIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA:
            raise ValueError("unsupported native contact declaration schema")
        if not _is_sha(self.source_sha256):
            raise ValueError("native contact declaration needs an exact source SHA")
        if not isinstance(self.forces, tuple) or not self.forces:
            raise ValueError(
                "native contact declaration needs an immutable nonempty set"
            )
        if not isinstance(self.geometries, tuple) or not self.geometries:
            raise ValueError(
                "native contact declaration needs exact geometry identities"
            )
        paths = tuple(item.absolute_path for item in self.forces)
        if paths != tuple(sorted(paths)) or len(set(paths)) != len(paths):
            raise ValueError(
                "native contact force identities must be unique and sorted"
            )
        geometry_paths = tuple(item.absolute_path for item in self.geometries)
        if geometry_paths != tuple(sorted(geometry_paths)) or len(
            set(geometry_paths)
        ) != len(geometry_paths):
            raise ValueError(
                "native contact geometry identities must be unique and sorted"
            )

    @property
    def force_paths(self) -> tuple[str, ...]:
        return tuple(item.absolute_path for item in self.forces)

    @property
    def declaration_sha256(self) -> str:
        payload = {
            "schema_version": self.schema_version,
            "source_sha256": self.source_sha256,
            "forces": [asdict(item) for item in self.forces],
            "geometries": [asdict(item) for item in self.geometries],
        }
        return _sha(json.dumps(payload, sort_keys=True, allow_nan=False).encode())


def declare_native_contact_execution(
    model_path: Path, source_sha256: str
) -> NativeContactExecutionDeclaration:
    """Capture supported force identities from exact pinned model bytes."""
    import opensim as osim

    path = Path(model_path)
    raw = path.read_bytes()
    if _sha(raw) != source_sha256:
        raise ValueError("native contact declaration source identity differs")
    model = osim.Model(str(path))
    model.finalizeConnections()
    forces = _force_identities(model)
    geometries = _geometry_identities(model)
    if not forces:
        raise ValueError("source has no supported native contact forces")
    if not geometries:
        raise ValueError("source has no supported native contact geometries")
    contact_sampler(model, tuple(item.absolute_path for item in forces))
    if path.read_bytes() != raw:
        raise ValueError("native contact source changed during declaration")
    return NativeContactExecutionDeclaration(_SCHEMA, source_sha256, forces, geometries)


def validate_native_contact_execution(
    model: Any,
    declaration: NativeContactExecutionDeclaration,
    expected_source_sha256: str,
) -> None:
    """Recheck exact force closure and native geometry restrictions on load."""
    if not isinstance(declaration, NativeContactExecutionDeclaration):
        raise TypeError("native contact declaration must be typed")
    if declaration.source_sha256 != expected_source_sha256:
        raise ValueError("native contact declaration source identity differs")
    actual = _force_identities(model)
    if actual != declaration.forces:
        raise ValueError("native contact force closure differs from declaration")
    if _geometry_identities(model) != declaration.geometries:
        raise ValueError("native contact geometry closure differs from declaration")
    contact_sampler(model, declaration.force_paths)


def contact_sampler(model: Any, paths: tuple[str, ...]) -> Any:
    """Create the existing native force-torque source under strict path audit."""
    if not paths:
        return None
    from src.engines.physics_engines.opensim.python.opensim_force_torque import (
        OpenSimForceTorqueSource,
    )

    sampler = OpenSimForceTorqueSource(model, include_muscles=False)
    sampler.validate_ground_contact_paths(paths)
    return sampler


def sample_native_contacts(
    sampler: Any, state: Any, paths: tuple[str, ...]
) -> tuple[Any, ...]:
    """Require complete finite contact wrenches for the declared native forces."""
    if sampler is None:
        return ()
    import numpy as np

    contacts = tuple(sampler.sample(state).by_kind(WrenchKind.CONTACT))
    expected = {"contact:" + path.rsplit("/", 1)[-1] for path in paths}
    if len(contacts) != len(paths) or {wrench.label for wrench in contacts} != expected:
        raise ValueError("native contact wrench evidence is incomplete or unsupported")
    for wrench in contacts:
        if wrench.force_n is None or wrench.torque_nm is None:
            raise ValueError("native contact wrench evidence needs force and torque")
        if not np.isfinite((*wrench.force_n, *wrench.torque_nm, *wrench.point_m)).all():
            raise RuntimeError("native contact wrench evidence must be finite")
    return contacts


__all__ = [
    "NativeContactExecutionDeclaration",
    "NativeContactForceIdentity",
    "NativeContactGeometryIdentity",
    "contact_sampler",
    "declare_native_contact_execution",
    "sample_native_contacts",
    "validate_native_contact_execution",
]

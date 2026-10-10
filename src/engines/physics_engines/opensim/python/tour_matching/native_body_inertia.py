"""Source-bound OpenSim body inertia admission without parameter repair."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
from pathlib import Path
from typing import Protocol, cast

from defusedxml import ElementTree
import numpy as np

from src.shared.python.core.contracts import PreconditionError
from src.shared.python.estimation.dime_global_calibration import (
    validate_physical_inertia,
)

Vec3 = tuple[float, float, float]
Inertia6 = tuple[float, float, float, float, float, float]


class _IndexedVector(Protocol):
    def get(self, index: int) -> float: ...


class _NativeInertia(Protocol):
    def getMoments(self) -> _IndexedVector: ...
    def getProducts(self) -> _IndexedVector: ...


class _NativeBody(Protocol):
    def getMass(self) -> float: ...
    def getMassCenter(self) -> _IndexedVector: ...
    def getInertia(self) -> _NativeInertia: ...


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class BodyInertiaFinding:
    """One exact source/native body-property observation in kg and kg·m²."""

    name: str
    mass_kg: float
    center_m: Vec3
    inertia_kg_m2: Inertia6
    source_native_exact: bool
    failure_reason: str | None


@dataclass(frozen=True)
class NativeBodyInertiaAudit:
    """Complete body inventory and physical-admission outcome for one source."""

    source_sha256: str
    adapter_sha256: str
    opensim_version: str
    native_simulation_sha256: str
    native_simbody_sha256: str
    native_actuators_sha256: str
    bodies: tuple[BodyInertiaFinding, ...]

    @property
    def body_count(self) -> int:
        return len(self.bodies)

    @property
    def admitted(self) -> bool:
        return bool(self.bodies) and all(
            body.source_native_exact and body.failure_reason is None
            for body in self.bodies
        )

    def require_admitted(self) -> None:
        """Reject physical use when any body lacks source/native admission."""
        if not self.admitted:
            names = ", ".join(body.name for body in self.bodies if body.failure_reason)
            raise ValueError(f"native body inertia admission failed: {names}")


def _tensor(values: tuple[float, ...]) -> np.ndarray:
    xx, yy, zz, xy, xz, yz = values
    return np.array([[xx, xy, xz], [xy, yy, yz], [xz, yz, zz]])


def _source_bodies(raw: bytes) -> tuple[tuple[str, float, Vec3, Inertia6], ...]:
    document = ElementTree.fromstring(raw)
    objects = document.find("./Model/BodySet/objects")
    if (
        objects is None
        or len(objects) == 0
        or any(node.tag != "Body" for node in objects)
    ):
        raise ValueError("unsupported or absent source BodySet inventory")
    rows: list[tuple[str, float, Vec3, Inertia6]] = []
    for node in objects:
        name = node.get("name")
        if not name or any(row[0] == name for row in rows):
            raise ValueError("source body identity missing or duplicated")
        try:
            mass = float(node.findtext("mass", ""))
            center = tuple(
                float(value) for value in node.findtext("mass_center", "").split()
            )
            inertia = tuple(
                float(value) for value in node.findtext("inertia", "").split()
            )
        except ValueError as exc:
            raise ValueError(f"malformed source body properties: {name}") from exc
        if (
            not np.isfinite(mass)
            or mass < 0
            or len(center) != 3
            or len(inertia) != 6
            or not np.all(np.isfinite(center))
            or not np.all(np.isfinite(inertia))
        ):
            raise ValueError(f"malformed source body mass, COM or inertia: {name}")
        if mass == 0 and any(value != 0 for value in inertia):
            raise ValueError(f"massless body has nonzero inertia: {name}")
        rows.append((name, mass, cast(Vec3, center), cast(Inertia6, inertia)))
    return tuple(rows)


def _native_body_properties(body: _NativeBody) -> tuple[float, Vec3, Inertia6]:
    mass = float(body.getMass())
    center = body.getMassCenter()
    native = body.getInertia()
    moments, products = native.getMoments(), native.getProducts()
    return (
        mass,
        cast(Vec3, tuple(float(center.get(i)) for i in range(3))),
        cast(
            Inertia6,
            tuple(float(moments.get(i)) for i in range(3))
            + tuple(float(products.get(i)) for i in range(3)),
        ),
    )


def audit_native_body_inertia(
    source_model_path: Path, expected_source_sha256: str
) -> NativeBodyInertiaAudit:
    """Check all source bodies against fresh native readback and shared physics law."""
    import opensim as osim

    raw = source_model_path.read_bytes()
    source_sha = hashlib.sha256(raw).hexdigest()
    if source_sha != expected_source_sha256:
        raise ValueError("source model hash mismatch")
    rows = _source_bodies(raw)
    model = osim.Model(str(source_model_path))
    model.initSystem()
    native_bodies = model.getBodySet()
    native_names = tuple(
        native_bodies.get(i).getName() for i in range(native_bodies.getSize())
    )
    if len(native_names) != len(rows) or set(native_names) != {row[0] for row in rows}:
        raise ValueError("native body inventory differs from exact source")
    findings = []
    for name, mass, center, inertia in rows:
        native_mass, native_center, native_inertia = _native_body_properties(
            native_bodies.get(name)
        )
        exact = (mass, center, inertia) == (native_mass, native_center, native_inertia)
        if not exact:
            raise ValueError(f"source/native body mass, COM or inertia differs: {name}")
        reason = None
        if mass > 0:
            try:
                validate_physical_inertia(_tensor(inertia))
            except PreconditionError as exc:
                reason = str(exc)
        findings.append(BodyInertiaFinding(name, mass, center, inertia, exact, reason))
    if _sha(source_model_path) != source_sha:
        raise ValueError("source model changed during native admission")
    return NativeBodyInertiaAudit(
        source_sha256=source_sha,
        adapter_sha256=_sha(Path(__file__)),
        opensim_version=osim.GetVersionAndDate(),
        native_simulation_sha256=_sha(Path(osim._simulation.__file__)),
        native_simbody_sha256=_sha(Path(osim._simbody.__file__)),
        native_actuators_sha256=_sha(Path(osim._actuators.__file__)),
        bodies=tuple(findings),
    )

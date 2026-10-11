"""Exact-source declaration and native observations for reviewed wrapped paths.

The declaration closes native WrapObject ownership, PathWrap references, and
ConditionalPathPoint chart ranges. It does not admit moving-path splines,
generalized joints, contact, or a physical/physiological endpoint.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
import math
from pathlib import Path
from typing import Any

_SCHEMA_VERSION = "native-wrapped-path-v1"
_SHAPES = {"WrapCylinder", "WrapEllipsoid", "WrapSphere"}


def _is_sha256(value: str) -> bool:
    return len(value) == 64 and all(char in "0123456789abcdef" for char in value)


def _finite(value: float, name: str) -> float:
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite")
    return result


def _sha_dump(component: Any) -> str:
    return hashlib.sha256(component.dump().encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WrapObjectIdentity:
    path: str
    concrete_class: str
    parent_frame_path: str
    translation_m: tuple[float, float, float]
    xyz_body_rotation_rad: tuple[float, float, float]
    active: bool
    quadrant: str
    shape_parameters: tuple[tuple[str, float], ...]
    serialized_sha256: str

    def __post_init__(self) -> None:
        if self.concrete_class not in _SHAPES:
            raise ValueError("wrapped path shape is outside the reviewed profile")
        if not self.path.startswith("/") or not self.parent_frame_path.startswith("/"):
            raise ValueError("wrap object and parent frame paths must be absolute")
        if len(self.translation_m) != 3 or len(self.xyz_body_rotation_rad) != 3:
            raise ValueError("wrap transform must contain three finite values")
        for value in (*self.translation_m, *self.xyz_body_rotation_rad):
            _finite(value, "wrap transform")
        names = [name for name, _ in self.shape_parameters]
        if len(names) != len(set(names)):
            raise ValueError("wrap shape parameters must have unique names")
        if not names:
            raise ValueError("wrap shape dimensions are required")
        for name, value in self.shape_parameters:
            if not name or _finite(value, "wrap shape dimension") <= 0:
                raise ValueError("wrap shape dimensions must be positive")
        if not _is_sha256(self.serialized_sha256):
            raise ValueError("wrap component requires serialized SHA-256")


@dataclass(frozen=True)
class PathWrapIdentity:
    path: str
    owner_path: str
    wrap_object_path: str
    wrap_object_name: str
    method: str
    range_indices: tuple[float, float]
    serialized_sha256: str

    def __post_init__(self) -> None:
        if not self.path.startswith("/") or not self.owner_path.startswith("/"):
            raise ValueError("PathWrap identities must use absolute paths")
        if not self.wrap_object_path.startswith("/") or not self.wrap_object_name:
            raise ValueError("PathWrap needs an exact wrap-object reference")
        if self.method != "hybrid":
            raise ValueError("PathWrap method is outside the reviewed hybrid profile")
        if len(self.range_indices) != 2:
            raise ValueError("PathWrap range must have two indices")
        lower, upper = (_finite(item, "PathWrap range") for item in self.range_indices)
        if (lower, upper) != (-1.0, -1.0) and (
            lower < 0
            or upper < lower
            or not lower.is_integer()
            or not upper.is_integer()
        ):
            raise ValueError(
                "PathWrap range must be -1/-1 or ordered nonnegative indices"
            )
        if not _is_sha256(self.serialized_sha256):
            raise ValueError("PathWrap component requires serialized SHA-256")


@dataclass(frozen=True)
class ConditionalPathPointIdentity:
    path: str
    owner_path: str
    coordinate_path: str
    active_range: tuple[float, float]
    serialized_sha256: str

    def __post_init__(self) -> None:
        if not all(
            value.startswith("/")
            for value in (self.path, self.owner_path, self.coordinate_path)
        ):
            raise ValueError("conditional path-point references must be absolute")
        if len(self.active_range) != 2:
            raise ValueError("conditional path-point range must have two values")
        lower, upper = (
            _finite(item, "conditional active range") for item in self.active_range
        )
        if lower >= upper:
            raise ValueError("conditional active range must be strictly ordered")
        if not _is_sha256(self.serialized_sha256):
            raise ValueError("conditional point requires serialized SHA-256")


@dataclass(frozen=True)
class WrappedPathDeclaration:
    schema_version: str
    source_sha256: str
    wrap_objects: tuple[WrapObjectIdentity, ...]
    path_wraps: tuple[PathWrapIdentity, ...]
    conditional_points: tuple[ConditionalPathPointIdentity, ...]

    def __post_init__(self) -> None:
        if self.schema_version != _SCHEMA_VERSION:
            raise ValueError("unsupported wrapped-path declaration schema")
        if not _is_sha256(self.source_sha256):
            raise ValueError("wrapped-path declaration requires exact source SHA-256")
        for values, label in (
            (self.wrap_objects, "wrap object"),
            (self.path_wraps, "PathWrap"),
            (self.conditional_points, "conditional path point"),
        ):
            paths = [item.path for item in values]
            if len(paths) != len(set(paths)):
                raise ValueError(f"duplicate {label} path in wrapped declaration")
        if not (self.wrap_objects or self.path_wraps or self.conditional_points):
            raise ValueError("wrapped declaration needs at least one path component")
        if bool(self.wrap_objects) != bool(self.path_wraps):
            raise ValueError(
                "wrap objects and PathWrap links must be declared together"
            )
        object_paths = {item.path for item in self.wrap_objects}
        if any(item.wrap_object_path not in object_paths for item in self.path_wraps):
            raise ValueError("PathWrap target is missing from declared wrap objects")

    @property
    def declaration_sha256(self) -> str:
        payload = json.dumps(
            {
                "schema_version": self.schema_version,
                "source_sha256": self.source_sha256,
                "wrap_objects": [asdict(item) for item in self.wrap_objects],
                "path_wraps": [asdict(item) for item in self.path_wraps],
                "conditional_points": [
                    asdict(item) for item in self.conditional_points
                ],
            },
            sort_keys=True,
            allow_nan=False,
        )
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class WrappedPathAudit:
    declaration_sha256: str
    source_sha256: str
    reference_observation: Any


@dataclass(frozen=True)
class NativeWrappedPathAudit:
    """Source-bound wrap findings for the shared native diagnostic contract."""

    source_sha256: str
    declaration_sha256: str | None
    reference_observation: Any
    findings: tuple[Any, ...]


def _diagnostic_finding(
    code_value: str,
    source_sha256: str,
    evidence_sha256: str,
    execution_effect: str,
    qualification_effect: str,
    component_path: str | None = None,
) -> Any:
    from .native_diagnostic_contract import (
        DiagnosticCode,
        ExecutionEffect,
        NativeDiagnosticFinding,
        QualificationEffect,
    )

    return NativeDiagnosticFinding(
        code=DiagnosticCode(code_value),
        source_sha256=source_sha256,
        component_path=component_path,
        evidence_sha256=evidence_sha256,
        execution_effect=ExecutionEffect[execution_effect],
        qualification_effect=QualificationEffect[qualification_effect],
    )


def _observation_sha256(observation: Any) -> str:
    return hashlib.sha256(repr(observation).encode("utf-8")).hexdigest()


def _wrapped_failure(
    source_sha256: str, code: str, evidence_sha256: str
) -> NativeWrappedPathAudit:
    finding = _diagnostic_finding(code, source_sha256, evidence_sha256, "BLOCK", "NONE")
    return NativeWrappedPathAudit(source_sha256, None, None, (finding,))


def _exception_failure(
    source_sha256: str, code: str, error: Exception
) -> NativeWrappedPathAudit:
    evidence = hashlib.sha256(
        f"{source_sha256}:{type(error).__name__}:{error}".encode()
    ).hexdigest()
    return _wrapped_failure(source_sha256, code, evidence)


def _load_native_wrapped_model(
    source: Path, source_sha256: str
) -> tuple[Any | None, NativeWrappedPathAudit | None]:
    """Load and connect the exact native model without exporting its ownership."""
    import opensim as osim

    try:
        model = osim.Model(str(source))
    except (RuntimeError, ValueError) as exc:
        return None, _exception_failure(source_sha256, "native_model_load_failed", exc)
    try:
        model.finalizeConnections()
    except (RuntimeError, ValueError) as exc:
        return None, _exception_failure(
            source_sha256, "wrapped_path_source_closure_invalid", exc
        )
    return model, None


def _validate_wrapped_source_declaration(
    source: Path,
    raw: bytes,
    source_sha256: str,
    model: Any,
    declaration: WrappedPathDeclaration,
) -> NativeWrappedPathAudit | None:
    try:
        validate_wrapped_path_declaration(model, declaration, source_sha256)
        if source.read_bytes() != raw:
            raise ValueError("native model source changed during wrap audit")
    except (RuntimeError, ValueError) as exc:
        return _exception_failure(
            source_sha256, "wrapped_path_source_closure_invalid", exc
        )
    return None


def _observe_declared_wrapped_route(
    source: Path,
    source_sha256: str,
    declaration: WrappedPathDeclaration,
    reference: Any,
    reference_frame_path: str,
) -> tuple[Any | None, Any]:
    try:
        audited = audit_wrapped_path_source(
            source, declaration, reference, reference_frame_path
        )
    except (RuntimeError, ValueError) as exc:
        evidence = hashlib.sha256(
            f"{source_sha256}:{type(exc).__name__}:{exc}".encode()
        ).hexdigest()
        return None, _diagnostic_finding(
            "wrapped_path_native_route_invalid",
            source_sha256,
            evidence,
            "BLOCK",
            "NONE",
            (
                declaration.path_wraps[0].owner_path
                if declaration.path_wraps
                else declaration.conditional_points[0].owner_path
            ),
        )
    observation = audited.reference_observation
    finding = _diagnostic_finding(
        "wrapped_path_native_route_observed",
        source_sha256,
        _observation_sha256(observation),
        "PROCEED",
        "OBSERVATION",
    )
    return observation, finding


def _related_wrapped_components(model: Any) -> tuple[Any, ...]:
    related_classes = _SHAPES | {"PathWrap", "PathWrapPoint", "ConditionalPathPoint"}
    return tuple(
        component
        for component in tuple(model.getComponentsList())
        if component.getConcreteClassName() in related_classes
    )


def _missing_wrapped_declaration(
    source_sha256: str, evidence_sha256: str, related: tuple[Any, ...]
) -> NativeWrappedPathAudit:
    findings = tuple(
        _diagnostic_finding(
            "wrapped_path_declaration_required",
            source_sha256,
            evidence_sha256,
            "BLOCK",
            "NONE",
            component.getAbsolutePathString(),
        )
        for component in related
    )
    return NativeWrappedPathAudit(source_sha256, None, None, findings)


def _closed_wrap_findings(
    source_sha256: str, declaration: WrappedPathDeclaration
) -> list[Any]:
    findings = [
        _diagnostic_finding(
            "wrapped_path_source_closed",
            source_sha256,
            declaration.declaration_sha256,
            "PROCEED",
            "OBSERVATION",
        )
    ]
    findings.extend(
        _diagnostic_finding(
            "wrapped_path_transition_unqualified",
            source_sha256,
            declaration.declaration_sha256,
            "REQUIRES_OPT_IN",
            "UNQUALIFIED",
            owner_path,
        )
        for owner_path in sorted(
            {item.owner_path for item in declaration.path_wraps}
            | {item.owner_path for item in declaration.conditional_points}
        )
    )
    return findings


def audit_native_wrapped_path_source(
    model_path: Path,
    source_sha256: str,
    declaration: WrappedPathDeclaration | None,
    reference: Any | None = None,
    reference_frame_path: str = "/ground",
) -> NativeWrappedPathAudit:
    """Audit a wrapped source and return typed shared-contract findings.

    A missing declaration blocks only when the source actually contains one of
    the wrap-related components in this profile. A successful source closure
    remains explicitly unqualified for global route/chart transitions.
    """
    if not _is_sha256(source_sha256):
        raise ValueError("wrapped-path audit requires exact source SHA-256")

    source = Path(model_path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    evidence = digest
    if digest != source_sha256:
        finding = _diagnostic_finding(
            "wrapped_path_source_closure_invalid",
            source_sha256,
            evidence,
            "BLOCK",
            "NONE",
        )
        return NativeWrappedPathAudit(source_sha256, None, None, (finding,))

    model, load_failure = _load_native_wrapped_model(source, source_sha256)
    if load_failure is not None:
        return load_failure
    assert model is not None
    related = _related_wrapped_components(model)
    if not related:
        return NativeWrappedPathAudit(source_sha256, None, None, ())

    if declaration is None:
        return _missing_wrapped_declaration(source_sha256, evidence, related)

    declaration_failure = _validate_wrapped_source_declaration(
        source, raw, source_sha256, model, declaration
    )
    if declaration_failure is not None:
        return NativeWrappedPathAudit(
            source_sha256,
            declaration.declaration_sha256,
            declaration_failure.reference_observation,
            declaration_failure.findings,
        )

    findings = _closed_wrap_findings(source_sha256, declaration)

    observation = None
    if reference is not None:
        observation, route_finding = _observe_declared_wrapped_route(
            source, source_sha256, declaration, reference, reference_frame_path
        )
        if route_finding.execution_effect.value == "block":
            findings.append(route_finding)
            return NativeWrappedPathAudit(
                source_sha256,
                declaration.declaration_sha256,
                None,
                tuple(findings),
            )
        findings.append(route_finding)

    if source.read_bytes() != raw:
        failed_evidence = hashlib.sha256(
            b"source-mutated-during-wrap-audit"
        ).hexdigest()
        findings.append(
            _diagnostic_finding(
                "wrapped_path_source_closure_invalid",
                source_sha256,
                failed_evidence,
                "BLOCK",
                "NONE",
            )
        )
    return NativeWrappedPathAudit(
        source_sha256,
        declaration.declaration_sha256,
        observation,
        tuple(findings),
    )


def _wrap_identity_from_component(wrap: Any, concrete_class: str) -> WrapObjectIdentity:
    parameters: tuple[tuple[str, float], ...]
    if concrete_class == "WrapCylinder":
        parameters = (
            ("length_m", _finite(wrap.get_length(), "cylinder length")),
            ("radius_m", _finite(wrap.get_radius(), "cylinder radius")),
        )
    elif concrete_class == "WrapSphere":
        parameters = (("radius_m", _finite(wrap.get_radius(), "sphere radius")),)
    else:
        dimensions = wrap.get_dimensions()
        parameters = tuple(
            (name, _finite(dimensions.get(i), "ellipsoid dimension"))
            for i, name in enumerate(("x_radius_m", "y_radius_m", "z_radius_m"))
        )
    return WrapObjectIdentity(
        wrap.getAbsolutePathString(),
        concrete_class,
        wrap.getFrame().getAbsolutePathString(),
        (
            _finite(wrap.get_translation().get(0), "wrap translation"),
            _finite(wrap.get_translation().get(1), "wrap translation"),
            _finite(wrap.get_translation().get(2), "wrap translation"),
        ),
        (
            _finite(wrap.get_xyz_body_rotation().get(0), "wrap rotation"),
            _finite(wrap.get_xyz_body_rotation().get(1), "wrap rotation"),
            _finite(wrap.get_xyz_body_rotation().get(2), "wrap rotation"),
        ),
        bool(wrap.get_active()),
        str(wrap.get_quadrant()),
        parameters,
        _sha_dump(wrap),
    )


def _path_wrap_identity_from_component(component: Any, osim: Any) -> PathWrapIdentity:
    path_wrap = osim.PathWrap.safeDownCast(component)
    if path_wrap is None:
        raise ValueError("PathWrap needs its native concrete class")
    target = path_wrap.getWrapObject()
    if target is None:
        raise ValueError("PathWrap has an unresolved wrap-object socket")
    return PathWrapIdentity(
        path_wrap.getAbsolutePathString(),
        path_wrap.getOwner().getAbsolutePathString(),
        target.getAbsolutePathString(),
        str(path_wrap.getWrapObjectName()),
        str(path_wrap.getMethodName()),
        (
            _finite(path_wrap.get_range(0), "PathWrap range"),
            _finite(path_wrap.get_range(1), "PathWrap range"),
        ),
        _sha_dump(path_wrap),
    )


def _conditional_path_identity_from_component(
    component: Any, osim: Any
) -> ConditionalPathPointIdentity:
    point = osim.ConditionalPathPoint.safeDownCast(component)
    if point is None or point.getCoordinate() is None:
        raise ValueError("conditional path point coordinate is unresolved")
    return ConditionalPathPointIdentity(
        point.getAbsolutePathString(),
        point.getOwner().getAbsolutePathString(),
        point.getCoordinate().getAbsolutePathString(),
        (
            _finite(point.get_range(0), "conditional active range"),
            _finite(point.get_range(1), "conditional active range"),
        ),
        _sha_dump(point),
    )


def _validate_path_wrap_owners(
    model: Any, declaration: WrappedPathDeclaration, osim: Any
) -> None:
    object_by_path = {item.path: item for item in declaration.wrap_objects}
    geometry_paths = {
        component.getAbsolutePathString()
        for component in tuple(model.getComponentsList())
        if osim.GeometryPath.safeDownCast(component) is not None
    }
    for link in declaration.path_wraps:
        target = object_by_path[link.wrap_object_path]
        if link.wrap_object_name != target.path.rsplit("/", 1)[-1]:
            raise ValueError("PathWrap name and exact wrap-object path disagree")
        if link.owner_path not in geometry_paths:
            raise ValueError("PathWrap owner is not a native GeometryPath")


def _identity_from_model(model: Any, source_sha256: str) -> WrappedPathDeclaration:
    import opensim as osim

    objects: list[WrapObjectIdentity] = []
    links: list[PathWrapIdentity] = []
    conditional: list[ConditionalPathPointIdentity] = []
    for component in tuple(model.getComponentsList()):
        concrete_class = component.getConcreteClassName()
        if concrete_class in _SHAPES:
            objects.append(_wrap_identity_from_component(component, concrete_class))
        elif concrete_class == "PathWrap":
            links.append(_path_wrap_identity_from_component(component, osim))
        elif concrete_class == "ConditionalPathPoint":
            conditional.append(
                _conditional_path_identity_from_component(component, osim)
            )
    objects.sort(key=lambda item: item.path)
    links.sort(key=lambda item: item.path)
    conditional.sort(key=lambda item: item.path)
    declaration = WrappedPathDeclaration(
        _SCHEMA_VERSION,
        source_sha256,
        tuple(objects),
        tuple(links),
        tuple(conditional),
    )
    _validate_path_wrap_owners(model, declaration, osim)
    return declaration


def declare_wrapped_path_source(model_path: Path) -> WrappedPathDeclaration:
    """Capture a typed declaration; callers must pin and retain its source hash."""
    import opensim as osim

    source = Path(model_path)
    raw = source.read_bytes()
    source_sha256 = hashlib.sha256(raw).hexdigest()
    model = osim.Model(str(source))
    model.finalizeConnections()
    result = _identity_from_model(model, source_sha256)
    if source.read_bytes() != raw:
        raise ValueError("native model source changed during wrap declaration")
    return result


def validate_wrapped_path_declaration(
    model: Any, declaration: WrappedPathDeclaration, expected_source_sha256: str
) -> None:
    """Require exact typed inventory equality before constrained replay."""
    if declaration.source_sha256 != expected_source_sha256:
        raise ValueError("wrapped-path declaration source identity differs")
    actual = _identity_from_model(model, expected_source_sha256)
    if actual != declaration:
        raise ValueError("wrapped-path declaration differs from native source")


def audit_wrapped_path_source(
    model_path: Path,
    declaration: WrappedPathDeclaration,
    reference: Any,
    reference_frame_path: str = "/ground",
) -> WrappedPathAudit:
    """Verify exact source closure, then reuse the maintained native route observer."""
    source = Path(model_path)
    raw = source.read_bytes()
    digest = hashlib.sha256(raw).hexdigest()
    if digest != declaration.source_sha256:
        raise ValueError("wrapped-path source identity differs from declaration")
    if reference.expected_source_sha256 != digest:
        raise ValueError("wrapped-path reference identity differs from declaration")
    declaration_now = declare_wrapped_path_source(source)
    if declaration_now != declaration:
        raise ValueError("wrapped-path declaration differs from native source")
    from .native_reference_conventions import audit_source_reference

    observation = audit_source_reference(source, reference, reference_frame_path)
    if source.read_bytes() != raw:
        raise ValueError("wrapped-path source changed during native audit")
    return WrappedPathAudit(declaration.declaration_sha256, digest, observation)


__all__ = [
    "ConditionalPathPointIdentity",
    "NativeWrappedPathAudit",
    "PathWrapIdentity",
    "WrapObjectIdentity",
    "WrappedPathAudit",
    "WrappedPathDeclaration",
    "audit_wrapped_path_source",
    "audit_native_wrapped_path_source",
    "declare_wrapped_path_source",
    "validate_wrapped_path_declaration",
]

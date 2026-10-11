"""Strict JSON entrypoint for the existing file-driven native Moco providers."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
import hashlib
import json
import math
from pathlib import Path
from collections.abc import Mapping
from types import MappingProxyType
from typing import TYPE_CHECKING, Any

import numpy as np

from .moco_initial_bindings import MocoInitialBindings
from .moco_tracking import MocoTrackingConfig
from .native_prepared_state import DeclaredColdStart
from .native_mixed_actuation import ActuationRole, MixedChannel, MixedActuationProfile
from .native_passive_readiness import MusclePassiveLimits, PassiveReadinessPolicy
from .native_diagnostic_contract import (
    DiagnosticCode,
    ExecutionEffect,
    NativeDiagnosticFinding,
    NativeDiagnosticPolicy,
    NativeQualificationContext,
    QualificationEffect,
    diagnostic_effects,
)
from .native_wrapped_path import (
    ConditionalPathPointIdentity,
    PathWrapIdentity,
    WrapObjectIdentity,
    WrappedPathDeclaration,
)
from .registration import CaptureRegistration

if TYPE_CHECKING:
    from .native_mechanics_profile import NativeMechanicsExecutionProfile
    from .native_contact_policy import NativeContactExecutionDeclaration


def _validate_request_policies(request: NativeMocoRequest) -> None:
    """Validate typed disposition and exact-source declarations together."""
    if not isinstance(request.allow_unqualified_passive_diagnostic, bool):
        raise TypeError("passive diagnostic disposition must be boolean")
    if request.diagnostic_policy is not None:
        if not isinstance(request.diagnostic_policy, NativeDiagnosticPolicy):
            raise TypeError("native diagnostic policy must be typed")
        if request.diagnostic_policy.source_sha256 != request.model_sha256:
            raise ValueError("native diagnostic policy source differs from model")
        if request.allow_unqualified_passive_diagnostic:
            raise ValueError("legacy and typed diagnostic policies cannot combine")
    if request.wrapped_path_declaration is not None:
        if not isinstance(request.wrapped_path_declaration, WrappedPathDeclaration):
            raise TypeError("wrapped-path declaration must be typed")
        if request.wrapped_path_declaration.source_sha256 != request.model_sha256:
            raise ValueError("wrapped-path declaration source differs from model")
    if request.qualification_context is not None:
        if not isinstance(request.qualification_context, NativeQualificationContext):
            raise TypeError("qualification context must be typed")
        if request.qualification_context.source_sha256 != request.model_sha256:
            raise ValueError("qualification context source differs from model")
    if request.native_mechanics_profile is not None:
        from .native_mechanics_profile import NativeMechanicsExecutionProfile

        if not isinstance(
            request.native_mechanics_profile, NativeMechanicsExecutionProfile
        ):
            raise TypeError("native mechanics profile must be typed")
        if request.native_mechanics_profile.source_sha256 != request.model_sha256:
            raise ValueError("native mechanics profile source differs from model")
        if (
            request.constrained_cold_start is None
            or request.mixed_actuation is not None
        ):
            raise ValueError(
                "native mechanics profile requires constrained-muscle execution"
            )
        if request.diagnostic_policy is None:
            raise ValueError(
                "native mechanics profile requires an explicit typed diagnostic policy"
            )
    if request.contact_declaration is not None:
        from .native_contact_policy import NativeContactExecutionDeclaration

        if not isinstance(
            request.contact_declaration, NativeContactExecutionDeclaration
        ):
            raise TypeError("native contact declaration must be typed")
        if request.contact_declaration.source_sha256 != request.model_sha256:
            raise ValueError("native contact declaration source differs from model")
        if (
            request.constrained_cold_start is None
            or request.mixed_actuation is not None
        ):
            raise ValueError(
                "native contact declaration requires constrained-muscle execution"
            )
        if request.diagnostic_policy is None:
            raise ValueError(
                "native contact declaration requires an explicit typed diagnostic policy"
            )
    if (
        request.allow_unqualified_passive_diagnostic
        and request.passive_policy is not None
    ):
        raise ValueError("diagnostic disposition requires unavailable passive policy")
    if request.mixed_actuation is not None:
        if not isinstance(request.mixed_actuation, MixedActuationProfile):
            raise TypeError("mixed Moco requires a typed actuation profile")
        if dict(request.bindings.control_bounds) != {
            channel.path: channel.control_bounds
            for channel in request.mixed_actuation.channels
        }:
            raise ValueError("mixed Moco control bounds differ from profile")


def _validate_request_inputs(request: NativeMocoRequest) -> None:
    """Validate immutable file identities, marker bindings, and state seed."""
    for name in ("model_sha256", "trc_sha256", "states_guess_sha256"):
        value = getattr(request, name)
        if len(value) != 64 or any(char not in "0123456789abcdef" for char in value):
            raise ValueError(f"{name} must be a lowercase SHA-256")
    if not request.reference_frame_path.startswith("/"):
        raise ValueError("Native reference frame path must be absolute")
    weights = dict(request.marker_weights)
    bindings = dict(request.marker_bindings)
    exclusions = dict(request.excluded_markers)
    for value in weights.values():
        if isinstance(value, bool) or not math.isfinite(value) or value <= 0:
            raise ValueError("Marker weights must be finite and positive")
    for label, placement in bindings.items():
        if (
            not label
            or len(placement) != 2
            or not isinstance(placement[0], str)
            or not placement[0].startswith("/")
            or len(placement[1]) != 3
            or not all(math.isfinite(float(value)) for value in placement[1])
        ):
            raise ValueError("Native marker placement requires frame and 3D offset")
    if any(not reason.strip() for reason in exclusions.values()):
        raise ValueError("Excluded marker reasons must be explicit")
    if set(exclusions) & set(weights):
        raise ValueError("A selected marker cannot also be excluded")
    if request.constrained_cold_start is not None:
        declared = request.constrained_cold_start
        if (
            declared.model_path != request.model_path.resolve()
            or declared.source_sha256 != request.model_sha256
            or dict(declared.named_state) != dict(request.bindings.initial_state)
            or declared.time_seconds != request.config.t_start_s
            or declared.constant_commands
            or declared.source_controller_replacement is not None
            or declared.allow_source_controllers_for_observation
        ):
            raise ValueError("constrained cold start differs from Moco source or state")


@dataclass(frozen=True)
class NativeMocoRequest:
    """Only the files and reviewed native bindings needed for one offline fit."""

    model_path: Path
    trc_path: Path
    states_guess_path: Path
    model_sha256: str
    trc_sha256: str
    states_guess_sha256: str
    bindings: MocoInitialBindings
    config: MocoTrackingConfig
    marker_weights: Mapping[str, float]
    marker_bindings: Mapping[str, tuple[str, tuple[float, float, float]]]
    registration: CaptureRegistration | None
    reference_frame_path: str
    passive_policy: PassiveReadinessPolicy | None
    excluded_markers: Mapping[str, str]
    constrained_cold_start: DeclaredColdStart | None = None
    mixed_actuation: MixedActuationProfile | None = None
    allow_unqualified_passive_diagnostic: bool = False
    diagnostic_policy: NativeDiagnosticPolicy | None = None
    wrapped_path_declaration: WrappedPathDeclaration | None = None
    qualification_context: NativeQualificationContext | None = None
    native_mechanics_profile: NativeMechanicsExecutionProfile | None = None
    contact_declaration: NativeContactExecutionDeclaration | None = None

    def __post_init__(self) -> None:
        self.config.validate()
        _validate_request_policies(self)
        _validate_request_inputs(self)
        weights = dict(self.marker_weights)
        bindings = dict(self.marker_bindings)
        exclusions = dict(self.excluded_markers)
        object.__setattr__(self, "marker_weights", MappingProxyType(weights))
        object.__setattr__(self, "marker_bindings", MappingProxyType(bindings))
        object.__setattr__(self, "excluded_markers", MappingProxyType(exclusions))

    @property
    def identity_sha256(self) -> str:
        """Bind every caller policy and placement across all execution stages."""
        registration = self.registration
        declared = self.constrained_cold_start
        payload: dict[str, Any] = {
            "model_path": str(self.model_path.resolve()),
            "trc_path": str(self.trc_path.resolve()),
            "guess_path": str(self.states_guess_path.resolve()),
            "source_hashes": (
                self.model_sha256,
                self.trc_sha256,
                self.states_guess_sha256,
            ),
            "state_bounds": dict(self.bindings.state_bounds),
            "initial_state": dict(self.bindings.initial_state),
            "control_bounds": dict(self.bindings.control_bounds),
            "config": asdict(self.config),
            "marker_weights": dict(self.marker_weights),
            "marker_bindings": dict(self.marker_bindings),
            "registration": None
            if registration is None
            else {
                "rotation": registration.rotation.tolist(),
                "translation": registration.translation.tolist(),
                "source_frame": registration.source_frame,
                "target_frame": registration.target_frame,
            },
            "reference_frame_path": self.reference_frame_path,
            "passive_policy": (
                None if self.passive_policy is None else asdict(self.passive_policy)
            ),
            "excluded_markers": dict(self.excluded_markers),
            "mixed_actuation": None
            if self.mixed_actuation is None
            else asdict(self.mixed_actuation),
            "constrained_cold_start": None
            if declared is None
            else {
                "lock_targets": dict(declared.lock_targets),
                "chart_bounds": dict(declared.chart_bounds),
                "linear_chart_bounds": {
                    name: (dict(terms), lower, upper)
                    for name, (
                        terms,
                        lower,
                        upper,
                    ) in declared.linear_chart_bounds.items()
                },
                "constraint_enforcement": dict(declared.constraint_enforcement),
                "residual_tolerance": declared.residual_tolerance,
            },
        }
        if self.diagnostic_policy is not None:
            payload["diagnostic_policy"] = {
                "source_sha256": self.diagnostic_policy.source_sha256,
                "allowed_unqualified_findings": sorted(
                    code.value
                    for code in self.diagnostic_policy.allowed_unqualified_findings
                ),
                "provenance": self.diagnostic_policy.provenance,
            }
        if self.wrapped_path_declaration is not None:
            payload["wrapped_path_declaration_sha256"] = (
                self.wrapped_path_declaration.declaration_sha256
            )
        if self.qualification_context is not None:
            payload["qualification_context"] = {
                "source_sha256": self.qualification_context.source_sha256,
                "findings": [
                    {
                        "code": finding.code.value,
                        "source_sha256": finding.source_sha256,
                        "component_path": finding.component_path,
                        "evidence_sha256": finding.evidence_sha256,
                        "execution_effect": finding.execution_effect.value,
                        "qualification_effect": finding.qualification_effect.value,
                    }
                    for finding in self.qualification_context.findings
                ],
                "provenance_references": [
                    {"evidence_sha256": evidence_sha, "references": references}
                    for evidence_sha, references in self.qualification_context.provenance_references
                ],
            }
        if self.native_mechanics_profile is not None:
            payload["native_mechanics_profile_sha256"] = (
                self.native_mechanics_profile.profile_sha256
            )
        if self.contact_declaration is not None:
            payload["native_contact_declaration_sha256"] = (
                self.contact_declaration.declaration_sha256
            )
        if self.allow_unqualified_passive_diagnostic:
            payload["allow_unqualified_passive_diagnostic"] = True
        return hashlib.sha256(
            json.dumps(payload, sort_keys=True, allow_nan=False).encode()
        ).hexdigest()

    @property
    def effective_diagnostic_policy(self) -> NativeDiagnosticPolicy | None:
        """Normalize the legacy passive-only switch into the typed policy."""
        if self.diagnostic_policy is not None:
            return self.diagnostic_policy
        if self.allow_unqualified_passive_diagnostic:
            return NativeDiagnosticPolicy(
                source_sha256=self.model_sha256,
                allowed_unqualified_findings=(
                    DiagnosticCode.PASSIVE_POLICY_UNAVAILABLE,
                ),
                provenance="legacy passive diagnostic compatibility",
            )
        return None


def _keys(value: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - allowed
    ):
        raise ValueError("Native Moco request has missing or unknown fields")
    return value


def _parse_registration(registration_data: Any) -> CaptureRegistration | None:
    if registration_data is None:
        return None
    checked = _keys(
        registration_data,
        {field.name for field in fields(CaptureRegistration)},
        {"rotation", "translation"},
    )
    return CaptureRegistration(
        np.asarray(checked["rotation"], dtype=float),
        np.asarray(checked["translation"], dtype=float),
        checked.get("source_frame", "capture"),
        checked.get("target_frame", "world"),
    )


def _parse_passive_policy(policy_data: Any) -> PassiveReadinessPolicy | None:
    if policy_data is None:
        return None
    checked_policy = _keys(
        policy_data,
        {field.name for field in fields(PassiveReadinessPolicy)},
        {field.name for field in fields(PassiveReadinessPolicy)},
    )
    limits = tuple(
        MusclePassiveLimits(
            **{
                **checked,
                "normalized_fiber_range": tuple(checked["normalized_fiber_range"]),
            }
        )
        for item in checked_policy["limits"]
        for checked in (
            _keys(
                item,
                {field.name for field in fields(MusclePassiveLimits)},
                {field.name for field in fields(MusclePassiveLimits)},
            ),
        )
    )
    return PassiveReadinessPolicy(
        checked_policy["loaded_model_sha256"],
        checked_policy["preparation_scope"],
        limits,
    )


def _parse_declared_cold_start(
    declared_data: Any,
    payload: dict[str, Any],
    binding: dict[str, Any],
    config: dict[str, Any],
    model_path: Path,
) -> DeclaredColdStart | None:
    if declared_data is None:
        return None
    checked = _keys(
        declared_data,
        {
            "version",
            "lock_targets",
            "chart_bounds",
            "linear_chart_bounds",
            "constraint_enforcement",
            "residual_tolerance",
        },
        {
            "version",
            "lock_targets",
            "chart_bounds",
            "linear_chart_bounds",
            "constraint_enforcement",
            "residual_tolerance",
        },
    )
    if checked["version"] != "declared-constrained-muscles/2.0.0":
        raise ValueError("Unknown constrained muscle policy version")
    return DeclaredColdStart(
        model_path,
        payload["model_sha256"],
        binding["initial_state"],
        float(config.get("t_start_s", 0.0)),
        checked["lock_targets"],
        {name: tuple(value) for name, value in checked["chart_bounds"].items()},
        checked["constraint_enforcement"],
        checked["residual_tolerance"],
        linear_chart_bounds={
            name: (terms, lower, upper)
            for name, (terms, lower, upper) in checked["linear_chart_bounds"].items()
        },
    )


def _parse_mixed_profile(mixed_data: Any) -> MixedActuationProfile | None:
    mixed = None
    if mixed_data is not None:
        mixed_data = _keys(mixed_data, {"channels"}, {"channels"})
        mixed = MixedActuationProfile(
            tuple(
                MixedChannel(
                    checked["path"],
                    ActuationRole(checked["role"]),
                    tuple(checked["control_bounds"]),
                )
                for item in mixed_data["channels"]
                for checked in (
                    _keys(
                        item,
                        {"path", "role", "control_bounds"},
                        {"path", "role", "control_bounds"},
                    ),
                )
            )
        )
    return mixed


def _parse_diagnostic_policy(data: Any) -> NativeDiagnosticPolicy | None:
    if data is None:
        return None
    checked = _keys(
        data,
        {"source_sha256", "allowed_unqualified_findings", "provenance"},
        {"source_sha256", "allowed_unqualified_findings", "provenance"},
    )
    findings = checked["allowed_unqualified_findings"]
    if not isinstance(findings, list) or any(
        not isinstance(code, str) for code in findings
    ):
        raise ValueError("Diagnostic allowlist must be a JSON string array")
    try:
        codes = tuple(DiagnosticCode(code) for code in findings)
    except ValueError as exc:
        raise ValueError(
            "Diagnostic allowlist contains an unknown finding code"
        ) from exc
    return NativeDiagnosticPolicy(
        source_sha256=checked["source_sha256"],
        allowed_unqualified_findings=codes,
        provenance=checked["provenance"],
    )


def _parse_qualification_context(data: Any) -> NativeQualificationContext | None:
    if data is None:
        return None
    checked = _keys(
        data,
        {"source_sha256", "findings", "provenance_references"},
        {"source_sha256", "findings", "provenance_references"},
    )
    if not isinstance(checked["findings"], list) or not isinstance(
        checked["provenance_references"], list
    ):
        raise ValueError("Qualification context findings and references must be arrays")
    findings: list[NativeDiagnosticFinding] = []
    for raw in checked["findings"]:
        item = _keys(
            raw,
            {
                "code",
                "source_sha256",
                "component_path",
                "evidence_sha256",
                "execution_effect",
                "qualification_effect",
            },
            {"code", "source_sha256", "evidence_sha256"},
        )
        try:
            code = DiagnosticCode(item["code"])
        except (ValueError, TypeError) as exc:
            raise ValueError(
                "Qualification context has an unknown finding code"
            ) from exc
        execution, qualification = diagnostic_effects(code)
        if "execution_effect" in item and item["execution_effect"] != execution.value:
            raise ValueError("Qualification context execution effect is not registered")
        if (
            "qualification_effect" in item
            and item["qualification_effect"] != qualification.value
        ):
            raise ValueError(
                "Qualification context qualification effect is not registered"
            )
        findings.append(
            NativeDiagnosticFinding(
                code=code,
                source_sha256=item["source_sha256"],
                component_path=item.get("component_path"),
                evidence_sha256=item["evidence_sha256"],
                execution_effect=execution,
                qualification_effect=qualification,
            )
        )
    provenance: list[tuple[str, tuple[str, ...]]] = []
    for raw_reference in checked["provenance_references"]:
        item = _keys(
            raw_reference,
            {"evidence_sha256", "references"},
            {"evidence_sha256", "references"},
        )
        if not isinstance(item["references"], list):
            raise ValueError("Qualification provenance references must be arrays")
        provenance.append((item["evidence_sha256"], tuple(item["references"])))
    return NativeQualificationContext(
        source_sha256=checked["source_sha256"],
        findings=tuple(findings),
        provenance_references=tuple(provenance),
    )


def _parse_wrapped_path_declaration(data: Any) -> WrappedPathDeclaration | None:
    if data is None:
        return None
    checked = _keys(
        data,
        {
            "schema_version",
            "source_sha256",
            "wrap_objects",
            "path_wraps",
            "conditional_points",
        },
        {
            "schema_version",
            "source_sha256",
            "wrap_objects",
            "path_wraps",
            "conditional_points",
        },
    )
    wraps = tuple(
        WrapObjectIdentity(
            **{
                **item,
                "translation_m": tuple(item["translation_m"]),
                "xyz_body_rotation_rad": tuple(item["xyz_body_rotation_rad"]),
                "shape_parameters": tuple(
                    (str(name), float(value))
                    for name, value in item["shape_parameters"]
                ),
            }
        )
        for raw in checked["wrap_objects"]
        for item in (
            _keys(
                raw,
                {field.name for field in fields(WrapObjectIdentity)},
                {field.name for field in fields(WrapObjectIdentity)},
            ),
        )
    )
    path_wraps = tuple(
        PathWrapIdentity(**{**item, "range_indices": tuple(item["range_indices"])})
        for raw in checked["path_wraps"]
        for item in (
            _keys(
                raw,
                {field.name for field in fields(PathWrapIdentity)},
                {field.name for field in fields(PathWrapIdentity)},
            ),
        )
    )
    conditional_points = tuple(
        ConditionalPathPointIdentity(
            **{**item, "active_range": tuple(item["active_range"])}
        )
        for raw in checked["conditional_points"]
        for item in (
            _keys(
                raw,
                {field.name for field in fields(ConditionalPathPointIdentity)},
                {field.name for field in fields(ConditionalPathPointIdentity)},
            ),
        )
    )
    return WrappedPathDeclaration(
        checked["schema_version"],
        checked["source_sha256"],
        wraps,
        path_wraps,
        conditional_points,
    )


def _parse_native_mechanics_profile(
    data: Any,
) -> NativeMechanicsExecutionProfile | None:
    if data is None:
        return None
    from .native_mechanics_profile import (
        NativeMechanicsComponentIdentity,
        NativeMechanicsExecutionProfile,
    )

    checked = _keys(
        data,
        {"schema_version", "source_sha256", "components", "profile_sha256"},
        {"schema_version", "source_sha256", "components", "profile_sha256"},
    )
    if not isinstance(checked["components"], list):
        raise ValueError("Native mechanics profile components must be an array")
    components = tuple(
        NativeMechanicsComponentIdentity(**item)
        for raw in checked["components"]
        for item in (
            _keys(
                raw,
                {field.name for field in fields(NativeMechanicsComponentIdentity)},
                {field.name for field in fields(NativeMechanicsComponentIdentity)},
            ),
        )
    )
    profile = NativeMechanicsExecutionProfile(
        checked["schema_version"], checked["source_sha256"], components
    )
    if checked["profile_sha256"] != profile.profile_sha256:
        raise ValueError("Native mechanics profile digest does not match its contents")
    return profile


def _parse_native_contact_declaration(
    data: Any,
) -> NativeContactExecutionDeclaration | None:
    if data is None:
        return None
    from .native_contact_policy import (
        NativeContactExecutionDeclaration,
        NativeContactForceIdentity,
        NativeContactGeometryIdentity,
    )

    checked = _keys(
        data,
        {
            "schema_version",
            "source_sha256",
            "forces",
            "geometries",
            "declaration_sha256",
        },
        {
            "schema_version",
            "source_sha256",
            "forces",
            "geometries",
            "declaration_sha256",
        },
    )
    if not isinstance(checked["forces"], list) or not isinstance(
        checked["geometries"], list
    ):
        raise ValueError("Native contact identities must be JSON arrays")
    forces = tuple(
        NativeContactForceIdentity(**item)
        for raw in checked["forces"]
        for item in (
            _keys(
                raw,
                {field.name for field in fields(NativeContactForceIdentity)},
                {field.name for field in fields(NativeContactForceIdentity)},
            ),
        )
    )
    geometries = tuple(
        NativeContactGeometryIdentity(**item)
        for raw in checked["geometries"]
        for item in (
            _keys(
                raw,
                {field.name for field in fields(NativeContactGeometryIdentity)},
                {field.name for field in fields(NativeContactGeometryIdentity)},
            ),
        )
    )
    declaration = NativeContactExecutionDeclaration(
        checked["schema_version"], checked["source_sha256"], forces, geometries
    )
    if checked["declaration_sha256"] != declaration.declaration_sha256:
        raise ValueError("Native contact declaration digest does not match contents")
    return declaration


def load_native_moco_request(path: Path) -> NativeMocoRequest:
    """Read explicit files, mapping and policies without deriving anatomy."""
    request_path = Path(path).resolve()
    payload = _keys(
        json.loads(request_path.read_text(encoding="utf-8")),
        {field.name for field in fields(NativeMocoRequest)},
        {
            field.name
            for field in fields(NativeMocoRequest)
            if field.default is MISSING and field.default_factory is MISSING
        },
    )
    binding = _keys(
        payload["bindings"],
        {field.name for field in fields(MocoInitialBindings)},
        {field.name for field in fields(MocoInitialBindings)},
    )
    config = _keys(
        payload["config"],
        {field.name for field in fields(MocoTrackingConfig)},
        set(),
    )
    registration = _parse_registration(payload["registration"])
    policy = _parse_passive_policy(payload["passive_policy"])
    placements = {
        label: (value[0], tuple(value[1]))
        for label, value in payload["marker_bindings"].items()
    }

    def source_path(key: str) -> Path:
        candidate = Path(payload[key])
        if not candidate.is_absolute():
            candidate = request_path.parent / candidate
        return candidate.resolve()

    declared = _parse_declared_cold_start(
        payload.get("constrained_cold_start"),
        payload,
        binding,
        config,
        source_path("model_path"),
    )

    return NativeMocoRequest(
        source_path("model_path"),
        source_path("trc_path"),
        source_path("states_guess_path"),
        payload["model_sha256"],
        payload["trc_sha256"],
        payload["states_guess_sha256"],
        MocoInitialBindings(**binding),
        MocoTrackingConfig(**config),
        payload["marker_weights"],
        placements,
        registration,
        payload["reference_frame_path"],
        policy,
        payload["excluded_markers"],
        declared,
        _parse_mixed_profile(payload.get("mixed_actuation")),
        payload.get("allow_unqualified_passive_diagnostic", False),
        _parse_diagnostic_policy(payload.get("diagnostic_policy")),
        _parse_wrapped_path_declaration(payload.get("wrapped_path_declaration")),
        _parse_qualification_context(payload.get("qualification_context")),
        _parse_native_mechanics_profile(payload.get("native_mechanics_profile")),
        _parse_native_contact_declaration(payload.get("contact_declaration")),
    )


__all__ = ["load_native_moco_request"]

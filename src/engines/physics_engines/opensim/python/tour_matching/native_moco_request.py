"""Strict JSON entrypoint for the existing file-driven native Moco providers."""

from __future__ import annotations

from dataclasses import MISSING, fields
import json
from pathlib import Path
from typing import Any

import numpy as np

from .moco_initial_bindings import MocoInitialBindings
from .moco_tracking import MocoTrackingConfig
from .native_moco_runner import NativeMocoRequest
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
    )


__all__ = ["load_native_moco_request"]

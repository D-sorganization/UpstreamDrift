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
from .native_passive_readiness import MusclePassiveLimits, PassiveReadinessPolicy
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
    )


__all__ = ["load_native_moco_request"]

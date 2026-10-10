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
from .registration import CaptureRegistration


def _keys(value: Any, allowed: set[str], required: set[str]) -> dict[str, Any]:
    if (
        not isinstance(value, dict)
        or not required <= value.keys()
        or value.keys() - allowed
    ):
        raise ValueError("Native Moco request has missing or unknown fields")
    return value


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
    registration_data = payload["registration"]
    registration = None
    if registration_data is not None:
        registration_data = _keys(
            registration_data,
            {field.name for field in fields(CaptureRegistration)},
            {"rotation", "translation"},
        )
        registration = CaptureRegistration(
            np.asarray(registration_data["rotation"], dtype=float),
            np.asarray(registration_data["translation"], dtype=float),
            registration_data.get("source_frame", "capture"),
            registration_data.get("target_frame", "world"),
        )
    policy_data = payload["passive_policy"]
    policy = None
    if policy_data is not None:
        policy_data = _keys(
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
            for item in policy_data["limits"]
            for checked in (
                _keys(
                    item,
                    {field.name for field in fields(MusclePassiveLimits)},
                    {field.name for field in fields(MusclePassiveLimits)},
                ),
            )
        )
        policy = PassiveReadinessPolicy(
            policy_data["loaded_model_sha256"], policy_data["preparation_scope"], limits
        )
    placements = {
        label: (value[0], tuple(value[1]))
        for label, value in payload["marker_bindings"].items()
    }

    def source_path(key: str) -> Path:
        candidate = Path(payload[key])
        if not candidate.is_absolute():
            candidate = request_path.parent / candidate
        return candidate.resolve()

    declared_data = payload.get("constrained_cold_start")
    declared = None
    if declared_data is not None:
        declared_data = _keys(
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
        if declared_data["version"] != "declared-constrained-muscles/2.0.0":
            raise ValueError("Unknown constrained muscle policy version")
        declared = DeclaredColdStart(
            source_path("model_path"),
            payload["model_sha256"],
            binding["initial_state"],
            float(config.get("t_start_s", 0.0)),
            declared_data["lock_targets"],
            {
                name: tuple(value)
                for name, value in declared_data["chart_bounds"].items()
            },
            declared_data["constraint_enforcement"],
            declared_data["residual_tolerance"],
            linear_chart_bounds={
                name: (terms, lower, upper)
                for name, (terms, lower, upper) in declared_data[
                    "linear_chart_bounds"
                ].items()
            },
        )

    mixed_data = payload.get("mixed_actuation")
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
        mixed,
    )


__all__ = ["load_native_moco_request"]

"""Strict JSON entrypoint for the existing file-driven native Moco providers."""

from __future__ import annotations

from dataclasses import fields
import json
from pathlib import Path
from typing import Any

import numpy as np

from .moco_initial_bindings import MocoInitialBindings
from .moco_tracking import MocoTrackingConfig
from .native_moco_runner import NativeMocoRequest
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
        {field.name for field in fields(NativeMocoRequest)},
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
    )


__all__ = ["load_native_moco_request"]

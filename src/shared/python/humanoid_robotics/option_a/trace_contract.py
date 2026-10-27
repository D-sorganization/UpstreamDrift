from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

TRACE_SCHEMA_VERSION = "humanoid_option_a.trace/v1"
FAILURE_DEFINITION_ID = "affine_drift.option_a.failure_definition/v0-placeholder"
RECOVERY_THRESHOLD_RAD = 0.40
TRACKING_ERROR_BASELINE_RAD = 0.28
_REQUIRED_ARRAYS = (
    "x_k",
    "u_k",
    "A_k",
    "B_k",
    "trial_index",
    "outcome_label",
    "baseline_flag_time_s",
    "control_effort",
)


def validate_trace_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema: {bundle.get('schema_version')!r}")
    if bundle.get("failure_definition_id") != FAILURE_DEFINITION_ID:
        raise ValueError("failure_definition_id mismatch")
    for key in _REQUIRED_ARRAYS:
        if key not in bundle:
            raise ValueError(f"missing trace field: {key}")
    x_k = np.asarray(bundle["x_k"], dtype=np.float64)
    u_k = np.asarray(bundle["u_k"], dtype=np.float64)
    a_k = np.asarray(bundle["A_k"], dtype=np.float64)
    b_k = np.asarray(bundle["B_k"], dtype=np.float64)
    if x_k.ndim != 2 or u_k.ndim != 2 or a_k.ndim != 3 or b_k.ndim != 3:
        raise ValueError("trace arrays have invalid rank")
    steps, nx = x_k.shape
    nu = u_k.shape[1]
    if u_k.shape[0] != steps or a_k.shape[0] != steps or b_k.shape[0] != steps:
        raise ValueError("step dimension mismatch across trace arrays")
    if a_k.shape[1:] != (nx, nx) or b_k.shape[1:] != (nx, nu):
        raise ValueError("Jacobian shape mismatch")
    for name, arr in (("x_k", x_k), ("u_k", u_k), ("A_k", a_k), ("B_k", b_k)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
    if int(np.asarray(bundle["trial_index"]).shape[0]) <= 0:
        raise ValueError("at least one trial is required")


def canonical_payload_bytes(bundle: Mapping[str, Any]) -> bytes:
    arrays = {key: np.asarray(bundle[key]) for key in _REQUIRED_ARRAYS}
    meta = {
        "schema_version": bundle["schema_version"],
        "failure_definition_id": bundle["failure_definition_id"],
        "seed": int(bundle["seed"]),
        "mujoco_version": str(bundle["mujoco_version"]),
        "timestep_s": float(bundle["timestep_s"]),
        "torque_limit_nm": float(bundle["torque_limit_nm"]),
        "perturbation_metadata": bundle["perturbation_metadata"],
    }
    header = json.dumps(meta, sort_keys=True, separators=(",", ":")).encode("utf-8")
    parts = [header]
    for key in _REQUIRED_ARRAYS:
        parts.extend((key.encode("ascii"), arrays[key].tobytes(order="C")))
    return b"".join(parts)


def payload_sha256(bundle: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_payload_bytes(bundle)).hexdigest()

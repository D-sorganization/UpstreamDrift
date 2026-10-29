from __future__ import annotations

import hashlib
import json
from typing import Any, Mapping

import numpy as np

TRACE_SCHEMA_VERSION = "humanoid_option_a.trace/v2"
FAILURE_DEFINITION: dict[str, Any] = {
    "id": "affine_drift.option_a.failure_definition/v1",
    "quantity": "terminal_distal_tip_position_error",
    "unit": "m",
    "tolerance": 0.05,
}
FAILURE_DEFINITION_ID = str(FAILURE_DEFINITION["id"])
RECOVERY_THRESHOLD_M = float(FAILURE_DEFINITION["tolerance"])
RECOVERY_THRESHOLD_RAD = 0.40
TRACKING_ERROR_BASELINE_RAD = 0.28
_REQUIRED_ARRAYS = (
    "x_k",
    "u_k",
    "A_k",
    "B_k",
    "q_ref_k",
    "v_ref_k",
    "trial_index",
    "outcome_label",
    "baseline_flag_time_s",
    "control_effort",
)


def validate_trace_bundle(bundle: Mapping[str, Any]) -> None:
    if bundle.get("schema_version") != TRACE_SCHEMA_VERSION:
        raise ValueError(f"unsupported trace schema: {bundle.get('schema_version')!r}")
    failure = bundle.get("failure_definition")
    if failure != FAILURE_DEFINITION:
        raise ValueError("failure_definition mismatch")
    if bundle.get("failure_definition_id") != FAILURE_DEFINITION_ID:
        raise ValueError("failure_definition_id mismatch")
    for key in _REQUIRED_ARRAYS:
        if key not in bundle:
            raise ValueError(f"missing trace field: {key}")
    x_k = np.asarray(bundle["x_k"], dtype=np.float64)
    u_k = np.asarray(bundle["u_k"], dtype=np.float64)
    a_k = np.asarray(bundle["A_k"], dtype=np.float64)
    b_k = np.asarray(bundle["B_k"], dtype=np.float64)
    q_ref = np.asarray(bundle["q_ref_k"], dtype=np.float64)
    v_ref = np.asarray(bundle["v_ref_k"], dtype=np.float64)
    if x_k.ndim != 2 or u_k.ndim != 2 or a_k.ndim != 3 or b_k.ndim != 3:
        raise ValueError("trace arrays have invalid rank")
    horizon_steps = q_ref.shape[0]
    if v_ref.shape != (horizon_steps, 2) or q_ref.ndim != 2 or q_ref.shape[1] != 2:
        raise ValueError("reference trajectory shape mismatch")
    trace_steps, state_dim = x_k.shape[0], x_k.shape[1]
    nu = u_k.shape[1]
    if (
        trace_steps != x_k.shape[0]
        or a_k.shape[0] != trace_steps
        or b_k.shape[0] != trace_steps
    ):
        raise ValueError("step dimension mismatch across trace arrays")
    if trace_steps % horizon_steps != 0:
        raise ValueError(
            "trace steps must be an integer multiple of the reference horizon"
        )
    if a_k.shape[1:] != (state_dim, state_dim) or b_k.shape[1:] != (state_dim, nu):
        raise ValueError("Jacobian shape mismatch")
    for name, arr in (("x_k", x_k), ("u_k", u_k), ("A_k", a_k), ("B_k", b_k)):
        if not np.isfinite(arr).all():
            raise ValueError(f"{name} contains non-finite values")
    if not np.isfinite(q_ref).all() or not np.isfinite(v_ref).all():
        raise ValueError("reference trajectory contains non-finite values")
    if int(np.asarray(bundle["trial_index"]).shape[0]) <= 0:
        raise ValueError("at least one trial is required")


def canonical_payload_bytes(bundle: Mapping[str, Any]) -> bytes:
    arrays = {key: np.asarray(bundle[key]) for key in _REQUIRED_ARRAYS}
    meta = {
        "schema_version": bundle["schema_version"],
        "failure_definition": bundle["failure_definition"],
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

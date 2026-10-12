"""Atomic, source-bound receipt for one returned native passive audit."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import hashlib
import json
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from src.engines.physics_engines.opensim.python.tour_matching.native_passive_readiness import (
        NativePassiveReadinessAudit,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_request import (
        NativeMocoRequest,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_moco_preparation import (
        _NativeModelPreparation,
    )

from src.shared.python.motion_matching.jobs.io_atomic import atomic_write_bytes


_RECEIPT_NAME = "native_passive_observation.json"
_SCHEMA = "native-passive-observation/1.0.0"


@dataclass(frozen=True)
class NativePassiveObservationReceiptReference:
    """Integrity and lineage fields recorded in native preparation."""

    relative_path: str
    size_bytes: int
    sha256: str
    audit_sha256: str
    request_sha256: str
    source_sha256: str
    loaded_model_sha256: str
    runtime_sha256: str
    prepared_state_sha256: str
    observer_sha256: str
    policy_sha256: str | None


def _canonical_json(value: Any) -> bytes:
    return (
        json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")


def _audit_digest(observation: Any) -> str:
    """Match NativePassiveReadinessAudit's established identity encoding."""
    return hashlib.sha256(
        json.dumps(observation, sort_keys=True, allow_nan=False).encode("utf-8")
    ).hexdigest()


def write_native_passive_observation_receipt(
    directory: Path,
    audit: NativePassiveReadinessAudit,
    *,
    request_sha256: str,
    source_sha256: str,
    runtime_sha256: str,
    observer_sha256: str,
) -> NativePassiveObservationReceiptReference:
    """Persist an existing audit object without invoking any native observer."""
    native = audit.native_state
    observation = asdict(audit)
    if audit.observer_sha256 != observer_sha256:
        raise ValueError("passive observer source identity mismatch")
    if not isinstance(audit.muscles, tuple) or not isinstance(native.actuators, tuple):
        raise ValueError("passive muscle or actuator inventory is malformed")
    if any(not isinstance(item.path, str) or not item.path for item in audit.muscles):
        raise ValueError("passive muscle paths must be nonempty strings")
    if any(
        not isinstance(item.path, str)
        or not item.path
        or not isinstance(item.is_muscle, bool)
        for item in native.actuators
    ):
        raise ValueError("native actuator paths and muscle flags are malformed")
    muscle_paths = [item.path for item in audit.muscles]
    native_muscle_paths = [item.path for item in native.actuators if item.is_muscle]
    if (
        len(set(muscle_paths)) != len(muscle_paths)
        or len(set(native_muscle_paths)) != len(native_muscle_paths)
        or set(muscle_paths) != set(native_muscle_paths)
    ):
        raise ValueError("passive muscle coverage does not match native actuators")
    raw = _canonical_json(
        {
            "schema_version": _SCHEMA,
            "request_sha256": request_sha256,
            "source_sha256": source_sha256,
            "loaded_model_sha256": native.loaded_model_sha256,
            "runtime_sha256": runtime_sha256,
            "prepared_state_sha256": native.observation_sha256,
            "observer_sha256": audit.observer_sha256,
            "muscle_count": len(audit.muscles),
            "policy_sha256": audit.policy_sha256,
            "observation_sha256": audit.observation_sha256,
            "observation": observation,
        }
    )
    path = directory / _RECEIPT_NAME
    atomic_write_bytes(path, raw)
    return NativePassiveObservationReceiptReference(
        relative_path=_RECEIPT_NAME,
        size_bytes=len(raw),
        sha256=hashlib.sha256(raw).hexdigest(),
        audit_sha256=audit.observation_sha256,
        request_sha256=request_sha256,
        source_sha256=source_sha256,
        loaded_model_sha256=native.loaded_model_sha256,
        runtime_sha256=runtime_sha256,
        prepared_state_sha256=native.observation_sha256,
        observer_sha256=audit.observer_sha256,
        policy_sha256=audit.policy_sha256,
    )


def _validate_passive_muscle_inventory(
    payload: dict[str, Any],
    observation: dict[str, Any],
    loaded_model_sha256: str,
    policy_sha256: str | None,
    observer_sha256: str,
    reference: NativePassiveObservationReceiptReference,
) -> None:
    native_state = observation.get("native_state")
    muscles = observation.get("muscles")
    if not isinstance(native_state, dict) or not isinstance(muscles, list):
        raise ValueError("receipt native state or muscle inventory is missing")
    if _audit_digest(native_state) != reference.prepared_state_sha256:
        raise ValueError("receipt prepared-state digest mismatch")
    if native_state.get("loaded_model_sha256") != loaded_model_sha256:
        raise ValueError("receipt native loaded-model identity mismatch")
    if observation.get("policy_sha256") != policy_sha256:
        raise ValueError("receipt passive-policy identity mismatch")
    if len(muscles) != payload.get("muscle_count") or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not item["path"]
        for item in muscles
    ):
        raise ValueError("receipt muscle inventory is malformed")
    muscle_paths = [item["path"] for item in muscles]
    actuators = native_state.get("actuators")
    if not isinstance(actuators, list) or any(
        not isinstance(item, dict)
        or not isinstance(item.get("path"), str)
        or not item["path"]
        or not isinstance(item.get("is_muscle"), bool)
        for item in actuators
    ):
        raise ValueError("receipt native actuator inventory is malformed")
    native_muscle_paths = [item["path"] for item in actuators if item["is_muscle"]]
    if (
        len(set(muscle_paths)) != len(muscle_paths)
        or len(set(native_muscle_paths)) != len(native_muscle_paths)
        or set(muscle_paths) != set(native_muscle_paths)
        or payload.get("muscle_count") != len(muscles)
    ):
        raise ValueError("receipt passive muscle coverage mismatch")


def validate_native_passive_observation_receipt(
    directory: Path,
    reference: NativePassiveObservationReceiptReference,
    *,
    request_sha256: str,
    source_sha256: str,
    loaded_model_sha256: str,
    runtime_sha256: str,
    policy_sha256: str | None,
    observer_sha256: str,
) -> dict[str, Any]:
    """Fail closed unless bytes and every expected source identity match."""
    if Path(reference.relative_path).name != reference.relative_path:
        raise ValueError("receipt path must stay relative to the owned directory")
    path = (directory / reference.relative_path).resolve()
    root = directory.resolve()
    if path.parent != root:
        raise ValueError("receipt path escapes the owned directory")
    raw = path.read_bytes()
    if (
        len(raw) != reference.size_bytes
        or hashlib.sha256(raw).hexdigest() != reference.sha256
    ):
        raise ValueError("receipt integrity mismatch")
    try:
        payload = json.loads(raw)
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError("receipt JSON is malformed") from exc
    if not isinstance(payload, dict):
        raise ValueError("receipt top-level JSON must be an object")
    # Canonical re-encoding rejects NaN/Infinity and noncanonical ambiguous bytes.
    if _canonical_json(payload) != raw:
        raise ValueError("receipt JSON is nonfinite or noncanonical")
    expected = {
        "schema_version": _SCHEMA,
        "request_sha256": request_sha256,
        "source_sha256": source_sha256,
        "loaded_model_sha256": loaded_model_sha256,
        "runtime_sha256": runtime_sha256,
        "policy_sha256": policy_sha256,
        "prepared_state_sha256": reference.prepared_state_sha256,
        "observer_sha256": observer_sha256,
        "observation_sha256": reference.audit_sha256,
    }
    for key, value in expected.items():
        if payload.get(key) != value:
            raise ValueError(f"receipt {key} identity mismatch")
    if (
        reference.request_sha256 != request_sha256
        or reference.source_sha256 != source_sha256
        or reference.loaded_model_sha256 != loaded_model_sha256
        or reference.runtime_sha256 != runtime_sha256
        or reference.policy_sha256 != policy_sha256
        or reference.observer_sha256 != observer_sha256
    ):
        raise ValueError("receipt reference identity mismatch")
    observation = payload.get("observation")
    if not isinstance(observation, dict):
        raise ValueError("receipt observation is missing")
    audit_sha = _audit_digest(observation)
    if (
        audit_sha != reference.audit_sha256
        or payload.get("observation_sha256") != reference.audit_sha256
    ):
        raise ValueError("receipt audit digest mismatch")
    if observation.get("observer_sha256") != reference.observer_sha256:
        raise ValueError("receipt observer identity mismatch")
    _validate_passive_muscle_inventory(
        payload,
        observation,
        loaded_model_sha256,
        policy_sha256,
        observer_sha256,
        reference,
    )
    return payload


def prepare_native_passive_observation(
    directory: Path,
    audits: list[NativePassiveReadinessAudit],
    request: NativeMocoRequest,
    model_result: _NativeModelPreparation,
) -> tuple[NativePassiveObservationReceiptReference | None, tuple[str, ...]]:
    """Validate and persist the one passive audit returned during preparation."""
    reference: NativePassiveObservationReceiptReference | None = None
    if audits:
        try:
            if len(audits) != 1:
                raise ValueError("passive audit sink received multiple observations")
            observed = audits[0]
            if (
                model_result.loaded_sha256 is None
                or model_result.runtime_sha256 is None
            ):
                raise ValueError(
                    "passive receipt requires loaded model and runtime identities"
                )
            if observed.native_state.loaded_model_sha256 != model_result.loaded_sha256:
                raise ValueError(
                    "passive audit loaded model identity does not match preparation"
                )
            if observed.observation_sha256 != model_result.passive_sha256:
                raise ValueError("passive audit digest does not match preparation")
            from .native_passive_readiness import audit_native_passive_readiness

            observer_sha256 = hashlib.sha256(
                Path(audit_native_passive_readiness.__code__.co_filename).read_bytes()
            ).hexdigest()
            reference = write_native_passive_observation_receipt(
                directory,
                observed,
                request_sha256=request.identity_sha256,
                source_sha256=request.model_sha256,
                runtime_sha256=model_result.runtime_sha256,
                observer_sha256=observer_sha256,
            )
        except (OSError, ValueError, TypeError):
            reference = None
    required = (
        model_result.passive_sha256 is not None
        or request.allow_unqualified_passive_diagnostic
    )
    blockers = (
        ("native-passive-observation-receipt-unavailable",)
        if required and reference is None
        else ()
    )
    return reference, blockers

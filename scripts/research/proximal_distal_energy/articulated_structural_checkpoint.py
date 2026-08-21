"""Persist structural atlas restart checkpoints without weakening identity gates."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import hashlib
import io
import json
from pathlib import Path
from typing import Any, TypeAlias
import zipfile

import numpy as np
from numpy.typing import NDArray

from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    StructuralExecutionIdentity,
    structural_checkpoint_metadata,
    validate_structural_checkpoint_metadata,
)

Array: TypeAlias = NDArray[Any]
ArraySignature: TypeAlias = tuple[tuple[int, ...], str]
ArrayContract: TypeAlias = dict[str, ArraySignature]
BranchContracts: TypeAlias = Mapping[tuple[str, int], Mapping[str, ArraySignature]]
CheckpointKey: TypeAlias = tuple[int, str, int]
CheckpointDescriptor: TypeAlias = tuple[int, tuple[int, int], str, int]
METADATA_FIELD = "__metadata__"


@dataclass(slots=True)
class StructuralCheckpointInventory:
    """Validated restart payloads and the exact remaining registered work."""

    audit: dict[str, Any]
    restored: dict[CheckpointKey, dict[str, Array]]
    pending: tuple[CheckpointDescriptor, ...]


def _field_name(name: object) -> str:
    if not isinstance(name, str) or not name or name == METADATA_FIELD:
        raise ValueError("checkpoint array names must be nonempty and nonreserved")
    return name


def _safe_array(name: str, value: Any) -> Array:
    array = np.asarray(value)
    if array.dtype.hasobject:
        raise ValueError(f"checkpoint array {name!r} may not require pickle")
    if array.dtype.kind not in "biufc":
        raise ValueError(f"checkpoint array {name!r} must be numeric or Boolean")
    if array.dtype.kind in "fc" and np.any(np.isinf(array)):
        raise ValueError(f"checkpoint array {name!r} may not contain infinity")
    return array


def structural_checkpoint_array_contract(arrays: Mapping[str, Any]) -> ArrayContract:
    """Return the exact field, shape, and dtype contract for one branch payload."""

    if not isinstance(arrays, Mapping) or not arrays:
        raise ValueError("checkpoint arrays must be a nonempty mapping")
    contract: ArrayContract = {}
    for raw_name, value in arrays.items():
        name = _field_name(raw_name)
        array = _safe_array(name, value)
        contract[name] = (array.shape, array.dtype.str)
    return contract


def _normalized_contract(
    expected_contract: Mapping[str, ArraySignature],
) -> ArrayContract:
    if not isinstance(expected_contract, Mapping) or not expected_contract:
        raise ValueError("checkpoint array contract must be a nonempty mapping")
    normalized: ArrayContract = {}
    for raw_name, signature in expected_contract.items():
        name = _field_name(raw_name)
        if (
            not isinstance(signature, tuple)
            or len(signature) != 2
            or not isinstance(signature[0], tuple)
            or not all(type(value) is int and value >= 0 for value in signature[0])
            or not isinstance(signature[1], str)
        ):
            raise ValueError("checkpoint array signature is invalid")
        try:
            dtype = np.dtype(signature[1])
        except (TypeError, ValueError) as error:
            raise ValueError("checkpoint array dtype is invalid") from error
        if dtype.hasobject or dtype.kind not in "biufc" or dtype.str != signature[1]:
            raise ValueError("checkpoint array dtype is not registered")
        normalized[name] = (signature[0], signature[1])
    return normalized


def _validated_arrays(
    arrays: Mapping[str, Any], expected_contract: Mapping[str, ArraySignature]
) -> dict[str, Array]:
    observed_contract = structural_checkpoint_array_contract(arrays)
    if _normalized_contract(expected_contract) != observed_contract:
        raise RuntimeError("structural checkpoint array contract does not reproduce")
    return {name: np.asarray(arrays[name]).copy() for name in sorted(observed_contract)}


def _metadata_array(metadata: dict[str, Any]) -> Array:
    return np.asarray(json.dumps(metadata, sort_keys=True, separators=(",", ":")))


def _read_metadata(source: Any) -> dict[str, Any]:
    try:
        encoded = np.asarray(source[METADATA_FIELD])
        if encoded.shape != () or encoded.dtype.kind not in "US":
            raise ValueError
        metadata = json.loads(str(encoded.item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("structural checkpoint metadata is invalid") from error
    if not isinstance(metadata, dict):
        raise RuntimeError("structural checkpoint metadata is invalid")
    return metadata


def write_structural_checkpoint(
    path: Path,
    identity: StructuralExecutionIdentity,
    *,
    state_slot: int,
    state: tuple[int, int],
    branch_kind: str,
    branch_slot: int,
    arrays: Mapping[str, Any],
    expected_contract: Mapping[str, ArraySignature],
) -> None:
    """Validate and atomically write one digest-bound structural checkpoint."""

    payload = _validated_arrays(arrays, expected_contract)
    metadata = structural_checkpoint_metadata(
        identity,
        state_slot=state_slot,
        state=state,
        branch_kind=branch_kind,
        branch_slot=branch_slot,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    try:
        with temporary.open("wb") as stream:
            np.savez_compressed(
                stream,
                **{METADATA_FIELD: _metadata_array(metadata), **payload},
            )
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _load_structural_checkpoint_payload(
    payload: bytes,
    identity: StructuralExecutionIdentity,
    *,
    state_slot: int,
    state: tuple[int, int],
    branch_kind: str,
    branch_slot: int,
    expected_contract: Mapping[str, ArraySignature],
) -> dict[str, Array]:
    contract = _normalized_contract(expected_contract)
    expected_fields = set(contract) | {METADATA_FIELD}
    try:
        with np.load(io.BytesIO(payload), allow_pickle=False) as source:
            if set(source.files) != expected_fields:
                raise RuntimeError("structural checkpoint fields do not reproduce")
            metadata = _read_metadata(source)
            arrays = {name: np.asarray(source[name]).copy() for name in contract}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as error:
        raise RuntimeError("structural checkpoint cannot be loaded safely") from error
    validate_structural_checkpoint_metadata(
        metadata,
        identity,
        state_slot=state_slot,
        state=state,
        branch_kind=branch_kind,
        branch_slot=branch_slot,
    )
    try:
        return _validated_arrays(arrays, contract)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "structural checkpoint payload does not reproduce"
        ) from error


def load_structural_checkpoint(
    path: Path,
    identity: StructuralExecutionIdentity,
    *,
    state_slot: int,
    state: tuple[int, int],
    branch_kind: str,
    branch_slot: int,
    expected_contract: Mapping[str, ArraySignature],
) -> dict[str, Array]:
    """Load one immutable byte snapshot and reproduce every checkpoint contract."""

    try:
        payload = path.read_bytes()
    except OSError as error:
        raise RuntimeError("structural checkpoint cannot be loaded safely") from error
    return _load_structural_checkpoint_payload(
        payload,
        identity,
        state_slot=state_slot,
        state=state,
        branch_kind=branch_kind,
        branch_slot=branch_slot,
        expected_contract=expected_contract,
    )


def structural_checkpoint_path(
    directory: Path,
    identity: StructuralExecutionIdentity,
    *,
    state_slot: int,
    branch_kind: str,
    branch_slot: int,
) -> Path:
    """Resolve the only registered filename for one state and branch."""

    if type(state_slot) is not int or not 0 <= state_slot < len(
        identity.registered_states
    ):
        raise ValueError("state_slot must select a registered state")
    structural_checkpoint_metadata(
        identity,
        state_slot=state_slot,
        state=identity.registered_states[state_slot],
        branch_kind=branch_kind,
        branch_slot=branch_slot,
    )
    return directory / f"state-{state_slot:02d}-{branch_kind}-{branch_slot:02d}.npz"


def restore_structural_checkpoint_directory(
    directory: Path,
    identity: StructuralExecutionIdentity,
    *,
    expected_contracts: BranchContracts,
) -> StructuralCheckpointInventory:
    """Restore valid branches and return the exact registered pending sequence."""

    if not isinstance(expected_contracts, Mapping) or set(expected_contracts) != set(
        identity.registered_branches
    ):
        raise ValueError("checkpoint contracts must cover every registered branch")
    contracts = {
        branch: _normalized_contract(expected_contracts[branch])
        for branch in identity.registered_branches
    }
    expected = {
        structural_checkpoint_path(
            directory,
            identity,
            state_slot=state_slot,
            branch_kind=branch_kind,
            branch_slot=branch_slot,
        ): (state_slot, state, branch_kind, branch_slot)
        for state_slot, state in enumerate(identity.registered_states)
        for branch_kind, branch_slot in identity.registered_branches
    }
    observed = set(directory.iterdir()) if directory.exists() else set()
    unknown = observed - set(expected)
    if unknown:
        raise RuntimeError("checkpoint directory contains unregistered files")

    content = hashlib.sha256()
    observed_states: set[int] = set()
    branch_count: dict[int, int] = {}
    restored: dict[CheckpointKey, dict[str, Array]] = {}
    for path in sorted(observed):
        state_slot, state, branch_kind, branch_slot = expected[path]
        try:
            payload = path.read_bytes()
        except OSError as error:
            raise RuntimeError(
                "structural checkpoint cannot be loaded safely"
            ) from error
        restored[(state_slot, branch_kind, branch_slot)] = (
            _load_structural_checkpoint_payload(
                payload,
                identity,
                state_slot=state_slot,
                state=state,
                branch_kind=branch_kind,
                branch_slot=branch_slot,
                expected_contract=contracts[(branch_kind, branch_slot)],
            )
        )
        observed_states.add(state_slot)
        branch_count[state_slot] = branch_count.get(state_slot, 0) + 1
        content.update(path.name.encode("utf-8"))
        content.update(hashlib.sha256(payload).digest())

    branches_per_state = len(identity.registered_branches)
    complete_states = sum(
        count == branches_per_state for count in branch_count.values()
    )
    pending = tuple(
        descriptor for path, descriptor in expected.items() if path not in observed
    )
    status = "empty" if not observed else "complete" if not pending else "partial"
    audit = {
        "schema_version": "articulated-structural-checkpoint-audit/v1",
        "status": status,
        "checkpoint_count": len(observed),
        "expected_checkpoint_count": len(expected),
        "observed_state_slot_count": len(observed_states),
        "complete_state_slot_count": complete_states,
        "checkpoint_set_sha256": content.hexdigest(),
        "release_evidence": False,
    }
    return StructuralCheckpointInventory(
        audit=audit,
        restored=restored,
        pending=pending,
    )


def audit_structural_checkpoint_directory(
    directory: Path,
    identity: StructuralExecutionIdentity,
    *,
    expected_contracts: BranchContracts,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate exact registered checkpoint coverage without promoting partial work."""

    inventory = restore_structural_checkpoint_directory(
        directory,
        identity,
        expected_contracts=expected_contracts,
    )
    if inventory.audit["status"] == "empty" or (
        not allow_partial and inventory.audit["status"] != "complete"
    ):
        raise RuntimeError("checkpoint directory is incomplete")
    return inventory.audit


__all__ = [
    "ArrayContract",
    "ArraySignature",
    "BranchContracts",
    "CheckpointDescriptor",
    "CheckpointKey",
    "METADATA_FIELD",
    "StructuralCheckpointInventory",
    "audit_structural_checkpoint_directory",
    "load_structural_checkpoint",
    "restore_structural_checkpoint_directory",
    "structural_checkpoint_array_contract",
    "structural_checkpoint_path",
    "write_structural_checkpoint",
]

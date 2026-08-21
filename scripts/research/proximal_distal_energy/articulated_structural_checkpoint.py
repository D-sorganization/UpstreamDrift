"""Persist structural atlas restart checkpoints without weakening identity gates."""

from __future__ import annotations

from collections.abc import Mapping
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
METADATA_FIELD = "__metadata__"


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


def _validated_arrays(
    arrays: Mapping[str, Any], expected_contract: Mapping[str, ArraySignature]
) -> dict[str, Array]:
    observed_contract = structural_checkpoint_array_contract(arrays)
    if dict(expected_contract) != observed_contract:
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
    """Load one checkpoint with pickle disabled and reproduce every contract."""

    expected_fields = set(expected_contract) | {METADATA_FIELD}
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != expected_fields:
                raise RuntimeError("structural checkpoint fields do not reproduce")
            metadata = _read_metadata(source)
            arrays = {
                name: np.asarray(source[name]).copy() for name in expected_contract
            }
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
        return _validated_arrays(arrays, expected_contract)
    except (TypeError, ValueError, RuntimeError) as error:
        raise RuntimeError(
            "structural checkpoint payload does not reproduce"
        ) from error


__all__ = [
    "ArrayContract",
    "ArraySignature",
    "METADATA_FIELD",
    "load_structural_checkpoint",
    "structural_checkpoint_array_contract",
    "write_structural_checkpoint",
]

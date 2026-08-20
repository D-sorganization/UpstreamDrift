"""Audit restart checkpoints without treating partial execution as evidence."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from collections.abc import Sequence
from typing import Any

import numpy as np

SCHEMA_VERSION = "articulated-ground-branch-checkpoint/v1"
BRANCHES = tuple(
    [("primary", slot) for slot in range(4)] + [("control", slot) for slot in range(2)]
)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _valid_sha256(value: str) -> bool:
    try:
        return len(value) == 64 and int(value, 16) >= 0
    except ValueError:
        return False


def _metadata(source: Any) -> dict[str, Any]:
    try:
        raw = np.asarray(source["__metadata__"])
        if raw.shape != () or raw.dtype.kind not in "US":
            raise ValueError
        value = json.loads(str(raw.item()))
    except (KeyError, TypeError, ValueError, json.JSONDecodeError) as error:
        raise RuntimeError("checkpoint metadata is invalid") from error
    if not isinstance(value, dict):
        raise RuntimeError("checkpoint metadata is invalid")
    return value


def _reference_contract(reference: Path) -> tuple[set[str], dict[str, tuple]]:
    try:
        with np.load(reference, allow_pickle=False) as source:
            fields = set(source.files)
            contract = {
                name: (
                    np.asarray(source[name]).shape,
                    np.asarray(source[name]).dtype.str,
                )
                for name in fields - {"__metadata__"}
            }
    except (OSError, ValueError) as error:
        raise RuntimeError("checkpoint reference cannot be loaded safely") from error
    if "__metadata__" not in fields or not contract:
        raise RuntimeError("checkpoint reference fields are incomplete")
    return fields, contract


def _validate_payload(
    path: Path, fields: set[str], contract: dict[str, tuple]
) -> dict[str, Any]:
    try:
        with np.load(path, allow_pickle=False) as source:
            if set(source.files) != fields:
                raise RuntimeError("checkpoint payload fields do not match")
            metadata = _metadata(source)
            for name, (shape, dtype) in contract.items():
                value = np.asarray(source[name])
                if value.shape != shape or value.dtype.str != dtype:
                    raise RuntimeError(
                        "checkpoint payload shape or dtype does not match"
                    )
                if value.dtype.hasobject:
                    raise RuntimeError("checkpoint payload may not require pickle")
                if value.dtype.kind in "fci" and not np.all(np.isfinite(value)):
                    raise RuntimeError("checkpoint payload must be finite")
    except (OSError, ValueError) as error:
        raise RuntimeError("checkpoint payload cannot be loaded safely") from error
    return metadata


def _identity(metadata: dict[str, Any], expected_state_count: int) -> tuple:
    required = {
        "state_slot",
        "state",
        "kind",
        "branch_slot",
        "schema_version",
        "design_digest",
    }
    if not required.issubset(metadata):
        raise RuntimeError("checkpoint identity metadata is incomplete")
    try:
        state_slot = int(metadata["state_slot"])
        branch_slot = int(metadata["branch_slot"])
        state = tuple(int(value) for value in metadata["state"])
    except (TypeError, ValueError) as error:
        raise RuntimeError("checkpoint identity metadata is invalid") from error
    kind = str(metadata["kind"])
    if (
        metadata["state_slot"] != state_slot
        or metadata["branch_slot"] != branch_slot
        or not 0 <= state_slot < expected_state_count
        or (kind, branch_slot) not in BRANCHES
        or len(state) != 2
    ):
        raise RuntimeError("checkpoint identity is not registered")
    return state_slot, state, kind, branch_slot


def audit_ground_checkpoint_directory(
    directory: Path,
    *,
    reference: Path,
    expected_count: int = 72,
    allow_partial: bool = False,
) -> dict[str, Any]:
    """Validate one corner checkpoint set and classify partial versus complete."""

    if expected_count <= 0 or expected_count % len(BRANCHES):
        raise ValueError("expected checkpoint count must contain complete states")
    files = sorted(directory.glob("*.npz"))
    if (
        not files
        or len(files) > expected_count
        or (not allow_partial and len(files) != expected_count)
    ):
        raise RuntimeError("checkpoint count is incomplete or exceeds registration")
    fields, contract = _reference_contract(reference)
    expected_state_count = expected_count // len(BRANCHES)
    identities = []
    states: dict[int, tuple[int, int]] = {}
    schemas = set()
    design_digests = set()
    content = hashlib.sha256()
    for path in files:
        metadata = _validate_payload(path, fields, contract)
        identity = _identity(metadata, expected_state_count)
        expected_name = f"state-{identity[0]:02d}-{identity[2]}-{identity[3]:02d}.npz"
        if path.name != expected_name:
            raise RuntimeError("checkpoint filename does not match its identity")
        identities.append(identity)
        if identity[0] in states and states[identity[0]] != identity[1]:
            raise RuntimeError("checkpoint state slot maps to inconsistent states")
        states[identity[0]] = identity[1]
        schemas.add(str(metadata["schema_version"]))
        design_digests.add(str(metadata["design_digest"]))
        content.update(path.name.encode("utf-8"))
        content.update(bytes.fromhex(_sha256(path)))
    if len(set(identities)) != len(identities):
        raise RuntimeError("checkpoint identities must be unique")
    if schemas != {SCHEMA_VERSION}:
        raise RuntimeError("checkpoint schema is not registered")
    if len(design_digests) != 1 or any(
        not _valid_sha256(value) for value in design_digests
    ):
        raise RuntimeError("checkpoint design digest is not uniform SHA-256")
    expected_identities = {
        (state_slot, states.get(state_slot), kind, branch_slot)
        for state_slot in range(expected_state_count)
        for kind, branch_slot in BRANCHES
    }
    if len(files) == expected_count and set(identities) != expected_identities:
        raise RuntimeError("complete checkpoint set does not cover registration")
    return {
        "schema_version": "articulated-checkpoint-audit/v1",
        "status": "complete" if len(files) == expected_count else "partial",
        "checkpoint_count": len(files),
        "expected_checkpoint_count": expected_count,
        "unique_identity_count": len(set(identities)),
        "completed_state_slot_count": len(states),
        "checkpoint_schema_version": next(iter(schemas)),
        "design_digest": next(iter(design_digests)),
        "reference_sha256": _sha256(reference),
        "checkpoint_set_sha256": content.hexdigest(),
    }


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--reference", type=Path, required=True)
    parser.add_argument("--expected-count", type=int, default=72)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args(argv)
    report = audit_ground_checkpoint_directory(
        args.directory,
        reference=args.reference,
        expected_count=args.expected_count,
        allow_partial=args.allow_partial,
    )
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()


__all__ = ["audit_ground_checkpoint_directory", "main"]

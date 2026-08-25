"""Tests for fail-closed structural checkpoint persistence."""

from __future__ import annotations

from dataclasses import replace
import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from scripts.research.proximal_distal_energy.articulated_atlas_authority import (
    ArticulatedAtlasAuthority,
)
from scripts.research.proximal_distal_energy.articulated_ground_atlas import (
    ArticulatedGroundAtlasConfig,
)
from scripts.research.proximal_distal_energy.articulated_scaled_authority import (
    load_scaled_authority,
)
from scripts.research.proximal_distal_energy.articulated_structural_checkpoint import (
    METADATA_FIELD,
    audit_structural_checkpoint_directory,
    load_structural_checkpoint,
    restore_structural_checkpoint_directory,
    structural_checkpoint_array_contract,
    structural_checkpoint_path,
    write_structural_checkpoint,
)
from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    resolve_structural_execution_identity,
)

pytestmark = pytest.mark.scientific

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"


def _identity():
    scaled = load_scaled_authority(
        DATA / "articulated_structural_authority_nominal.json",
        DATA / "articulated_structural_authority_nominal.npz",
    )
    return resolve_structural_execution_identity(
        ArticulatedAtlasAuthority.from_scaled(scaled),
        corner_id="nominal",
        pathway="ground",
        configuration=ArticulatedGroundAtlasConfig(),
    )


def _arrays() -> dict[str, np.ndarray]:
    return {
        "finite": np.asarray([1.0, 2.0]),
        "structural_nan": np.asarray([np.nan, 3.0]),
        "count": np.asarray([1, 2], dtype=np.int64),
        "passed": np.asarray([True, False]),
    }


def _write(path: Path) -> tuple[object, dict]:
    identity = _identity()
    arrays = _arrays()
    contract = structural_checkpoint_array_contract(arrays)
    write_structural_checkpoint(
        path,
        identity,
        state_slot=0,
        state=(0, 0),
        branch_kind="primary",
        branch_slot=0,
        arrays=arrays,
        expected_contract=contract,
    )
    return identity, contract


def _repack(path: Path, mutation) -> None:
    with np.load(path, allow_pickle=False) as source:
        payload = {name: np.asarray(source[name]).copy() for name in source.files}
    mutation(payload)
    with path.open("wb") as stream:
        np.savez_compressed(stream, **payload)


def test_checkpoint_round_trip_is_atomic_exact_and_detached(tmp_path) -> None:
    path = tmp_path / "nested/checkpoint.npz"
    identity, contract = _write(path)

    loaded = load_structural_checkpoint(
        path,
        identity,
        state_slot=0,
        state=(0, 0),
        branch_kind="primary",
        branch_slot=0,
        expected_contract=contract,
    )

    assert set(loaded) == set(_arrays())
    assert np.array_equal(
        loaded["structural_nan"], _arrays()["structural_nan"], equal_nan=True
    )
    loaded["finite"][0] = 99.0
    assert (
        load_structural_checkpoint(
            path,
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            expected_contract=contract,
        )["finite"][0]
        == 1.0
    )
    assert not path.with_suffix(".npz.tmp").exists()


@pytest.mark.parametrize(
    "arrays",
    [
        {},
        {METADATA_FIELD: np.asarray([1.0])},
        {"object": np.asarray([object()], dtype=object)},
        {"text": np.asarray(["not a branch result"])},
        {"infinite": np.asarray([np.inf])},
    ],
)
def test_array_contract_rejects_unsafe_payloads(arrays) -> None:
    with pytest.raises(ValueError):
        structural_checkpoint_array_contract(arrays)


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype"])
def test_write_rejects_array_contract_drift(tmp_path, mutation: str) -> None:
    identity = _identity()
    arrays = _arrays()
    contract = structural_checkpoint_array_contract(arrays)
    if mutation == "missing":
        arrays.pop("count")
    elif mutation == "extra":
        arrays["extra"] = np.asarray([1])
    elif mutation == "shape":
        arrays["count"] = np.asarray([[1, 2]], dtype=np.int64)
    else:
        arrays["count"] = arrays["count"].astype(np.int32)

    with pytest.raises(RuntimeError, match="array contract"):
        write_structural_checkpoint(
            tmp_path / "checkpoint.npz",
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            arrays=arrays,
            expected_contract=contract,
        )


def test_atomic_failure_preserves_prior_checkpoint(tmp_path, monkeypatch) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, contract = _write(path)
    before = path.read_bytes()

    def fail(*args, **kwargs):
        raise OSError("injected write failure")

    monkeypatch.setattr(np, "savez_compressed", fail)
    with pytest.raises(OSError, match="injected"):
        write_structural_checkpoint(
            path,
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            arrays=_arrays(),
            expected_contract=contract,
        )
    assert path.read_bytes() == before
    assert not path.with_suffix(".npz.tmp").exists()


def test_load_rejects_tampered_metadata(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, contract = _write(path)

    def mutation(payload) -> None:
        metadata = json.loads(str(payload[METADATA_FIELD].item()))
        metadata["corner_id"] = "tampered"
        payload[METADATA_FIELD] = np.asarray(json.dumps(metadata))

    _repack(path, mutation)
    with pytest.raises(RuntimeError, match="corner_id"):
        load_structural_checkpoint(
            path,
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            expected_contract=contract,
        )


@pytest.mark.parametrize("mutation", ["missing", "extra", "shape", "dtype", "infinite"])
def test_load_rejects_payload_drift(tmp_path, mutation: str) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, contract = _write(path)

    def alter(payload) -> None:
        if mutation == "missing":
            payload.pop("count")
        elif mutation == "extra":
            payload["extra"] = np.asarray([1])
        elif mutation == "shape":
            payload["count"] = np.asarray([[1, 2]], dtype=np.int64)
        elif mutation == "dtype":
            payload["count"] = payload["count"].astype(np.int32)
        else:
            payload["finite"][0] = np.inf

    _repack(path, alter)
    with pytest.raises(RuntimeError):
        load_structural_checkpoint(
            path,
            identity,
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            expected_contract=contract,
        )


def test_load_rejects_wrong_execution_identity(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    identity, contract = _write(path)
    wrong = replace(identity, plan_contract_sha256="0" * 64)

    with pytest.raises(RuntimeError, match="plan_contract_sha256"):
        load_structural_checkpoint(
            path,
            wrong,
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            expected_contract=contract,
        )


def test_load_rejects_corrupt_archive(tmp_path) -> None:
    path = tmp_path / "checkpoint.npz"
    path.write_bytes(b"not an npz")

    with pytest.raises(RuntimeError, match="cannot be loaded safely"):
        load_structural_checkpoint(
            path,
            _identity(),
            state_slot=0,
            state=(0, 0),
            branch_kind="primary",
            branch_slot=0,
            expected_contract=structural_checkpoint_array_contract(_arrays()),
        )


def _branch_contracts(identity) -> dict:
    contract = structural_checkpoint_array_contract(_arrays())
    return dict.fromkeys(identity.registered_branches, contract)


def _write_registered_set(directory: Path, identity, count: int) -> None:
    contract = structural_checkpoint_array_contract(_arrays())
    identities = [
        (state_slot, state, branch_kind, branch_slot)
        for state_slot, state in enumerate(identity.registered_states)
        for branch_kind, branch_slot in identity.registered_branches
    ]
    for state_slot, state, branch_kind, branch_slot in identities[:count]:
        write_structural_checkpoint(
            structural_checkpoint_path(
                directory,
                identity,
                state_slot=state_slot,
                branch_kind=branch_kind,
                branch_slot=branch_slot,
            ),
            identity,
            state_slot=state_slot,
            state=state,
            branch_kind=branch_kind,
            branch_slot=branch_slot,
            arrays=_arrays(),
            expected_contract=contract,
        )


def test_checkpoint_set_audit_distinguishes_partial_from_complete(tmp_path) -> None:
    identity = _identity()
    contracts = _branch_contracts(identity)
    _write_registered_set(tmp_path, identity, 7)

    partial = audit_structural_checkpoint_directory(
        tmp_path,
        identity,
        expected_contracts=contracts,
        allow_partial=True,
    )
    assert partial == {
        "schema_version": "articulated-structural-checkpoint-audit/v1",
        "status": "partial",
        "checkpoint_count": 7,
        "expected_checkpoint_count": 72,
        "observed_state_slot_count": 2,
        "complete_state_slot_count": 1,
        "checkpoint_set_sha256": partial["checkpoint_set_sha256"],
        "release_evidence": False,
    }
    assert len(partial["checkpoint_set_sha256"]) == 64
    expected_digest = hashlib.sha256()
    for path in sorted(tmp_path.iterdir()):
        expected_digest.update(path.name.encode("utf-8"))
        expected_digest.update(hashlib.sha256(path.read_bytes()).digest())
    assert partial["checkpoint_set_sha256"] == expected_digest.hexdigest()
    with pytest.raises(RuntimeError, match="incomplete"):
        audit_structural_checkpoint_directory(
            tmp_path,
            identity,
            expected_contracts=contracts,
        )

    _write_registered_set(tmp_path, identity, 72)
    complete = audit_structural_checkpoint_directory(
        tmp_path,
        identity,
        expected_contracts=contracts,
    )
    assert complete["status"] == "complete"
    assert complete["checkpoint_count"] == 72
    assert complete["complete_state_slot_count"] == 12
    assert complete["release_evidence"] is False


def test_restart_inventory_returns_exact_restored_and_pending_work(tmp_path) -> None:
    identity = _identity()
    contracts = _branch_contracts(identity)

    empty = restore_structural_checkpoint_directory(
        tmp_path,
        identity,
        expected_contracts=contracts,
    )
    assert empty.audit["status"] == "empty"
    assert empty.restored == {}
    assert len(empty.pending) == 72
    assert empty.pending[0] == (0, (0, 0), "primary", 0)

    _write_registered_set(tmp_path, identity, 7)
    partial = restore_structural_checkpoint_directory(
        tmp_path,
        identity,
        expected_contracts=contracts,
    )
    assert partial.audit["status"] == "partial"
    assert len(partial.restored) == 7
    assert len(partial.pending) == 65
    assert partial.pending[0] == (1, (0, 6), "primary", 1)
    assert np.array_equal(
        partial.restored[(0, "primary", 0)]["structural_nan"],
        _arrays()["structural_nan"],
        equal_nan=True,
    )


def test_checkpoint_set_audit_rejects_unregistered_files(tmp_path) -> None:
    identity = _identity()
    _write_registered_set(tmp_path, identity, 1)
    (tmp_path / "interrupted.npz.tmp").write_bytes(b"torn")

    with pytest.raises(RuntimeError, match="unregistered files"):
        audit_structural_checkpoint_directory(
            tmp_path,
            identity,
            expected_contracts=_branch_contracts(identity),
            allow_partial=True,
        )


def test_checkpoint_set_audit_requires_every_branch_contract(tmp_path) -> None:
    identity = _identity()
    contracts = _branch_contracts(identity)
    contracts.pop(next(iter(contracts)))

    with pytest.raises(ValueError, match="every registered branch"):
        audit_structural_checkpoint_directory(
            tmp_path,
            identity,
            expected_contracts=contracts,
            allow_partial=True,
        )


def test_checkpoint_set_audit_validates_absent_branch_contracts(tmp_path) -> None:
    identity = _identity()
    _write_registered_set(tmp_path, identity, 1)
    contracts = _branch_contracts(identity)
    absent = identity.registered_branches[-1]
    contracts[absent] = {"unsafe": ((1,), "|O")}

    with pytest.raises(ValueError, match="dtype is not registered"):
        audit_structural_checkpoint_directory(
            tmp_path,
            identity,
            expected_contracts=contracts,
            allow_partial=True,
        )


def test_checkpoint_path_rejects_unregistered_identity(tmp_path) -> None:
    identity = _identity()

    with pytest.raises(ValueError, match="registered state"):
        structural_checkpoint_path(
            tmp_path,
            identity,
            state_slot=12,
            branch_kind="primary",
            branch_slot=0,
        )
    with pytest.raises(ValueError, match="registered branch"):
        structural_checkpoint_path(
            tmp_path,
            identity,
            state_slot=0,
            branch_kind="activation",
            branch_slot=0,
        )

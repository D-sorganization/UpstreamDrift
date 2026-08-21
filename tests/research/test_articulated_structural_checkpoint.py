"""Tests for fail-closed structural checkpoint persistence."""

from __future__ import annotations

from dataclasses import replace
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
    load_structural_checkpoint,
    structural_checkpoint_array_contract,
    write_structural_checkpoint,
)
from scripts.research.proximal_distal_energy.articulated_structural_execution_identity import (
    resolve_structural_execution_identity,
)

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

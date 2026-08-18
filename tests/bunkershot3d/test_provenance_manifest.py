"""Run manifest for BunkerShot3D results (issue #8617, finding B18).

A result file that cannot say *what produced it* is not an audit artifact. The
manifest records the config hash, every RNG seed, library versions, git SHA +
dirty flag, solver and fidelity tier, the validity verdict, wall clock and
host. It is written both as HDF5 root attributes and as a sibling JSON file.
"""

from __future__ import annotations

import json
from pathlib import Path

import h5py
import pytest

from bunkershot3d.provenance import (
    PROVENANCE_SUFFIX,
    RunManifest,
    Validity,
    root_seed_sequence,
    seed_record,
)

pytestmark = pytest.mark.unit


def _manifest() -> RunManifest:
    return RunManifest(
        config_hash="a" * 64,
        physics_hash="b" * 64,
        seeds=(
            seed_record(root_seed_sequence(entropy=11), "grains"),
            seed_record(root_seed_sequence(entropy=12), "noise"),
        ),
        solver="drft",
        fidelity_tier="F0",
        validity=Validity.OUT_OF_ENVELOPE,
        validity_reason="entry speed 31 m/s above calibrated 27 m/s",
        library_versions={"numpy": "2.2.6", "h5py": "3.11.0"},
        git_commit="0" * 40,
        git_branch="feat/bunker-schema",
        git_dirty=True,
        python_version="3.13.0",
        platform="Windows-11",
        hostname="desk",
        started_at_utc="2026-08-13T00:00:00Z",
        wall_clock_s=1.25,
    )


def test_manifest_round_trips_through_dict() -> None:
    manifest = _manifest()
    assert RunManifest.from_dict(manifest.to_dict()) == manifest


def test_manifest_dict_is_json_serialisable() -> None:
    payload = json.dumps(_manifest().to_dict())
    assert '"validity": "out_of_envelope"' in payload


def test_manifest_round_trips_through_hdf5_attrs(tmp_path: Path) -> None:
    path = tmp_path / "m.h5"
    manifest = _manifest()
    with h5py.File(path, "w") as handle:
        manifest.write_attrs(handle)
    with h5py.File(path, "r") as handle:
        restored = RunManifest.read_attrs(handle)
    assert restored == manifest


def test_read_attrs_returns_none_when_absent(tmp_path: Path) -> None:
    path = tmp_path / "empty.h5"
    with h5py.File(path, "w") as handle:
        handle.attrs["schema_version"] = 2
    with h5py.File(path, "r") as handle:
        assert RunManifest.read_attrs(handle) is None


def test_sidecar_json_names_the_artifact_and_checksums_it(tmp_path: Path) -> None:
    artifact = tmp_path / "run.h5"
    with h5py.File(artifact, "w") as handle:
        handle.attrs["schema_version"] = 2

    sidecar = _manifest().write_sidecar(artifact)

    assert sidecar == artifact.parent / f"{artifact.name}{PROVENANCE_SUFFIX}"
    payload = json.loads(sidecar.read_text(encoding="utf-8"))
    assert payload["artifact"]["path"] == artifact.name
    assert payload["artifact"]["format"] == "hdf5"
    assert payload["artifact"]["checksum_algorithm"] == "sha256"
    assert len(payload["artifact"]["checksum_sha256"]) == 64
    assert payload["run_manifest"]["config_hash"] == "a" * 64
    assert payload["run_manifest"]["seeds"][0]["name"] == "grains"


def test_manifest_requires_every_seed_to_be_recorded() -> None:
    with pytest.raises(ValueError, match="seeds"):
        RunManifest(
            config_hash="a" * 64,
            physics_hash="b" * 64,
            seeds=(),
            solver="drft",
            fidelity_tier="F0",
            validity=Validity.VALID,
        )


def test_manifest_rejects_unknown_validity_verdict() -> None:
    with pytest.raises(ValueError, match="validity"):
        RunManifest(
            config_hash="a" * 64,
            physics_hash="b" * 64,
            seeds=(seed_record(root_seed_sequence(entropy=1), "s"),),
            solver="drft",
            fidelity_tier="F0",
            validity="probably fine",  # type: ignore[arg-type]
        )


def test_capture_fills_environment_and_keeps_supplied_fields() -> None:
    manifest = RunManifest.capture(
        config_hash="c" * 64,
        physics_hash="d" * 64,
        seeds=(seed_record(root_seed_sequence(entropy=3), "grains"),),
        solver="drft",
        fidelity_tier="F0",
        validity=Validity.VALID,
        wall_clock_s=0.5,
    )
    assert manifest.config_hash == "c" * 64
    assert manifest.hostname
    assert manifest.python_version
    assert manifest.library_versions["numpy"]
    assert manifest.started_at_utc.endswith("Z")
    assert isinstance(manifest.git_dirty, bool)

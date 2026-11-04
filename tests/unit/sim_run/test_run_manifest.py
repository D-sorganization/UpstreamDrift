"""Tests for the run-manifest schema and validator (Issue #14431 / Epic #14436)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

jsonschema = pytest.importorskip("jsonschema")

from src.sim_run.manifest import (
    RunManifest,
    RunManifestError,
    export_run_manifest_schema,
    validate_run_manifest,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SCHEMA_PATH = _REPO_ROOT / "schemas" / "run-manifest-v1.json"
_FIXTURE_PATH = (
    _REPO_ROOT / "tests" / "fixtures" / "sim_run" / "run_manifest_valid_v1.json"
)

# Every provenance field the issue requires, as a path into the manifest.
_REQUIRED_PATHS = [
    ("schema_version",),
    ("model",),
    ("model", "model_id"),
    ("model", "revision"),
    ("parameters",),
    ("input_data",),
    ("input_data", "sha256"),
    ("seed",),
    ("engine",),
    ("engine", "name"),
    ("engine", "version"),
    ("ud_commit",),
    ("created_at",),
]


def _valid() -> dict[str, Any]:
    return json.loads(_FIXTURE_PATH.read_text(encoding="utf-8"))


def _without(path: tuple[str, ...]) -> dict[str, Any]:
    manifest = _valid()
    parent = manifest
    for key in path[:-1]:
        parent = parent[key]
    del parent[path[-1]]
    return manifest


def _with(path: tuple[str, ...], value: Any) -> dict[str, Any]:
    manifest = _valid()
    parent = manifest
    for key in path[:-1]:
        parent = parent[key]
    parent[path[-1]] = value
    return manifest


def _schema_validator() -> Any:
    schema = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    return jsonschema.Draft202012Validator(schema)


def test_checked_in_schema_equals_export_and_is_draft_2020_12() -> None:
    assert _SCHEMA_PATH.is_file(), f"Missing {_SCHEMA_PATH}"
    checked_in = json.loads(_SCHEMA_PATH.read_text(encoding="utf-8"))
    jsonschema.Draft202012Validator.check_schema(checked_in)
    assert checked_in == export_run_manifest_schema()


def test_valid_fixture_passes_validator_and_schema() -> None:
    manifest = validate_run_manifest(_valid())
    assert isinstance(manifest, RunManifest)
    assert manifest.schema_version == "1.0.0"
    assert manifest.seed == 20261104
    assert manifest.engine.version == "3.3.7"
    assert not list(_schema_validator().iter_errors(_valid()))


def test_validator_does_not_mutate_its_input() -> None:
    data = _valid()
    snapshot = copy.deepcopy(data)
    validate_run_manifest(data)
    assert data == snapshot


@pytest.mark.parametrize("path", _REQUIRED_PATHS, ids=".".join)
def test_missing_provenance_field_fails_with_its_name(path: tuple[str, ...]) -> None:
    with pytest.raises(RunManifestError, match=path[-1]):
        validate_run_manifest(_without(path))
    assert list(_schema_validator().iter_errors(_without(path)))


@pytest.mark.parametrize(
    ("path", "ref"),
    [
        (("model", "revision"), "main"),
        (("model", "revision"), "feature/hand-fix"),
        (("model", "revision"), "v1.2.0"),
        (("model", "revision"), "468442b"),
        (("ud_commit",), "main"),
        (("ud_commit",), "HEAD"),
        (("ud_commit",), "468442b4b9"),
    ],
)
def test_mutable_ref_fails_with_clear_error(path: tuple[str, ...], ref: str) -> None:
    with pytest.raises(RunManifestError, match=r"mutable ref") as excinfo:
        validate_run_manifest(_with(path, ref))
    assert ref in str(excinfo.value)
    assert list(_schema_validator().iter_errors(_with(path, ref)))


@pytest.mark.parametrize(
    "revision",
    [
        "468442b4b9a6735a74214ae04ca7d470ab21a7db",
        "sha256:" + "ab" * 32,
    ],
)
def test_revision_accepts_git_sha_or_file_hash(revision: str) -> None:
    manifest = validate_run_manifest(_with(("model", "revision"), revision))
    assert manifest.model.revision == revision


@pytest.mark.parametrize("digest", ["", "abc", "z" * 64, "AB" * 31])
def test_input_data_must_be_a_sha256_digest(digest: str) -> None:
    with pytest.raises(RunManifestError, match="sha256"):
        validate_run_manifest(_with(("input_data", "sha256"), digest))


def test_raw_input_data_is_rejected() -> None:
    manifest = _with(("input_data", "samples"), [[0.0, 1.0, 2.0]])
    with pytest.raises(RunManifestError, match="samples"):
        validate_run_manifest(manifest)
    assert list(_schema_validator().iter_errors(manifest))


@pytest.mark.parametrize(
    ("path", "value"),
    [
        (("seed",), -1),
        (("seed",), 1.5),
        (("engine", "name"), ""),
        (("engine", "version"), " "),
        (("model", "model_id"), ""),
        (("schema_version",), "2.0.0"),
        (("created_at",), "2026-11-04T01:30:00"),
        (("created_at",), "yesterday"),
    ],
)
def test_invalid_values_fail(path: tuple[str, ...], value: Any) -> None:
    with pytest.raises(RunManifestError, match=path[-1]):
        validate_run_manifest(_with(path, value))


def test_non_mapping_input_is_rejected() -> None:
    with pytest.raises(TypeError, match="mapping"):
        validate_run_manifest(["not", "a", "manifest"])  # type: ignore[arg-type]

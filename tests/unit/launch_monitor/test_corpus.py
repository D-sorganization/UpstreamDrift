"""Tests for the private-corpus loader over synthetic Parquet fixtures.

Covers the ADR-0031 canonicalisation the loader has always done and the
ADR-0048 D30 manifest gate it gained from Tools#4907's P19 canonical merge.
Every refusal is asserted **by name** - which of the five checks fired - so a
check that silently stops firing cannot pass as a different one.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

pytest.importorskip("pyarrow")

from src.tools.launch_monitor_model.corpus import (
    MANIFEST_FILENAME,
    MAX_RETAINED_ROWS,
    SUPPORTED_MANIFEST_SCHEMA_VERSION,
    CorpusManifest,
    corpus_dataset_path,
    load_private_corpus,
    read_corpus_manifest,
    validate_corpus_manifest,
)


pytestmark = pytest.mark.unit


def _synthetic_checkout(tmp_path: Path) -> Path:
    checkout = tmp_path / "checkout"
    dataset = checkout / "data" / "authority" / "database" / "shot_corpus_parquet"
    rows = pd.DataFrame(
        {
            "monitor": ["TrackMan", "FlightScope Mevo+"],
            "file": ["a.csv", "b.csv"],
            "row_index": [0, 0],
            "club": ["Driver", "7 Iron"],
            "club_speed_mph": [100.0, 80.0],
            "ball_speed_mph": [150.0, 110.0],
            "smash_factor": [1.5, 1.375],
            "launch_angle_deg": [12.0, 18.0],
            "launch_direction_deg": [1.0, -0.5],
            "spin_rate_rpm": [2700.0, 6500.0],
            "back_spin_rpm": [2600.0, 6400.0],
            "side_spin_rpm": [300.0, -200.0],
            "spin_axis_deg": [4.0, -2.0],
            "attack_angle_deg": [-1.2, -4.0],
            "club_path_deg": [0.5, 1.5],
            "face_angle_deg": [0.2, 0.8],
            "carry_yd": [250.0, 165.0],
            "total_yd": [270.0, 172.0],
            "apex_native": [95.0, 28.0],
            "descent_angle_deg": [38.0, 45.0],
            "native_json": ["{}", "{}"],
        }
    )
    for source_id, group in (
        ("synthetic_trackman", rows.iloc[:1]),
        ("synthetic_mevo", rows.iloc[1:]),
    ):
        partition = dataset / f"source_id={source_id}"
        partition.mkdir(parents=True)
        group.to_parquet(partition / "part-0.parquet", index=False)
    _write_manifest(dataset)
    return checkout


def _write_manifest(dataset: Path, manifest: dict[str, Any] | None = None) -> Path:
    """Write the authority manifest the D30 gate validates against."""
    if manifest is None:
        sources = sorted(
            entry.name.split("=", 1)[1]
            for entry in dataset.iterdir()
            if entry.is_dir() and entry.name.startswith("source_id=")
        )
        manifest = {
            "schema_version": SUPPORTED_MANIFEST_SCHEMA_VERSION,
            "sources": dict.fromkeys(sources, {}),
            "total_rows": sum(
                len(pd.read_parquet(path))
                for path in sorted(dataset.rglob("*.parquet"))
            ),
        }
    path = dataset / MANIFEST_FILENAME
    path.write_bytes(
        json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode("utf-8")
    )
    return path


def test_load_private_corpus_converts_to_canonical_units(tmp_path: Path) -> None:
    frame = load_private_corpus(root=_synthetic_checkout(tmp_path))

    assert len(frame) == 2
    row = frame.set_index("session_id").loc["synthetic_trackman"]
    assert row["ball_speed"] == pytest.approx(150.0 * 0.44704)
    assert row["launch_angle"] == pytest.approx(math.radians(12.0))
    assert row["spin_rate"] == pytest.approx(2700.0 * math.pi / 30.0)
    assert row["carry_distance"] == pytest.approx(250.0 * 0.9144)
    assert row["monitor_vendor"] == "TrackMan"
    assert row["observation_kind"] == "shot"
    assert "apex_native" not in frame.columns
    assert frame["shot_id"].nunique() == 2


def test_source_and_metric_selection(tmp_path: Path) -> None:
    checkout = _synthetic_checkout(tmp_path)
    frame = load_private_corpus(
        root=checkout,
        sources=["synthetic_mevo"],
        metrics=["ball_speed", "carry_distance"],
    )
    assert set(frame["session_id"].astype(str)) == {"synthetic_mevo"}
    assert "ball_speed" in frame.columns
    assert "spin_rate" not in frame.columns
    with pytest.raises(ValueError, match="Unknown corpus sources"):
        load_private_corpus(root=checkout, sources=["nope"])
    with pytest.raises(ValueError, match="Unknown corpus metrics"):
        load_private_corpus(root=checkout, metrics=["warp_speed"])


def test_missing_root_and_missing_dataset_fail_closed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("LAUNCH_MONITOR_DATA_ROOT", raising=False)
    with pytest.raises(FileNotFoundError, match="LAUNCH_MONITOR_DATA_ROOT"):
        corpus_dataset_path()
    with pytest.raises(FileNotFoundError, match="shot corpus dataset not found"):
        load_private_corpus(root=tmp_path / "empty")


def test_lateral_flight_and_capture_columns_reach_canonical_schema(
    tmp_path: Path,
) -> None:
    """The #18/#19 corpus columns convert into the canonical schema."""
    checkout = tmp_path / "checkout"
    dataset = checkout / "data" / "authority" / "database" / "shot_corpus_parquet"
    partition = dataset / "source_id=synthetic_new"
    partition.mkdir(parents=True)
    rows = pd.DataFrame(
        {
            "monitor": ["TrackMan"],
            "file": ["a.csv"],
            "row_index": [0],
            "club": ["Driver"],
            "club_speed_mph": [100.0],
            "ball_speed_mph": [150.0],
            "smash_factor": [1.5],
            "launch_angle_deg": [12.0],
            "launch_direction_deg": [1.0],
            "spin_rate_rpm": [2700.0],
            "back_spin_rpm": [2600.0],
            "side_spin_rpm": [300.0],
            "spin_axis_deg": [4.0],
            "attack_angle_deg": [-1.2],
            "club_path_deg": [0.5],
            "face_angle_deg": [0.2],
            "carry_yd": [250.0],
            "total_yd": [270.0],
            "apex_native": [95.0],
            "descent_angle_deg": [38.0],
            "lateral_carry_yd": [-12.5],
            "flight_time_s": [6.2],
            "captured_at": ["2023-08-07T00:00:00"],
            "native_json": ["{}"],
        }
    )
    rows.to_parquet(partition / "part-0.parquet", index=False)
    _write_manifest(dataset)

    frame = load_private_corpus(root=checkout)

    row = frame.iloc[0]
    assert row["lateral_carry"] == pytest.approx(-12.5 * 0.9144)  # yards -> m
    assert row["flight_time"] == pytest.approx(6.2)
    assert row["captured_at"] == "2023-08-07T00:00:00"


def test_corpus_predating_the_new_columns_still_loads(tmp_path: Path) -> None:
    """An older pinned corpus lacks the columns; the loader must not fail."""
    frame = load_private_corpus(root=_synthetic_checkout(tmp_path))

    assert len(frame) == 2
    assert "lateral_carry" not in frame.columns
    assert "captured_at" not in frame.columns


# --- ADR-0048 D30: the fail-closed manifest gate ----------------------------
#
# Ported from the canonical Tools module
# ``src/shared/python/launch_monitor/corpus.py`` (Tools#4907, P19), which
# merged this repo's canonicalisation with ``rate_of_closure``'s validation.
# Each case names the check it exercises; the loader must refuse, and refuse
# for that reason.


def test_gate_refuses_a_corpus_with_no_manifest(tmp_path: Path) -> None:
    """missing_manifest: bytes with no published description are not a corpus."""
    checkout = _synthetic_checkout(tmp_path)
    dataset = corpus_dataset_path(checkout)
    (dataset / MANIFEST_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_private_corpus(root=checkout)


def test_gate_refuses_an_unsupported_manifest_schema(tmp_path: Path) -> None:
    """unsupported_schema: a future schema is refused, not best-effort read."""
    checkout = _synthetic_checkout(tmp_path)
    dataset = corpus_dataset_path(checkout)
    _write_manifest(
        dataset,
        {
            "schema_version": SUPPORTED_MANIFEST_SCHEMA_VERSION + 1,
            "sources": {},
            "total_rows": 2,
        },
    )

    with pytest.raises(ValueError, match="manifest schema is unsupported"):
        load_private_corpus(root=checkout)


def test_gate_refuses_a_manifest_whose_sources_are_not_an_object(
    tmp_path: Path,
) -> None:
    """unsupported_schema: ``sources`` must be a mapping, not a list."""
    checkout = _synthetic_checkout(tmp_path)
    _write_manifest(
        corpus_dataset_path(checkout),
        {"schema_version": 1, "sources": ["synthetic_mevo"], "total_rows": 2},
    )

    with pytest.raises(ValueError, match="manifest schema is unsupported"):
        load_private_corpus(root=checkout)


@pytest.mark.parametrize("declared", [-1, MAX_RETAINED_ROWS + 1])
def test_gate_refuses_a_row_count_outside_the_desktop_cap(
    tmp_path: Path, declared: int
) -> None:
    """row_cap_exceeded: the retained-data limit is a refusal, not a warning."""
    checkout = _synthetic_checkout(tmp_path)
    _write_manifest(
        corpus_dataset_path(checkout),
        {
            "schema_version": 1,
            "sources": dict.fromkeys(("synthetic_trackman", "synthetic_mevo"), {}),
            "total_rows": declared,
        },
    )

    with pytest.raises(ValueError, match="outside the desktop retained-"):
        load_private_corpus(root=checkout)


def test_gate_refuses_a_row_count_that_disagrees_with_the_corpus(
    tmp_path: Path,
) -> None:
    """row_count_mismatch: the manifest must describe the rows on disk."""
    checkout = _synthetic_checkout(tmp_path)
    _write_manifest(
        corpus_dataset_path(checkout),
        {
            "schema_version": 1,
            "sources": dict.fromkeys(("synthetic_trackman", "synthetic_mevo"), {}),
            "total_rows": 99,
        },
    )

    with pytest.raises(ValueError, match=r"row count mismatch: expected 99, loaded 2"):
        load_private_corpus(root=checkout)


def test_gate_refuses_a_source_set_that_disagrees_with_the_partitions(
    tmp_path: Path,
) -> None:
    """source_set_mismatch: the partition tree must be the declared one."""
    checkout = _synthetic_checkout(tmp_path)
    _write_manifest(
        corpus_dataset_path(checkout),
        {"schema_version": 1, "sources": {"someone_elses_corpus": {}}, "total_rows": 2},
    )

    with pytest.raises(ValueError, match="source IDs do not match the manifest"):
        load_private_corpus(root=checkout)


def test_gate_basis_is_the_whole_corpus_not_the_caller_selection(
    tmp_path: Path,
) -> None:
    """A selection cannot buy its way past the gate.

    The manifest below describes exactly the ``synthetic_mevo`` slice - one row,
    one source - which is precisely what the selection asks for. The gate still
    refuses, because it counts the unfiltered dataset and reads the partition
    directory names, not the loaded frame.
    """
    checkout = _synthetic_checkout(tmp_path)
    _write_manifest(
        corpus_dataset_path(checkout),
        {"schema_version": 1, "sources": {"synthetic_mevo": {}}, "total_rows": 1},
    )

    with pytest.raises(ValueError, match="row count mismatch: expected 1, loaded 2"):
        load_private_corpus(
            root=checkout, sources=["synthetic_mevo"], metrics=["carry_distance"]
        )


def test_gate_runs_before_an_unknown_selection_is_reported(tmp_path: Path) -> None:
    """The gate is first: an invalid corpus is refused before a bad request."""
    checkout = _synthetic_checkout(tmp_path)
    (corpus_dataset_path(checkout) / MANIFEST_FILENAME).unlink()

    with pytest.raises(FileNotFoundError, match="manifest not found"):
        load_private_corpus(root=checkout, metrics=["warp_speed"])


def test_read_corpus_manifest_content_addresses_the_exact_bytes(
    tmp_path: Path,
) -> None:
    """The digest is over the manifest bytes; the corpus rows are never hashed."""
    import hashlib

    checkout = _synthetic_checkout(tmp_path)
    manifest_path = corpus_dataset_path(checkout) / MANIFEST_FILENAME

    manifest = read_corpus_manifest(corpus_dataset_path(checkout))

    assert isinstance(manifest, CorpusManifest)
    assert manifest.schema_version == SUPPORTED_MANIFEST_SCHEMA_VERSION
    assert manifest.total_rows == 2
    assert manifest.source_count == 2
    assert (
        manifest.manifest_sha256
        == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    )
    assert len(manifest.manifest_sha256) == 64


def test_validate_corpus_manifest_accepts_a_corpus_its_manifest_describes() -> None:
    """The happy path is silent: a matching corpus raises nothing."""
    manifest = CorpusManifest(
        schema_version=1,
        sources={"a": {}, "b": {}},
        total_rows=4,
        manifest_sha256="0" * 64,
    )

    assert (
        validate_corpus_manifest(manifest, observed_rows=4, observed_sources={"a", "b"})
        is None
    )


def test_desktop_row_cap_matches_the_authority_limit() -> None:
    """``MAX_RETAINED_ROWS`` is redefined here, not imported; pin its value.

    The cross-stack seam - equality with ``rate_of_closure``'s constant - is
    pinned by ``tests/integration/launch_monitor_drift/test_corpus_drift.py``,
    which is the only place allowed to import the vendored package.
    """
    assert MAX_RETAINED_ROWS == 300_000

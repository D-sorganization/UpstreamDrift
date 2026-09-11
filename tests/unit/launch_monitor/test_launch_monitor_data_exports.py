"""Contract tests for importing public Launch-Monitor-Data canonical exports.

The public ``D-sorganization/Launch-Monitor-Data`` repository publishes two
canonical shapes: shot-level frames from ``load_shots()`` (SI or native units)
and the long-format ``upstreamdrift_aggregate_metrics.csv`` of published
group means. Neither is a vendor export, so the header-fingerprint profiles
mis-map them; these tests pin the dedicated importer instead (#8365).
"""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.tools.launch_monitor_model import (
    AGGREGATE_EXPORT_PROFILE_ID,
    SHOT_EXPORT_PROFILE_ID,
    FlexibleAnalysisRequest,
    LaunchMonitorProject,
    analyze_variables,
    detect_launch_monitor_data_export,
    import_launch_monitor_data_export,
)

pytestmark = [pytest.mark.unit, pytest.mark.headless_safe]

AGGREGATE_COLUMNS = (
    "observation_id",
    "source_id",
    "monitor_vendor",
    "monitor_model",
    "software_version",
    "environment",
    "cohort",
    "club",
    "metric",
    "aggregation_level",
    "observation_kind",
    "sample_count",
    "measurement_status",
    "reported_mean",
    "reported_sd",
    "reported_unit",
    "canonical_mean",
    "canonical_sd",
    "canonical_unit",
    "matched_shots",
)


def _shot_export(count: int = 24, *, canonical_units: bool = True) -> pd.DataFrame:
    rng = np.random.default_rng(8365)
    club_speed = np.linspace(40.0, 50.0, count)
    ball_speed = 1.47 * club_speed + rng.normal(0.0, 0.3, count)
    frame = pd.DataFrame(
        {
            "source_id": np.where(np.arange(count) % 2, "src_b", "src_a"),
            "monitor": np.where(np.arange(count) % 2, "FlightScope", "TrackMan"),
            "club": "Driver",
            "file": "shots.csv",
            "row_index": np.arange(count),
            "captured_at": "2025-01-01T00:00:00+00:00",
            "observation_kind": "shot",
            "apex_native": rng.normal(30.0, 2.0, count),
        }
    )
    if canonical_units:
        frame["club_speed"] = club_speed
        frame["ball_speed"] = ball_speed
        frame["lateral_carry"] = rng.normal(0.0, 5.0, count)
    else:
        frame["club_speed_mph"] = club_speed / 0.44704
        frame["ball_speed_mph"] = ball_speed / 0.44704
        frame["lateral_carry_yd"] = rng.normal(0.0, 5.0, count) / 0.9144
    return frame


def _aggregate_rows() -> list[dict[str, object]]:
    def row(source: str, club: str, metric: str, index: int) -> dict[str, object]:
        canonical_unit = {
            "club_speed": "m/s",
            "ball_speed": "m/s",
            "carry_distance": "m",
        }[metric]
        return {
            "observation_id": f"obs-{index}",
            "source_id": source,
            "monitor_vendor": "TrackMan",
            "monitor_model": "TrackMan 4",
            "software_version": "",
            "environment": "outdoor",
            "cohort": "PGA Tour",
            "club": club,
            "metric": metric,
            "aggregation_level": "group_mean",
            "observation_kind": "aggregate",
            "sample_count": 100 + index,
            "measurement_status": "reported",
            "reported_mean": 100.0 + index,
            "reported_sd": 3.0,
            "reported_unit": "mph" if canonical_unit == "m/s" else "yd",
            "canonical_mean": 40.0 + index,
            "canonical_sd": 1.3,
            "canonical_unit": canonical_unit,
            "matched_shots": "",
        }

    rows: list[dict[str, object]] = []
    index = 0
    for source, club in (("pga", "Driver"), ("pga", "7 Iron"), ("lpga", "Driver")):
        for metric in ("club_speed", "ball_speed", "carry_distance"):
            rows.append(row(source, club, metric, index))
            index += 1
    return rows


def _write_aggregate(path: Path, rows: list[dict[str, object]]) -> Path:
    pd.DataFrame(rows, columns=list(AGGREGATE_COLUMNS)).to_csv(path, index=False)
    return path


def test_detects_both_public_export_shapes() -> None:
    assert detect_launch_monitor_data_export(list(_shot_export().columns)) == "shots"
    assert (
        detect_launch_monitor_data_export(
            list(_shot_export(canonical_units=False).columns)
        )
        == "shots"
    )
    assert detect_launch_monitor_data_export(list(AGGREGATE_COLUMNS)) == "aggregates"
    assert detect_launch_monitor_data_export(["Shot", "Club Speed (mph)"]) is None
    with pytest.raises(ValueError, match="at least one column"):
        detect_launch_monitor_data_export([])


def test_canonical_shot_export_keeps_si_units_and_corpus_identity(
    tmp_path: Path,
) -> None:
    export = _shot_export()
    path = tmp_path / "shots_si.csv"
    export.to_csv(path, index=False)

    session = import_launch_monitor_data_export(path)
    shots = session.shots

    assert session.manifest.profile_id == SHOT_EXPORT_PROFILE_ID
    assert session.manifest.vendor == "Launch-Monitor-Data"
    assert session.manifest.file_sha256 == hashlib.sha256(path.read_bytes()).hexdigest()
    # SI columns are already canonical: no conversion may be applied.
    assert shots["club_speed"].to_numpy() == pytest.approx(
        export["club_speed"].to_numpy()
    )
    assert session.manifest.source_units["club_speed"] == "m/s"
    assert session.manifest.unit_evidence["club_speed"] == "mapping"
    assert "lateral_carry" in shots.columns
    # Identity mirrors the private corpus loader so the same shot gets the
    # same id whichever public surface it arrives through.
    expected_shot_id = hashlib.sha256(b"src_a\x1fshots.csv\x1f0").hexdigest()[:20]
    assert shots.loc[0, "shot_id"] == expected_shot_id
    assert shots.loc[0, "session_id"] == "src_a"
    assert shots.loc[1, "session_id"] == "src_b"
    assert shots.loc[0, "monitor_vendor"] == "TrackMan"
    assert shots.loc[1, "monitor_vendor"] == "FlightScope"
    assert set(shots["observation_kind"]) == {"shot"}
    assert shots.loc[0, "source_row"] == 2
    # Every source column is retained verbatim, including the pass-through
    # apex whose unit varies by source and therefore is never converted.
    assert "source::apex_native" in shots.columns
    assert "apex_height" not in shots.columns
    assert set(shots["status::club_speed"]) == {"reported"}
    assert session.metadata["export_kind"] == "shots"


def test_native_unit_shot_export_converts_with_corpus_units(tmp_path: Path) -> None:
    export = _shot_export(canonical_units=False)
    path = tmp_path / "shots_native.csv"
    export.to_csv(path, index=False)

    session = import_launch_monitor_data_export(path)

    assert session.shots["club_speed"].to_numpy() == pytest.approx(
        export["club_speed_mph"].to_numpy() * 0.44704
    )
    assert session.shots["lateral_carry"].to_numpy() == pytest.approx(
        export["lateral_carry_yd"].to_numpy() * 0.9144
    )
    assert session.manifest.metric_sources["club_speed"] == "club_speed_mph"
    assert session.manifest.source_units["club_speed"] == "mph"
    assert "source::club_speed_mph" in session.shots.columns


def test_shot_export_rejects_mixed_unit_columns(tmp_path: Path) -> None:
    export = _shot_export()
    export["club_speed_mph"] = export["club_speed"] / 0.44704
    path = tmp_path / "mixed.csv"
    export.to_csv(path, index=False)

    with pytest.raises(ValueError, match="both native and canonical"):
        import_launch_monitor_data_export(path)


def test_shot_export_rejects_unexpected_metric_column(tmp_path: Path) -> None:
    export = _shot_export()
    export["Height (ft)"] = 90.0
    path = tmp_path / "extra.csv"
    export.to_csv(path, index=False)

    with pytest.raises(ValueError, match="assumed unit"):
        import_launch_monitor_data_export(path)


def test_aggregate_export_pivots_to_one_observation_per_group(tmp_path: Path) -> None:
    path = _write_aggregate(
        tmp_path / "upstreamdrift_aggregate_metrics.csv", _aggregate_rows()
    )

    session = import_launch_monitor_data_export(path)
    shots = session.shots

    assert session.manifest.profile_id == AGGREGATE_EXPORT_PROFILE_ID
    assert session.manifest.row_count == 3
    assert session.metadata["export_kind"] == "aggregates"
    assert list(shots["session_id"]) == ["lpga", "pga", "pga"]
    assert list(shots["club"]) == ["Driver", "7 Iron", "Driver"]
    assert set(shots["observation_kind"]) == {"aggregate"}
    assert set(shots["monitor_vendor"]) == {"TrackMan"}
    assert set(shots["monitor_model"]) == {"TrackMan 4"}
    pga_driver = shots[(shots["session_id"] == "pga") & (shots["club"] == "Driver")]
    assert pga_driver["club_speed"].item() == pytest.approx(40.0)
    assert pga_driver["carry_distance"].item() == pytest.approx(42.0)
    # Each published row's cells are retained verbatim under its metric.
    assert pga_driver["source::club_speed::observation_id"].item() == "obs-0"
    assert pga_driver["source::club_speed::reported_mean"].item() == pytest.approx(
        100.0
    )
    assert pga_driver["source::club_speed::reported_unit"].item() == "mph"
    assert pga_driver["source::club_speed::sample_count"].item() == 100
    assert pga_driver["source::carry_distance::sample_count"].item() == 102
    assert pga_driver["status::club_speed"].item() == "reported"
    assert pga_driver["source::cohort"].item() == "PGA Tour"
    assert session.manifest.metric_sources["club_speed"] == "canonical_mean"
    assert session.manifest.source_units["club_speed"] == "m/s"
    assert session.manifest.unit_evidence["club_speed"] == "canonical_unit"
    assert len(set(shots["shot_id"])) == 3


def test_aggregate_export_import_is_deterministic(tmp_path: Path) -> None:
    rows = _aggregate_rows()
    first = import_launch_monitor_data_export(
        _write_aggregate(tmp_path / "a.csv", rows)
    )
    second = import_launch_monitor_data_export(
        _write_aggregate(tmp_path / "b.csv", list(reversed(rows)))
    )

    pd.testing.assert_frame_equal(
        first.shots.drop(columns=["source_row"]),
        second.shots.drop(columns=["source_row"]),
    )
    assert list(first.shots["shot_id"]) == list(second.shots["shot_id"])


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ({"observation_kind": "shot"}, "must not be expanded"),
        ({"aggregation_level": "per_shot"}, "aggregation_level"),
        ({"canonical_unit": "mph"}, "unit"),
        ({"metric": "launch_monitor_score"}, "Unknown canonical metric"),
        ({"canonical_mean": "n/a"}, "non-numeric"),
    ],
)
def test_aggregate_export_contract_rejects_invalid_rows(
    tmp_path: Path, mutation: dict[str, object], message: str
) -> None:
    rows = _aggregate_rows()
    rows[0] = {**rows[0], **mutation}

    with pytest.raises(ValueError, match=message):
        import_launch_monitor_data_export(_write_aggregate(tmp_path / "bad.csv", rows))


def test_aggregate_export_rejects_duplicate_metric_in_group(tmp_path: Path) -> None:
    rows = _aggregate_rows()
    rows.append({**rows[0], "observation_id": "obs-dup"})

    with pytest.raises(ValueError, match="more than once"):
        import_launch_monitor_data_export(_write_aggregate(tmp_path / "dup.csv", rows))


def test_aggregate_export_rejects_missing_columns(tmp_path: Path) -> None:
    rows = [
        {key: value for key, value in row.items() if key != "canonical_unit"}
        for row in _aggregate_rows()
    ]
    path = tmp_path / "short.csv"
    pd.DataFrame(rows).to_csv(path, index=False)

    with pytest.raises(ValueError, match="not a Launch-Monitor-Data"):
        import_launch_monitor_data_export(path)


def test_unrecognised_export_fails_closed(tmp_path: Path) -> None:
    path = tmp_path / "vendor.csv"
    pd.DataFrame({"Shot": [1], "Club Speed (mph)": [90.0]}).to_csv(path, index=False)

    with pytest.raises(ValueError, match="not a Launch-Monitor-Data"):
        import_launch_monitor_data_export(path)


def test_imported_aggregates_never_enter_shot_level_fits(tmp_path: Path) -> None:
    aggregates = import_launch_monitor_data_export(
        _write_aggregate(tmp_path / "agg.csv", _aggregate_rows())
    )
    shots_path = tmp_path / "shots.csv"
    _shot_export().to_csv(shots_path, index=False)
    shots = import_launch_monitor_data_export(shots_path)

    project = LaunchMonitorProject("public")
    project.add_session(shots)
    project.add_session(aggregates)
    pooled = project.combined_shots()
    request = FlexibleAnalysisRequest(
        outcome="ball_speed",
        predictors=("club_speed",),
        analysis_mode="regression",
        min_samples=5,
    )

    shot_only = analyze_variables(shots.shots, request)
    assert shot_only.regression is not None
    assert shot_only.dataset.observation_kinds == ("shot",)
    with pytest.raises(ValueError, match="Aggregate observations"):
        analyze_variables(pooled, request)
    with pytest.raises(ValueError, match="Aggregate observations cannot enter"):
        analyze_variables(aggregates.shots, request)

    descriptive = analyze_variables(
        aggregates.shots,
        FlexibleAnalysisRequest(
            outcome="ball_speed",
            predictors=("club_speed",),
            analysis_mode="correlation",
            allow_aggregate=True,
            min_samples=3,
        ),
    )
    assert any("ecological" in warning.lower() for warning in descriptive.warnings)


def test_project_round_trip_preserves_export_lineage(tmp_path: Path) -> None:
    session = import_launch_monitor_data_export(
        _write_aggregate(tmp_path / "agg.csv", _aggregate_rows())
    )
    project = LaunchMonitorProject("public")
    project.add_session(session)
    saved = project.save(tmp_path / "public.lmproject")

    loaded = LaunchMonitorProject.load(saved)

    restored = loaded.sessions[0]
    assert restored.manifest == session.manifest
    assert restored.metadata["export_kind"] == "aggregates"
    assert set(restored.shots["observation_kind"]) == {"aggregate"}
    assert list(restored.shots["source::club_speed::observation_id"]) == list(
        session.shots["source::club_speed::observation_id"]
    )

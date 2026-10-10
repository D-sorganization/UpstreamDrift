"""Frozen corrected-club experimental matrix contracts."""

import pytest

from src.tools.shot_pattern_analysis.matrix import (
    corrected_matrix,
    matrix_shard,
    source_snapshot,
)

pytestmark = pytest.mark.unit


def test_corrected_matrix_has_24_paired_10k_scenarios() -> None:
    scenarios = corrected_matrix()
    assert len(scenarios) == 24
    assert len({name for name, _ in scenarios}) == 24
    assert all(config.n_shots == 10_000 for _, config in scenarios)
    assert {config.seed for _, config in scenarios} == {20_261_008}
    assert {config.club_id for _, config in scenarios} == {
        "driver",
        "seven_iron",
        "pitching_wedge",
    }
    assert {config.delivery_mode for _, config in scenarios} == {
        "fixed_loft",
        "shaft_rotation",
    }
    assert {config.face_sd_deg for _, config in scenarios} == {1.0, 2.0}
    assert {config.curve_scale for _, config in scenarios} == {1.0, 2.0}
    assert scenarios[0][1].club_id == "driver"
    assert scenarios[1][1].club_id == "seven_iron"
    assert scenarios[2][1].club_id == "pitching_wedge"


def test_four_shards_partition_cells_without_overlap() -> None:
    cells = corrected_matrix()
    shards = [matrix_shard(cells, index=i, count=4) for i in range(4)]
    assert sorted(name for shard in shards for name, _ in shard) == sorted(
        name for name, _ in cells
    )
    assert all(len(shard) == 6 for shard in shards)


def test_execution_snapshot_covers_scientific_sources_and_native_binary() -> None:
    snapshot = source_snapshot()
    sources = snapshot["source_sha256"]
    assert "src/shared/python/physics/impact_model/models.py" in sources
    assert "src/tools/shot_pattern_analysis/core.py" in sources
    assert "src/tools/shot_pattern_analysis/scoring_cache.py" in sources
    assert "rust_core/upstream-physics/src/ball_flight.rs" in sources
    assert snapshot["cargo_lock_present"] == ("Cargo.lock" in sources)
    assert len(snapshot["native_binary_sha256"]) == 64

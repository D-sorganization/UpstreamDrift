"""Tests for the split ORG-01 capability baseline layout (issue #12848)."""

from __future__ import annotations

import pytest

from scripts import generate_capability_baseline as gen
from src.config.capability_migration import load_migration_inventory

DOC_BUDGET_BYTES = 51_200


@pytest.fixture(scope="module")
def inventory():
    return load_migration_inventory(gen.MIGRATION_JSON_PATH)


@pytest.mark.unit
def test_every_entry_appears_in_exactly_one_part(inventory) -> None:
    parts = gen.plan_inventory_parts(inventory)
    seen = [e.id for _, _, entries in parts for e in entries]
    assert sorted(seen) == sorted(inventory.entries)
    assert len(seen) == len(set(seen))


@pytest.mark.unit
def test_parts_respect_row_limit_and_are_single_workspace(inventory) -> None:
    for _, _, entries in gen.plan_inventory_parts(inventory):
        assert 0 < len(entries) <= gen.ROWS_PER_PART
        assert len({e.primary_workspace for e in entries}) == 1


@pytest.mark.unit
def test_index_links_every_part_and_part_links_back(inventory) -> None:
    files = gen.render_baseline_files(inventory)
    index = files[gen.BASELINE_PATH]
    for path, content in files.items():
        if path == gen.BASELINE_PATH:
            continue
        assert f"{gen.PARTS_DIR.name}/{path.name}" in index
        assert f"../{gen.BASELINE_PATH.name}" in content


@pytest.mark.unit
def test_index_keeps_required_sections(inventory) -> None:
    index = gen.render_baseline_files(inventory)[gen.BASELINE_PATH]
    assert "## Complete Capability Baseline Inventory" in index
    assert "## Representative Golden Preservation Fixtures" in index
    assert "## Legacy Alias Resolution Graph" in index


@pytest.mark.unit
def test_committed_files_under_doc_budget_and_check_passes() -> None:
    inv = load_migration_inventory(gen.MIGRATION_JSON_PATH)
    files = gen.render_baseline_files(inv)
    for path in files:
        assert path.is_file()
        assert path.stat().st_size < DOC_BUDGET_BYTES, path
    assert gen.main(["--check"]) == 0


@pytest.mark.unit
def test_invalid_rows_per_part_raises(inventory, monkeypatch) -> None:
    monkeypatch.setattr(gen, "ROWS_PER_PART", 0)
    with pytest.raises(ValueError):
        gen.plan_inventory_parts(inventory)

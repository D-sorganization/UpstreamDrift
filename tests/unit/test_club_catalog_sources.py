"""Offline catalog qualification and review without overwriting player evidence."""

from __future__ import annotations

import pytest

from src.shared.python.club_data.catalog import PropertyClaim, SpecificationSource
from src.shared.python.club_data.catalog_sources import (
    catalog_diff,
    load_public_catalog,
    summarize_record,
    with_player_overrides,
)

pytestmark = pytest.mark.unit


def test_seed_catalog_has_inspectable_evidence_and_partial_data() -> None:
    records = load_public_catalog()
    assert len({r.identity.manufacturer for r in records}) >= 2
    assert len({r.identity.club_type for r in records}) >= 2
    assert any(r.physical_value("mass", "head") is None for r in records)
    for record in records:
        for claim in record.claims:
            assert claim.source.url and claim.source.retrieved_at
            assert claim.source.license and claim.status == "published"
        assert record.physical_value("moi", "head") is None


def test_manufacturer_columns_are_associated_with_the_correct_build() -> None:
    records = load_public_catalog()
    iron = next(
        r
        for r in records
        if r.identity.manufacturer == "Titleist" and r.identity.number == "7"
    )
    assert iron.identity.release_year == 2023
    assert iron.physical_value("length", "assembled") == pytest.approx(0.9398)
    assert iron.physical_value("loft", "head") == pytest.approx(
        32 * 3.141592653589793 / 180
    )
    driver = next(r for r in records if r.identity.manufacturer == "PING")
    assert driver.physical_value("mass", "head") == pytest.approx(0.196)


def test_review_diff_is_stable_and_detects_all_change_kinds() -> None:
    old = load_public_catalog()
    changed = old[0].model_copy(update={"notes": "Reviewed source update"})
    candidate = (changed, *old[1:-1])
    changes = catalog_diff(old, candidate)
    assert {c.kind for c in changes} == {"changed", "removed"}
    assert changes == catalog_diff(tuple(reversed(old)), tuple(reversed(candidate)))
    assert catalog_diff((), old)[0].kind == "added"
    assert catalog_diff(old, old) == ()
    with pytest.raises(ValueError, match="duplicate"):
        catalog_diff(old, (old[0], old[0]))


def test_source_update_preserves_manual_override_without_mutation() -> None:
    base = load_public_catalog()[0]
    override = PropertyClaim(
        property="length",
        component="assembled",
        value=36,
        unit="in",
        status="measured",
        source=SpecificationSource(
            kind="player",
            title="Player tape measure",
            license="user-provided",
            method="Measured assembled club length",
        ),
    )
    updated = base.model_copy(update={"notes": "New source review"})
    first = with_player_overrides(base, (override,))
    second = with_player_overrides(updated, (override,))
    assert first.physical_value("length", "assembled") == second.physical_value(
        "length", "assembled"
    )
    assert override not in base.claims
    assert first.catalog_id == base.catalog_id and first.revision != base.revision
    with pytest.raises(ValueError, match="player"):
        with_player_overrides(base, (base.claims[0],))


def test_summary_displays_missing_values_as_unknown() -> None:
    iron = next(
        r for r in load_public_catalog() if r.identity.manufacturer == "Titleist"
    )
    summary = summarize_record(iron)
    assert "Head Mass: Unknown" in summary
    assert "Head MOI: Unknown" in summary
    assert "published" in summary


def test_offline_review_command_is_read_only(monkeypatch, tmp_path, capsys) -> None:
    from scripts.review_club_catalog import main
    from src.shared.python.club_data.catalog_io import export_json

    candidate = tmp_path / "candidate.json"
    original = export_json(load_public_catalog())
    candidate.write_text(original, encoding="utf-8")
    monkeypatch.setattr("sys.argv", ["review_club_catalog", str(candidate)])
    assert main() == 0
    assert capsys.readouterr().out.strip() == "[]"
    assert candidate.read_text(encoding="utf-8") == original
    assert list(tmp_path.iterdir()) == [candidate]

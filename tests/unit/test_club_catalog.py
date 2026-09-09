"""Contract tests for provenance-aware specifications in the club-data authority."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest
from pydantic import ValidationError

from src.shared.python.club_data.catalog import (
    ClubIdentity,
    ClubRecord,
    PropertyClaim,
    SpecificationSource,
)
from src.shared.python.club_data.catalog_io import (
    export_csv,
    export_json,
    import_csv,
    import_json,
)
from src.shared.python.club_data.catalog_legacy import import_legacy_specification

pytestmark = pytest.mark.unit


def source() -> SpecificationSource:
    return SpecificationSource(
        kind="manufacturer",
        title="Example specification",
        url="https://example.com/specs",
        retrieved_at="2026-09-09T00:00:00Z",
        license="redistribution-unverified",
    )


def identity(**kwargs: object) -> ClubIdentity:
    return ClubIdentity(model="Personal 7-Iron", club_type="iron", number="7", **kwargs)


def claim(**kwargs: object) -> PropertyClaim:
    values = {
        "property": "length",
        "component": "assembled",
        "value": 37.0,
        "unit": "in",
        "status": "published",
        "source": source(),
    }
    values.update(kwargs)
    return PropertyClaim(**values)


def test_catalog_import_does_not_load_gui_or_excel_stack() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            (
                "import sys; from src.shared.python.club_data import ClubRecord; "
                "assert not any(x in sys.modules for x in ['PyQt6', 'pandas', 'openpyxl'])"
            ),
        ],
        capture_output=True,
        text=True,
        check=False,
        timeout=30,
        cwd=Path(__file__).resolve().parents[2],
    )
    assert result.returncode == 0, result.stderr


def test_unknowns_and_custom_builds_have_stable_identity() -> None:
    record = ClubRecord(identity=identity())
    assert record.physical_value("mass", "head") is None
    assert record.catalog_id == ClubRecord(identity=identity()).catalog_id
    custom = ClubRecord(identity=identity(build="shaft shortened by player"))
    assert record.catalog_id != custom.catalog_id
    assert record.revision != custom.revision


def test_si_conversion_keeps_original_units_and_provenance() -> None:
    length = claim()
    record = ClubRecord(identity=identity(), claims=(length,))
    assert record.physical_value("length", "assembled") == pytest.approx(0.9398)
    assert length.value == 37 and length.unit == "in"
    assert length.source.license == "redistribution-unverified"
    mass = claim(property="mass", component="head", value=250, unit="g")
    assert mass.si_value() == pytest.approx(0.25)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), -1, 0, 3000, True])
def test_invalid_lengths_are_rejected(value: float) -> None:
    with pytest.raises(ValidationError):
        claim(value=value)


def test_swing_weight_is_never_interpreted_as_mass() -> None:
    with pytest.raises(ValidationError):
        claim(property="mass", value=2, unit="D2")
    record = ClubRecord(identity=identity(), swing_weight="D2")
    assert record.physical_value("mass", "assembled") is None


def test_estimates_require_explicit_consumer_opt_in() -> None:
    record = ClubRecord(identity=identity(), claims=(claim(status="estimated"),))
    with pytest.raises(ValueError, match="estimated"):
        record.physical_value("length", "assembled")
    assert record.physical_value(
        "length", "assembled", allow_estimates=True
    ) == pytest.approx(0.9398)


def test_ambiguous_moi_is_retained_but_not_given_to_physics() -> None:
    moi = claim(property="moi", component="head", value=5000, unit="g*cm^2")
    record = ClubRecord(identity=identity(), claims=(moi,))
    with pytest.raises(ValueError, match="axis.*frame.*origin"):
        record.physical_value("moi", "head")
    defined = moi.model_copy(
        update={"axis": "z", "frame": "head-local", "origin": "center of mass"}
    )
    assert ClubRecord(identity=identity(), claims=(defined,)).physical_value(
        "moi", "head"
    ) == pytest.approx(0.0005)


def test_conflicting_sources_are_preserved_and_require_resolution() -> None:
    record = ClubRecord(identity=identity(), claims=(claim(), claim(value=37.5)))
    with pytest.raises(ValueError, match="conflicting"):
        record.physical_value("length", "assembled")
    assert len(import_json(export_json((record,)))[0].claims) == 2


def test_source_requires_attribution_and_public_url() -> None:
    with pytest.raises(ValidationError):
        SpecificationSource(kind="manufacturer", title="", license="unknown")
    with pytest.raises(ValidationError):
        SpecificationSource(
            kind="manufacturer", title="Specs", license="unknown", url="file:///secret"
        )


def test_revision_changes_without_changing_build_id() -> None:
    first = ClubRecord(identity=identity(), claims=(claim(),))
    second = ClubRecord(identity=identity(), claims=(claim(value=38),))
    assert first.catalog_id == second.catalog_id
    assert first.revision != second.revision


@pytest.mark.parametrize(
    "exporter, importer", [(export_json, import_json), (export_csv, import_csv)]
)
def test_lossless_exchange_preserves_unknown_and_sourced_claims(
    exporter, importer
) -> None:
    records = (
        ClubRecord(
            identity=identity(),
            claims=(
                claim(),
                claim(
                    property="mass",
                    component="shaft",
                    value=None,
                    unit="g",
                    status="unknown",
                ),
            ),
        ),
        ClubRecord(
            identity=identity(build="custom"), notes="Instructor, note\nsecond line"
        ),
    )
    assert importer(exporter(records)) == records


def test_import_rejects_versions_duplicates_and_forged_ids() -> None:
    record = ClubRecord(identity=identity())
    data = json.loads(export_json((record,)))
    data["schema_version"] = "future"
    with pytest.raises(ValueError):
        import_json(json.dumps(data))
    with pytest.raises(ValueError, match="duplicate"):
        export_json((record, record))
    csv_text = export_csv((record,)).replace(record.catalog_id, "forged")
    with pytest.raises(ValueError, match="identity"):
        import_csv(csv_text)


def test_legacy_defaults_never_become_measurements() -> None:
    from src.shared.python.club_data.loader import ClubSpecification

    legacy = ClubSpecification(name="Unknown Iron", club_type="Iron")
    record = import_legacy_specification(legacy)
    assert record.claims
    assert all(c.status == "unverified" for c in record.claims)
    with pytest.raises(ValueError, match="unverified"):
        record.physical_value("length", "assembled")
    assert "defaults" in record.notes


def test_records_are_immutable() -> None:
    with pytest.raises(ValidationError):
        identity().model = "Mutated"


@pytest.mark.parametrize(
    "change",
    [
        {"value": None},
        {"status": "unknown"},
        {"confidence": float("nan")},
        {"confidence": 1.1},
        {"unit": "kg"},
        {"property": "swing_weight"},
    ],
)
def test_invalid_evidence_combinations_are_rejected(change: dict) -> None:
    with pytest.raises(ValidationError):
        claim(**change)


def test_csv_rejects_inconsistent_record_metadata() -> None:
    record = ClubRecord(identity=identity(), claims=(claim(), claim(value=38)))
    text = export_csv((record,))
    text = text.replace("Personal 7-Iron", "Personal 8-Iron", 1)
    with pytest.raises(ValueError, match="identity"):
        import_csv(text)


def test_exchange_refuses_oversized_and_unknown_fields() -> None:
    with pytest.raises(ValueError, match="8 MiB"):
        import_json(" " * (8 * 1024 * 1024 + 1))
    document = json.loads(export_json((ClubRecord(identity=identity()),)))
    document["records"][0]["invented_mass"] = 1
    with pytest.raises(ValidationError):
        import_json(json.dumps(document))

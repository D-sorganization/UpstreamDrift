"""Executable contracts for numeric claim evidence (#8918)."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy import (
    build_claim_numeric_comparison_evidence as comparison_builder,
)
from scripts.research.proximal_distal_energy.claim_numeric_audit import (
    audit_claim_numeric_evidence,
    audit_registry_numeric_evidence,
    extract_numeric_literals,
)


pytestmark = pytest.mark.unit


def test_comparison_evidence_check_is_json_format_independent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    record = {
        "source_sha256": "abc123",
        "spatial_forward_contact": {
            "reference": [1.0, 2.0],
            "candidate": [1.001, 2.001],
        },
    }
    output = tmp_path / "comparison.json"
    output.write_text(json.dumps(record, separators=(",", ":")), encoding="utf-8")
    monkeypatch.setattr(comparison_builder, "OUTPUT", output)
    monkeypatch.setattr(comparison_builder, "build_record", lambda: record)

    result = comparison_builder.validate_record()

    assert result["comparison_sample_count"] == 2


def _claim(*entries: dict[str, object]) -> dict[str, object]:
    return {
        "claim_id": "PD-CLAIM-TEST",
        "statement": "The run used 12 cases and reached -7.50 N m twice: -7.50.",
        "evidence_artifacts": ["result.json"],
        "numeric_evidence": list(entries),
    }


def _entry(
    literal_id: str,
    pointer: str,
    *,
    scale: float = 1.0,
    offset: float = 0.0,
    atol: float = 0.0,
) -> dict[str, object]:
    return {
        "literal_id": literal_id,
        "artifact": "result.json",
        "json_pointer": pointer,
        "evidence_scope": "local_json_value",
        "scale": scale,
        "offset": offset,
        "atol": atol,
        "rtol": 0.0,
    }


@pytest.mark.unit
def test_numeric_literals_have_stable_occurrence_identifiers() -> None:
    assert extract_numeric_literals(
        "Counts 1,944 and 2.0e-3 repeat 2.0e-3; M^-1 is retained."
    ) == [
        {"literal_id": "1,944#1", "text": "1,944", "value": 1944.0},
        {"literal_id": "2.0e-3#1", "text": "2.0e-3", "value": 0.002},
        {"literal_id": "2.0e-3#2", "text": "2.0e-3", "value": 0.002},
        {"literal_id": "-1#1", "text": "-1", "value": -1.0},
    ]


@pytest.mark.unit
def test_numeric_literals_do_not_merge_comma_delimited_array_values() -> None:
    assert extract_numeric_literals("translation [0.2,-0.1,0.05] m") == [
        {"literal_id": "0.2#1", "text": "0.2", "value": 0.2},
        {"literal_id": "-0.1#1", "text": "-0.1", "value": -0.1},
        {"literal_id": "0.05#1", "text": "0.05", "value": 0.05},
    ]


@pytest.mark.unit
def test_numeric_literals_treat_double_hyphen_as_unsigned_range_separator() -> None:
    assert extract_numeric_literals("range 9.05--9.46 and 13--20 percent") == [
        {"literal_id": "9.05#1", "text": "9.05", "value": 9.05},
        {"literal_id": "9.46#1", "text": "9.46", "value": 9.46},
        {"literal_id": "13#1", "text": "13", "value": 13.0},
        {"literal_id": "20#1", "text": "20", "value": 20.0},
    ]


@pytest.mark.unit
def test_numeric_audit_resolves_pointers_and_declared_unit_transform(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "counts": {"retained": 12},
                "minimum": {"couple_nm": -7.5},
                "encoded/field": {"~value": -7500.0},
            }
        ),
        encoding="utf-8",
    )
    claim = _claim(
        _entry("12#1", "/counts/retained"),
        _entry("-7.50#1", "/minimum/couple_nm"),
        _entry("-7.50#2", "/encoded~1field/~0value", scale=0.001),
    )

    result = audit_claim_numeric_evidence(claim, repository_root=tmp_path)

    assert result == {
        "claim_id": "PD-CLAIM-TEST",
        "literal_count": 3,
        "verified_count": 3,
        "nondegenerate_comparison_count": 0,
        "evidence_scope_counts": {"local_json_value": 3},
    }


@pytest.mark.unit
def test_numeric_audit_rejects_missing_duplicate_literal_declaration(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text(
        '{"counts":{"retained":12},"minimum":{"couple_nm":-7.5}}',
        encoding="utf-8",
    )
    claim = _claim(
        _entry("12#1", "/counts/retained"),
        _entry("-7.50#1", "/minimum/couple_nm"),
    )

    with pytest.raises(ValueError, match=r"missing=.*-7.50#2"):
        audit_claim_numeric_evidence(claim, repository_root=tmp_path)


@pytest.mark.unit
@pytest.mark.parametrize(
    ("mutator", "message"),
    [
        (lambda entry: entry.update(artifact="undeclared.json"), "not declared"),
        (lambda entry: entry.update(json_pointer="counts/retained"), "JSON Pointer"),
        (lambda entry: entry.update(json_pointer="/missing"), "does not resolve"),
        (lambda entry: entry.update(atol=-1.0), "non-negative"),
        (lambda entry: entry.update(scale="one"), "finite number"),
        (lambda entry: entry.update(evidence_scope="independent_validation"), "scope"),
    ],
)
def test_numeric_audit_fails_closed_on_invalid_contracts(
    tmp_path: Path,
    mutator: object,
    message: str,
) -> None:
    (tmp_path / "result.json").write_text(
        '{"counts":{"retained":12},"minimum":{"couple_nm":-7.5}}',
        encoding="utf-8",
    )
    entry = _entry("12#1", "/counts/retained")
    mutator(entry)  # type: ignore[operator]
    claim = _claim(
        entry,
        _entry("-7.50#1", "/minimum/couple_nm"),
        _entry("-7.50#2", "/minimum/couple_nm"),
    )

    with pytest.raises(ValueError, match=message):
        audit_claim_numeric_evidence(claim, repository_root=tmp_path)


@pytest.mark.unit
def test_numeric_audit_rejects_false_numeric_claim(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        '{"counts":{"retained":11},"minimum":{"couple_nm":-7.5}}',
        encoding="utf-8",
    )
    claim = _claim(
        _entry("12#1", "/counts/retained"),
        _entry("-7.50#1", "/minimum/couple_nm"),
        _entry("-7.50#2", "/minimum/couple_nm"),
    )

    with pytest.raises(ValueError, match="numeric evidence mismatch"):
        audit_claim_numeric_evidence(claim, repository_root=tmp_path)


@pytest.mark.unit
def test_numeric_audit_rejects_exact_zero_parity_as_degenerate(
    tmp_path: Path,
) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "counts": {"retained": 12},
                "minimum": {"couple_nm": -7.5},
                "parity": {"reference": [1.0, 2.0], "candidate": [1.0, 2.0]},
            }
        ),
        encoding="utf-8",
    )
    claim = _claim(
        _entry("12#1", "/counts/retained"),
        _entry("-7.50#1", "/minimum/couple_nm"),
        _entry("-7.50#2", "/minimum/couple_nm"),
    )
    claim["numeric_comparisons"] = [
        {
            "comparison_id": "native-versus-independent",
            "artifact": "result.json",
            "reference_pointer": "/parity/reference",
            "candidate_pointer": "/parity/candidate",
            "require_nondegenerate": True,
            "atol": 1e-9,
            "rtol": 1e-9,
        }
    ]

    with pytest.raises(ValueError, match="degenerate exact-zero comparison"):
        audit_claim_numeric_evidence(claim, repository_root=tmp_path)


@pytest.mark.unit
def test_numeric_audit_accepts_close_but_nondegenerate_parity(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        json.dumps(
            {
                "counts": {"retained": 12},
                "minimum": {"couple_nm": -7.5},
                "parity": {
                    "reference": [1.0, 2.0],
                    "candidate": [1.0 + 1e-12, 2.0 - 1e-12],
                },
            }
        ),
        encoding="utf-8",
    )
    claim = _claim(
        _entry("12#1", "/counts/retained"),
        _entry("-7.50#1", "/minimum/couple_nm"),
        _entry("-7.50#2", "/minimum/couple_nm"),
    )
    claim["numeric_comparisons"] = [
        {
            "comparison_id": "native-versus-independent",
            "artifact": "result.json",
            "reference_pointer": "/parity/reference",
            "candidate_pointer": "/parity/candidate",
            "require_nondegenerate": True,
            "atol": 1e-9,
            "rtol": 1e-9,
        }
    ]

    result = audit_claim_numeric_evidence(claim, repository_root=tmp_path)

    assert result["nondegenerate_comparison_count"] == 1


@pytest.mark.unit
def test_reported_numeric_evidence_requires_explicit_nonvalidation_boundary(
    tmp_path: Path,
) -> None:
    artifact = "claim_numeric_reported_values.json"
    record = {
        "literal_id": "12#1",
        "value": 12,
        "evidence_scope": "reported_external_value",
        "source_references": ["https://doi.org/10.0000/example"],
        "source_locations": ["paper.qmd:12"],
        "independent_validation": False,
        "boundary": "Reported transcription; not independent validation.",
    }
    (tmp_path / artifact).write_text(
        json.dumps({"claims": {"PD-CLAIM-REPORTED": [record]}}), encoding="utf-8"
    )
    claim = {
        "claim_id": "PD-CLAIM-REPORTED",
        "statement": "The cited sample contained 12 participants.",
        "evidence_artifacts": [artifact],
        "numeric_evidence": [
            {
                **_entry("12#1", "/claims/PD-CLAIM-REPORTED/0/value"),
                "artifact": artifact,
                "evidence_scope": "reported_external_value",
            }
        ],
    }

    result = audit_claim_numeric_evidence(claim, repository_root=tmp_path)
    assert result["verified_count"] == 1

    record.pop("boundary")
    (tmp_path / artifact).write_text(
        json.dumps({"claims": {"PD-CLAIM-REPORTED": [record]}}), encoding="utf-8"
    )
    with pytest.raises(ValueError, match="boundary"):
        audit_claim_numeric_evidence(claim, repository_root=tmp_path)


@pytest.mark.unit
def test_registry_numeric_audit_reports_complete_coverage(tmp_path: Path) -> None:
    (tmp_path / "result.json").write_text(
        '{"counts":{"retained":12},"minimum":{"couple_nm":-7.5}}',
        encoding="utf-8",
    )
    numeric = _claim(
        _entry("12#1", "/counts/retained"),
        _entry("-7.50#1", "/minimum/couple_nm"),
        _entry("-7.50#2", "/minimum/couple_nm"),
    )
    narrative = {
        "claim_id": "PD-CLAIM-NARRATIVE",
        "statement": "This boundary contains no numerical literal.",
        "evidence_artifacts": ["result.json"],
    }
    path = tmp_path / "registry.json"
    path.write_text(json.dumps({"claims": [numeric, narrative]}), encoding="utf-8")

    result = audit_registry_numeric_evidence(path, repository_root=tmp_path)

    assert result == {
        "claim_count": 2,
        "numeric_claim_count": 1,
        "numeric_literal_count": 3,
        "verified_numeric_literal_count": 3,
        "nondegenerate_comparison_count": 0,
        "evidence_scope_counts": {"local_json_value": 3},
        "completion_status": "complete",
    }

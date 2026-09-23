"""TDD: anthropometric ground-support receipt provenance chain (#10271)."""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.anthropometry import de_leva_table_sha256
from src.shared.python.motion_matching.full_body_spec import canonical_sha256
from src.shared.python.motion_matching.pipeline.receipt_provenance import (
    CHAIN_CONTRACT_VERSION,
    ProvenanceChainInputs,
    raw_file_sha256,
    validate_receipt_provenance_chain,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"
EVIDENCE = DOCS / "evidence" / "ground_support"

CURRENT_BASELINE_CAPTURES = (
    (
        "anthro_driver",
        DOCS / "full_body_spec_anthro_driver.json",
        EVIDENCE / "anthro_driver",
    ),
    (
        "anthro_iron",
        DOCS / "full_body_spec_anthro_iron7.json",
        EVIDENCE / "anthro_iron",
    ),
)


def _write_json(path: Path, document: dict[str, Any]) -> None:
    path.write_text(
        json.dumps(document, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _minimal_receipt(
    *,
    base_spec: dict[str, Any],
    final_spec: dict[str, Any],
    final_bytes: bytes,
    include_table: bool = True,
    include_canonical_spec: bool = True,
) -> dict[str, Any]:
    table = de_leva_table_sha256()
    receipt: dict[str, Any] = {
        "base_spec_sha256": canonical_sha256(base_spec),
        "base_spec_file": "base.json",
        "spec_file": "final.json",
        "hipcal_spec_file": "hipcal.json",
        "spec_sha256": hashlib.sha256(final_bytes).hexdigest(),
        "capture": "driver",
        "capture_sha256": "a" * 64,
        "candidate_sha256": "b" * 64,
    }
    if include_table:
        receipt["de_leva_table_sha256"] = table
    if include_canonical_spec:
        receipt["spec_canonical_sha256"] = canonical_sha256(final_spec)
    return receipt


def test_chain_contract_version_is_stable() -> None:
    assert CHAIN_CONTRACT_VERSION == "receipt-provenance-chain/1"


def test_valid_complete_chain_passes(tmp_path: Path) -> None:
    base = {
        "name": "base",
        "de_leva_table_sha256": de_leva_table_sha256(),
        "anthropometry": {"segments": {"shank": {"length_m": 0.4}}},
    }
    final = {
        "name": "final",
        "de_leva_table_sha256": de_leva_table_sha256(),
        "value": 1.0,
    }
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    final_bytes = (
        json.dumps(final, sort_keys=True, separators=(",", ":"), allow_nan=False) + "\n"
    ).encode("utf-8")
    final_path.write_bytes(final_bytes)
    receipt = _minimal_receipt(
        base_spec=base, final_spec=final, final_bytes=final_bytes
    )
    validate_receipt_provenance_chain(
        ProvenanceChainInputs(
            receipt=receipt,
            base_spec_path=base_path,
            final_spec_path=final_path,
            hipcal_spec_path=None,
        )
    )


def test_missing_receipt_table_hash_fails(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "final"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base,
        final_spec=final,
        final_bytes=final_path.read_bytes(),
        include_table=False,
    )
    with pytest.raises(ValueError, match="de_leva_table_sha256"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )


def test_mismatched_receipt_table_hash_fails(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "final"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base, final_spec=final, final_bytes=final_path.read_bytes()
    )
    receipt["de_leva_table_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="de_leva_table_sha256"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )


def test_mismatched_canonical_base_fails(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "final"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base, final_spec=final, final_bytes=final_path.read_bytes()
    )
    receipt["base_spec_sha256"] = "f" * 64
    with pytest.raises(ValueError, match="base_spec_sha256"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )


def test_missing_final_spec_fails(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    base_path = tmp_path / "base.json"
    _write_json(base_path, base)
    receipt = {
        "base_spec_sha256": canonical_sha256(base),
        "de_leva_table_sha256": de_leva_table_sha256(),
        "spec_sha256": "c" * 64,
        "spec_canonical_sha256": "d" * 64,
    }
    with pytest.raises(ValueError, match="final spec"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=tmp_path / "missing_final.json",
            )
        )


def test_mismatched_final_spec_raw_without_canonical_fails(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "final"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base,
        final_spec=final,
        final_bytes=final_path.read_bytes(),
        include_canonical_spec=False,
    )
    receipt["spec_sha256"] = "e" * 64
    with pytest.raises(ValueError, match="spec_sha256"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )


def test_formatting_only_reserialization_passes_via_canonical(tmp_path: Path) -> None:
    """Pretty-print changes raw bytes but not canonical semantic identity."""
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "z": 1, "a": 2}
    compact = json.dumps(
        final, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")
    pretty = (json.dumps(final, indent=2, allow_nan=False) + "\n").encode("utf-8")
    assert hashlib.sha256(compact).hexdigest() != hashlib.sha256(pretty).hexdigest()
    assert canonical_sha256(json.loads(compact)) == canonical_sha256(json.loads(pretty))

    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    final_path.write_bytes(pretty)
    receipt = _minimal_receipt(base_spec=base, final_spec=final, final_bytes=compact)
    validate_receipt_provenance_chain(
        ProvenanceChainInputs(
            receipt=receipt,
            base_spec_path=base_path,
            final_spec_path=final_path,
        )
    )


def test_numeric_change_fails_even_with_stale_canonical_claim(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    original = {"de_leva_table_sha256": de_leva_table_sha256(), "value": 1.0}
    changed = {"de_leva_table_sha256": de_leva_table_sha256(), "value": 2.0}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, changed)
    receipt = _minimal_receipt(
        base_spec=base,
        final_spec=original,
        final_bytes=json.dumps(original, separators=(",", ":")).encode("utf-8"),
    )
    with pytest.raises(ValueError, match="spec_canonical_sha256|spec_sha256"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )


def test_absent_intermediate_hipcal_is_allowed_by_policy(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "final"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base, final_spec=final, final_bytes=final_path.read_bytes()
    )
    validate_receipt_provenance_chain(
        ProvenanceChainInputs(
            receipt=receipt,
            base_spec_path=base_path,
            final_spec_path=final_path,
            hipcal_spec_path=tmp_path / "full_body_spec_hipcal.json",
            require_hipcal_document=False,
        )
    )


def test_required_hipcal_document_missing_fails(tmp_path: Path) -> None:
    base = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "base"}
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "final"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base, final_spec=final, final_bytes=final_path.read_bytes()
    )
    with pytest.raises(ValueError, match="hipcal"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=receipt,
                base_spec_path=base_path,
                final_spec_path=final_path,
                hipcal_spec_path=tmp_path / "full_body_spec_hipcal.json",
                require_hipcal_document=True,
            )
        )


def test_dbc_rejects_non_mapping_receipt(tmp_path: Path) -> None:
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, {"de_leva_table_sha256": de_leva_table_sha256()})
    _write_json(final_path, {"de_leva_table_sha256": de_leva_table_sha256()})
    with pytest.raises(TypeError, match="receipt"):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt="not a mapping",  # type: ignore[arg-type]
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )


def test_raw_file_sha256_matches_hashlib(tmp_path: Path) -> None:
    path = tmp_path / "blob.bin"
    path.write_bytes(b"abc")
    assert raw_file_sha256(path) == hashlib.sha256(b"abc").hexdigest()


@pytest.mark.parametrize(
    ("_label", "base_path", "capture_dir"),
    CURRENT_BASELINE_CAPTURES,
    ids=[row[0] for row in CURRENT_BASELINE_CAPTURES],
)
def test_current_baseline_receipts_have_intact_provenance_chain(
    _label: str, base_path: Path, capture_dir: Path
) -> None:
    """CI gate: designated hip-calibrated baselines must chain to current bases."""
    receipt_path = capture_dir / "receipt.json"
    final_path = capture_dir / "full_body_spec_hipcal_scaled.json"
    assert receipt_path.is_file()
    assert base_path.is_file()
    assert final_path.is_file()
    receipt = json.loads(receipt_path.read_text(encoding="utf-8"))
    validate_receipt_provenance_chain(
        ProvenanceChainInputs(
            receipt=receipt,
            base_spec_path=base_path,
            final_spec_path=final_path,
            hipcal_spec_path=capture_dir / "full_body_spec_hipcal.json",
            require_hipcal_document=False,
        )
    )


def test_ho8_style_broken_receipt_fixture_fails_closed(tmp_path: Path) -> None:
    """Reproduce the #10271 gap: missing table hash + stale base digest."""
    base = json.loads(
        (DOCS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    final = {"de_leva_table_sha256": de_leva_table_sha256(), "name": "scaled"}
    base_path = tmp_path / "base.json"
    final_path = tmp_path / "final.json"
    _write_json(base_path, base)
    _write_json(final_path, final)
    receipt = _minimal_receipt(
        base_spec=base, final_spec=final, final_bytes=final_path.read_bytes()
    )
    broken = copy.deepcopy(receipt)
    broken.pop("de_leva_table_sha256", None)
    broken["base_spec_sha256"] = (
        "a73d9e0623e79c5becd2b304cbc72940d16451097c4da0474a7e17d15fe3717a"
    )
    with pytest.raises(ValueError):
        validate_receipt_provenance_chain(
            ProvenanceChainInputs(
                receipt=broken,
                base_spec_path=base_path,
                final_spec_path=final_path,
            )
        )

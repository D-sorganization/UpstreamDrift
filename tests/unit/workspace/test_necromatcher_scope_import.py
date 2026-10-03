"""Raw reviewed windows register through the canonical immutable asset spine."""

import hashlib
import json
from types import SimpleNamespace
from pathlib import Path
from typing import Any

import pytest

from tests.unit.workspace.test_necromatcher_scope_receipts import (
    scope_asset_case as imported_scope_case,
)
from src.shared.python.workspace import import_fit_source_scope_review

pytestmark = pytest.mark.unit
scope_asset_case = imported_scope_case


def test_exact_bytes_portable_registration_and_checked_reuse(
    scope_asset_case: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    library, source, identity = scope_asset_case
    monkeypatch.setattr(
        library,
        "load_fit",
        lambda _: {
            "capture_id": identity.capture_id,
            "capture_hash": identity.capture_hash,
        },
    )
    raw = source.read_bytes() + b"\n"
    scope = import_fit_source_scope_review(library, "parent", raw)
    review_id = "scope-review-" + hashlib.sha256(raw).hexdigest()
    assert scope.review.artifact.artifact_id == review_id
    stored = Path(library.load_asset(review_id).path)
    assert stored.read_bytes() == raw
    assert scope.review.artifact.path == f"assets/{review_id}.json"
    before = {str(p): p.read_bytes() for p in library.root.rglob("*") if p.is_file()}
    assert import_fit_source_scope_review(library, "parent", raw) == scope
    assert before == {
        str(p): p.read_bytes() for p in library.root.rglob("*") if p.is_file()
    }


@pytest.mark.parametrize(
    "raw",
    [b"", b"[]", b"{}", b"\xff", b" " * (1024 * 1024 + 1), "{}"],
    ids=["empty", "array", "missing", "utf8", "oversize", "string"],
)
def test_malformed_or_oversized_receipt_never_reads_library(raw: Any) -> None:
    with pytest.raises(ValueError):
        import_fit_source_scope_review(SimpleNamespace(), "parent", raw)


def test_other_capture_review_rejected_without_registration(
    scope_asset_case: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    library, source, identity = scope_asset_case
    monkeypatch.setattr(
        library,
        "load_fit",
        lambda _: {"capture_id": "foreign", "capture_hash": identity.capture_hash},
    )
    before = library.assets("practice")
    with pytest.raises(ValueError, match="capture"):
        import_fit_source_scope_review(library, "parent", source.read_bytes())
    assert library.assets("practice") == before


def test_registered_tamper_never_overwritten(
    scope_asset_case: tuple, monkeypatch: pytest.MonkeyPatch
) -> None:
    library, source, identity = scope_asset_case
    monkeypatch.setattr(
        library,
        "load_fit",
        lambda _: {
            "capture_id": identity.capture_id,
            "capture_hash": identity.capture_hash,
        },
    )
    raw = source.read_bytes()
    scope = import_fit_source_scope_review(library, "parent", raw)
    stored = Path(library.load_asset(scope.review.artifact.artifact_id).path)
    stored.write_bytes(b"tampered")
    with pytest.raises(ValueError):
        import_fit_source_scope_review(library, "parent", raw)
    assert stored.read_bytes() == b"tampered"


def test_calibrated_claim_rejects_before_library_read(scope_asset_case: tuple) -> None:
    _, source, _ = scope_asset_case
    record = json.loads(source.read_bytes())
    record["contact_calibrated"] = True
    with pytest.raises(ValueError):
        import_fit_source_scope_review(
            SimpleNamespace(), "parent", json.dumps(record).encode()
        )

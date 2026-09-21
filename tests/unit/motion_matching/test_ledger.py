"""Unit tests for matched-swing run ledger (MS-02, #10323).

Validates discovery, classification, determinism, freshness, and fail-safe handling
of all committed receipts under evidence roots into one browsable index.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.ledger import (
    DEFAULT_ROOTS,
    Ledger,
    LedgerRow,
    default_ledger_path,
    scan,
)
from src.tools.motion_matching.pipeline import list_runs

REPO_ROOT = Path(__file__).resolve().parents[3]


@pytest.mark.unit
def test_scan_finds_every_committed_receipt():
    """Count of ledger rows equals the count of receipt*.json files under evidence roots."""
    roots = (
        REPO_ROOT / "docs" / "development" / "full_body_models" / "evidence",
        REPO_ROOT
        / "docs"
        / "development"
        / "simscape_tour_matching"
        / "native_evidence",
        REPO_ROOT / "docs" / "development" / "opensim_tour_matching" / "evidence",
    )
    disk_receipts = [
        p.resolve()
        for root in roots
        for p in root.rglob("*.json")
        if p.is_file() and "receipt" in p.name.lower()
    ]
    assert len(disk_receipts) >= 40, (
        f"Expected at least 40 receipts on disk, found {len(disk_receipts)}"
    )

    ledger = scan(roots=roots)
    assert len(ledger.rows) == len(disk_receipts)

    # Also verify default roots scan (which includes evidence/ if present)
    full_ledger = scan()
    assert len(full_ledger.rows) >= len(disk_receipts)


@pytest.mark.unit
def test_row_classification():
    """Specific landmark receipts classify into correct engine, capture, and lane."""
    ledger = scan()
    rows_by_subpath = {row.receipt_path.replace("\\", "/"): row for row in ledger.rows}

    # 1. anthro_driver_shoot_g025/receipt.json -> engine mujoco, capture driver, lane ground_support
    match_gs = [
        row
        for path, row in rows_by_subpath.items()
        if "anthro_driver_shoot_g025/receipt.json" in path
    ]
    assert len(match_gs) == 1
    r_gs = match_gs[0]
    assert r_gs.engine == "mujoco"
    assert r_gs.capture == "driver"
    assert r_gs.lane == "ground_support"

    # 2. two_window_fit_9967_102/receipt.json -> engine simscape, lane native
    match_ss = [
        row
        for path, row in rows_by_subpath.items()
        if "two_window_fit_9967_102/receipt.json" in path
    ]
    assert len(match_ss) == 1
    r_ss = match_ss[0]
    assert r_ss.engine == "simscape"
    assert r_ss.lane == "native"

    # 3. os3b_scale_ik/receipt.json -> opensim, lane tour_matching
    match_os = [
        row
        for path, row in rows_by_subpath.items()
        if "os3b_scale_ik/receipt.json" in path
    ]
    assert len(match_os) == 1
    r_os = match_os[0]
    assert r_os.engine == "opensim"
    assert r_os.lane == "tour_matching"


@pytest.mark.unit
def test_ledger_is_deterministic():
    """Two successive scans produce byte-identical rows and totals."""
    ledger1 = scan()
    ledger2 = scan()
    d1 = json.loads(ledger1.to_json())
    d2 = json.loads(ledger2.to_json())
    d1.pop("generated_at", None)
    d2.pop("generated_at", None)
    assert d1 == d2


@pytest.mark.unit
def test_ledger_freshness():
    """Committed reports/matched_swing_ledger.json equals a fresh scan."""
    path = default_ledger_path()
    assert path.is_file(), (
        f"Committed ledger missing at {path}. Run 'python -m src.shared.python.motion_matching ledger --write'"
    )

    committed_data = json.loads(path.read_text(encoding="utf-8"))
    fresh_ledger = scan()
    fresh_data = json.loads(fresh_ledger.to_json())

    # Ignore timestamp for freshness comparison
    committed_data.pop("generated_at", None)
    fresh_data.pop("generated_at", None)
    assert committed_data == fresh_data


@pytest.mark.unit
def test_unknown_receipt_shape_is_listed_not_dropped(tmp_path: Path):
    """A receipt that fails standard classification appears with lane='unclassified' and a reason."""
    dummy_evidence = tmp_path / "evidence_misc"
    dummy_evidence.mkdir(parents=True)
    bad_receipt = dummy_evidence / "weird_receipt.json"
    bad_receipt.write_text(
        json.dumps({"arbitrary_key": "some_value"}), encoding="utf-8"
    )

    ledger = scan(roots=[dummy_evidence])
    assert len(ledger.rows) == 1
    row = ledger.rows[0]
    assert row.lane == "unclassified"
    assert row.reason is not None and len(row.reason) > 0
    assert row.sha256 is not None and len(row.sha256) == 64


@pytest.mark.unit
def test_postcondition_contract():
    """Every row has receipt_path relative to repo root and valid sha256."""
    ledger = scan()
    assert len(ledger.rows) > 0
    for row in ledger.rows:
        assert not Path(row.receipt_path).is_absolute()
        assert len(row.sha256) == 64
        assert all(c in "0123456789abcdef" for c in row.sha256.lower())


@pytest.mark.unit
def test_pipeline_list_runs():
    """src.tools.motion_matching.pipeline.list_runs is backed by ledger."""
    runs = list_runs()
    assert len(runs) >= 40
    assert isinstance(runs[0], LedgerRow)


@pytest.mark.unit
def test_self_reported_acceptance_is_never_passed() -> None:
    """MS-100: a receipt's own ``accepted`` flag cannot mark a ledger row accepted."""
    from src.shared.python.motion_matching.ledger import extract_acceptance

    self_reported = {"receipt": {"accepted": True, "status": "terminal"}}
    block = extract_acceptance("x/receipt.json", self_reported, None)
    assert block is not None
    assert block["is_physically_accepted"] is False
    assert block["status"] == "UNVERIFIED"

    gateless = {"acceptance": {"is_physically_accepted": True, "status": "PASSED"}}
    block = extract_acceptance("y/receipt.json", gateless, None)
    assert block is not None and block["status"] == "UNVERIFIED"

    evaluated = {
        "acceptance": {
            "horizon": "G1",
            "is_physically_accepted": True,
            "status": "PASSED",
            "gates": [{"name": "whole", "status": "PASSED"}],
        }
    }
    block = extract_acceptance("z/receipt.json", evaluated, None)
    assert block is not None and block["status"] == "PASSED"

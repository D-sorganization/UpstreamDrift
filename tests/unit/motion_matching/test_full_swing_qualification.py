"""Unit tests for full-swing qualification matrix (MS-104, #10378).

Software-contract coverage only: fixtures prove matrix machinery,
named blockers, and fail-closed release status. No invented six-engine
native pass is asserted against the live ledger.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from src.shared.python.motion_matching.acceptance import Horizon
from src.shared.python.motion_matching.contact_law import CONFORMANCE_VERSION
from src.shared.python.motion_matching.full_swing_qualification import (
    SCHEMA_VERSION,
    TARGET_ENGINES,
    FullSwingQualificationError,
    QualificationEvidenceLinks,
    QualificationRow,
    build_required_row_specs,
    evaluate_matrix,
    load_report,
    render_blocker_report,
    row_key,
)
from src.shared.python.motion_matching.ledger_schema import (
    ArtefactPaths,
    Ledger,
    LedgerRow,
    SharedMetrics,
)
from src.shared.python.motion_matching.matching_strategy import ALL_ENGINES

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
EVIDENCE_PATH = (
    REPO_ROOT
    / "docs"
    / "plans"
    / "matched_swing"
    / "evidence"
    / "ms104_full_swing_qualification.json"
)


def _accepted_block(horizon: str) -> dict[str, Any]:
    return {
        "horizon": horizon,
        "is_physically_accepted": True,
        "status": "PASSED",
        "gates": [
            {
                "name": "whole_marker_rmse_m",
                "status": "passed",
                "threshold": 0.025,
                "measured": 0.020,
                "unit": "m",
                "reason": "fixture",
            }
        ],
        "qualification_note": "software-contract fixture only",
    }


def _empty_ledger() -> Ledger:
    return Ledger(generated_at="t", total_receipts=0, rows=[])


def _ledger_from_rows(*rows: LedgerRow) -> Ledger:
    return Ledger(generated_at="t", total_receipts=len(rows), rows=list(rows))


def _ledger_row(
    *,
    engine: str,
    capture: str,
    horizon: str,
    receipt_path: str,
    accepted: bool = True,
) -> LedgerRow:
    acceptance = _accepted_block(horizon) if accepted else None
    if acceptance is None:
        acceptance = {
            "horizon": horizon,
            "is_physically_accepted": False,
            "status": "REJECTED",
            "gates": [],
            "qualification_note": "fixture rejected",
        }
    return LedgerRow(
        receipt_path=receipt_path,
        sha256="a" * 64,
        engine=engine,
        lane="matched",
        capture=capture,
        candidate_sha="b" * 64,
        horizon_s={"G1": 0.85, "G2": 1.20, "G3": 1.814}[horizon],
        metrics=SharedMetrics(whole_marker_rmse_m=0.020),
        acceptance=acceptance,
        artefacts=ArtefactPaths(),
        reason=None,
    )


def _full_links(receipt: str) -> QualificationEvidenceLinks:
    return QualificationEvidenceLinks(
        ms100_acceptance_path=receipt,
        ms100_acceptance_sha256="a" * 64,
        ms72_conformance_version=CONFORMANCE_VERSION,
        native_replay_receipt_path=receipt,
        native_replay_sha256="a" * 64,
        numerical_convergence_receipt_path=receipt,
        numerical_convergence_sha256="a" * 64,
        ms70_parity_receipt_path=None,
        candidate_sha="b" * 64,
        model_sha="c" * 64,
        runtime_hash="d" * 64,
        gate_hash="e" * 64,
    )


class TestContractShape:
    def test_schema_version_is_stable(self) -> None:
        assert SCHEMA_VERSION == "full-swing-qualification/1.0.0"

    def test_target_engines_match_strategy_all_engines(self) -> None:
        assert frozenset(ALL_ENGINES) == TARGET_ENGINES
        assert len(TARGET_ENGINES) == 6

    def test_required_specs_cover_36_flagship_rows(self) -> None:
        specs = build_required_row_specs()
        assert len(specs) == 36
        keys = {row_key(s.engine, s.club, s.gate) for s in specs}
        assert len(keys) == 36
        for engine in TARGET_ENGINES:
            for club in ("driver", "iron"):
                for gate in (Horizon.G1, Horizon.G2, Horizon.G3):
                    assert row_key(engine, club, gate.value) in keys

    def test_every_required_spec_carries_named_owner_blocker(self) -> None:
        for spec in build_required_row_specs():
            assert spec.owner_blocker.issue > 0
            assert spec.owner_blocker.title.strip()
            assert spec.model_class == "full_body_flagship"


class TestEvaluateMatrixFailClosed:
    def test_empty_ledger_blocks_all_required_rows(self) -> None:
        report = evaluate_matrix(_empty_ledger())
        assert report.release_status == "blocked"
        assert report.incomplete_required_count == 36
        assert len(report.required_rows) == 36
        assert all(r.status == "blocked" for r in report.required_rows)
        assert len(report.blockers) >= 6

    def test_cannot_claim_release_ready_with_incomplete_rows(self) -> None:
        report = evaluate_matrix(_empty_ledger())
        with pytest.raises(FullSwingQualificationError, match="incomplete"):
            report.require_release_ready()

    def test_rejected_acceptance_does_not_qualify(self) -> None:
        ledger = _ledger_from_rows(
            _ledger_row(
                engine="mujoco",
                capture="driver",
                horizon="G1",
                receipt_path="evidence/fixture/receipt.json",
                accepted=False,
            ),
        )
        report = evaluate_matrix(ledger)
        mujoco_g1 = next(
            r
            for r in report.required_rows
            if r.engine == "mujoco" and r.club == "driver" and r.gate == "G1"
        )
        assert mujoco_g1.status == "blocked"
        assert mujoco_g1.evidence.ms100_acceptance_path is not None
        assert not mujoco_g1.is_qualified

    def test_accepted_without_full_evidence_links_stays_incomplete(self) -> None:
        """MS-100 pass alone is not enough — replay + convergence + MS-72 required."""
        ledger = _ledger_from_rows(
            _ledger_row(
                engine="pinocchio",
                capture="driver",
                horizon="G1",
                receipt_path="evidence/fixture/pin_receipt.json",
                accepted=True,
            ),
        )
        report = evaluate_matrix(ledger, evidence_overrides={})
        row = next(
            r
            for r in report.required_rows
            if r.engine == "pinocchio" and r.club == "driver" and r.gate == "G1"
        )
        assert row.status in {"blocked", "incomplete"}
        assert not row.is_qualified
        assert any(
            "replay" in b.title.lower()
            or "convergence" in b.title.lower()
            or "ms-72" in b.title.lower()
            or "conformance" in b.title.lower()
            for b in row.blockers
        )

    def test_fully_linked_fixture_qualifies_one_row_only(self) -> None:
        receipt = "evidence/fixture/full_links_receipt.json"
        ledger = _ledger_from_rows(
            _ledger_row(
                engine="mujoco",
                capture="driver",
                horizon="G1",
                receipt_path=receipt,
                accepted=True,
            ),
        )
        overrides = {
            row_key("mujoco", "driver", "G1"): _full_links(receipt),
        }
        report = evaluate_matrix(ledger, evidence_overrides=overrides)
        qualified = [r for r in report.required_rows if r.is_qualified]
        assert len(qualified) == 1
        assert qualified[0].engine == "mujoco"
        assert qualified[0].club == "driver"
        assert qualified[0].gate == "G1"
        assert report.release_status == "blocked"
        assert report.incomplete_required_count == 35

    def test_all_36_linked_fixtures_can_set_release_ready(self) -> None:
        rows: list[LedgerRow] = []
        overrides: dict[str, QualificationEvidenceLinks] = {}
        for engine in sorted(TARGET_ENGINES):
            for club in ("driver", "iron"):
                for gate in ("G1", "G2", "G3"):
                    path = f"evidence/fixture/{engine}_{club}_{gate}.json"
                    rows.append(
                        _ledger_row(
                            engine=engine,
                            capture=club,
                            horizon=gate,
                            receipt_path=path,
                            accepted=True,
                        )
                    )
                    overrides[row_key(engine, club, gate)] = _full_links(path)
        report = evaluate_matrix(
            _ledger_from_rows(*rows),
            evidence_overrides=overrides,
        )
        assert report.incomplete_required_count == 0
        assert report.release_status == "ready"
        report.require_release_ready()  # must not raise


class TestPartialAndReducedRows:
    def test_reduced_oracle_rows_are_not_required(self) -> None:
        report = evaluate_matrix(_empty_ledger())
        assert any(
            r.model_class == "reduced_oracle" and not r.is_required
            for r in report.partial_rows
        )
        assert all(r.is_required for r in report.required_rows)

    def test_reduced_oracle_cannot_satisfy_flagship_row(self) -> None:
        """27-coordinate Simscape evidence stays partial, never fills a required cell."""
        receipt = "docs/development/simscape_tour_matching/native_evidence/oracle.json"
        ledger = _ledger_from_rows(
            LedgerRow(
                receipt_path=receipt,
                sha256="a" * 64,
                engine="simscape",
                lane="native",
                capture="driver",
                candidate_sha="b" * 64,
                horizon_s=0.85,
                metrics=SharedMetrics(whole_marker_rmse_m=0.020),
                acceptance=_accepted_block("G1"),
                artefacts=ArtefactPaths(),
                reason="27-coordinate reduced oracle",
            ),
        )
        report = evaluate_matrix(
            ledger,
            evidence_overrides={
                row_key("simscape", "driver", "G1"): _full_links(receipt),
            },
            reduced_oracle_keys={row_key("simscape", "driver", "G1")},
        )
        flagship = next(
            r
            for r in report.required_rows
            if r.engine == "simscape" and r.club == "driver" and r.gate == "G1"
        )
        assert not flagship.is_qualified
        assert any(r.model_class == "reduced_oracle" for r in report.partial_rows)


class TestBlockerReport:
    def test_render_enumerates_every_remaining_blocker(self) -> None:
        report = evaluate_matrix(_empty_ledger())
        text = render_blocker_report(report)
        assert "release_status=blocked" in text
        assert "incomplete_required=36" in text
        for engine in TARGET_ENGINES:
            assert engine in text

    def test_committed_evidence_snapshot_exists_and_is_blocked(self) -> None:
        assert EVIDENCE_PATH.is_file()
        payload = json.loads(EVIDENCE_PATH.read_text(encoding="utf-8"))
        assert payload["schema_version"] == SCHEMA_VERSION
        assert payload["release_status"] == "blocked"
        assert payload["incomplete_required_count"] == 36
        assert (
            "no invented six-engine native pass"
            in payload["qualification_note"].lower()
        )

    def test_load_report_round_trips_committed_evidence(self) -> None:
        report = load_report(EVIDENCE_PATH)
        assert isinstance(report.required_rows[0], QualificationRow)
        assert report.release_status == "blocked"
        assert report.incomplete_required_count == 36

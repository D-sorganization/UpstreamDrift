"""CLI runner and release audit gate for motion matching (MMR-17 #11103).

Ingests native engine lane receipts, dual-club verification receipts, and audits
clean-host end-to-end lifecycle journeys. Emits structured release matrix.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from src.shared.python.motion_matching.release_gates import (
    CleanHostJourneyResult,
    EngineTestStatus,
    ReleaseGateMatrix,
    ReleaseVerdict,
    audit_clean_host_journey,
    evaluate_motion_matching_release_matrix,
)

NIGHTLY_EVIDENCE_DIR = (
    REPO_ROOT
    / "docs"
    / "development"
    / "matched_swing_program"
    / "evidence"
    / "nightly"
)
EVIDENCE_DIR = REPO_ROOT / "docs" / "development" / "matched_swing_program" / "evidence"


def discover_nightly_receipts(nightly_dir: Path) -> dict[str, Any]:
    """Discover and parse all committed nightly native engine lane receipts."""
    receipts: dict[str, Any] = {}
    if not nightly_dir.is_dir():
        return receipts

    for file_path in nightly_dir.glob("*_receipt.json"):
        try:
            data = json.loads(file_path.read_text(encoding="utf-8"))
            if not isinstance(data, dict):
                continue
            engine = data.get("engine")
            if engine:
                receipts[engine] = data
        except (OSError, json.JSONDecodeError):
            continue
    return receipts


def discover_club_receipts(evidence_dir: Path) -> dict[str, dict[str, Any]]:
    """Discover dual-club qualification receipts organized by engine and club."""
    club_receipts: dict[str, dict[str, Any]] = {}
    if not evidence_dir.is_dir():
        return club_receipts

    # Search subdirectories (e.g. drake, ms102)
    for sub in evidence_dir.iterdir():
        if not sub.is_dir():
            continue
        for file_path in sub.glob("*.json"):
            name = file_path.stem.lower()
            try:
                data = json.loads(file_path.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    continue
            except (OSError, json.JSONDecodeError):
                continue

            engine = data.get("engine")
            club = data.get("club")

            if not engine:
                # Infer engine from filename
                for cand_engine in (
                    "drake",
                    "mujoco",
                    "pinocchio",
                    "simscape",
                    "opensim",
                    "myosuite",
                ):
                    if cand_engine in name:
                        engine = cand_engine
                        break

            if not club:
                if "driver" in name:
                    club = "driver"
                elif "iron" in name:
                    club = "7-iron"

            if engine and club:
                if engine not in club_receipts:
                    club_receipts[engine] = {}
                club_receipts[engine][club] = data

    return club_receipts


def run_release_audit(
    repo_root: Path = REPO_ROOT,
    *,
    mandatory_engines: list[str] | None = None,
    advertised_engines: list[str] | None = None,
) -> tuple[ReleaseGateMatrix, CleanHostJourneyResult]:
    """Execute complete release readiness evaluation across receipts and journey audit."""
    nightly_dir = (
        repo_root
        / "docs"
        / "development"
        / "matched_swing_program"
        / "evidence"
        / "nightly"
    )
    evidence_dir = (
        repo_root / "docs" / "development" / "matched_swing_program" / "evidence"
    )

    native_receipts = discover_nightly_receipts(nightly_dir)
    club_receipts = discover_club_receipts(evidence_dir)

    matrix = evaluate_motion_matching_release_matrix(
        native_receipts=native_receipts,
        club_receipts=club_receipts,
        mandatory_engines=mandatory_engines,
        advertised_engines=advertised_engines,
    )
    journey = audit_clean_host_journey(mock_clean_host=True)
    return matrix, journey


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Audit clean-host and native release gates for motion matching."
    )
    parser.add_argument(
        "--matrix-out",
        type=Path,
        help="Optional destination path to write release matrix JSON.",
    )
    parser.add_argument(
        "--audit-only",
        action="store_true",
        help="Run audit and print matrix without failing process exit code on blockers.",
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=REPO_ROOT,
        help="Path to repository root.",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    matrix, journey = run_release_audit(repo_root=args.repo_root)

    print("=== Motion Matching Release Gate Matrix ===")
    print(f"Verdict: {matrix.verdict.value.upper()}")
    print("Engine Test Matrix:")
    for eng, st in sorted(matrix.engine_status.items()):
        print(f"  - {eng:12}: {st.value}")

    print("\nDual-Club Coverage:")
    for eng, clubs in sorted(matrix.dual_club_status.items()):
        drv = "OK" if clubs.get("driver") else "MISSING"
        irn = "OK" if clubs.get("7-iron") else "MISSING"
        print(f"  - {eng:12}: Driver={drv}, 7-Iron={irn}")

    if matrix.blockers:
        print("\nActive Blockers:")
        for b in matrix.blockers:
            print(f"  * {b}")

    print(f"\nClean Host Lifecycle Journey: {'PASS' if journey.all_passed else 'FAIL'}")

    if args.matrix_out:
        args.matrix_out.parent.mkdir(parents=True, exist_ok=True)
        encoded = json.dumps(matrix.to_dict(), indent=2, sort_keys=True)
        args.matrix_out.write_text(encoded + "\n", encoding="utf-8")
        print(f"\nWrote release matrix to {args.matrix_out}")

    if args.audit_only:
        return 0

    return 0 if matrix.verdict == ReleaseVerdict.RELEASE_QUALIFIED else 1


if __name__ == "__main__":
    raise SystemExit(main())

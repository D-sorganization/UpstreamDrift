"""Write the participant-twin cohort record and calibration evidence bundle."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .participant_twin_calibration import build_synthetic_cohort
from .participant_twin_evaluation import build_evidence_record

ROOT = Path(__file__).resolve().parents[3]
ARTICLE = ROOT / "docs/research/proximal_distal_energy_transfer"
COHORT = ARTICLE / "data/participant_twin_cohort.json"
EVIDENCE = ARTICLE / "data/participant_twin_calibration.json"


def _write(path: Path, record: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    """Regenerate the frozen cohort and evidence bundle, or validate them."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("write", "validate"))
    args = parser.parse_args()
    cohort = build_synthetic_cohort()
    record = build_evidence_record()
    if args.command == "write":
        _write(COHORT, cohort)
        _write(EVIDENCE, record)
        print(json.dumps({"cohort": str(COHORT), "evidence": str(EVIDENCE)}, indent=2))
        return
    committed = json.loads(EVIDENCE.read_text(encoding="utf-8"))
    if json.loads(COHORT.read_text(encoding="utf-8")) != cohort:
        raise ValueError("the committed cohort record does not reproduce")
    if committed != record:
        raise ValueError("the committed evidence bundle does not reproduce")
    print(json.dumps(record["evaluation_ledger"]["gates"], indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

"""Write the unified cross-engine perturbation receipt (SAP-6, #13795).

Run from the repository root:
``PYTHONPATH=.:src/engines/pendulum_models/python python3 -m
scripts.validation.generate_cross_engine_perturbation_receipt``
"""

from __future__ import annotations

import json
from pathlib import Path

from src.shared.python.pendulum_simulator.unified_perturbation_study import (
    run_unified_study,
)

RECEIPT = Path("docs/validation/cross_engine_perturbation_receipt.json")

if __name__ == "__main__":
    receipt = run_unified_study()
    RECEIPT.write_text(json.dumps(receipt, indent=2) + "\n", encoding="utf-8")
    print(f"wrote {RECEIPT} ({receipt['status']})")

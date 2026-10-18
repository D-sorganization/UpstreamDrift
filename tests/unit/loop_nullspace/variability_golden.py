"""The checked-in NSX variability golden (NSX-99, #12728) and its regeneration.

One file shared by both UIs: the PyQt tabs and the React panel read
``["plot_model"]`` (version 2.0.0, with the ``sensitivity`` panels) and neither
keeps a copy.  ``["report"]`` is the schema 2.0.0 report it was built from.

* File: ``tests/fixtures/loop_nullspace/goldens/report_v2_variability.json``
* Regenerate: ``python3 -m tests.unit.loop_nullspace.variability_golden --write``
* Check:      ``python3 -m tests.unit.loop_nullspace.variability_golden --check``

Source.  ``analyze()`` on the planar NSX-00 fixture with the pendulum variance
fixture of :mod:`tests.unit.loop_nullspace.sensitivity_support` (white torque
noise, additive, no signal law) and decisions from ``display_decision``
(research view).  The numbers demonstrate the mechanism; they are not a golfer
measurement.  Floats are rounded to ``DECIMALS`` places so solver noise cannot
make the comparison platform dependent.  The family-dependent arm-S1-free
point is withheld by the real decision (one allocation is not a family band),
so the withheld path is present without any override.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

from src.shared.python.loop_nullspace.facade import AnalysisOptions, analyze
from src.shared.python.loop_nullspace.plot_model import build_plot_model
from src.shared.python.loop_nullspace.plot_model_decisions import report_decisions
from tests.unit.loop_nullspace import sensitivity_support as support
from tests.unit.loop_nullspace.report_support import planar_trajectory

GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
    / "report_v2_variability.json"
)
PRESET = "envelope_only"
DECIMALS = 8
REGENERATE = "python3 -m tests.unit.loop_nullspace.variability_golden --write"


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, DECIMALS) + 0.0
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def fixture_report() -> dict[str, Any]:
    """The real ``analyze()`` report with the sensitivity stage, rounded."""
    options = AnalysisOptions(
        disturbance_spec=support.disturbance(), mechanisms=frozenset()
    )
    report = analyze(planar_trajectory(), support.sensitivity_engine(), PRESET, options)
    return _round(json.loads(json.dumps(report)))


def build_golden() -> dict[str, Any]:
    """The golden document: provenance block, the report and the plot model."""
    report = fixture_report()
    model = build_plot_model(report, report_decisions(report, "research_view"))
    return {
        "fixture": {
            "id": "report_v2_variability",
            "issue": "D-sorganization/UpstreamDrift#12728",
            "source": "analyze() on the planar NSX-00 fixture with the pendulum "
            "variance fixture, preset envelope_only; decisions from "
            "display_decision (research_view)",
            "report_schema_version": report["schema_version"],
            "decimals": DECIMALS,
            "regenerate": REGENERATE,
            "consumers": ["NSX-43 (PyQt tabs)", "NSX-44 (React panel)"],
        },
        "report": report,
        "plot_model": _round(model.to_dict()),
    }


def _render(doc: dict[str, Any]) -> str:
    return json.dumps(doc, indent=2, ensure_ascii=False) + "\n"


def main(argv: list[str] | None = None) -> int:
    """``--write`` regenerates the golden, ``--check`` fails when it is stale."""
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--write", action="store_true")
    mode.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = _render(build_golden())
    if args.write:
        GOLDEN_PATH.parent.mkdir(parents=True, exist_ok=True)
        GOLDEN_PATH.write_text(text, encoding="utf-8")
        return 0
    current = GOLDEN_PATH.read_text(encoding="utf-8") if GOLDEN_PATH.exists() else ""
    if current != text:
        sys.stderr.write(f"stale golden; run: {REGENERATE}\n")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

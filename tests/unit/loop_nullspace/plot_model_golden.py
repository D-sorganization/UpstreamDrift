"""The checked-in NSX plot-model golden (NSX-42, #12574) and its regeneration.

The golden is the cross-UI fixture: NSX-43 (PyQt) and NSX-44 (Vitest) read this
one file and render ``["plot_model"]``; neither keeps a copy.

* File: ``tests/fixtures/loop_nullspace/goldens/plot_model_v1.json``
* Regenerate: ``python3 -m tests.unit.loop_nullspace.plot_model_golden --write``
* Check:      ``python3 -m tests.unit.loop_nullspace.plot_model_golden --check``

Source.  The report is the real ``analyze()`` output of the planar NSX-00
fixture (preset ``envelope_only``, with the allocation), the decisions are the
real ``display_decision`` answers for it, and the model is
``build_plot_model``.  Four deterministic overrides make every renderer path
present in the one fixture; they are listed in ``fixture.overrides`` of the
file itself and are fixture-only -- they edit report *bounds* and one decision,
never a record, so they are not a claim about a swing:

1. ``tau_3`` lower bound set to +10 N m at the first two frames, sign class
   ``positive`` (``DETERMINED_SIGN`` segment);
2. ``tau_2`` upper bound ``UNBOUNDED`` at the third frame (``UNBOUNDED``
   segment; with the real NSX-41 decision an envelope unbounded in any frame is
   ``UNAVAILABLE`` and is a reason only, so this path is not reachable from a
   real report today);
3. a hand-split band ``s1_axial_pull_positive_squeeze_negative`` copied from
   ``tau_1`` scaled by 0.1, in newtons (the facade does not emit hand-split
   envelopes yet);
4. the ``tau_4`` decision replaced by ``WITHHOLD`` (the withheld path).

Floats are rounded to ``DECIMALS`` places so solver noise at 1e-12 cannot make
the comparison platform dependent.
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from src.shared.python.loop_nullspace.facade import AnalysisOptions, analyze
from src.shared.python.loop_nullspace.output_policy_integration import (
    DisplayAction,
    DisplayDecision,
)
from src.shared.python.loop_nullspace.plot_model import decision_key
from src.shared.python.loop_nullspace.plot_model_decisions import plot_model_payload
from tests.unit.loop_nullspace.report_support import planar_engine, planar_trajectory

GOLDEN_PATH = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "fixtures"
    / "loop_nullspace"
    / "goldens"
    / "plot_model_v1.json"
)
PRESET = "envelope_only"
SPLIT_NAME = "s1_axial_pull_positive_squeeze_negative"
DECIMALS = 8
REGENERATE = "python3 -m tests.unit.loop_nullspace.plot_model_golden --write"
OVERRIDES = (
    "tau_3 lower bound +10 N m at the first two frames, sign class positive",
    "tau_2 upper bound UNBOUNDED at the third frame",
    f"{SPLIT_NAME} band copied from tau_1 scaled by 0.1 (N)",
    "tau_4 decision replaced by WITHHOLD",
)


def _override_bounds(report: dict[str, Any]) -> None:
    bounds = report["envelopes"]["bounds"]
    tau3 = bounds["tau_3"]
    for k in (0, 1):
        assert tau3["upper"][k] > 10.0
        tau3["lower"][k] = 10.0
        tau3["sign_class"][k] = "positive"
    tau2 = bounds["tau_2"]
    tau2["upper"][2] = None
    tau2["status_upper"][2] = "UNBOUNDED"


def _add_hand_split(report: dict[str, Any]) -> None:
    env = report["envelopes"]
    split = copy.deepcopy(env["bounds"]["tau_1"])
    split["units"] = "N"
    for side in ("lower", "upper"):
        split[side] = [0.1 * v for v in split[side]]
    env["bounds"][SPLIT_NAME] = split
    record = copy.deepcopy(env["quantities"]["tau_1"])
    record["units"] = "N"
    record["values"] = [[0.1 * v for v in row] for row in record["values"]]
    env["quantities"][SPLIT_NAME] = record


def _round(obj: Any) -> Any:
    if isinstance(obj, float):
        return round(obj, DECIMALS) + 0.0
    if isinstance(obj, dict):
        return {k: _round(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_round(v) for v in obj]
    return obj


def fixture_report() -> dict[str, Any]:
    """The real planar report with the fixture-only bound overrides."""
    report = analyze(
        planar_trajectory(),
        planar_engine(),
        PRESET,
        AnalysisOptions(include_allocation=True),
    )
    report = _round(json.loads(json.dumps(report)))
    _override_bounds(report)
    _add_hand_split(report)
    return report


def build_golden() -> dict[str, Any]:
    """The golden document: provenance block plus the plot model."""
    report = fixture_report()
    withheld = {
        decision_key("envelopes", "tau_4"): DisplayDecision(
            "research_view",
            DisplayAction.WITHHOLD,
            "fixture: withheld to exercise the withheld path",
        )
    }
    model = plot_model_payload(report, decision_overrides=withheld)
    return {
        "fixture": {
            "id": "plot_model_v1",
            "issue": "D-sorganization/UpstreamDrift#12574",
            "source": "analyze() on the planar NSX-00 fixture, preset envelope_only, "
            "with allocation; decisions from display_decision (research_view)",
            "report_schema_version": report["schema_version"],
            "decimals": DECIMALS,
            "overrides": list(OVERRIDES),
            "regenerate": REGENERATE,
            "consumers": ["NSX-43 (PyQt tabs)", "NSX-44 (React panel)"],
        },
        "plot_model": _round(model),
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

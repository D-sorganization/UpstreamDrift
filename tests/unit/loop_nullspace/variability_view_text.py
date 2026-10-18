"""The text both Variability & Consistency views must show (NSX-100, #12729).

``expected_strings`` reads the raw JSON of a v2 plot model (no view code) and
lists every string a UI must show verbatim: decision labels, badges and chips,
withheld / unavailable reasons, point and arm labels, assumptions and
non-claims.  The PyQt6 and the Vitest tests both read the committed file and
assert each string is on screen, so the two UIs show the same text set.

* File: ``tests/fixtures/loop_nullspace/goldens/variability_view_text.json``
* Regenerate: ``python3 -m tests.unit.loop_nullspace.variability_view_text --write``
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
from pathlib import Path
from typing import Any

from src.shared.python.loop_nullspace.derivative_capability import default_registry
from src.shared.python.loop_nullspace.plot_model import build_plot_model
from src.shared.python.loop_nullspace.plot_model_decisions import report_decisions
from tests.unit.loop_nullspace.variability_golden import GOLDEN_PATH

TEXT_PATH = GOLDEN_PATH.with_name("variability_view_text.json")
REGENERATE = "python3 -m tests.unit.loop_nullspace.variability_view_text --write"
LIMITED_ENGINE = "simscape"


def _decision_strings(decision: dict[str, Any] | None) -> list[str]:
    if not decision:
        return []
    return [
        decision["label"],
        *decision["badges"],
        *decision["hypothesis_chips"],
        *decision["conditioning_chips"],
    ]


def _item_strings(item: dict[str, Any]) -> list[str]:
    out = _decision_strings(item.get("decision"))
    if not item.get("shown", True) and item.get("reason"):
        out.append(item["reason"])
    return out


def expected_strings(plot_model: dict[str, Any]) -> list[str]:
    """Every string the sensitivity view shows verbatim (order-stable, unique)."""
    sens = plot_model["sensitivity"]
    out: list[str] = [*sens["non_claims"], *sens["noise_hypotheses"]]
    out += sens["declared_mechanisms"]
    if sens["status"] != "complete":
        out.append(sens["reason"])
    heat, budget = sens["kernel_heatmap"], sens["variance_budget"]
    pareto, disp = sens["strategy_pareto"], sens["face_dispersion"]
    for panel in (heat, budget):
        out += _item_strings(panel)
    for point in pareto["points"]:
        out += [point["label"], *_item_strings(point)]
    for skipped in pareto["skipped"]:
        out += [skipped["label"], skipped["reason"]]
    out += pareto["assumptions"]
    for key in ("non_claim", "unavailable_reason", "winner"):
        if pareto[key]:
            out.append(pareto[key])
    if disp["unavailable_reason"]:
        out.append(disp["unavailable_reason"])
    entries = [disp["linear"], *disp["arms"], *disp["mc"]]
    for entry in entries:
        if entry:
            out += _item_strings(entry)
    return list(dict.fromkeys(s for s in out if s))


def limited_plot_model(golden: dict[str, Any]) -> dict[str, Any]:
    """The golden's report on an engine that cannot supply derivatives.

    The sensitivity stage is declared ``unavailable`` with the capability
    registry's own reason for ``LIMITED_ENGINE``; the plot model is built by the
    real ``build_plot_model``.
    """
    report = copy.deepcopy(golden["report"])
    reason = default_registry().declaration(LIMITED_ENGINE).reason
    report["sensitivity"] = {"status": "unavailable", "reason": reason}
    report["variance_budget"] = None
    report["strategies"] = None
    decisions = report_decisions(report, "research_view")
    model = build_plot_model(report, decisions).to_dict()
    return json.loads(json.dumps(model))


def build() -> dict[str, Any]:
    golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))
    limited = limited_plot_model(golden)
    return {
        "regenerate": REGENERATE,
        "complete": {"expected": expected_strings(golden["plot_model"])},
        "unavailable": {
            "plot_model": limited,
            "expected": expected_strings(limited),
        },
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--write", action="store_true")
    group.add_argument("--check", action="store_true")
    args = parser.parse_args(argv)
    text = json.dumps(build(), indent=1, sort_keys=True) + "\n"
    if args.write:
        TEXT_PATH.write_text(text, encoding="utf-8")
        return 0
    return 0 if TEXT_PATH.read_text(encoding="utf-8") == text else 1


if __name__ == "__main__":
    sys.exit(main())

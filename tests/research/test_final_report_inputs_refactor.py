"""Fixed synthetic report inputs preserve source identity and qualification order."""

import copy
import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.shot_pattern_analysis import final_report_inputs as module

pytestmark = pytest.mark.unit
REFERENCE = Path(__file__).with_name("fixtures") / "final_report_inputs_reference.json"


def _inputs():
    primary, entries = [], []
    for club in ("driver", "seven_iron", "pitching_wedge"):
        for curve in (1, 2):
            for rotation in (1, 2):
                for profile in ("shaft_only", "combined2"):
                    i = len(primary)
                    supplemental = club == "pitching_wedge" and curve == 2
                    name = f"{club}_lean0_curve{curve}_rotation{rotation}_{profile}"
                    interval = {
                        "estimate": i * 0.001,
                        "lower_95": i * 0.001 - 0.002,
                        "upper_95": i * 0.001 + 0.002,
                    }
                    cell = {
                        "cell": name,
                        "status": "completed",
                        "config": {
                            "club": {"club_id": club, "shaft_lean_deg": 0},
                            "curve_scale": curve,
                            "noise": {
                                "standard_deviations": [rotation, 0, 0, 0, 0, 0, 0]
                            },
                            "n_shots": 10000,
                            "match_face_bounds_deg": [-6.0, 6.0]
                            if supplemental
                            else None,
                            "match_speed_factors": [0.75, 1.25],
                        },
                        "summary": {
                            pattern: {"total_n": 10000, "supported_n": 9990 - i}
                            for pattern in ("Straight", "Draw", "Fade")
                        },
                        "paired_comparisons": {"paired_n": 9900 - i},
                        "course_scoring": {
                            "total_n": 10000,
                            "paired_n": 9900 - i,
                            "status": "conditional_only",
                            "paired_benefit_vs_straight": {
                                "Draw": interval,
                                "Fade": interval,
                            },
                        },
                        "source_manifest_sha256": hashlib.sha256(
                            name.encode()
                        ).hexdigest(),
                    }
                    primary.append(
                        {**cell, "status": "unavailable"} if supplemental else cell
                    )
                    if supplemental:
                        entries.append(
                            {
                                "cell": name,
                                "status": "completed",
                                "source_protocol": "pw-primary-face-bounds/1",
                                "summary": cell,
                            }
                        )
    return {"cells": primary}, {"report": {"unique_cells": entries}}


def _resolved_receipt():
    primary, supplement = _inputs()
    resolved = module.assemble_resolved_selected_cells(primary, supplement)
    cases = {}
    for mode in (
        "duplicate_primary",
        "duplicate_supplement",
        "missing_primary",
        "missing_supplement",
        "primary_support",
        "primary_paired",
        "supplement_support",
        "supplement_paired",
        "supplement_incomplete",
        "invalid_status",
        "protocol",
    ):
        p, s = copy.deepcopy((primary, supplement))
        if mode == "duplicate_primary":
            p["cells"].append(p["cells"][0])
        elif mode == "duplicate_supplement":
            s["report"]["unique_cells"].append(s["report"]["unique_cells"][0])
        elif mode == "missing_primary":
            p["cells"].pop(0)
        elif mode == "missing_supplement":
            s["report"]["unique_cells"].pop(0)
        elif mode == "primary_support":
            p["cells"][0]["summary"]["Straight"].pop("supported_n")
        elif mode == "primary_paired":
            p["cells"][0]["paired_comparisons"].pop("paired_n")
        elif mode == "supplement_support":
            s["report"]["unique_cells"][0]["summary"]["summary"]["Draw"].pop(
                "supported_n"
            )
        elif mode == "supplement_paired":
            s["report"]["unique_cells"][0]["summary"]["paired_comparisons"].pop(
                "paired_n"
            )
        elif mode == "supplement_incomplete":
            s["report"]["unique_cells"][0]["status"] = "unavailable"
        elif mode == "invalid_status":
            p["cells"][0]["status"] = "not_run"
        else:
            s["report"]["unique_cells"][0]["source_protocol"] = "unreviewed/1"
        with pytest.raises(ValueError) as caught:
            module.assemble_resolved_selected_cells(p, s)
        cases[mode] = str(caught.value)
    return {"resolved": resolved, "errors": cases}


def _qualified_receipt(root, monkeypatch, mode):
    from src.tools.shot_pattern_analysis import matched_interpretation

    primary, supplement = _inputs()
    for original in list(primary["cells"]):
        if (
            original["config"]["curve_scale"] == 1
            and original["config"]["noise"]["standard_deviations"][0] == 2
            and original["cell"].endswith("combined2")
        ):
            for lean in (5, 10):
                cell = copy.deepcopy(original)
                cell["cell"] = cell["cell"].replace("lean0", f"lean{lean}")
                cell["config"]["club"]["shaft_lean_deg"] = lean
                primary["cells"].append(cell)
    primary["cells"].reverse()
    supplement.update(
        comparisons={"schema": "matched-supplement-comparisons/1", "comparisons": []},
        figure_ledger={"schema": "matched-supplement-figure-ledger/1", "figures": []},
    )
    supplement["report"].update(
        schema="matched-supplement-report/1",
        unique_executed_cells=26,
        aliases=[{"proof_status": "verified"}, {"proof_status": "unavailable"}],
    )
    supplement_dir = root / "supplement"
    supplement_dir.mkdir()
    receipt = supplement_dir / "REPORT_SUPPLEMENT_PIPELINE.json"
    receipt.write_text("synthetic receipt")
    digest = hashlib.sha256(receipt.read_bytes()).hexdigest()
    if mode in ("explicit_identity", "identity_changed"):
        supplement["publication_identity"] = {
            "kind": "synthetic",
            "receipt_filename": receipt.name,
            "receipt_sha256": digest if mode == "explicit_identity" else "0" * 64,
            "parent_receipt_sha256": "a" * 64,
        }
    if mode == "missing_condition":
        primary["cells"].pop(0)
    if mode == "comparison_schema":
        supplement["comparisons"]["schema"] = "bad"
    if mode == "ledger_schema":
        supplement["figure_ledger"]["schema"] = "bad"
    if mode == "inventory_count":
        supplement["report"]["unique_executed_cells"] = 25
    events = []

    def description(path):
        events.append(["describe", path.name])
        return {"cell": path.name}

    def score(cell):
        events.append(["score", cell["cell"]])
        return {"n": cell["course_scoring"]["paired_n"]}

    monkeypatch.setattr(matched_interpretation, "describe_saved_cell", description)
    monkeypatch.setattr(matched_interpretation, "describe_scoring", score)

    def stub(name, value):
        def call(*args, **kwargs):
            events.append([name, kwargs.get("protocol")])
            return copy.deepcopy(value)

        monkeypatch.setattr(module, name, call)

    stub("select_representative_interaction_pairs", [{"effect": 0.125}])
    stub("select_bounds_control_pairs", ([{"effect": -0.25}], ["missing control"]))
    stub("verified_original_figure_links", [{"relative_path": "synthetic.png"}])
    stub("verified_original_figure_inventory", "availability.csv")
    stub("supplement_figure_links", [{"relative_path": "supplement.png"}])
    try:
        result = module.assemble_qualified_interpretation(
            root / "primary",
            primary,
            root / "control",
            {},
            supplement,
            assets=module.InterpretationAssets(
                root / "primary_figures", root / "control_figures", supplement_dir, root
            ),
        )
        error = None
    except ValueError as exc:
        result, error = None, str(exc)
    return {"result": result, "error": error, "events": events}


def test_resolved_values_order_protocols_and_errors_remain_exact():
    assert _resolved_receipt() == json.loads(REFERENCE.read_text())["resolved"]


@pytest.mark.parametrize(
    "mode",
    [
        "pipeline",
        "explicit_identity",
        "identity_changed",
        "missing_condition",
        "comparison_schema",
        "ledger_schema",
        "inventory_count",
    ],
)
def test_qualified_inputs_and_failure_order_remain_exact(tmp_path, monkeypatch, mode):
    actual = _qualified_receipt(tmp_path, monkeypatch, mode)
    assert json.dumps(actual, allow_nan=False) == json.dumps(
        json.loads(REFERENCE.read_text())["qualified"][mode], allow_nan=False
    )


def test_asset_bundle_retains_explicit_directories(tmp_path):
    assets = module.InterpretationAssets(
        tmp_path / "p", tmp_path / "c", tmp_path / "s", tmp_path
    )
    assert (
        assets.primary_figures,
        assets.control_figures,
        assets.supplement_dir,
        assets.report_root,
    ) == (tmp_path / "p", tmp_path / "c", tmp_path / "s", tmp_path)


def test_future_report_inventory_binds_interpretation_helpers():
    from scripts.research.shot_pattern_analysis.build_final_report import (
        _report_source_hashes,
    )

    key = "scripts/research/shot_pattern_analysis/final_report_inputs.py"
    assert (
        _report_source_hashes()[key]
        == hashlib.sha256(Path(module.__file__).read_bytes()).hexdigest()
    )

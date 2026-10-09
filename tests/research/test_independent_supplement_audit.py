"""Fail-closed protocol checks for the separate bounds supplement auditor."""

from copy import deepcopy
import json
from pathlib import Path
import pytest

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/shot_pattern_analysis/phase2"


def _api():
    from docs.research.shot_pattern_analysis.phase2.validation import (
        supplement_audit_protocol as api,
    )

    return api


def _inputs():
    inventory = json.loads(
        (
            EVIDENCE / "review_followups/SUPPLEMENTAL_EXECUTION_INVENTORY.json"
        ).read_text()
    )
    original = {
        "cells": {
            row["cell"]: {
                "club": {
                    "club_speed_mps": 45.0
                    if row["cell"].startswith("driver")
                    else 32.0,
                    "cor": 0.83,
                },
                "noise": {"standard_deviations": [1, 0, 0, 0, 0, 0, 0]},
                "seed": 20261009,
                "n_shots": 10000,
                "target_x_m": None,
                "match_tolerance_m": 0.05,
            }
            for row in inventory["entries"]
        }
    }
    return inventory, original


@pytest.fixture(autouse=True)
def reviewed_inputs(monkeypatch):
    monkeypatch.setattr(_api(), "_load_review_inputs", lambda evidence: _inputs())


def _plan(stage="pilot", protocol="pw-primary-face-bounds/1"):
    inventory, original = _inputs()
    cells = {}
    for entry in inventory["entries"]:
        if entry["protocol"] != _api().PROTOCOLS[protocol]:
            continue
        config = deepcopy(original["cells"][entry["cell"]])
        config.update(
            n_shots=32 if stage == "pilot" else 10000,
            match_face_bounds_deg=[-6, 6],
            match_speed_factors=[
                value / config["club"]["club_speed_mps"]
                for value in entry["speed_bounds_mps"]
            ],
            target_x_m=entry["target_x_m"],
        )
        cells[entry["cell"]] = config
    return {
        "stage": stage,
        "supplement_protocol": protocol,
        "selected_cell_names": list(cells),
        "cells": cells,
    }


@pytest.mark.parametrize(
    "protocol",
    [
        "pw-primary-face-bounds/1",
        "pw-control-face-bounds/1",
        "driver-control-speed-bounds/1",
    ],
)
def test_exact_supplement_plan_and_explicit_stage(protocol):
    api = _api()
    for stage in ("pilot", "final"):
        receipt = api.validate_supplement_plan(_plan(stage, protocol), stage, EVIDENCE)
        assert receipt["unique_cells"] == (
            18 if protocol == "pw-primary-face-bounds/1" else 4
        )
    with pytest.raises(ValueError):
        api.validate_supplement_plan(_plan(), "final", EVIDENCE)


@pytest.mark.parametrize(
    "mutation",
    [
        "missing",
        "extra",
        "duplicate",
        "face",
        "speed",
        "target",
        "seed",
        "noise",
        "count",
        "club",
    ],
)
def test_changed_protocol_is_rejected(mutation):
    api = _api()
    plan = _plan()
    key = next(iter(plan["cells"]))
    config = plan["cells"][key]
    if mutation == "missing":
        plan["cells"].pop(key)
    elif mutation == "extra":
        plan["cells"]["extra"] = deepcopy(config)
    elif mutation == "duplicate":
        plan["selected_cell_names"].append(key)
    elif mutation == "face":
        config["match_face_bounds_deg"] = [-7, 6]
    elif mutation == "speed":
        config["match_speed_factors"] = [0.75, 1.3]
    elif mutation == "target":
        config["target_x_m"] = 77.0
    elif mutation == "seed":
        config["seed"] += 1
    elif mutation == "noise":
        config["noise"]["standard_deviations"][1] += 0.1
    elif mutation == "count":
        config["n_shots"] = 31
    elif mutation == "club":
        config["club"]["cor"] = 0.8
    with pytest.raises(ValueError):
        api.validate_supplement_plan(plan, "pilot", EVIDENCE)


def test_duplicate_json_keys_rejected(tmp_path):
    path = tmp_path / "plan.json"
    path.write_text('{"cells":{},"cells":{}}')
    with pytest.raises(ValueError):
        _api().load_json_strict(path)


def _aliases(tmp_path):
    import hashlib

    plan = _plan()
    root = tmp_path / "primary"
    root.mkdir()

    def save(path, value):
        path.write_text(json.dumps(value))

    def sha(path):
        return hashlib.sha256(path.read_bytes()).hexdigest()

    save(root / "matrix_plan.json", plan)
    inventory, _ = _inputs()
    refs = []
    for name in inventory["control_aliases_to_primary"]:
        folder = root / name
        folder.mkdir()
        config = plan["cells"][name]
        summary = {
            "config": config,
            "target_x_m": 77.40247724866299,
            "nominals": {
                p: {"outcome": {"carry_x_m": 77.40247724866299, "carry_y_m": 0.0}}
                for p in ("Straight", "Draw", "Fade")
            },
        }
        save(folder / "summary.json", summary)
        (folder / "shots.csv.gz").write_bytes(b"saved input/output bytes")
        save(
            folder / "manifest.json",
            {
                "files_sha256": {
                    f: sha(folder / f) for f in ("summary.json", "shots.csv.gz")
                }
            },
        )
        request = deepcopy(config)
        request["target_x_m"] = summary["target_x_m"]
        refs.append(
            {
                "protocol_id": "supplement_control_alias",
                "alias_cell": name,
                "alias_of_primary": name,
                "execution_kind": "provenance_alias",
                "source_root": str(root),
                "source_plan_sha256": sha(root / "matrix_plan.json"),
                "source_cell_manifest_sha256": sha(folder / "manifest.json"),
                "source_csv_sha256": sha(folder / "shots.csv.gz"),
                "requested_config": request,
            }
        )
    envelope = {"schema": "matched-study-control-aliases/1", "references": refs}
    return envelope


@pytest.mark.parametrize(
    "mutation", [None, "duplicate", "independent", "request", "csv", "target"]
)
def test_aliases_are_exact_verified_references(tmp_path, mutation):
    api = _api()
    envelope = _aliases(tmp_path)
    first = envelope["references"][0]
    if mutation == "duplicate":
        envelope["references"][1] = deepcopy(first)
    elif mutation == "independent":
        first["receipt"] = {"execution": "invented"}
    elif mutation == "request":
        first["requested_config"]["noise"]["standard_deviations"][1] = 0.7
    elif mutation == "csv":
        (
            Path(first["source_root"]) / first["alias_of_primary"] / "shots.csv.gz"
        ).write_bytes(b"changed")
    elif mutation == "target":
        first["requested_config"]["target_x_m"] += 1
    path = tmp_path / "aliases.json"
    path.write_text(json.dumps(envelope))
    if mutation is None:
        assert (
            api.validate_control_aliases(path, "pilot", EVIDENCE)["verified_aliases"]
            == 4
        )
    else:
        with pytest.raises(ValueError):
            api.validate_control_aliases(path, "pilot", EVIDENCE)


def _acceptance(tmp_path):
    import hashlib

    execution = {"source_sha256": {"reviewed": "same"}}
    protocols = {}
    evidence = {}
    for protocol in _api().PROTOCOLS:
        final = _plan("final", protocol)
        pilot = _plan("pilot", protocol)
        pilot["execution"] = execution
        name = protocol.replace("/", "_") + ".json"
        path = tmp_path / name
        path.write_text(json.dumps(pilot))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        protocols[protocol] = {
            "final_cells": final["cells"],
            "pilot_plan": name,
            "pilot_plan_sha256": digest,
        }
        evidence[name] = digest
    for name in ("review.txt", "numerics.txt"):
        path = tmp_path / name
        path.write_text("fixture reviewed evidence")
        evidence[name] = hashlib.sha256(path.read_bytes()).hexdigest()
    receipt = {
        "status": "accepted_for_conditional_simulation",
        "supplement_stage": "final",
        "execution": execution,
        "checks": dict.fromkeys(("pilot", "numerical_refinement", "independent_review", "rendered_figures", "software_tests"), True),
        "evidence_files": evidence,
        "supplement_protocols": protocols,
    }
    plan = _plan("final")
    plan["execution"] = execution
    return receipt, plan


@pytest.mark.parametrize(
    "mutation", [None, "source", "missing_protocol", "pilot_config", "gate"]
)
def test_final_audit_requires_source_and_protocol_bound_acceptance(tmp_path, mutation):
    import hashlib

    api = _api()
    receipt, plan = _acceptance(tmp_path)
    if mutation == "source":
        receipt["execution"] = {"source_sha256": {"reviewed": "stale"}}
    elif mutation == "missing_protocol":
        receipt["supplement_protocols"].pop("driver-control-speed-bounds/1")
    elif mutation == "gate":
        receipt["checks"]["independent_review"] = False
    elif mutation == "pilot_config":
        item = receipt["supplement_protocols"]["pw-primary-face-bounds/1"]
        path = tmp_path / item["pilot_plan"]
        pilot = json.loads(path.read_text())
        next(iter(pilot["cells"].values()))["seed"] += 1
        path.write_text(json.dumps(pilot))
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        item["pilot_plan_sha256"] = digest
        receipt["evidence_files"][item["pilot_plan"]] = digest
    path = tmp_path / "acceptance.json"
    path.write_text(json.dumps(receipt))
    if mutation is None:
        assert (
            api.validate_execution_acceptance(plan, path, EVIDENCE)[
                "verified_protocols"
            ]
            == 3
        )
    else:
        with pytest.raises(ValueError):
            api.validate_execution_acceptance(plan, path, EVIDENCE)

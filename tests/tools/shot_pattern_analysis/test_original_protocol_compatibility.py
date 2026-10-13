"""Historical protocol defaults must not widen the accepted experiment."""

from copy import deepcopy
from dataclasses import asdict
import hashlib
import json

import pytest

from src.tools.shot_pattern_analysis.matched_matrix import matched_matrix_cells
from src.tools.shot_pattern_analysis.matched_report import (
    load_reviewed_common_targets,
    validate_original_matrix_protocol,
)

pytestmark = pytest.mark.unit
_TARGETS = {
    "driver": 178.94875782337786,
    "seven_iron": 127.88194166336235,
    "pitching_wedge": 77.40247724866299,
}


def _historical_cells(protocol, n_shots):
    control = protocol == "original_control"
    cells = {
        name: json.loads(json.dumps(asdict(config)))
        for name, config in matched_matrix_cells(
            n_shots=n_shots,
            seed=20261009,
            common_targets=_TARGETS if control else None,
            profiles=("shaft_only", "combined2") if control else None,
        )
    }
    for config in cells.values():
        config.pop("match_face_bounds_deg", None)
        config.pop("match_speed_factors", None)
    return cells


@pytest.mark.parametrize("protocol", ["original_primary", "original_control"])
def test_original_final_protocol_accepts_only_missing_historical_bounds(protocol):
    cells = _historical_cells(protocol, 10_000)
    report = {
        "cells": [{"cell": name, "config": config} for name, config in cells.items()]
    }
    before = deepcopy(report)
    validate_original_matrix_protocol(
        report,
        protocol=protocol,
        common_targets=_TARGETS if protocol == "original_control" else None,
    )
    assert report == before


@pytest.mark.parametrize(
    "field,value",
    [
        ("match_face_bounds_deg", [-6.0, 6.0]),
        ("match_speed_factors", [0.75, 1.30]),
        ("seed", 20261010),
        ("match_tolerance_m", 0.1),
    ],
)
def test_original_protocol_rejects_explicit_bounds_or_other_drift(field, value):
    cells = _historical_cells("original_control", 10_000)
    next(iter(cells.values()))[field] = value
    report = {
        "cells": [{"cell": name, "config": config} for name, config in cells.items()]
    }
    with pytest.raises(ValueError, match="reviewed protocol"):
        validate_original_matrix_protocol(
            report, protocol="original_control", common_targets=_TARGETS
        )


def test_hash_bound_historical_pilot_accepts_defaults_without_rewriting_evidence(
    tmp_path,
):
    pilot = tmp_path / "control_pilot_v1/matrix_plan.json"
    pilot.parent.mkdir()
    pilot.write_text(
        json.dumps(
            {"stage": "pilot", "cells": _historical_cells("original_control", 32)}
        )
    )
    audit = tmp_path / "validation/independent_control_pilot_audit.json"
    audit.parent.mkdir()
    audit.write_text(
        json.dumps(
            {
                "pass": True,
                "failures": [],
                "plan_sha256": hashlib.sha256(pilot.read_bytes()).hexdigest(),
            }
        )
    )
    acceptance = tmp_path / "FINAL_RUN_ACCEPTANCE.md"
    acceptance.write_text(
        "validation/independent_control_pilot_audit.json: "
        + hashlib.sha256(audit.read_bytes()).hexdigest()
    )
    targets = tmp_path / "common_targets.json"
    targets.write_text(json.dumps(_TARGETS))
    before = {path: path.read_bytes() for path in (pilot, audit, acceptance, targets)}
    loaded, provenance = load_reviewed_common_targets(tmp_path)
    assert loaded == _TARGETS
    assert provenance["pilot_plan_sha256"] == hashlib.sha256(before[pilot]).hexdigest()
    assert all(path.read_bytes() == value for path, value in before.items())

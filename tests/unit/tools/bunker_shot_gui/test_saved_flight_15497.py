"""Saved flight traces must be coherent before a view can consume them (#15497)."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import pytest

from src.tools.bunker_shot_gui.export import (
    ExportIntegrityError,
    load_workbench_evaluation,
    save_workbench_evaluation,
)

pytestmark = pytest.mark.unit


def _record() -> dict[str, Any]:
    return {
        "schema_version": "bunker-workbench-evaluation/1",
        "predictive": False,
        "refused": False,
        "verdict": {"status": "beyond_validation"},
        "carry": {"carry_m": 5.0, "verdict_status": "beyond_validation"},
        "flight": {
            "flight_m": [[0, 0, 0], [1, 1, 2], [3, 4, 0]],
            "rollout_m": [[3, 4, 0], [4, 5, 0.1], [5, 6, 0]],
            "carry_m": 5.0,
            "assumption": "Uncalibrated launch; assumed flat green",
        },
    }


def _save(tmp_path: Path, record: dict[str, Any]) -> Path:
    # Save computes a valid digest, so failures below concern semantic validity.
    return save_workbench_evaluation(record, tmp_path / "flight.json")


def test_valid_saved_trace_round_trips_without_losing_assumptions(
    tmp_path: Path,
) -> None:
    record = _record()
    path = _save(tmp_path, record)
    before = path.read_bytes()
    loaded = load_workbench_evaluation(path)
    assert loaded["flight"] == record["flight"]
    assert loaded["carry"] == record["carry"]
    assert path.read_bytes() == before


@pytest.mark.parametrize("legacy", [True, False])
def test_legacy_or_unavailable_trace_stays_loadable(
    tmp_path: Path, legacy: bool
) -> None:
    record = _record()
    record["refused"] = True
    record["carry"] = None
    record["verdict"]["status"] = "refused"
    if legacy:
        record.pop("flight")
    else:
        record["flight"] = None
    assert load_workbench_evaluation(_save(tmp_path, record))["refused"] is True


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("refused", True),
        ("refused", "false"),
        ("verdict", {"status": "refused"}),
        ("verdict", None),
        ("verdict", {"status": []}),
        ("verdict", {"status": "unknown"}),
        ("carry", None),
        ("carry", {"verdict_status": "beyond_validation"}),
        ("carry", {"carry_m": True, "verdict_status": "beyond_validation"}),
        ("carry", {"carry_m": 6.0, "verdict_status": "beyond_validation"}),
        ("carry", {"carry_m": 5.0, "verdict_status": "refused"}),
    ],
)
def test_refused_or_unqualified_saved_flight_is_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    record = _record()
    record[field] = value
    with pytest.raises(ExportIntegrityError, match="flight"):
        load_workbench_evaluation(_save(tmp_path, record))


@pytest.mark.parametrize(
    ("field", "value"),
    [
        ("flight_m", []),
        ("flight_m", [[0, 0, 0], [3, 4, 1]]),
        ("flight_m", [[0, 0], [3, 4, 0]]),
        ("flight_m", [[0, 0, -1], [3, 4, 0]]),
        ("flight_m", [[False, 0, 0], [3, 4, 0]]),
        ("flight_m", [["0", 0, 0], [3, 4, 0]]),
        ("flight_m", None),
        ("flight_m", [None, [3, 4, 0]]),
        ("rollout_m", [[0, 0, 0]]),
        ("rollout_m", [[3, 4, 0], [5, 6, 1]]),
        ("carry_m", 6.0),
        ("carry_m", "5.0"),
        ("assumption", ""),
        ("assumption", "   "),
        ("assumption", None),
    ],
)
def test_incoherent_saved_trace_is_rejected(
    tmp_path: Path, field: str, value: Any
) -> None:
    record = _record()
    record["flight"][field] = value
    with pytest.raises(ExportIntegrityError, match="flight"):
        load_workbench_evaluation(_save(tmp_path, record))


@pytest.mark.parametrize("value", [[], "trace", 0, False, {}])
def test_malformed_flight_object_is_rejected(tmp_path: Path, value: Any) -> None:
    record = _record()
    record["flight"] = value
    with pytest.raises(ExportIntegrityError, match="flight"):
        load_workbench_evaluation(_save(tmp_path, record))


def test_digest_still_covers_trace_points(tmp_path: Path) -> None:
    record = _record()
    path = _save(tmp_path, record)
    tampered = json.loads(path.read_text(encoding="utf-8"))
    tampered["flight"]["flight_m"][1][2] += 1
    path.write_text(json.dumps(tampered), encoding="utf-8")
    with pytest.raises(ExportIntegrityError, match="digest"):
        load_workbench_evaluation(path)


def test_loading_does_not_mutate_record_data(tmp_path: Path) -> None:
    record = _record()
    original = copy.deepcopy(record)
    load_workbench_evaluation(_save(tmp_path, record))
    assert record == original

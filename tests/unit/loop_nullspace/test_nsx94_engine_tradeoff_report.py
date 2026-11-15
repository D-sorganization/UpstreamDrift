"""NSX-94 (#12723): committed cross-engine co-contraction trade-off report.

The evidence file is produced offline when MuJoCo and OpenSim are available;
default CI validates the schema and harness invariants without those engines.
"""

from __future__ import annotations

import copy
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.nsx94_tradeoff_report import (
    DEFAULT_EVIDENCE_PATH,
    build_tradeoff_report,
    load_tradeoff_report,
    validate_tradeoff_report,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]


def test_nsx94_cocontraction_tradeoff_reported_for_muscle_engines() -> None:
    """Acceptance: MyoSuite and OpenSim trade-off curves in the evidence bundle."""
    assert DEFAULT_EVIDENCE_PATH.is_file(), (
        f"missing evidence artifact: {DEFAULT_EVIDENCE_PATH.relative_to(ROOT)}"
    )
    doc = load_tradeoff_report()
    validate_tradeoff_report(doc)
    engines = {entry["engine"] for entry in doc["entries"]}
    assert engines == {"opensim", "myosuite"}
    assert doc["validation_status"] == "UNVALIDATED"
    for entry in doc["entries"]:
        assert entry["sweep_status"] == "COMPUTED"
        assert len(entry["levels"]) >= 3
        noise = np.asarray(entry["joint_torque_noise_variance_N2m2"], dtype=np.float64)
        stiffness = np.asarray(
            entry["joint_stiffness_trace_Nm_per_rad"], dtype=np.float64
        )
        assert noise.shape == stiffness.shape == (len(entry["levels"]),)
        assert np.all(np.isfinite(noise)) and np.all(np.isfinite(stiffness))
        assert entry["noise_monotone_increasing"]
        assert entry["stiffness_monotone_increasing"]
        assert float(entry["torque_residual_max"]) < 1e-12


def test_harness_rejects_missing_engine() -> None:
    doc = load_tradeoff_report()
    bad = copy.deepcopy(doc)
    bad["entries"] = [e for e in bad["entries"] if e["engine"] != "myosuite"]
    with pytest.raises(ValueError, match="myosuite"):
        validate_tradeoff_report(bad)


def test_harness_rejects_not_qualified_entries() -> None:
    doc = load_tradeoff_report()
    bad = copy.deepcopy(doc)
    bad["entries"].append(
        {
            "engine": "fixture",
            "sweep_status": "NOT_QUALIFIED",
            "levels": [0.0, 1.0, 2.0],
            "joint_torque_noise_variance_N2m2": [1.0, 2.0, 3.0],
            "joint_stiffness_trace_Nm_per_rad": [1.0, 2.0, 3.0],
        }
    )
    with pytest.raises(ValueError, match="NOT_QUALIFIED"):
        validate_tradeoff_report(bad)


def test_harness_rejects_non_monotone_trace() -> None:
    doc = load_tradeoff_report()
    bad = copy.deepcopy(doc)
    bad["entries"][0]["noise_monotone_increasing"] = False
    with pytest.raises(ValueError, match="monotone"):
        validate_tradeoff_report(bad)


@pytest.mark.requires_opensim
def test_regenerated_opensim_entry_matches_schema() -> None:
    from tests.helpers.opensim_support import require_opensim

    require_opensim()
    doc = build_tradeoff_report(engines=("opensim",))
    assert len(doc["entries"]) == 1
    validate_tradeoff_report(
        {
            **doc,
            "entries": doc["entries"]
            + [
                e
                for e in load_tradeoff_report()["entries"]
                if e["engine"] == "myosuite"
            ],
        }
    )


@pytest.mark.requires_mujoco
def test_regenerated_myosuite_entry_matches_schema() -> None:
    pytest.importorskip("mujoco")
    doc = build_tradeoff_report(engines=("myosuite",))
    assert len(doc["entries"]) == 1
    validate_tradeoff_report(
        {
            **doc,
            "entries": doc["entries"]
            + [
                e for e in load_tradeoff_report()["entries"] if e["engine"] == "opensim"
            ],
        }
    )

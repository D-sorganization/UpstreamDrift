"""Evidence gates for the committed articulated LHS uncertainty record."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
RECORD = DATA / "articulated_uncertainty_study.json"
ARRAYS = DATA / "articulated_uncertainty_study.npz"
pytestmark = pytest.mark.scientific

EXPECTED_PARAMETERS = (
    "height_scale",
    "body_mass_scale",
    "joint_limit_scale",
    "grip_stiffness_n_m",
    "grip_damping_n_s_m",
    "friction_coefficient",
    "club_mass_kg",
    "club_moi_scale",
    "initial_velocity_m_s",
)


def _record() -> dict[str, object]:
    return json.loads(RECORD.read_text(encoding="utf-8"))


def test_committed_lhs_record_matches_registered_design() -> None:
    record = _record()

    assert record["schema_version"] == "articulated-uncertainty-study/v2"
    assert record["design"]["method"] == "deterministic_latin_hypercube"
    assert record["configuration"]["sample_count"] == 40
    assert tuple(record["uncertainty_parameters"]) == EXPECTED_PARAMETERS
    assert record["design"]["parameter_count"] == len(EXPECTED_PARAMETERS)
    assert record["results"]["sample_count"] == 40
    assert sum(record["results"]["failure_distribution"].values()) == 40


def test_committed_lhs_arrays_retain_every_sample_and_status() -> None:
    record = _record()
    with np.load(ARRAYS) as arrays:
        assert tuple(arrays["parameter_names"].tolist()) == EXPECTED_PARAMETERS
        assert arrays["parameter_samples"].shape == (40, len(EXPECTED_PARAMETERS))
        assert arrays["response_matrix"].shape == (40, 5)
        assert arrays["failure_classes"].shape == (40,)
        included = np.asarray(arrays["analysis_included"], dtype=bool)
        assert included.shape == (40,)
        assert (
            int(np.count_nonzero(included))
            == record["results"]["analysis_included_count"]
        )
        classes, counts = np.unique(arrays["failure_classes"], return_counts=True)
        observed = {
            str(name): int(count) for name, count in zip(classes, counts, strict=True)
        }
        assert observed == record["results"]["failure_distribution"]


def test_committed_lhs_record_binds_sources_and_inference_boundaries() -> None:
    record = _record()

    for relative, expected in record["source_sha256"].items():
        observed = hashlib.sha256((ROOT / relative).read_bytes()).hexdigest()
        assert observed == expected, relative
    limitations = record["limitations"]
    assert "does not propagate" in limitations["headline_estimands"]
    assert "not measured participant" in limitations["calibration"]
    assert "does not support" in limitations["human_inference"]
    assert "exploratory" in limitations["prcc"]

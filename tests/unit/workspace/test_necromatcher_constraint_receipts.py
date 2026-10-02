"""Refit receipts preserve explicit constraint options and finite probe evidence."""

from dataclasses import asdict
import json

import numpy as np
import pytest

from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.historical_fit import (
    ImageFitConfig,
    ImageFitResult,
)
from src.shared.python.workspace.necromatcher_fit_worker import _build_fit_payload

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("constraints", [False, True])
def test_refit_receipt_decodes_json_and_keeps_probe_evidence(constraints):
    options = ConstraintOptions(GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1)
    config = ImageFitConfig(
        constraint_options=options if constraints else None,
        interior_fractions=(0.5,) if constraints else (),
    )
    times = np.array([0.0, 0.5, 1.0]) if constraints else np.empty(0)
    rows = np.array([[0.1], [0.7], [0.2]]) if constraints else np.empty((0, 0))
    labels = ("ground:left",) if constraints else ()
    result = ImageFitResult(
        np.array([0.0, 1.0]),
        np.zeros((2, 1)),
        2,
        3,
        np.zeros((2, 1)),
        2,
        "model",
        ("angle",),
        True,
        "converged",
        np.array([0.0, 1.0]),
        np.zeros(4),
        ("angle",),
        times,
        rows,
        labels,
    )
    request = {
        "source_fit_id": "parent",
        "source_fit_hash": "hash",
        "execution_stamp": {},
        "options": {
            "frame_indices": [0, 1],
            "config": json.loads(json.dumps(asdict(config))),
        },
    }
    source = {
        "provenance": {"native_definition": {}},
        "evidence": {"original_fit": {"camera": {}, "attachments": {}}},
    }
    stamp = dict.fromkeys(("started_at_utc", "source_sha256", "runtime_sha256"), "test")
    output = _build_fit_payload(
        request, source, result, ((0, 1), [{}, {}], result.q), stamp, 0
    )
    evidence = output["evidence"]["original_fit"]
    decoded = ImageFitConfig.from_record(json.loads(json.dumps(evidence["config"])))
    assert decoded == config
    assert evidence["constraint_assessment"] == {
        "tested_times": times.tolist(),
        "row_labels": list(labels),
        "scaled_residuals": rows.tolist(),
        "maximum_dimensionless_residual": 0.7 if constraints else 0.0,
        "continuous_certified": False,
    }
    assert "anatomical_ranges_not_enforced" in output["evidence"]["rejection_reasons"]
    assert output["q"] == result.q.tolist()

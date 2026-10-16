"""NSX-89 (#12718): Simscape linearization payload gate and NONE capability.

No payload here comes from MATLAB: the documents are synthetic dictionaries
that exercise the loader's refusal rules.  Real acceptance is the deferred
validation record DV-12718 (a MATLAB R2025b per-frame ``linearize`` export).
"""

from __future__ import annotations

import copy
from typing import Any

import numpy as np
import pytest

from src.shared.python.loop_nullspace import derivative_capability as dc
from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.simscape_linearization import (
    LINEARIZATION_SCHEMA_ID,
    load_simscape_linearization_doc,
    parse_simscape_linearization,
)

pytestmark = pytest.mark.unit

SHA = "b" * 64


def _doc(release: str = "R2025b", scope: str = "golf_model") -> dict[str, Any]:
    return {
        "schema": LINEARIZATION_SCHEMA_ID,
        "provenance": {
            "provenance_kind": "matlab_export",
            "model_scope": scope,
            "matlab_release": release,
            "matlab_version": f"25.2.0.1 ({release}) Update 5",
            "model_file": "GS3DX_probe.slx",
            "model_sha256": SHA,
            "exported_at": "2026-10-15T00:00:00Z",
        },
        "solver_settings": {"solver": "ode15s", "rel_tol": 1e-8, "abs_tol": 1e-10},
        "state_mapping": {
            "q_names": ["a", "b"],
            "v_names": ["da", "db"],
            "input_names": ["tau_a"],
        },
        "frames": [
            {
                "q": [0.0, 0.1],
                "v": [0.0, 0.0],
                "tau": [0.0],
                "A": np.eye(4).tolist(),
                "B": np.zeros((4, 1)).tolist(),
            }
        ],
    }


def test_accepts_golf_model_r2025b_payload() -> None:
    lin = parse_simscape_linearization(_doc())
    assert len(lin.frames) == 1
    assert lin.frames[0].A.shape == (4, 4)
    assert lin.provenance.matlab_release == "R2025b"


@pytest.mark.parametrize("release", ["R2026a", "R2024b", "R2025a"])
def test_refuses_mismatched_release(release: str) -> None:
    with pytest.raises(ValueError, match="R2025b"):
        parse_simscape_linearization(_doc(release=release))


def test_refuses_verification_rig_scope() -> None:
    with pytest.raises(ValueError, match="golf_model"):
        parse_simscape_linearization(_doc(scope="verification_rig"))


def test_refuses_synthetic_provenance() -> None:
    doc = _doc()
    doc["provenance"] = {
        "provenance_kind": "synthetic_test_fixture",
        "model_scope": "verification_rig",
    }
    with pytest.raises(ValueError, match="matlab_export"):
        parse_simscape_linearization(doc)


def test_refuses_wrong_schema_and_missing_solver_or_mapping() -> None:
    bad = _doc()
    bad["schema"] = "other/v1"
    with pytest.raises(ValueError, match="schema"):
        parse_simscape_linearization(bad)
    for key in ("solver_settings", "state_mapping", "frames"):
        doc = copy.deepcopy(_doc())
        del doc[key]
        with pytest.raises(ValueError, match=key):
            parse_simscape_linearization(doc)


def test_refuses_shape_mismatch_with_state_mapping() -> None:
    doc = _doc()
    doc["frames"][0]["A"] = np.eye(3).tolist()
    with pytest.raises(ValueError, match="A"):
        parse_simscape_linearization(doc)


def test_loader_refuses_missing_file(tmp_path: Any) -> None:
    with pytest.raises(ValueError, match="not valid JSON"):
        bad = tmp_path / "bad.json"
        bad.write_text("{", encoding="utf-8")
        load_simscape_linearization_doc(bad)


def test_simscape_derivative_capability_is_none_with_reason() -> None:
    decl = dc.default_registry().declaration("simscape")
    assert decl.method is dc.DerivativeMethod.NONE
    for phrase in ("snapshot", "perturbed", "linearize", "R2025b", "DV-12718"):
        assert phrase in decl.reason


def test_requesting_simscape_derivatives_raises_with_reason() -> None:
    reg = dc.default_registry()
    with pytest.raises(dc.DerivativeCapabilityUnavailable, match="DV-12718"):
        reg.require("simscape")
    with pytest.raises(dc.DerivativeCapabilityUnavailable, match="DV-12718"):
        reg.load("simscape")


def test_variability_output_for_simscape_is_unavailable_with_reason() -> None:
    rec = dc.unavailable_derivative_record(
        "simscape",
        shape=(2, 2),
        units="1/s^2",
        model_provenance=ModelProvenance("simscape", "none", ""),
        input_provenance=InputProvenance("test", "unit"),
    )
    assert rec is not None
    assert rec.allocation_dependence is AllocationDependence.UNAVAILABLE
    assert np.isnan(rec.values).all()
    assert rec.values.shape == (2, 2)
    assert "DV-12718" in str(rec.reason)


def test_available_engine_yields_no_unavailable_record() -> None:
    rec = dc.unavailable_derivative_record(
        "reference_fd",
        shape=(2,),
        units="1",
        model_provenance=ModelProvenance("x", "1", ""),
        input_provenance=InputProvenance("test", "unit"),
    )
    assert rec is None

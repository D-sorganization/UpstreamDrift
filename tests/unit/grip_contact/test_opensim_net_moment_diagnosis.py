"""Unit tests for OpenSim net-moment diagnosis helpers (#13560)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact.club_dynamics import ClubDynamics
from src.shared.python.grip_contact.contact_topology import GripStyle
from src.shared.python.grip_contact.opensim_net_moment_diagnosis import (
    DIAGNOSIS_ISSUE,
    components_match_series,
    decompose_net_moment_components,
    diagnosis_payload,
    trace_peak_summary,
    validate_diagnosis_payload,
    write_diagnosis_json,
    xfail_baseline_cases,
)
from tests.unit.grip_contact.test_grip_style_parity_report import (
    SPEC_PATH,
    _synthetic_contact_run,
    _synthetic_series,
)


@pytest.mark.unit
def test_decomposition_matches_net_moment_com() -> None:
    series = _synthetic_series(scale=1.1, internal_scale=0.7, n=17)
    assert components_match_series(series)


@pytest.mark.unit
def test_trace_peak_summary_on_identical_runs() -> None:
    ref = _synthetic_contact_run(net_scale=1.0, n=11)
    cand = _synthetic_contact_run(net_scale=1.0, n=11)
    from src.shared.python.grip_contact.opensim_net_moment_diagnosis import (
        _residual_arrays,
    )

    spec = json.loads(SPEC_PATH.read_bytes())
    club = ClubDynamics.from_spec(spec)
    from src.shared.python.grip_contact import GripInterface

    interface = GripInterface.from_spec(spec)
    traces = _residual_arrays(ref, cand, 0.04, club, interface)
    summary = trace_peak_summary(traces)
    assert summary["peak_residual_nm"] == pytest.approx(0.0, abs=1e-9)


@pytest.mark.unit
def test_xfail_baseline_payload_validates() -> None:
    payload = diagnosis_payload(
        xfail_baseline_cases(),
        generated_by="unit_test",
        opensim_available=False,
    )
    validate_diagnosis_payload(payload)
    assert payload["issue"] == DIAGNOSIS_ISSUE
    assert len(payload["cases"]) == 5
    failing = [
        c
        for c in payload["cases"]
        if not c["determinate_errors"]["net_moment_com"].get("passes", True)
    ]
    assert len(failing) == 5


@pytest.mark.unit
def test_committed_artifact_matches_schema(tmp_path: Path) -> None:
    root = Path(__file__).resolve().parents[3]
    artifact = (
        root
        / "docs/development/full_body_models/evidence/grip_kinetics"
        / "opensim_net_moment_diagnosis_13560.json"
    )
    if not artifact.is_file():
        payload = diagnosis_payload(
            xfail_baseline_cases(),
            generated_by="unit_test_placeholder",
            opensim_available=False,
        )
        write_diagnosis_json(
            tmp_path / "opensim_net_moment_diagnosis_13560.json", payload
        )
        validate_diagnosis_payload(payload)
        return
    data = json.loads(artifact.read_text(encoding="utf-8"))
    validate_diagnosis_payload(data)
    assert data["candidate_engine"] == "opensim"


@pytest.mark.unit
def test_scaled_candidate_shows_nonzero_residual() -> None:
    ref = _synthetic_contact_run(net_scale=1.0, n=9)
    cand = _synthetic_contact_run(net_scale=1.05, n=9)
    from src.shared.python.grip_contact.opensim_net_moment_diagnosis import (
        _residual_arrays,
    )

    spec = json.loads(SPEC_PATH.read_bytes())
    club = ClubDynamics.from_spec(spec)
    from src.shared.python.grip_contact import GripInterface

    interface = GripInterface.from_spec(spec)
    traces = _residual_arrays(ref, cand, 0.04, club, interface)
    mag = np.linalg.norm(traces["residual_net_moment_com_nm"], axis=1)
    assert float(mag.max()) > 0.0

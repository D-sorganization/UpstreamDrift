"""A prospective authority amendment preserves historical qualification."""

import hashlib
import json
from pathlib import Path

import pytest

from scripts.research.proximal_distal_energy.nonlinear_controller_registration import (
    REPORT_PATH,
    build_registration,
)

pytestmark = pytest.mark.scientific
ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "docs/research/proximal_distal_energy_transfer/data"
PREDECESSOR = DATA / "nonlinear_controller_comparison_registration.json"
AMENDMENT = DATA / "nonlinear_controller_registration_amendment_20261011.json"
PREDECESSOR_SHA256 = "4856de8cabd4a820c22c2bb3e25f9f97852c22bbb7f08d405702f9b10d64d7bf"


def test_historical_registration_remains_byte_identical():
    assert hashlib.sha256(PREDECESSOR.read_bytes()).hexdigest() == PREDECESSOR_SHA256


def test_current_registration_has_a_separate_identity():
    assert REPORT_PATH != PREDECESSOR
    assert json.loads(REPORT_PATH.read_text()) == build_registration(ROOT)


def test_amendment_changes_authorities_without_changing_protocol():
    previous = json.loads(PREDECESSOR.read_text())
    current = json.loads(REPORT_PATH.read_text())
    authority_fields = {"environment_authority", "plant_and_solver_source_authorities"}
    assert {k: v for k, v in previous.items() if k not in authority_fields} == {
        k: v for k, v in current.items() if k not in authority_fields
    }
    amendment = json.loads(AMENDMENT.read_text())
    assert amendment["predecessor"]["sha256"] == PREDECESSOR_SHA256
    assert (
        amendment["current"]["sha256"]
        == hashlib.sha256(REPORT_PATH.read_bytes()).hexdigest()
    )
    assert amendment["qualification_status"] == "fresh_qualification_required"
    assert amendment["controller_evaluation_count"] == 0
    assert amendment["ranking_eligible_method_count"] == 0
    changes = amendment["reviewed_authority_changes"]
    assert len(changes) == 3
    assert {item["path"] for item in changes} == {
        "requirements.lock",
        "src/shared/python/simulation_backends/ode_backend.py",
        "src/engines/pendulum_models/python/double_pendulum_model/physics/double_pendulum.py",
    }
    previous_authorities = {
        item["path"]: item["sha256"]
        for item in [
            previous["environment_authority"],
            *previous["plant_and_solver_source_authorities"],
        ]
    }
    current_authorities = {
        item["path"]: item["sha256"]
        for item in [
            current["environment_authority"],
            *current["plant_and_solver_source_authorities"],
        ]
    }
    for item in changes:
        assert item["previous_sha256"] == previous_authorities[item["path"]]
        assert item["current_sha256"] == current_authorities[item["path"]]
        assert item["previous_sha256"] != item["current_sha256"]


def test_historical_qualifications_keep_their_registration_identity():
    historical_bindings = {
        "nonlinear_controller_solver_qualification.json": PREDECESSOR_SHA256,
        "nonlinear_controller_plant_transport.json": "e964b68173c062559a6daad35a39240d4289b9bb5c50244883e3c251a059f11f",
    }
    for name, historical_sha256 in historical_bindings.items():
        report = json.loads((DATA / name).read_text())
        assert (
            report["registration_authority"]["path"]
            == PREDECESSOR.relative_to(ROOT).as_posix()
        )
        assert report["registration_authority"]["sha256"] == historical_sha256

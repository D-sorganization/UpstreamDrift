"""SHOW-C1 (#14557): PhysiologicalEnvelope from the NSX-04 table + cited YAML."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.physiological_envelope import (
    DEFAULT_LIMITS_PATH,
    PhysiologicalEnvelope,
    load_limits,
)
from src.shared.python.loop_nullspace.capacity import CapacitySubject, capacity_for

SPEC = json.loads(
    Path(
        "docs/development/full_body_models/full_body_spec_anthro_driver.json"
    ).read_text(encoding="utf-8")
)
SUBJECT = CapacitySubject(mass_kg=78.0, stature_m=1.71)


@pytest.fixture(scope="module")
def env() -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope.from_spec(
        SPEC, SUBJECT, load_limits(DEFAULT_LIMITS_PATH)
    )


def test_torque_limits_come_from_nsx04_table(env: PhysiologicalEnvelope) -> None:
    cap = capacity_for(tuple(SPEC["coordinate_order"]), SUBJECT)
    assert env.names == cap.actuated_names
    np.testing.assert_allclose(env.tau_max_negative_nm, cap.tau_max_negative_nm)
    np.testing.assert_allclose(env.tau_max_positive_nm, cap.tau_max_positive_nm)


def test_rom_comes_from_spec_ranges(env: PhysiologicalEnvelope) -> None:
    i = env.names.index("LEInput")
    assert env.rom_deg[i].tolist() == SPEC["coordinate_ranges_deg"]["LEInput"]


def test_limits_are_ordered_finite_and_positive(env: PhysiologicalEnvelope) -> None:
    assert np.all(env.rom_deg[:, 0] < env.rom_deg[:, 1])
    for a in (env.tau_max_negative_nm, env.tau_max_positive_nm, env.vel_max_deg_s):
        assert np.all(np.isfinite(a) & (a > 0.0))


def test_unverified_limits_are_flagged_never_silent(env: PhysiologicalEnvelope) -> None:
    assert env.assumed_velocity_slots
    assert set(env.assumed_velocity_slots) <= set(env.names)


def test_missing_joint_fails_closed() -> None:
    limits = dict(load_limits(DEFAULT_LIMITS_PATH))
    limits.pop("LWInputX")
    with pytest.raises(ValueError, match="LWInputX"):
        PhysiologicalEnvelope.from_spec(SPEC, SUBJECT, limits)


def test_yaml_entry_without_citation_rejected(tmp_path: Path) -> None:
    p = tmp_path / "limits.yaml"
    p.write_text(
        "velocity_limits_deg_s:\n  - coordinates: [LWInputX]\n    value: 1000\n",
        encoding="utf-8",
    )
    with pytest.raises(ContractViolationError):
        load_limits(p)


def test_utilisation_and_excess(env: PhysiologicalEnvelope) -> None:
    n = len(env.names)
    assert np.all(env.utilisation(np.zeros(n)) == 0.0)
    assert np.all(env.torque_excess(np.zeros(n)) == 0.0)
    over = env.tau_max_positive_nm * 1.5
    assert np.allclose(env.utilisation(over), 1.5)
    np.testing.assert_allclose(env.torque_excess(over), 0.5 * env.tau_max_positive_nm)
    assert np.allclose(env.utilisation(-env.tau_max_negative_nm * 0.5), 0.5)


def test_binding_joints_report(env: PhysiologicalEnvelope) -> None:
    tau = np.zeros(len(env.names))
    tau[3] = env.tau_max_positive_nm[3] * 0.99
    assert env.binding_joints(tau, threshold=0.98) == (env.names[3],)
    assert env.binding_joints(np.zeros_like(tau)) == ()


def test_wrong_length_torque_rejected(env: PhysiologicalEnvelope) -> None:
    with pytest.raises(ContractViolationError):
        env.utilisation(np.zeros(3))

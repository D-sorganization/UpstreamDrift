"""SHOW-C1 (#14557): PhysiologicalEnvelope from the NSX-04 table + cited YAML."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest


from src.shared.python.contracts import ContractViolationError
from src.shared.python.impact_optimization.physiological_envelope import (
    DEFAULT_LIMITS_PATH,
    ROOT_COORDINATES,
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

pytestmark = pytest.mark.unit


def _no_rom() -> frozenset[str]:
    """Non-root coordinates the spec gives no range for (the shoulders)."""
    declared = set(SPEC["coordinate_ranges_deg"])
    names = capacity_for(tuple(SPEC["coordinate_order"]), SUBJECT).actuated_names
    return frozenset(n for n in names if n not in declared) - ROOT_COORDINATES


@pytest.fixture(scope="module")
def env() -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope.from_spec(
        SPEC,
        SUBJECT,
        load_limits(DEFAULT_LIMITS_PATH),
        allow_unbounded_rom=_no_rom(),
    )


def test_missing_rom_fails_closed_naming_the_shoulders() -> None:
    with pytest.raises(ValueError, match="LSInputX") as exc:
        PhysiologicalEnvelope.from_spec(SPEC, SUBJECT, load_limits(DEFAULT_LIMITS_PATH))
    assert "RSInputZ" in str(exc.value)
    assert "HipInput" not in str(exc.value)  # root is the explicit exemption


def test_unbounded_rom_must_be_listed_explicitly() -> None:
    assert _no_rom() == {f"{s}SInput{a}" for s in "LR" for a in "XYZ"}


def test_torque_limits_come_from_nsx04_table(env: PhysiologicalEnvelope) -> None:
    cap = capacity_for(tuple(SPEC["coordinate_order"]), SUBJECT)
    assert env.names == cap.actuated_names
    np.testing.assert_allclose(env.tau_max_negative_nm, cap.tau_max_negative_nm)
    np.testing.assert_allclose(env.tau_max_positive_nm, cap.tau_max_positive_nm)


def test_rom_comes_from_spec_ranges(env: PhysiologicalEnvelope) -> None:
    i = env.names.index("LEInput")
    assert env.rom_deg[i].tolist() == SPEC["coordinate_ranges_deg"]["LEInput"]


def test_undeclared_rom_is_unbounded_and_listed_not_invented(
    env: PhysiologicalEnvelope,
) -> None:
    declared = set(SPEC["coordinate_ranges_deg"])
    assert set(env.unbounded_rom_slots) == set(env.names) - declared
    i = env.names.index(env.unbounded_rom_slots[0])
    assert np.isinf(env.rom_deg[i]).all()


def test_limits_are_ordered_finite_and_positive(env: PhysiologicalEnvelope) -> None:
    assert np.all(env.rom_deg[:, 0] < env.rom_deg[:, 1])
    assert not np.any(np.isnan(env.rom_deg))
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


# ---- v2 trunk: soft_limits_deg fallback (D35, #12553) ----------------------------
SPEC_V2 = json.loads(
    Path(
        "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
    ).read_text(encoding="utf-8")
)
_TRUNK6 = tuple(SPEC_V2["soft_limits_deg"])
_SHOULDERS = {f"{s}SInput{a}" for s in "LR" for a in "XYZ"}


def _v2_env(spec: dict = SPEC_V2) -> PhysiologicalEnvelope:
    return PhysiologicalEnvelope.from_spec(
        spec,
        SUBJECT,
        load_limits(DEFAULT_LIMITS_PATH),
        allow_unbounded_rom=frozenset(_SHOULDERS),
    )


def test_v2_missing_rom_names_only_the_shoulders() -> None:
    """Trunk ROM is declared (soft_limits_deg); only the shoulders still fail closed."""
    with pytest.raises(ValueError, match="LSInputX") as exc:
        PhysiologicalEnvelope.from_spec(
            SPEC_V2, SUBJECT, load_limits(DEFAULT_LIMITS_PATH)
        )
    msg = str(exc.value)
    assert "lumbar" not in msg and "thoracic" not in msg


def test_v2_trunk_rom_comes_from_soft_limits_and_is_labelled() -> None:
    env = _v2_env()
    assert len(_TRUNK6) == 6
    assert set(env.soft_rom_slots) == set(_TRUNK6)
    for name in _TRUNK6:
        i = env.names.index(name)
        assert env.rom_deg[i].tolist() == list(SPEC_V2["soft_limits_deg"][name])
    assert set(env.unbounded_rom_slots) == _SHOULDERS | set(ROOT_COORDINATES)


def test_hard_range_wins_over_a_soft_limit_for_the_same_coordinate() -> None:
    spec = json.loads(json.dumps(SPEC_V2))
    spec["coordinate_ranges_deg"]["lumbar_flexion"] = [-5.0, 10.0]
    env = _v2_env(spec)
    i = env.names.index("lumbar_flexion")
    assert env.rom_deg[i].tolist() == [-5.0, 10.0]
    assert "lumbar_flexion" not in env.soft_rom_slots


def test_envelope_subset_keeps_the_soft_rom_label() -> None:
    from src.shared.python.impact_optimization.full_body_rollout import (
        envelope_subset,
    )

    sub = envelope_subset(_v2_env(), ("thoracic_flexion", "LEInput"))
    assert sub.soft_rom_slots == ("thoracic_flexion",)


@pytest.mark.parametrize("kind", ["flexion", "lateral_bending", "axial_rotation"])
def test_series_trunk_range_is_the_sum_of_the_regional_ranges(kind: str) -> None:
    """Lumbar and thoracic joints are in series: whole-trunk motion is the SUM of the
    regional ranges (documented, not enforced); each joint is limited on its own."""
    env = _v2_env()
    lum = env.rom_deg[env.names.index(f"lumbar_{kind}")]
    tho = env.rom_deg[env.names.index(f"thoracic_{kind}")]
    total = lum + tho
    assert total[0] < 0.0 < total[1]
    assert total.tolist() == [
        SPEC_V2["soft_limits_deg"][f"lumbar_{kind}"][0]
        + SPEC_V2["soft_limits_deg"][f"thoracic_{kind}"][0],
        SPEC_V2["soft_limits_deg"][f"lumbar_{kind}"][1]
        + SPEC_V2["soft_limits_deg"][f"thoracic_{kind}"][1],
    ]


def test_limit_is_assumed_answers_per_joint_and_kind_in_one_place() -> None:
    env = _v2_env()
    assert env.limit_is_assumed("lumbar_flexion", "ROM")
    assert env.limit_is_assumed("lumbar_flexion", "velocity")  # trunk velocity prior
    assert not env.limit_is_assumed("LEInput", "ROM")  # hard, declared
    assert ("thoracic_axial_rotation", "ROM") in env.assumed_limits()
    with pytest.raises(ContractViolationError):
        env.limit_is_assumed("LEInput", "torque")


def test_soft_rom_slots_must_be_known_and_not_unbounded() -> None:
    import dataclasses

    env = _v2_env()
    with pytest.raises(ContractViolationError):
        dataclasses.replace(env, soft_rom_slots=("not_a_joint",))
    with pytest.raises(ContractViolationError):
        dataclasses.replace(env, soft_rom_slots=(env.unbounded_rom_slots[0],))

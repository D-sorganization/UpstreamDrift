"""Torque-capacity loader tests for NSX-04 (#12553)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import pytest

from src.shared.python.loop_nullspace.capacity import (
    CapacitySubject,
    CapacityVector,
    capacity_for,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPECS = ROOT / "docs" / "development" / "full_body_models"
SPEC_FILES = ("full_body_spec_anthro_driver.json", "full_body_spec_anthro_iron7.json")
G = 9.80665


def _spec(name: str) -> dict:
    return json.loads((SPECS / name).read_text(encoding="utf-8"))


@pytest.fixture(params=SPEC_FILES)
def spec(request: pytest.FixtureRequest) -> dict:
    return _spec(request.param)


def _subject(spec: dict) -> CapacitySubject:
    return CapacitySubject.from_spec_subject(spec["subject"])


def test_covers_all_44_coordinates_of_both_specs(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    assert len(order) == 44
    vec = capacity_for(order, _subject(spec))
    assert isinstance(vec, CapacityVector)
    # Root translations are forces, not joint torques.
    assert len(vec.actuated_names) == 41
    assert all(not n.startswith("Translation") for n in vec.actuated_names)
    for arr in (vec.tau_max_negative_nm, vec.tau_max_positive_nm):
        assert arr.shape == (41,)
        assert (arr > 0).all() and bool(arr.dtype.kind == "f")
        assert all(math.isfinite(float(v)) for v in arr)
    assert [order[i] for i in vec.actuated_indices] == list(vec.actuated_names)


def test_result_is_frozen_and_read_only(spec: dict) -> None:
    vec = capacity_for(tuple(spec["coordinate_order"]), _subject(spec))
    with pytest.raises(AttributeError):
        vec.assumed_slots = ()  # type: ignore[misc]
    with pytest.raises(ValueError):
        vec.tau_max_positive_nm[0] = 1.0


def test_assumed_slots_are_surfaced_not_silent(spec: dict) -> None:
    vec = capacity_for(tuple(spec["coordinate_order"]), _subject(spec))
    assert vec.has_assumed
    assert set(vec.assumed_slots) | set(vec.verified_slots) == set(vec.actuated_names)
    assert not set(vec.assumed_slots) & set(vec.verified_slots)
    assert "TorsoInput" in vec.verified_slots
    assert "LEInput" in vec.verified_slots and "knee_angle_r" in vec.verified_slots
    assert "hip_rotation_r" in vec.assumed_slots
    assert vec.warnings and "ASSUMED" in vec.warnings[0]


def test_missing_slots_raise_listing_each(spec: dict) -> None:
    order = tuple(spec["coordinate_order"]) + ("PhantomInputA", "PhantomInputB")
    with pytest.raises(ValueError) as exc:
        capacity_for(order, _subject(spec))
    msg = str(exc.value)
    assert "PhantomInputA" in msg and "PhantomInputB" in msg


def test_unactuated_coordinate_need_not_be_covered(spec: dict) -> None:
    order = tuple(spec["coordinate_order"]) + ("PhantomInputA",)
    vec = capacity_for(order, _subject(spec), actuated=spec["coordinate_order"][3:])
    assert "PhantomInputA" not in vec.actuated_names


def test_preconditions(spec: dict) -> None:
    subj = _subject(spec)
    order = tuple(spec["coordinate_order"])
    with pytest.raises(ValueError, match="empty"):
        capacity_for((), subj)
    with pytest.raises(ValueError, match="duplicate"):
        capacity_for(order + (order[5],), subj)
    with pytest.raises(ValueError, match="not in coordinate_order"):
        capacity_for(order, subj, actuated=("Nope",))
    with pytest.raises(ValueError, match="mass_kg"):
        CapacitySubject(mass_kg=-1.0, stature_m=1.7)
    with pytest.raises(ValueError, match="stature_m"):
        CapacitySubject(mass_kg=70.0, stature_m=float("nan"))
    with pytest.raises(ValueError, match="sex"):
        capacity_for(order, CapacitySubject(70.0, 1.7, sex="female"))
    with pytest.raises(TypeError):
        capacity_for(order, {"mass_kg": 70.0})  # type: ignore[arg-type]


def test_anderson_scaling_reproduces_cited_example() -> None:
    """Anderson 2007 Table 2 mean young male (72.8 kg, 174.8 cm) times Table 3
    C1 = 0.163 (knee extension) gives 0.163 * 72.8 * g * 1.748 N m."""
    expected = 0.163 * 72.8 * G * 1.748
    vec = capacity_for(
        ("knee_angle_r",), CapacitySubject(72.8, 1.748), actuated=("knee_angle_r",)
    )
    assert vec.tau_max_positive_nm[0] == pytest.approx(expected, rel=1e-9)
    # Extension torque sits inside one SD of the independent Leblanc 2015
    # men's isometric knee extension mean 235.6 (SD 63.4).
    assert abs(vec.tau_max_positive_nm[0] - 235.6) < 63.4


def test_anderson_scaling_is_proportional_to_weight_times_height() -> None:
    a = capacity_for(("hip_flexion_l",), CapacitySubject(60.0, 1.60))
    b = capacity_for(("hip_flexion_l",), CapacitySubject(90.0, 1.80))
    ratio = (90.0 * 1.80) / (60.0 * 1.60)
    assert b.tau_max_positive_nm[0] / a.tau_max_positive_nm[0] == pytest.approx(ratio)
    assert b.tau_max_negative_nm[0] / a.tau_max_negative_nm[0] == pytest.approx(ratio)


def test_unscaled_population_values_are_returned_verbatim() -> None:
    vec = capacity_for(("TorsoInput", "LEInput"), CapacitySubject(50.0, 1.5))
    assert vec.tau_max_negative_nm[0] == pytest.approx(68.7)
    assert vec.tau_max_positive_nm[0] == pytest.approx(68.7)
    # Elbow: negative = flexion 67.9, positive = extension 59.6 (Leblanc 2015).
    assert vec.tau_max_negative_nm[1] == pytest.approx(67.9)
    assert vec.tau_max_positive_nm[1] == pytest.approx(59.6)


def test_reference_subject_reproduces_stored_reference_values(spec: dict) -> None:
    ref = CapacitySubject(78.0, 1.71)
    vec = capacity_for(tuple(spec["coordinate_order"]), ref)
    reg = json.loads(
        (ROOT / "src" / "config" / "loop_nullspace_registry.json").read_text("utf-8")
    )
    stored = {
        (e["joint_slot"], e["direction"]): e["tau_max_nm"]
        for e in reg["torque_capacity"]
    }
    for i, name in enumerate(vec.actuated_names):
        assert vec.tau_max_negative_nm[i] == pytest.approx(
            stored[(name, "negative")], abs=0.01
        )
        assert vec.tau_max_positive_nm[i] == pytest.approx(
            stored[(name, "positive")], abs=0.01
        )


def test_force_velocity_parameters_exposed_for_lower_limb() -> None:
    vec = capacity_for(("knee_angle_r",), CapacitySubject(78.0, 1.71))
    fv = vec.force_velocity["knee_angle_r"]
    assert fv["positive"]["omega_75pct_rad_s"] == pytest.approx(1.517)
    assert fv["negative"]["omega_50pct_rad_s"] == pytest.approx(5.233)

"""Torque-limit provider tests for CAP-2 (#14041, epic #14038)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.capacity import CapacitySubject, capacity_for
from src.shared.python.loop_nullspace.torque_limits import (
    TorqueLimits,
    torque_limits_for,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


def _vec(spec: dict):
    return capacity_for(
        tuple(spec["coordinate_order"]),
        CapacitySubject.from_spec_subject(spec["subject"]),
    )


def test_selection_matrix_places_columns_at_model_dof_index(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    dofs = tuple(reversed(order))  # caller ordering differs from the spec
    lim = torque_limits_for(spec, dofs)
    vec = _vec(spec)
    assert lim.B.shape == (len(dofs), 41)
    assert lim.tau_lo.shape == lim.tau_hi.shape == (41,)
    for col, name in enumerate(vec.actuated_names):
        column = lim.B[:, col]
        assert column.sum() == 1.0 and column[dofs.index(name)] == 1.0


def test_root_translations_are_zero_rows_never_invented(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    lim = torque_limits_for(spec, order)
    for i, name in enumerate(order):
        if name.startswith("Translation"):
            assert not lim.B[i].any()


def test_sign_convention(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    lim = torque_limits_for(spec, order)
    vec = _vec(spec)
    np.testing.assert_array_equal(lim.tau_lo, -vec.tau_max_negative_nm)
    np.testing.assert_array_equal(lim.tau_hi, vec.tau_max_positive_nm)
    assert (lim.tau_lo < 0).all() and (lim.tau_hi > 0).all()


def test_provenance_counts_carried_through(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    lim = torque_limits_for(spec, order)
    vec = _vec(spec)
    assert lim.assumed_slots == len(vec.assumed_slots)
    assert lim.verified_slots == len(vec.verified_slots)
    assert lim.assumed_slots + lim.verified_slots == 41
    assert lim.source_id == "loop_nullspace_registry"
    assert lim.warnings == vec.warnings


def test_subject_only_uses_dof_names_with_registry_entries() -> None:
    subject = CapacitySubject(78.0, 1.71)
    lim = torque_limits_for(subject, ("pelvis_tx", "hip_flexion_r", "knee_angle_r"))
    assert lim.B.shape == (3, 2)
    assert not lim.B[0].any()
    assert lim.B[1, 0] == 1.0 and lim.B[2, 1] == 1.0


def test_actuated_coordinate_missing_from_dofs_raises(spec: dict) -> None:
    order = tuple(spec["coordinate_order"])
    with pytest.raises(ValueError, match="missing from dof_names"):
        torque_limits_for(spec, order[:-1] if order[-1] != "Translation" else order[1:])


def test_duplicate_dof_names_raise(spec: dict) -> None:
    with pytest.raises(ValueError, match="duplicate"):
        torque_limits_for(spec, ("a", "a"))


def test_arrays_are_immutable(spec: dict) -> None:
    lim = torque_limits_for(spec, tuple(spec["coordinate_order"]))
    for arr in (lim.B, lim.tau_lo, lim.tau_hi):
        with pytest.raises(ValueError):
            arr[0] = 1.0
    with pytest.raises(AttributeError):
        lim.source_id = "x"  # type: ignore[misc]


def _make(source: str, lo: list[float], hi: list[float]) -> TorqueLimits:
    n = len(lo)
    return TorqueLimits(
        B=np.eye(n),
        tau_lo=np.array(lo),
        tau_hi=np.array(hi),
        source_id=source,
        assumed_slots=0,
        verified_slots=n,
        warnings=(),
    )


def test_uniform_100nm_placeholder_is_rejected() -> None:
    with pytest.raises(ValueError, match="uniform"):
        _make("model", [-100.0, -100.0, -100.0], [100.0, 100.0, 100.0])


@pytest.mark.parametrize("source", ["control_interface", "placeholder"])
def test_placeholder_sources_are_rejected(source: str) -> None:
    with pytest.raises(ValueError, match="placeholder"):
        _make(source, [-50.0, -60.0], [70.0, 80.0])


def test_non_finite_or_wrong_sign_rejected() -> None:
    with pytest.raises(ValueError):
        _make("model", [-1.0, 0.5], [2.0, 3.0])
    with pytest.raises(ValueError):
        _make("model", [-1.0, -2.0], [np.nan, 3.0])


def test_shipped_default_is_not_the_placeholder() -> None:
    """The control_interface default stays the value this provider refuses."""
    from src.shared.python.control_interface import JointInfo

    assert JointInfo(index=0, name="j").torque_limit == 100.0

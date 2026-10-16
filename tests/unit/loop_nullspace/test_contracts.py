"""Tests for NSX-10 contracts: snapshot, topology and evidence axes."""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import (
    AllocationDependence,
    ClosedLoopSnapshot,
    ClosureKind,
    FootContact,
    GripFrame,
    InputProvenance,
    LoopTopology,
    ModelProvenance,
    QuantityRecord,
    SpatialInertia,
    ValidationStatus,
)

pytestmark = pytest.mark.unit

GOLDEN = (
    Path(__file__).resolve().parents[2]
    / "fixtures/loop_nullspace/goldens/planar_two_arm_full.json"
)

MODEL = ModelProvenance("planar_golfer", "1")
INPUT = InputProvenance("synthetic", "golden")


def _inertia(mass: float = 0.3) -> SpatialInertia:
    return SpatialInertia(mass, np.zeros(3), np.diag([0.02, 0.03, 0.04]))


def _golden() -> dict:
    return json.loads(GOLDEN.read_text())


def _kwargs() -> dict:
    g = _golden()
    mat, model = g["matrices"], g["model"]
    topo = LoopTopology(
        ClosureKind[model["closure_kind"]],
        "trail",
        "lead",
        tuple(model["actuated_indices"]),
        tuple(model["unactuated_indices"]),
        (0, 1, 2),
        tuple(model["coordinate_order"]),
    )
    return {
        "q": g["state"]["q"],
        "v": g["state"]["v"],
        "a": g["state"]["a"],
        "M": mat["M"],
        "h": mat["h"],
        "S": mat["S"],
        "J_c": mat["J_c"],
        "Jdot_v": mat["Jdot_v"],
        "grip_frame": GripFrame(np.eye(3), np.zeros(3)),
        "bare_club": _inertia(),
        "welded_hand_inertias": (_inertia(1.0),),
        "model_provenance": MODEL,
        "input_provenance": INPUT,
        "topology": topo,
    }


def _snap(**over) -> ClosedLoopSnapshot:
    return ClosedLoopSnapshot(**{**_kwargs(), **over})


def _topo(**over) -> LoopTopology:
    base = {
        "closure_kind": ClosureKind.BALL3,
        "closing_side": "a",
        "holding_side": "b",
        "actuated_indices": (0, 1),
        "unactuated_indices": (2,),
        "loop_support_indices": (0, 2),
        "coordinate_order": ("x", "y", "z"),
    }
    return LoopTopology(**{**base, **over})


def _record(**over) -> QuantityRecord:
    base = {
        "values": [1.0, 2.0],
        "units": "N m",
        "allocation_dependence": AllocationDependence.INVARIANT,
        "model_provenance": MODEL,
        "input_provenance": INPUT,
    }
    return QuantityRecord(**{**base, **over})


# --- golden-backed snapshot -------------------------------------------------


def test_planar_golden_snapshot_valid_and_load_matches_b() -> None:
    snap = _snap()
    assert snap.topology.closure_kind is ClosureKind.PLANAR_POINT2
    np.testing.assert_allclose(
        snap.generalized_load(), _golden()["matrices"]["b"], atol=1e-12
    )
    assert snap.closure_rows().shape == (2, 5)
    assert snap.actuated_rows().shape == (5, 5)


def test_closure_kind_row_counts() -> None:
    rows = {k.name: k.rows for k in ClosureKind}
    assert rows == {
        "WELD6": 6,
        "SHAFT_FREE5": 5,
        "BALL3": 3,
        "PLANAR_POINT2": 2,
        "PLANAR_WELD3": 3,
    }


def test_arrays_copied_and_readonly() -> None:
    kw = _kwargs()
    m_in = np.array(kw["M"])
    snap = _snap(M=m_in)
    m_in[0, 0] = 99.0
    assert snap.M[0, 0] != 99.0
    for name in ("q", "v", "a", "M", "h", "S", "J_c", "Jdot_v"):
        with pytest.raises(ValueError):
            getattr(snap, name)[...] = 0.0
    with pytest.raises(dataclasses.FrozenInstanceError):
        snap.q = np.zeros(5)  # type: ignore[misc]


def test_optional_contact_block_valid_and_readonly() -> None:
    contact = FootContact("lf", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.8)
    snap = _snap(J_g=np.ones((3, 5)), grf_observed=np.ones(3), foot_contacts=[contact])
    assert isinstance(snap.foot_contacts, tuple)
    with pytest.raises(ValueError):
        snap.J_g[0, 0] = 1.0  # type: ignore[index]
    with pytest.raises(ValueError):
        snap.grf_observed[0] = 1.0  # type: ignore[index]


# --- snapshot rejections ----------------------------------------------------


@pytest.mark.parametrize("name", ["q", "v", "a", "M", "h", "S", "J_c", "Jdot_v"])
@pytest.mark.parametrize("bad", [np.nan, np.inf])
def test_rejects_non_finite(name: str, bad: float) -> None:
    arr = np.array(_kwargs()[name], dtype=float)
    arr.flat[0] = bad
    with pytest.raises(ValueError, match="finite"):
        _snap(**{name: arr})


def test_rejects_non_finite_optional() -> None:
    with pytest.raises(ValueError, match="finite"):
        _snap(J_g=np.full((3, 5), np.nan))


def test_rejects_asymmetric_mass_matrix() -> None:
    m = np.array(_kwargs()["M"])
    m[0, 1] += 0.5
    with pytest.raises(ValueError, match="symmetric"):
        _snap(M=m)


def test_symmetry_tolerance_is_scaled() -> None:
    m = np.array(_kwargs()["M"]) * 1e6
    m[0, 1] += 1e-4  # relative 1e-10 of the scale: accepted
    _snap(M=m)


def test_rejects_indefinite_mass_matrix() -> None:
    m = np.array(_kwargs()["M"])
    m[0, 0] = -1.0
    with pytest.raises(ValueError, match="positive definite"):
        _snap(M=m)


@pytest.mark.parametrize(
    ("name", "shape"),
    [
        ("q", (4,)),
        ("v", (6,)),
        ("a", (5, 1)),
        ("h", (4,)),
        ("M", (4, 4)),
        ("S", (4, 5)),
        ("J_c", (3, 5)),
        ("J_c", (2, 4)),
        ("Jdot_v", (3,)),
    ],
)
def test_rejects_wrong_shape(name: str, shape: tuple[int, ...]) -> None:
    with pytest.raises(ValueError, match="shape"):
        _snap(**{name: np.ones(shape)})


def test_rejects_bad_optional_blocks() -> None:
    with pytest.raises(ValueError, match="columns"):
        _snap(J_g=np.ones((3, 4)))
    with pytest.raises(ValueError, match="shape"):
        _snap(J_g=np.ones((3, 5)), grf_observed=np.ones(2))
    with pytest.raises(ValueError, match="require J_g"):
        _snap(grf_observed=np.ones(3))
    contact = FootContact("lf", np.zeros(3), np.array([0.0, 0.0, 1.0]), 0.8)
    with pytest.raises(ValueError, match="require J_g"):
        _snap(foot_contacts=(contact,))
    assert _snap(J_g=np.ones((3, 5))).grf_observed is None


def test_rejects_s_not_a_selection() -> None:
    s = np.eye(5)
    s[0, 0] = 0.5
    with pytest.raises(ValueError, match="select"):
        _snap(S=s)
    swapped = np.eye(5)[[1, 0, 2, 3, 4]]
    with pytest.raises(ValueError, match="select"):
        _snap(S=swapped)


# --- value types ------------------------------------------------------------


def test_grip_rotation_rejections() -> None:
    GripFrame(np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="orthonormal"):
        GripFrame(2.0 * np.eye(3), np.zeros(3))
    with pytest.raises(ValueError, match="determinant"):
        GripFrame(np.diag([1.0, 1.0, -1.0]), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        GripFrame(np.eye(2), np.zeros(3))
    with pytest.raises(ValueError, match="shape"):
        GripFrame(np.eye(3), np.zeros(2))
    with pytest.raises(ValueError, match="finite"):
        GripFrame(np.eye(3), np.array([np.nan, 0, 0]))


def test_spatial_inertia_rejections() -> None:
    _inertia()
    for mass in (0.0, -1.0, float("nan")):
        with pytest.raises(ValueError, match="mass"):
            SpatialInertia(mass, np.zeros(3), np.eye(3))
    with pytest.raises(ValueError, match="symmetric"):
        SpatialInertia(1.0, np.zeros(3), np.array([[1, 0.5, 0], [0, 1, 0], [0, 0, 1]]))
    with pytest.raises(ValueError, match="semidefinite"):
        SpatialInertia(1.0, np.zeros(3), np.diag([1.0, 1.0, -0.5]))
    with pytest.raises(ValueError, match="triangle"):
        SpatialInertia(1.0, np.zeros(3), np.diag([1.0, 1.0, 5.0]))
    with pytest.raises(ValueError, match="shape"):
        SpatialInertia(1.0, np.zeros(2), np.eye(3))
    with pytest.raises(ValueError, match="shape"):
        SpatialInertia(1.0, np.zeros(3), np.eye(2))


def test_spatial_inertia_readonly() -> None:
    with pytest.raises(ValueError):
        _inertia().inertia[0, 0] = 1.0


def test_foot_contact_rejections() -> None:
    n = np.array([0.0, 0.0, 1.0])
    FootContact("f", np.zeros(3), n, 0.5)
    with pytest.raises(ValueError, match="name"):
        FootContact("", np.zeros(3), n, 0.5)
    with pytest.raises(ValueError, match="unit"):
        FootContact("f", np.zeros(3), 2 * n, 0.5)
    with pytest.raises(ValueError, match="mu"):
        FootContact("f", np.zeros(3), n, 0.0)
    with pytest.raises(ValueError, match="shape"):
        FootContact("f", np.zeros(2), n, 0.5)
    with pytest.raises(ValueError, match="shape"):
        FootContact("f", np.zeros(3), np.ones(2), 0.5)


def test_provenance_rejections() -> None:
    with pytest.raises(ValueError):
        ModelProvenance("", "1")
    with pytest.raises(ValueError):
        InputProvenance("synthetic", "")


# --- topology ---------------------------------------------------------------


def test_topology_valid_properties() -> None:
    topo = _topo()
    assert (topo.n, topo.n_actuated) == (3, 2)


def test_topology_rejections() -> None:
    with pytest.raises(TypeError):
        _topo(closure_kind="BALL3")
    with pytest.raises(ValueError, match="repeat"):
        _topo(actuated_indices=(0, 0), unactuated_indices=(1, 2))
    with pytest.raises(ValueError, match="disjoint"):
        _topo(actuated_indices=(0, 1), unactuated_indices=(1, 2))
    with pytest.raises(ValueError, match="cover"):
        _topo(actuated_indices=(0,), unactuated_indices=(2,))
    with pytest.raises(ValueError, match="loop_support"):
        _topo(loop_support_indices=(7,))


# --- QuantityRecord ---------------------------------------------------------


def test_record_valid_cases() -> None:
    inv = _record()
    assert inv.hypotheses == ()
    assert inv.validation_status is ValidationStatus.UNVALIDATED
    fam = _record(
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        hypotheses=["min_metric_torque"],
    )
    assert fam.hypotheses == ("min_metric_torque",)
    na = _record(
        values=[np.nan],
        allocation_dependence=AllocationDependence.UNAVAILABLE,
        reason="solver backend missing",
    )
    assert np.isnan(na.values).all()
    with pytest.raises(ValueError):
        inv.values[0] = 0.0


def test_record_rejections() -> None:
    with pytest.raises(ValueError, match="units"):
        _record(units="")
    with pytest.raises(ValueError, match="INVARIANT"):
        _record(hypotheses=("h",))
    with pytest.raises(ValueError, match="at least one hypothesis"):
        _record(allocation_dependence=AllocationDependence.FAMILY_DEPENDENT)
    with pytest.raises(ValueError, match="reason"):
        _record(values=[np.nan], allocation_dependence=AllocationDependence.UNAVAILABLE)
    with pytest.raises(ValueError, match="NaN"):
        _record(
            values=[1.0],
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            reason="r",
        )
    with pytest.raises(ValueError, match="finite"):
        _record(values=[np.nan])


# --- import hygiene ---------------------------------------------------------


def test_import_pulls_no_physics_engine() -> None:
    code = (
        "import sys\n"
        "import src.shared.python.loop_nullspace.contracts\n"
        "bad = [m for m in ('pinocchio','mujoco','pydrake','opensim','PyQt6')"
        " if m in sys.modules]\n"
        "assert not bad, bad\n"
    )
    root = Path(__file__).resolve().parents[3]
    res = subprocess.run(
        [sys.executable, "-c", code], cwd=root, capture_output=True, text=True
    )
    assert res.returncode == 0, res.stderr


# --- QuantityRecord conditioning axis (NSX-81, #12710) ----------------------


def test_record_conditioning_defaults_empty_and_old_records_unchanged() -> None:
    inv = _record()
    assert inv.conditioning == ()
    fam = _record(
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        hypotheses=["min_metric_torque"],
    )
    assert fam.conditioning == ()
    assert fam.hypotheses == ("min_metric_torque",)


def test_invariant_record_may_be_conditioned_on_hypotheses() -> None:
    # Ids here are free text, as for ``hypotheses``; no registry lookup occurs.
    rec = _record(conditioning=["H-TORQUE-CAPACITY"])
    assert rec.allocation_dependence is AllocationDependence.INVARIANT
    assert rec.conditioning == ("H-TORQUE-CAPACITY",)
    assert rec.hypotheses == ()


def test_conditioning_is_orthogonal_to_allocation_hypotheses() -> None:
    fam = _record(
        allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
        hypotheses=["min_metric_torque"],
        conditioning=("H-TORQUE-CAPACITY",),
    )
    assert fam.hypotheses == ("min_metric_torque",)
    assert fam.conditioning == ("H-TORQUE-CAPACITY",)
    with pytest.raises(ValueError, match="INVARIANT"):
        _record(hypotheses=("h",), conditioning=("H-TORQUE-CAPACITY",))


def test_conditioning_rejects_bare_string_and_empty_ids() -> None:
    with pytest.raises(TypeError, match="conditioning"):
        _record(conditioning="H-TORQUE-CAPACITY")
    with pytest.raises(ValueError, match="conditioning"):
        _record(conditioning=("",))

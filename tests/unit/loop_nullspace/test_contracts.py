"""Unit tests for loop_nullspace contracts (#12555, NSX-10)."""

from __future__ import annotations

import numpy as np
import pytest

from src.shared.python.loop_nullspace.contracts import (
    CLOSURE_KIND_ROWS,
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


def _make_valid_parts() -> dict[str, object]:
    topology = LoopTopology(
        closure_kind=ClosureKind.PLANAR_POINT2,
        closing_side="right",
        holding_side="left",
        actuated_indices=(0, 1, 3, 4),
        unactuated_indices=(2,),
        loop_support_indices=(0, 1, 2, 3, 4),
        coordinate_order=("q_L1", "q_L2", "q_club", "q_R1", "q_R2"),
    )
    grip_frame = GripFrame(
        rotation=np.eye(3, dtype=np.float64),
        origin=np.array([0.4, -0.2, 0.9], dtype=np.float64),
    )
    bare_club = SpatialInertia(
        mass_kg=0.313,
        com_m=np.array([0.35, 0.0, 0.0], dtype=np.float64),
        inertia_com_kg_m2=np.diag([0.001, 0.045, 0.045]),
    )
    hand_l = SpatialInertia(
        mass_kg=0.5591,
        com_m=np.array([-0.04, 0.0, 0.0], dtype=np.float64),
        inertia_com_kg_m2=np.diag([0.0005, 0.0008, 0.0008]),
    )
    hand_r = SpatialInertia(
        mass_kg=0.5491,
        com_m=np.array([0.04, 0.0, 0.0], dtype=np.float64),
        inertia_com_kg_m2=np.diag([0.0005, 0.0008, 0.0008]),
    )
    model_prov = ModelProvenance(
        spec_sha256="a" * 64,
        inertial_source="full_body_spec_anthro_driver",
        bare_club_bench_id="bench-driver-01",
        engine_name="pinocchio",
    )
    input_prov = InputProvenance(
        kinematic_source="synthetic_oracle",
        derivative_method="analytical",
        time_sync_status="SYNCHRONIZED",
    )
    S = np.zeros((4, 5), dtype=np.float64)
    for row_idx, dof_idx in enumerate(topology.actuated_indices):
        S[row_idx, dof_idx] = 1.0
    return {
        "q": np.array([0.1, 0.2, -0.1, 0.15, 0.25], dtype=np.float64),
        "v": np.array([0.5, -0.3, 0.2, 0.4, -0.1], dtype=np.float64),
        "a": np.array([1.0, -0.5, 0.25, 0.75, -0.25], dtype=np.float64),
        "M": np.diag([2.5, 1.5, 0.8, 2.0, 1.2]),
        "h": np.array([0.5, -0.25, 0.1, 0.3, -0.2], dtype=np.float64),
        "S": S,
        "J_c": np.array(
            [
                [1.0, 0.5, 0.25, -1.0, -0.5],
                [0.0, 0.75, 0.5, 0.0, -0.75],
            ],
            dtype=np.float64,
        ),
        "Jdot_v": np.array([0.125, -0.08333333333333333], dtype=np.float64),
        "grip_frame": grip_frame,
        "bare_club": bare_club,
        "welded_hand_inertias": (hand_l, hand_r),
        "model_provenance": model_prov,
        "input_provenance": input_prov,
        "topology": topology,
    }


def test_closure_kind_row_mapping() -> None:
    expected = {
        ClosureKind.WELD6: 6,
        ClosureKind.SHAFT_FREE5: 5,
        ClosureKind.BALL3: 3,
        ClosureKind.PLANAR_POINT2: 2,
        ClosureKind.PLANAR_WELD3: 3,
    }
    assert expected == CLOSURE_KIND_ROWS
    for kind, rows in expected.items():
        assert kind.expected_rows() == rows


def test_loop_topology_validation() -> None:
    topo = LoopTopology(
        closure_kind="WELD6",
        closing_side="right",
        holding_side="left",
        actuated_indices=(0, 1, 2),
        unactuated_indices=(3,),
        loop_support_indices=(0, 1, 2),
        coordinate_order=("q0", "q1", "q2", "q3"),
    )
    assert topo.closure_kind is ClosureKind.WELD6
    assert topo.expected_closure_rows() == 6

    with pytest.raises(ValueError, match="Unsupported closure_kind"):
        LoopTopology(
            closure_kind="INVALID",  # type: ignore[arg-type]
            closing_side="right",
            holding_side="left",
            actuated_indices=(0,),
            unactuated_indices=(),
            loop_support_indices=(0,),
            coordinate_order=("q0",),
        )

    with pytest.raises(ValueError, match="disjoint"):
        LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="right",
            holding_side="left",
            actuated_indices=(0, 1),
            unactuated_indices=(1, 2),
            loop_support_indices=(0, 1),
            coordinate_order=("q0", "q1", "q2"),
        )

    with pytest.raises(ValueError, match="partition"):
        LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="right",
            holding_side="left",
            actuated_indices=(0,),
            unactuated_indices=(1,),
            loop_support_indices=(0, 1),
            coordinate_order=("q0", "q1", "q2"),
        )

    with pytest.raises(ValueError, match="closing_side"):
        LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="left",
            holding_side="left",
            actuated_indices=(0, 1),
            unactuated_indices=(),
            loop_support_indices=(0, 1),
            coordinate_order=("q0", "q1"),
        )

    with pytest.raises(ValueError, match="out of bounds"):
        LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="right",
            holding_side="left",
            actuated_indices=(0, 5),
            unactuated_indices=(),
            loop_support_indices=(0,),
            coordinate_order=("q0", "q1"),
        )

    with pytest.raises(ValueError, match="duplicate"):
        LoopTopology(
            closure_kind=ClosureKind.WELD6,
            closing_side="right",
            holding_side="left",
            actuated_indices=(0, 0),
            unactuated_indices=(1,),
            loop_support_indices=(0,),
            coordinate_order=("q0", "q1"),
        )


def test_spatial_inertia_and_foot_contact_validation() -> None:
    inertia = SpatialInertia(
        mass_kg=0.313,
        com_m=np.array([0.1, 0.2, 0.3]),
        inertia_com_kg_m2=np.eye(3) * 0.01,
        solid_names=("Clubhead",),
    )
    np.testing.assert_array_equal(inertia.com_local_m, inertia.com_m)
    np.testing.assert_array_equal(inertia.inertia_com_kgm2, inertia.inertia_com_kg_m2)

    with pytest.raises(ValueError, match="mass_kg"):
        SpatialInertia(
            mass_kg=0.0,
            com_m=np.zeros(3),
            inertia_com_kg_m2=np.eye(3),
        )

    with pytest.raises(ValueError, match="symmetric"):
        SpatialInertia(
            mass_kg=1.0,
            com_m=np.zeros(3),
            inertia_com_kg_m2=np.array(
                [[1.0, 2.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]
            ),
        )

    with pytest.raises(ValueError, match="positive semidefinite"):
        SpatialInertia(
            mass_kg=1.0,
            com_m=np.zeros(3),
            inertia_com_kg_m2=np.diag([1.0, -0.1, 1.0]),
        )

    with pytest.raises(ValueError, match="unit vector"):
        FootContact(
            foot_id="left",
            point_world_m=np.zeros(3),
            normal_world=np.array([0.0, 0.0, 2.0]),
            jacobian=np.ones((3, 6)),
            friction_coeff=0.5,
        )

    with pytest.raises(ValueError, match="friction_coeff"):
        FootContact(
            foot_id="left",
            point_world_m=np.zeros(3),
            normal_world=np.array([0.0, 0.0, 1.0]),
            jacobian=np.ones((3, 6)),
            friction_coeff=-0.1,
        )


def test_rejects_non_spd_mass_matrix() -> None:
    parts = _make_valid_parts()
    asymmetric = np.eye(5)
    asymmetric[0, 1] = 2.0
    parts_asym = {**parts, "M": asymmetric}
    with pytest.raises(ValueError, match="symmetric"):
        ClosedLoopSnapshot(**parts_asym)  # type: ignore[arg-type]

    not_pd = np.diag([2.0, -0.5, 1.0, 1.0, 1.0])
    parts_npd = {**parts, "M": not_pd}
    with pytest.raises(ValueError, match="positive definite"):
        ClosedLoopSnapshot(**parts_npd)  # type: ignore[arg-type]


def test_rejects_non_orthonormal_grip_frame() -> None:
    with pytest.raises(ValueError, match="orthonormal"):
        GripFrame(
            rotation=2.0 * np.eye(3),
            origin=np.zeros(3),
        )

    reflection = np.diag([1.0, 1.0, -1.0])
    with pytest.raises(ValueError, match="det"):
        GripFrame(
            rotation=reflection,
            origin=np.zeros(3),
        )


def test_rejects_wrong_closure_row_count() -> None:
    parts = _make_valid_parts()
    wrong_jc = np.ones((3, 5), dtype=np.float64)
    with pytest.raises(ValueError, match="J_c"):
        ClosedLoopSnapshot(**{**parts, "J_c": wrong_jc})  # type: ignore[arg-type]

    wrong_jdot = np.ones(3, dtype=np.float64)
    with pytest.raises(ValueError, match="Jdot_v"):
        ClosedLoopSnapshot(**{**parts, "Jdot_v": wrong_jdot})  # type: ignore[arg-type]


def test_rejects_non_finite_values() -> None:
    parts = _make_valid_parts()
    bad_v = np.array([0.0, np.nan, 0.0, 0.0, 0.0])
    with pytest.raises(ValueError, match="finite"):
        ClosedLoopSnapshot(**{**parts, "v": bad_v})  # type: ignore[arg-type]

    with pytest.raises(ValueError, match="finite"):
        GripFrame(rotation=np.eye(3), origin=np.array([0.0, np.inf, 0.0]))

    with pytest.raises(ValueError, match="finite"):
        QuantityRecord(
            values=np.array([1.0, np.nan]),
            units="N*m",
            allocation_dependence=AllocationDependence.INVARIANT,
            conventions=("C-FRAME-GRIP",),
            hypotheses=(),
            model_provenance=parts["model_provenance"],  # type: ignore[arg-type]
            input_provenance=parts["input_provenance"],  # type: ignore[arg-type]
        )


def test_quantity_record_rejects_hypotheses_on_invariant() -> None:
    parts = _make_valid_parts()
    with pytest.raises(ValueError, match="INVARIANT"):
        QuantityRecord(
            values=np.array([10.0, -5.0]),
            units="N",
            allocation_dependence=AllocationDependence.INVARIANT,
            conventions=("C-FRAME-GRIP",),
            hypotheses=("H-TORQUE-CAPACITY-BOX",),
            model_provenance=parts["model_provenance"],  # type: ignore[arg-type]
            input_provenance=parts["input_provenance"],  # type: ignore[arg-type]
        )


def test_quantity_record_rejects_family_dependent_without_hypotheses() -> None:
    parts = _make_valid_parts()
    with pytest.raises(ValueError, match="FAMILY_DEPENDENT"):
        QuantityRecord(
            values=np.array([10.0, -5.0]),
            units="N*m",
            allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
            conventions=("C-METRIC-CAPACITY",),
            hypotheses=(),
            model_provenance=parts["model_provenance"],  # type: ignore[arg-type]
            input_provenance=parts["input_provenance"],  # type: ignore[arg-type]
        )


def test_quantity_record_rejects_unavailable_without_reason_or_non_nan() -> None:
    parts = _make_valid_parts()
    with pytest.raises(ValueError, match="reason"):
        QuantityRecord(
            values=np.array([np.nan, np.nan]),
            units="N*m",
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            conventions=(),
            hypotheses=(),
            model_provenance=parts["model_provenance"],  # type: ignore[arg-type]
            input_provenance=parts["input_provenance"],  # type: ignore[arg-type]
            reason=None,
        )

    with pytest.raises(ValueError, match="NaN"):
        QuantityRecord(
            values=np.array([1.0, np.nan]),
            units="N*m",
            allocation_dependence=AllocationDependence.UNAVAILABLE,
            conventions=(),
            hypotheses=(),
            model_provenance=parts["model_provenance"],  # type: ignore[arg-type]
            input_provenance=parts["input_provenance"],  # type: ignore[arg-type]
            reason="Residual exceeded gate",
        )


def test_immutability_and_lod_accessors() -> None:
    parts = _make_valid_parts()
    foot = FootContact(
        foot_id="left_foot",
        point_world_m=np.array([-0.2, 0.0, 0.0]),
        normal_world=np.array([0.0, 0.0, 1.0]),
        jacobian=np.ones((2, 5)),
        friction_coeff=0.6,
    )
    J_g = np.ones((2, 5), dtype=np.float64)
    grf = np.array([300.0, 400.0], dtype=np.float64)
    snapshot = ClosedLoopSnapshot(
        **parts,  # type: ignore[arg-type]
        J_g=J_g,
        foot_contacts=(foot,),
        grf_observed=grf,
    )

    for arr in (
        snapshot.q,
        snapshot.v,
        snapshot.a,
        snapshot.M,
        snapshot.h,
        snapshot.S,
        snapshot.J_c,
        snapshot.Jdot_v,
        snapshot.J_g,
        snapshot.grf_observed,
        snapshot.grip_frame.rotation,
        snapshot.grip_frame.origin,
        snapshot.grip_frame_rotation,
        snapshot.grip_origin,
        snapshot.bare_club.com_m,
        snapshot.bare_club.inertia_com_kg_m2,
        snapshot.foot_contacts[0].point_world_m,
        snapshot.foot_contacts[0].normal_world,
        snapshot.foot_contacts[0].jacobian,
        snapshot.generalized_load(),
        snapshot.actuated_rows(),
        snapshot.unactuated_rows(),
    ):
        assert arr is not None
        assert arr.flags.writeable is False
        with pytest.raises(ValueError):
            arr.flat[0] = 999.0  # type: ignore[index]

    expected_b = parts["M"] @ parts["a"] + parts["h"]  # type: ignore[operator]
    np.testing.assert_allclose(snapshot.generalized_load(), expected_b, atol=1e-12)
    assert snapshot.closure_rows() == 2
    assert snapshot.nc() == 2
    assert snapshot.nv() == 5
    assert snapshot.nu() == 1
    np.testing.assert_array_equal(snapshot.actuated_rows(), np.array([0, 1, 3, 4]))
    np.testing.assert_array_equal(snapshot.unactuated_rows(), np.array([2]))

    record = QuantityRecord(
        values=np.array([1.0, 2.0]),
        units="W",
        allocation_dependence=AllocationDependence.INVARIANT,
        conventions=("C-FRAME-GRIP",),
        hypotheses=(),
        model_provenance=parts["model_provenance"],  # type: ignore[arg-type]
        input_provenance=parts["input_provenance"],  # type: ignore[arg-type]
        validation_status=ValidationStatus.UNVALIDATED,
    )
    assert record.values.flags.writeable is False
    with pytest.raises(ValueError):
        record.values[0] = 0.0

"""Original native force/length counterexample and fail-closed audits (#11856)."""

from __future__ import annotations

import hashlib
import math
from pathlib import Path

import pytest

from src.engines.physics_engines.opensim.python.tour_matching import (
    compute_path_length_finite_difference_moment_arm,
    validate_moment_arm_consistency,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("bad", [math.nan, math.inf, -math.inf])
def test_nonfinite_path_length_cannot_pass_gate(bad: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        validate_moment_arm_consistency(0.1, lambda q: bad, 0.0)


@pytest.mark.parametrize("tol", [math.nan, math.inf, -1.0, 0.0])
def test_invalid_tolerance_cannot_pass_gate(tol: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        validate_moment_arm_consistency(0.1, lambda q: 1.0 - q * 0.1, 0.0, tol)


@pytest.mark.parametrize(
    "q,dq", [(math.nan, 1e-5), (0.0, math.inf), (1e20, 1e-5), (0.0, 1e308)]
)
def test_invalid_or_unresolved_stencil_rejected(q: float, dq: float) -> None:
    with pytest.raises((ValueError, AssertionError)):
        compute_path_length_finite_difference_moment_arm(lambda value: 1.0, q, dq)


def test_original_native_probe_records_workless_identity_failure(
    tmp_path: Path,
) -> None:
    pytest.importorskip("opensim")
    from scripts.diagnostics.native_path_work import TOLERANCE_M, run_probe

    receipt = run_probe(tmp_path)
    assert receipt["scientific_status"] == "unqualified"
    assert receipt["source_binary_equivalence"] == "unverified"
    assert receipt["runtime"]["version"]
    assert receipt["runtime"]["extension_sha256"]
    fixed, moving = receipt["cases"]
    assert fixed["gate"] == "workless_identity_satisfied_at_tested_samples"
    assert moving["gate"] == "workless_identity_not_satisfied"
    assert (
        receipt["interpretation"] == "workless-identity-audit-not-runtime-defect-proof"
    )
    assert moving["guide_work_model"] == "unmodeled-moving-guide"
    for case in (fixed, moving):
        assert len(case["source_sha256"]) == 64
        assert len(case["loaded_sha256"]) == 64
        assert case["source_unchanged"] is True
        assert case["loaded_unchanged"] is True
        assert (case["native_nq"], case["native_nu"], case["native_nz"]) == (1, 1, 0)
        assert case["native_constraints"] == case["native_controllers"] == 0
        assert len(case["samples"]) == 9
        for row in case["samples"]:
            assert row["native_tension_n"] == pytest.approx(1.0)
            assert row["native_moment_arm_m"] == pytest.approx(
                row["native_acceleration_rad_s2"] * case["inertia_kg_m2"], abs=1e-11
            )
            assert row["coordinate_span_rad"] == pytest.approx(2 * row["step_rad"])
            assert row["full_minus_native_m"] == pytest.approx(
                row["same_body_segment_arm_m"], abs=TOLERANCE_M
            )
            assert row["length_speed_m_s"] == pytest.approx(
                -row["full_length_arm_m"], abs=TOLERANCE_M
            )
            if case["moving"]:
                assert row["full_minus_native_m"] == pytest.approx(
                    0.03, abs=TOLERANCE_M
                )
                assert row["gate_rejected"] is True
            else:
                assert abs(row["full_minus_native_m"]) < TOLERANCE_M
                assert row["gate_rejected"] is False
            if row["step_rad"] == 1e-5:
                assert row["full_minus_native_m"] == pytest.approx(
                    0.03 if case["moving"] else 0.0, abs=1e-10
                )


@pytest.mark.parametrize("moving", [False, True])
def test_generic_audit_consumes_native_path_and_independent_force_fixture(
    tmp_path: Path, moving: bool
) -> None:
    osim = pytest.importorskip("opensim")
    from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
        audit_native_path_work,
        inspect_native_path_work_identity,
    )
    from src.engines.physics_engines.opensim.python.tour_matching.native_prepared_state import (
        DeclaredColdStart,
    )
    from scripts.diagnostics.native_path_work import INERTIA_KG_M2, _build_fixture

    source_path = tmp_path / ("moving.osim" if moving else "fixed.osim")
    source_model = _build_fixture(osim, moving)
    if moving:
        source_cable = osim.PathActuator.safeDownCast(
            source_model.updForceSet().get("cable")
        )
        source_path_point = osim.MovingPathPoint.safeDownCast(
            source_cable.updGeometryPath().getPathPointSet().get(1)
        )
        spline = osim.SimmSpline()
        spline.addPoint(-0.3, 0.091)
        spline.addPoint(0.3, 0.109)
        source_path_point.set_x_location(spline)
    source_model.printToXML(str(source_path))
    source_bytes = source_path.read_bytes()
    model = osim.Model(str(source_path))
    initial = model.initSystem()
    coordinate = model.updCoordinateSet().get("angle")
    cable = osim.PathActuator.safeDownCast(model.updForceSet().get("cable"))
    path = cable.getGeometryPath()
    points = path.getPathPointSet()
    state_names = model.getStateVariableNames()
    named_state = {
        state_names.get(index): float(
            model.getStateVariableValue(initial, state_names.get(index))
        )
        for index in range(state_names.getSize())
    }
    declaration = DeclaredColdStart(
        model_path=source_path,
        source_sha256=hashlib.sha256(source_bytes).hexdigest(),
        named_state=named_state,
        time_seconds=0.0,
        lock_targets={},
        chart_bounds={coordinate.getAbsolutePathString(): (-0.5, 0.5)},
        constraint_enforcement={},
        residual_tolerance=1e-9,
    )
    if moving:
        identity = inspect_native_path_work_identity(
            model,
            declaration,
            path_component=path.getAbsolutePathString(),
            moving_point_path=points.get(1).getAbsolutePathString(),
            coordinate_path=coordinate.getAbsolutePathString(),
            osim=osim,
        )
        assert identity.domain_provenance == "native-loaded-source"
        assert identity.function_bounds == pytest.approx((-0.3, 0.3))
    else:
        from src.engines.physics_engines.opensim.python.tour_matching.native_path_work_audit import (
            NativePathWorkIdentity,
        )

        component_paths = (
            path.getAbsolutePathString(),
            coordinate.getAbsolutePathString(),
            *(
                points.get(index).getAbsolutePathString()
                for index in range(points.getSize())
            ),
        )
        identity = NativePathWorkIdentity(
            source_sha256=declaration.source_sha256,
            path_component=path.getAbsolutePathString(),
            coordinate_path=coordinate.getAbsolutePathString(),
            point_paths=tuple(
                points.get(index).getAbsolutePathString()
                for index in range(points.getSize())
            ),
            source_component_paths=component_paths,
            chart_bounds=(-0.5, 0.5),
            active_bounds=(-0.5, 0.5),
            function_bounds=(-1.0, 1.0),
        )
        assert identity.function_bounds == pytest.approx((-1.0, 1.0))

    def sample_path(q: float) -> tuple[float, float]:
        state = osim.State(initial)
        coordinate.setValue(state, q, False)
        model.realizePosition(state)
        return float(coordinate.getValue(state)), float(path.getLength(state))

    def force_observation(q: float) -> tuple[float, float]:
        state = osim.State(initial)
        coordinate.setValue(state, q, False)
        coordinate.setSpeedValue(state, 0.0)
        model.realizeVelocity(state)
        model.setControls(state, osim.Vector(1, 1.0))
        model.realizeAcceleration(state)
        native_arm = float(path.computeMomentArm(state, coordinate))
        force_arm = float(
            coordinate.getAccelerationValue(state)
            * INERTIA_KG_M2
            / cable.getActuation(state)
        )
        return native_arm, force_arm

    audit = audit_native_path_work(
        identity,
        source_bytes,
        q_values=(-0.2, 0.0, 0.2),
        step_sizes=(1e-4, 1e-5),
        sample_path=sample_path,
        native_moment_arm=lambda q: force_observation(q)[0],
        force_moment_arm=lambda q: force_observation(q)[1],
    )
    assert all(
        sample.achieved_span_rad == pytest.approx(2 * sample.step_rad)
        for sample in audit.samples
    )
    assert max(sample.force_residual_m for sample in audit.samples) < 1e-11
    if moving:
        assert audit.status == "workless_identity_not_satisfied"
        assert all(
            sample.workless_residual_m == pytest.approx(0.03, abs=1e-10)
            for sample in audit.samples
        )
    else:
        assert audit.status == "workless_identity_satisfied_at_tested_samples"
        assert max(sample.workless_residual_m for sample in audit.samples) < 1e-7


@pytest.mark.parametrize("level", ["warn", "off"])
def test_evidence_gate_cannot_be_disabled(
    monkeypatch: pytest.MonkeyPatch, level: str
) -> None:
    monkeypatch.setenv("DBC_LEVEL", level)
    with pytest.raises(ValueError, match="finite"):
        validate_moment_arm_consistency(0.1, lambda q: math.nan, 0.0)


def test_finite_difference_overflow_rejected() -> None:
    with pytest.raises(ValueError, match="finite-difference moment arm"):
        compute_path_length_finite_difference_moment_arm(
            lambda q: 1e308 if q > 0 else -1e308, 0.0
        )

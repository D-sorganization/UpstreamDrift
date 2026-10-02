"""Independent auditing evaluates preserved splines without optimizing them."""

from dataclasses import replace
from types import SimpleNamespace
from typing import Any
from numpy.typing import NDArray
import numpy as np
import pytest

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.motion_matching.historical_fit import ImageFitResult
from src.shared.python.motion_matching.constraint_kinematics import (
    ConstraintLinearization,
    ConstraintOptions,
)
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.workspace.necromatcher_ranges import AuthoredCoordinateBounds

pytestmark = pytest.mark.unit


def _fixture() -> tuple[
    ImageFitResult, AuthoredCoordinateBounds, list[tuple[int, dict]]
]:
    times = np.array([0.0, 2.0])
    coefficients = CubicHermiteSplineTrajectory(times, 1).pack(
        np.zeros((2, 1)), np.array([[2.0], [-2.0]])
    )
    result = ImageFitResult(
        times,
        np.zeros((2, 2)),
        1.0,
        1.0,
        np.zeros((2, 1)),
        2,
        "fixture",
        ("joint", "locked"),
        False,
        "authored seed",
        times,
        coefficients,
        ("joint",),
        optimizer_ran=False,
    )
    bounds = AuthoredCoordinateBounds(
        {"joint": (-0.5, 0.5)}, ("locked",), "sha256:" + "1" * 64, "sha256:" + "2" * 64
    )
    frames = [
        (
            index,
            {
                "pts_ticks": tick,
                "timebase_numerator": 1,
                "timebase_denominator": 1,
                "frame_id": str(index),
                "frame_sha256": str(index) * 64,
            },
        )
        for index, tick in ((0, 0), (1, 2))
    ]
    return result, bounds, frames


class _NativeProbe:
    def constraint_residual_jacobian(
        self, q: NDArray[np.float64], options: ConstraintOptions
    ) -> ConstraintLinearization:
        residual = np.array([q[0], 0, 0, q[0] / 10, 0, 0, min(0.0, -q[0])])
        labels = (
            "grip_position:x",
            "grip_position:y",
            "grip_position:z",
            "grip_rotation:x",
            "grip_rotation:y",
            "grip_rotation:z",
            "ground:foot",
        )
        return ConstraintLinearization(
            residual, np.zeros((7, 2)), ("joint", "locked"), labels
        )

    def closure_error(self, q: NDArray[np.float64]) -> tuple[float, float]:
        return abs(q[0]), abs(q[0]) / 10

    def sphere_heights(
        self, q: NDArray[np.float64], ground: GroundPlane
    ) -> dict[str, float]:
        return {"foot": -float(q[0])}


def test_audit_detects_between_frame_extrema_and_keeps_certificates_separate():
    from docs.development.historical_capture.audit_bounded_trial import audit_trajectory

    fit, bounds, frames = _fixture()
    before = fit.q.copy(), fit.spline_coefficients.copy()
    report = audit_trajectory(
        fit, bounds, frames, _NativeProbe(), GroundPlane((0, 0, 1), 0)
    )
    assert (
        report["q_bound_certificate"]["conservative_bernstein_q_domain_verified"]
        is False
    )
    assert report["coordinate_extrema"]["violating_coordinates"] == ["joint"]
    assert report["source_frame_count"] == 2
    midpoint = next(
        row
        for row in report["sampled_native_constraints"]
        if row["kind"] == "adjacent_source_midpoint"
    )
    assert midpoint["pts_rational"] == {"numerator": 1, "denominator": 1}
    assert midpoint["source_frame_indices"] == [0, 1]
    assert midpoint["grip_gap_m"] == pytest.approx(1)
    assert midpoint["ground_penetration_m"] == pytest.approx(1)
    assert any(
        row["kind"] == "coordinate_global_extremum" and row["source_time"] == 1
        for row in report["sampled_native_constraints"]
    )
    assert report["continuous_nonlinear_certified"] is False
    assert report["optimization_performed"] is False
    assert report["saved_optimizer_ran"] is False
    np.testing.assert_array_equal(fit.q, before[0])
    np.testing.assert_array_equal(fit.spline_coefficients, before[1])


def test_constant_feasible_seed_has_q_certificate_without_nonlinear_acceptance():
    from docs.development.historical_capture.audit_bounded_trial import audit_trajectory

    fit, bounds, frames = _fixture()
    fit = replace(fit, spline_coefficients=np.zeros(4))
    report = audit_trajectory(
        fit, bounds, frames, _NativeProbe(), GroundPlane((0, 0, 1), 0)
    )
    assert (
        report["q_bound_certificate"]["conservative_bernstein_q_domain_verified"]
        is True
    )
    assert report["scientific_acceptance"] is False


def test_duplicate_pts_or_saved_sample_disagreement_cannot_be_audited():
    from docs.development.historical_capture.audit_bounded_trial import audit_trajectory

    fit, bounds, frames = _fixture()
    frames[1][1]["pts_ticks"] = 0
    with pytest.raises(ValueError):
        audit_trajectory(fit, bounds, frames, _NativeProbe(), GroundPlane((0, 0, 1), 0))
    _, _, frames = _fixture()
    with pytest.raises(ValueError, match="samples"):
        audit_trajectory(
            replace(fit, q=np.ones((2, 2))),
            bounds,
            frames,
            _NativeProbe(),
            GroundPlane((0, 0, 1), 0),
        )


def test_dense_source_frames_and_midpoints_keep_exact_rational_identities():
    from docs.development.historical_capture.audit_bounded_trial import audit_trajectory

    fit, bounds, frames = _fixture()
    interior = dict(frames[0][1], pts_ticks=1, timebase_denominator=3, frame_id="5")
    frames = [frames[0], (5, interior), (9, frames[1][1])]
    report = audit_trajectory(
        fit, bounds, frames, _NativeProbe(), GroundPlane((0, 0, 1), 0)
    )
    rows = report["sampled_native_constraints"]
    assert report["source_frame_count"] == 3
    assert rows[0]["time_representation"] == "exact_container_pts"
    assert rows[0]["pts_rational"] == {"numerator": 0, "denominator": 1}
    midpoints = [row for row in rows if row["kind"] == "adjacent_source_midpoint"]
    assert [row["pts_rational"] for row in midpoints] == [
        {"numerator": 1, "denominator": 6},
        {"numerator": 7, "denominator": 6},
    ]
    assert [row["source_frame_indices"] for row in midpoints] == [[0, 5], [5, 9]]
    assert all(row["image_objective_point_added"] is False for row in rows)


@pytest.mark.parametrize("changed", ["source_sha256", "runtime_sha256", "script"])
def test_audit_rejects_source_runtime_or_script_changes(changed: str):
    from docs.development.historical_capture.audit_bounded_trial import (
        _verify_audit_stamp,
    )

    before = {"source_sha256": "source", "runtime_sha256": "runtime"}
    after = dict(before)
    script_after = "script"
    if changed == "script":
        script_after = "changed"
    else:
        after[changed] = "changed"
    with pytest.raises(ValueError, match="changed during assessment"):
        _verify_audit_stamp(before, after, "script", script_after)
    _verify_audit_stamp(before, before, "script", "script")


@pytest.mark.parametrize("changed", ["fit", "model", "capture"])
def test_public_parent_hash_change_rejects_audit(changed: str):
    from docs.development.historical_capture.audit_bounded_trial import _check_parents

    class Library:
        def load_asset(self, identity: str) -> Any:
            return SimpleNamespace(
                metadata={"hash": "changed" if identity == changed else identity}
            )

        def load_fit(self, identity: str) -> dict[str, Any]:
            return {}

    binding = SimpleNamespace(
        fit_id="fit",
        fit_hash="fit",
        model_id="model",
        model_hash="model",
        fit={"capture_id": "capture", "capture_hash": "capture"},
    )
    with pytest.raises(ValueError, match="parent identity changed"):
        _check_parents(Library(), binding)

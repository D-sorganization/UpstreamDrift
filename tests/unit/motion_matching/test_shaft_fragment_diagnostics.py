from dataclasses import FrozenInstanceError
import sys
import subprocess
import numpy as np
import pytest

pytestmark = pytest.mark.unit
from src.shared.python.motion_matching.historical_fit import (
    shaft_fragment_diagnostics as fd,
)
from src.shared.python.motion_matching.historical_fit.contracts import CameraProjection
from src.shared.python.motion_matching.historical_fit.shaft_observations import (
    ShaftAxisSegment,
)
from src.shared.python.motion_matching.marker_kinematics import MarkerLinearization


def options(**kwargs):
    fields = {
        "coordinate_order": ("angle", "offset", "locked"),
        "coordinate_units": ("rad", "rad", "m"),
        "selected_coordinates": ("angle", "offset"),
        "steps": (1e-5, 1e-6, 1e-7),
        "bounds": (None, None, None),
        "marker_labels": ("shaft_a", "shaft_b"),
    }
    fields.update(kwargs)
    return fd.FragmentDerivativeOptions(**fields)


def fragment(points=((510, 410), (525, 450))):
    return ShaftAxisSegment(
        "observed", points, "synthetic", "synthetic fixture", 0.5, None, 2.0
    )


class Provider:
    def __init__(self, degenerate=False):
        self.calls = 0
        self.degenerate = degenerate

    def marker_linearization(self, q):
        self.calls += 1
        angle, offset, _ = q
        a = np.array([0.2 + 0.1 * offset, -0.1, 2.0 + 0.08 * offset])
        b = a + np.array([np.cos(angle), np.sin(angle), 0.3])
        jac = np.zeros((2, 3, 3))
        jac[:, 0, 1] = 0.1
        jac[:, 2, 1] = 0.08
        jac[1, :, 0] = [-np.sin(angle), np.cos(angle), 0]
        if self.degenerate:
            b = a.copy()
            jac[1] = jac[0]
        return MarkerLinearization(
            np.array([a, b]), jac, ("shaft_a", "shaft_b"), ("angle", "offset", "locked")
        )


def camera():
    angle = 0.2
    rotation = np.array(
        [
            [np.cos(angle), 0, np.sin(angle)],
            [0, 1, 0],
            [-np.sin(angle), 0, np.cos(angle)],
        ]
    )
    return CameraProjection(
        np.array([[500.0, 0, 640], [0, 510, 360], [0, 0, 1]]),
        rotation,
        np.array([0, 0, 0.5]),
    )


def independently_project(q):
    a = np.array([0.2 + 0.1 * q[1], -0.1, 2 + 0.08 * q[1]])
    b = a + np.array([np.cos(q[0]), np.sin(q[0]), 0.3])
    c = camera()
    rows = np.array([c.rotation @ a + c.translation, c.rotation @ b + c.translation])
    return np.column_stack(
        (500 * rows[:, 0] / rows[:, 2] + 640, 510 * rows[:, 1] / rows[:, 2] + 360)
    )


def independent_distances(q):
    pixels = independently_project(q)
    delta = pixels[1] - pixels[0]
    normal = np.array([-delta[1], delta[0]]) / np.linalg.norm(delta)
    return (np.array([[510, 410], [525, 450]]) - pixels[0]) @ normal


def test_perspective_chain_independent_derivatives_and_units() -> None:
    provider = Provider()
    q = np.array([0.4, 0.2, 0.0])
    result = fd.assess_fragment_derivatives(
        provider, camera(), q, fragment(), options()
    )
    np.testing.assert_allclose(
        result.raw_distances_px, independent_distances(q), atol=1e-12, rtol=0
    )
    assert len(result.checks) == 6 and all(
        row.units == "px_per_rad" for row in result.checks
    )
    for check in result.checks:
        column = ("angle", "offset").index(check.coordinate)
        h = check.step
        plus = q.copy()
        minus = q.copy()
        plus[column] += h
        minus[column] -= h
        expected = (independent_distances(plus) - independent_distances(minus)) / (
            2 * h
        )
        np.testing.assert_allclose(check.central, expected, rtol=0, atol=1e-6)
        np.testing.assert_allclose(check.analytic, expected, rtol=0, atol=2e-6)
    assert provider.calls == 13


def test_zero_response_is_real_zero_not_unavailable() -> None:
    class Fixed(Provider):
        def marker_linearization(self, q):
            row = super().marker_linearization(np.zeros(3))
            return MarkerLinearization(
                row.positions,
                np.zeros_like(row.jacobian),
                row.marker_labels,
                row.coordinate_order,
            )

    result = fd.assess_fragment_derivatives(
        Fixed(), camera(), [0, 0, 0], fragment(), options()
    )
    assert all(
        row.status == "available"
        and row.analytic == (0.0, 0.0)
        and row.central == (0.0, 0.0)
        for row in result.checks
    )


def test_disabled_and_abstention_make_no_provider_calls() -> None:
    p = Provider()
    assert fd.optional_fragment_derivatives(None, None, None, None, None, False) is None
    abstain = ShaftAxisSegment("ambiguous", None, "reviewer", "blur", None, None, None)
    result = fd.assess_fragment_derivatives(p, camera(), [0, 0, 0], abstain, options())
    assert (
        result.status == "explicit_abstention"
        and result.raw_distances_px is None
        and result.checks == ()
        and p.calls == 0
    )
    with pytest.raises(ValueError):
        fd.optional_fragment_derivatives(None, None, None, None, None, 1)


def test_degenerate_and_bounds_are_null_without_clipping() -> None:
    result = fd.assess_fragment_derivatives(
        Provider(True), camera(), [0, 0, 0], fragment(), options()
    )
    assert result.status == "degenerate_projection" and result.raw_distances_px is None
    p = Provider()
    q = [0, 0, 0]
    result = fd.assess_fragment_derivatives(
        p, camera(), q, fragment(), options(bounds=((0.0, 1.0), None, None))
    )
    limited = [row for row in result.checks if row.coordinate == "angle"]
    assert all(
        row.status == "central_step_outside_authored_bound"
        and row.central is None
        and row.max_abs_error is None
        for row in limited
    )
    assert p.calls == 7 and q == [0, 0, 0]


@pytest.mark.parametrize(
    "kwargs",
    [
        {"steps": (True,)},
        {"steps": (float("nan"),)},
        {"steps": (0.0,)},
        {"coordinate_order": ("angle", "angle")},
        {"selected_coordinates": ("foreign",)},
        {"selected_coordinates": ("locked",)},
        {"bounds": ((False, 1), None, None)},
        {"bounds": ((2, 1), None, None)},
    ],
)
def test_strict_options(kwargs) -> None:
    with pytest.raises(ValueError):
        options(**kwargs)


@pytest.mark.parametrize("q", [[True, 0, 0], [float("nan"), 0, 0], [0, 0]])
def test_strict_pose(q) -> None:
    p = Provider()
    with pytest.raises(ValueError):
        fd.assess_fragment_derivatives(p, camera(), q, fragment(), options())
    assert p.calls == 0


def test_order_shape_depth_capability_fail_closed() -> None:
    class Wrong(Provider):
        def marker_linearization(self, q):
            row = super().marker_linearization(q)
            return MarkerLinearization(
                row.positions,
                row.jacobian,
                row.marker_labels,
                ("offset", "angle", "locked"),
            )

    with pytest.raises(ValueError):
        fd.assess_fragment_derivatives(
            Wrong(), camera(), [0, 0, 0], fragment(), options()
        )

    class Missing:
        def marker_linearization(self, q):
            raise NotImplementedError

    result = fd.assess_fragment_derivatives(
        Missing(), camera(), [0, 0, 0], fragment(), options()
    )
    assert result.status == "marker_capability_unavailable"
    behind = CameraProjection(np.eye(3), np.eye(3), np.array([0, 0, -10.0]))
    with pytest.raises(ValueError, match="front"):
        fd.assess_fragment_derivatives(
            Provider(), behind, [0, 0, 0], fragment(), options()
        )


def test_copied_immutable_options_results_and_reject_forged_check() -> None:
    names = ["angle", "offset", "locked"]
    steps = [1e-5]
    bounds = [None, None, None]
    config = options(coordinate_order=names, steps=steps, bounds=bounds)
    names[0] = "foreign"
    steps[0] = 9
    assert config.coordinate_order[0] == "angle" and config.steps == (1e-5,)
    with pytest.raises(FrozenInstanceError):
        config.steps = (2.0,)
    with pytest.raises(ValueError):
        fd.FragmentDerivativeCheck(
            "angle", True, "px_per_rad", (0, 0), (0, 0), 0, "available"
        )
    with pytest.raises(ValueError):
        fd.FragmentDerivativeCheck(
            "angle", 0.1, "px_per_rad", (0, 0), (1, 1), 0, "available"
        )


def test_import_is_sdk_free_in_cold_process() -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "from src.shared.python.motion_matching.historical_fit import shaft_fragment_diagnostics; import sys; assert not any(n in sys.modules for n in ('mujoco','pydrake','pinocchio','opensim'))",
        ],
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr


@pytest.mark.parametrize(
    "points", [[(1, 2), (1, 2)], [(True, 2), (3, 4)], [(float("nan"), 2), (3, 4)]]
)
def test_observed_fragment_contract_rejects_bad_points(points) -> None:
    with pytest.raises(ValueError):
        fragment(points)


def test_result_nested_alias_and_invalid_shape_status() -> None:
    analytic = [1.0, 2.0]
    central = [1.0, 2.0]
    row = fd.FragmentDerivativeCheck(
        "angle", 0.1, "px_per_rad", analytic, central, 0.0, "available"
    )
    analytic[0] = 9.0
    central[0] = 8.0
    assert row.analytic == (1.0, 2.0) and row.central == (1.0, 2.0)
    with pytest.raises(ValueError):
        fd.FragmentDerivativeCheck(
            "angle", 0.1, "px_per_rad", (float("nan"), 0), (0, 0), 0, "available"
        )
    with pytest.raises(ValueError):
        fd.FragmentDerivativeCheck(
            "angle", 0.1, "px_per_rad", (0, 0), (0, 0), 0, "unknown"
        )
    with pytest.raises(ValueError):
        fd.FragmentDerivativeAssessment(options(), "available", (0, 0), ())
    with pytest.raises(ValueError):
        fd.FragmentDerivativeAssessment(options(), "explicit_abstention", (0, 0), ())
    with pytest.raises(FrozenInstanceError):
        row.central = (0.0, 0.0)


def test_curated_exports_and_execution_fingerprint() -> None:
    import hashlib
    from pathlib import Path
    from src.shared.python.motion_matching import historical_fit
    from src.shared.python.workspace.necromatcher_fit_jobs import fit_execution_stamp

    for name in (
        "FragmentDerivativeOptions",
        "FragmentDerivativeCheck",
        "FragmentDerivativeAssessment",
        "assess_fragment_derivatives",
        "optional_fragment_derivatives",
    ):
        assert getattr(historical_fit, name) is getattr(fd, name)
        assert name in historical_fit.__all__
    key = (
        "src/shared/python/motion_matching/historical_fit/shaft_fragment_diagnostics.py"
    )
    assert (
        fit_execution_stamp()["source_files"][key]
        == "sha256:" + hashlib.sha256(Path(fd.__file__).read_bytes()).hexdigest()
    )

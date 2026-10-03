"""The optional image seam permits independent residual families and fixed rows."""

from dataclasses import replace
import numpy as np
import pytest
from src.shared.python.estimation import (
    CubicHermiteSplineTrajectory,
    finite_difference_jacobian,
)
from src.shared.python.motion_matching.historical_fit import (
    ImageFitInputs,
    ImageFitConfig,
    ImageResidualAssessment,
)
from src.shared.python.motion_matching.historical_fit.solver import _Fit
from tests.unit.motion_matching.test_shaft_residuals import setup

pytestmark = pytest.mark.unit


class ThreeRowTerm:
    source_times = (0.25, 0.75)

    def validate(self, native, identity):
        pass

    def residual(self, native, camera, poses):
        return np.array([poses[0, 0] ** 2, poses[1, 1], poses[0, 0] + 2 * poses[1, 1]])

    def assess(self, native, camera, poses):
        return ImageResidualAssessment(
            evidence_sha256="sha256:" + "a" * 64,
            frame_indices=(0, 1),
            source_times=self.source_times,
            raw_rms_pixels=0.0,
        )


def test_generic_three_rows_chain_both_independent_pose_bases():
    native, camera, _, bundle = setup()
    inputs = ImageFitInputs(
        np.array([0.0, 1.0]),
        np.zeros((2, 1, 2)),
        np.ones((2, 1)),
        np.zeros(2),
        np.ones(2),
        ("x", "angle"),
    )
    fit = _Fit(
        native,
        {"body": ("club", (0, 1, 4))},
        camera,
        inputs,
        ImageFitConfig(closure_weight=0),
        replace(bundle, terms=(ThreeRowTerm(),)),
    )
    trajectory = CubicHermiteSplineTrajectory(inputs.source_times, 2)
    coefficients = trajectory.initial_coefficients_from_samples(
        inputs.source_times, np.array([[0.1, 0.2], [0.3, 0.4]])
    )
    evaluation = trajectory.evaluate(coefficients, fit.map_evaluation_times)
    expected = finite_difference_jacobian(
        lambda c: fit.residual(trajectory.evaluate(c, fit.map_evaluation_times), {}),
        coefficients,
    )
    np.testing.assert_allclose(
        fit.jacobian(evaluation, {}, None), expected, rtol=2e-5, atol=2e-6
    )
    assert fit._additional_jacobian(
        evaluation, fit.additional_images.terms[0]
    ).shape == (3, len(coefficients))

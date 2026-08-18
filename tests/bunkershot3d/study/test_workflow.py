"""End-to-end design-study workflow and manifest replay (#8615).

Exercises the intended sequence on a stand-in objective: screen with Morris,
quantify with Sobol', fit a surrogate, then rank candidate designs with their
uncertainty -- and re-run the whole thing from the recorded manifests to show
a sweep is reproducible from its artifact.
"""

from __future__ import annotations

import json

import numpy as np
import pytest
from bunkershot3d.study import (
    DesignSpace,
    GaussianProcess,
    StudyManifest,
    compare_predicted_designs,
    morris_screening,
    sobol_analysis,
)

pytestmark = pytest.mark.unit

#: A stand-in for a wedge-sole objective: two live factors, one inert.
WEDGE_SPACE = DesignSpace.from_bounds(
    {
        "sole_width_mm": (10.0, 22.0),
        "bounce_deg": (4.0, 14.0),
        "shaft_paint_thickness_um": (5.0, 40.0),
    },
    {
        "sole_width_mm": "mm",
        "bounce_deg": "deg",
        "shaft_paint_thickness_um": "um",
    },
)


def surrogate_objective(points: np.ndarray) -> np.ndarray:
    """A smooth, deterministic stand-in for a solver run.

    Depends on sole width and bounce (with an interaction) and not at all on
    the third factor, which exists to be screened out.

    Args:
        points: ``(n, 3)`` design points in physical units.

    Returns:
        A ``(n,)`` array of pseudo-objective values (lower is better).
    """
    width = (points[:, 0] - 16.0) / 6.0
    bounce = (points[:, 1] - 9.0) / 5.0
    return width**2 + 0.8 * bounce**2 + 0.5 * width * bounce


class TestScreeningThenSensitivity:
    """Morris first, Sobol' on the survivors."""

    def test_screening_drops_the_inert_factor(self) -> None:
        result = morris_screening(
            WEDGE_SPACE, surrogate_objective, n_trajectories=12, seed=1
        )
        assert result.inert() == ("shaft_paint_thickness_um",)
        assert result.n_evaluations == 12 * 4

    def test_sobol_agrees_that_the_inert_factor_contributes_nothing(self) -> None:
        result = sobol_analysis(
            WEDGE_SPACE, surrogate_objective, 2**12, seed=2, n_bootstrap=100
        )
        index = result.names.index("shaft_paint_thickness_um")
        assert abs(result.first_order[index]) < 1e-3
        assert result.total_order[index] < 1e-3
        assert result.ranked()[-1] == "shaft_paint_thickness_um"

    def test_sobol_reports_the_width_bounce_interaction(self) -> None:
        result = sobol_analysis(WEDGE_SPACE, surrogate_objective, 2**13, seed=3)
        interaction = result.interaction_strength()
        assert interaction[0] > 0.01
        assert interaction[1] > 0.01

    def test_screening_is_two_orders_of_magnitude_cheaper(self) -> None:
        screen = morris_screening(
            WEDGE_SPACE, surrogate_objective, n_trajectories=12, seed=4
        )
        full = sobol_analysis(WEDGE_SPACE, surrogate_objective, 2**12, seed=4)
        assert screen.n_evaluations * 100 < full.n_evaluations


class TestSurrogateAssistedComparison:
    """Rank candidate grinds from a surrogate, with uncertainty."""

    def test_ranks_candidate_grinds_with_uncertainty(self) -> None:
        training = WEDGE_SPACE.sample(64, "sobol", seed=5)
        gp = GaussianProcess(space=WEDGE_SPACE).fit(
            training.values,
            surrogate_objective(training.values),
            optimize=True,
            n_restarts=2,
            seed=6,
        )

        candidates = np.array(
            [
                [16.0, 9.0, 20.0],
                [12.0, 6.0, 20.0],
                [21.0, 13.0, 20.0],
            ]
        )
        mean, std = gp.predict(candidates, return_std=True)
        comparison = compare_predicted_designs(
            ("neutral", "low_bounce", "wide_high_bounce"), mean, std, seed=7
        )

        assert comparison.best == "neutral"
        assert comparison.probability_best.sum() == pytest.approx(1.0)
        # The surrogate is trained on 64 points of a smooth function, so it
        # should be confident enough to separate these three.
        assert comparison.probability_best[0] > 0.9

    def test_surrogate_tracks_the_objective_on_held_out_points(self) -> None:
        training = WEDGE_SPACE.sample(64, "sobol", seed=8)
        gp = GaussianProcess(space=WEDGE_SPACE).fit(
            training.values,
            surrogate_objective(training.values),
            optimize=True,
            n_restarts=2,
            seed=9,
        )
        held_out = WEDGE_SPACE.sample(32, "halton", seed=10)
        predicted = gp.predict(held_out.values)
        truth = surrogate_objective(held_out.values)
        assert np.max(np.abs(predicted - truth)) < 0.05 * (truth.max() - truth.min())


class TestManifestReplay:
    """A sweep must be reproducible from its recorded manifest alone."""

    def test_sobol_analysis_replays_from_a_serialised_manifest(self) -> None:
        original = sobol_analysis(WEDGE_SPACE, surrogate_objective, 2**10, seed=None)
        assert original.manifest is not None

        payload = json.dumps(original.manifest.to_dict(), allow_nan=False)
        manifest = StudyManifest.from_dict(json.loads(payload))
        assert manifest.numpy_version == np.__version__
        assert manifest.extra["n_base"] == 2**10

        replay = sobol_analysis(
            WEDGE_SPACE,
            surrogate_objective,
            manifest.extra["n_base"],
            seed=manifest.seed.entropy,
        )
        np.testing.assert_array_equal(original.first_order, replay.first_order)
        np.testing.assert_array_equal(original.total_order, replay.total_order)

    def test_screening_replays_from_a_serialised_manifest(self) -> None:
        original = morris_screening(
            WEDGE_SPACE, surrogate_objective, n_trajectories=8, seed=None
        )
        assert original.manifest is not None
        manifest = StudyManifest.from_dict(
            json.loads(json.dumps(original.manifest.to_dict(), allow_nan=False))
        )
        replay = morris_screening(
            WEDGE_SPACE,
            surrogate_objective,
            n_trajectories=manifest.extra["n_trajectories"],
            n_levels=manifest.extra["n_levels"],
            seed=manifest.seed.entropy,
        )
        np.testing.assert_array_equal(original.mu_star, replay.mu_star)

    def test_manifests_record_the_parameter_names_in_column_order(self) -> None:
        result = sobol_analysis(WEDGE_SPACE, surrogate_objective, 2**8, seed=11)
        assert result.manifest is not None
        assert result.manifest.parameter_names == WEDGE_SPACE.names
        assert result.names == WEDGE_SPACE.names

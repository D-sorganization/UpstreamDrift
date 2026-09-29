"""Sobol' sensitivity indices against published analytic values (#8615).

The acceptance criterion for this module is numerical, not structural: the
Saltelli/Jansen estimators must land on the closed-form indices of the
Ishigami function and Sobol's g-function.
"""

from __future__ import annotations

import numpy as np
import pytest
from bunkershot3d.study import (
    DesignSpace,
    ishigami,
    ishigami_indices,
    ishigami_space,
    saltelli_design,
    sobol_analysis,
    sobol_g,
    sobol_g_indices,
    sobol_g_space,
    sobol_indices_from_outputs,
)

pytestmark = pytest.mark.unit

#: Published Ishigami indices for a = 7, b = 0.1 (Saltelli et al. 2010, Table 3).
PUBLISHED_ISHIGAMI_S1 = np.array([0.3139, 0.4424, 0.0])
PUBLISHED_ISHIGAMI_ST = np.array([0.5576, 0.4424, 0.2437])

#: Accuracy the estimator is required to reach at N = 2**15. The worst error
#: measured over five seeds is 2.4e-4, so this is a ~20x margin.
ISHIGAMI_TOLERANCE = 5e-3
#: Same, for the g-function at N = 2**14 (worst measured error 1.5e-3).
G_FUNCTION_TOLERANCE = 1e-2

#: Standard g-function coefficients: two dominant, two moderate, two inert.
G_COEFFICIENTS = np.array([0.0, 0.5, 3.0, 9.0, 99.0, 99.0])


def g_model(points: np.ndarray) -> np.ndarray:
    """Evaluate the g-function with the standard coefficients.

    Args:
        points: ``(n, 6)`` design matrix.

    Returns:
        A ``(n,)`` array of outputs.
    """
    return sobol_g(points, G_COEFFICIENTS)


class TestAnalyticReference:
    """The derived reference must agree with the published decimals."""

    def test_ishigami_matches_published_values(self) -> None:
        exact = ishigami_indices()
        np.testing.assert_allclose(exact.first_order, PUBLISHED_ISHIGAMI_S1, atol=1e-4)
        np.testing.assert_allclose(exact.total_order, PUBLISHED_ISHIGAMI_ST, atol=1e-4)

    def test_ishigami_third_factor_has_no_first_order_effect(self) -> None:
        exact = ishigami_indices()
        assert exact.first_order[2] == 0.0
        # ...but it is not inert: it acts only through the x1-x3 interaction.
        assert exact.total_order[2] > 0.2

    def test_g_function_indices_are_ordered_by_coefficient(self) -> None:
        exact = sobol_g_indices(G_COEFFICIENTS)
        assert np.all(np.diff(exact.first_order) <= 0.0)
        assert np.all(exact.total_order >= exact.first_order - 1e-12)

    def test_g_function_rejects_negative_coefficients(self) -> None:
        with pytest.raises(ValueError, match="non-negative"):
            sobol_g_indices(np.array([-1.0, 1.0]))


class TestIshigamiAccuracy:
    """Estimated indices versus the closed form."""

    @pytest.mark.scientific
    @pytest.mark.parametrize("seed", [11, 2029])
    def test_matches_analytic_first_and_total_order(self, seed: int) -> None:
        exact = ishigami_indices()
        result = sobol_analysis(ishigami_space(), ishigami, 2**15, seed=seed)

        np.testing.assert_allclose(
            result.first_order, exact.first_order, atol=ISHIGAMI_TOLERANCE
        )
        np.testing.assert_allclose(
            result.total_order, exact.total_order, atol=ISHIGAMI_TOLERANCE
        )
        np.testing.assert_allclose(
            result.first_order, PUBLISHED_ISHIGAMI_S1, atol=ISHIGAMI_TOLERANCE
        )
        np.testing.assert_allclose(
            result.total_order, PUBLISHED_ISHIGAMI_ST, atol=ISHIGAMI_TOLERANCE
        )

    @pytest.mark.scientific
    def test_recovers_the_total_variance(self) -> None:
        exact = ishigami_indices()
        result = sobol_analysis(ishigami_space(), ishigami, 2**15, seed=3)
        assert result.variance == pytest.approx(exact.variance, rel=0.02)

    def test_cost_is_n_times_dimension_plus_two(self) -> None:
        result = sobol_analysis(ishigami_space(), ishigami, 2**10, seed=1)
        assert result.n_base == 2**10
        assert result.n_evaluations == 2**10 * (3 + 2)

    def test_ranks_x2_and_x1_above_x3_by_first_order(self) -> None:
        result = sobol_analysis(ishigami_space(), ishigami, 2**13, seed=5)
        assert result.ranked(use_total=False)[-1] == "x3"

    def test_reports_the_x1_x3_interaction(self) -> None:
        result = sobol_analysis(ishigami_space(), ishigami, 2**14, seed=5)
        interaction = result.interaction_strength()
        assert interaction[1] < 0.02
        assert interaction[0] > 0.15
        assert interaction[2] > 0.15

    @pytest.mark.scientific
    def test_error_shrinks_as_the_sample_grows(self) -> None:
        exact = ishigami_indices()

        def worst_error(n_base: int) -> float:
            errors = [
                float(
                    np.max(
                        np.abs(
                            sobol_analysis(
                                ishigami_space(), ishigami, n_base, seed=seed
                            ).first_order
                            - exact.first_order
                        )
                    )
                )
                for seed in (0, 1, 2, 3)
            ]
            return float(np.median(errors))

        assert worst_error(2**14) < worst_error(2**8)


class TestGFunctionAccuracy:
    """The g-function is the harder, non-smooth benchmark."""

    @pytest.mark.scientific
    def test_matches_analytic_indices(self) -> None:
        exact = sobol_g_indices(G_COEFFICIENTS)
        space = sobol_g_space(G_COEFFICIENTS.size)
        result = sobol_analysis(space, g_model, 2**14, seed=404)

        np.testing.assert_allclose(
            result.first_order, exact.first_order, atol=G_FUNCTION_TOLERANCE
        )
        np.testing.assert_allclose(
            result.total_order, exact.total_order, atol=G_FUNCTION_TOLERANCE
        )

    def test_identifies_the_inert_factors(self) -> None:
        space = sobol_g_space(G_COEFFICIENTS.size)
        result = sobol_analysis(space, g_model, 2**13, seed=7)
        assert result.ranked()[:2] == ("x1", "x2")
        assert set(result.ranked()[-2:]) == {"x5", "x6"}
        assert np.all(result.total_order[4:] < 0.01)


class TestBootstrapIntervals:
    """Confidence intervals must bracket the truth and be ordered."""

    @pytest.mark.scientific
    def test_intervals_contain_the_analytic_values(self) -> None:
        exact = ishigami_indices()
        result = sobol_analysis(
            ishigami_space(), ishigami, 2**13, seed=99, n_bootstrap=300
        )
        assert result.first_order_ci is not None
        assert result.total_order_ci is not None
        for i in range(3):
            low, high = result.first_order_ci[i]
            assert low <= exact.first_order[i] <= high
            low, high = result.total_order_ci[i]
            assert low <= exact.total_order[i] <= high

    def test_intervals_are_ordered_and_bracket_the_estimate(self) -> None:
        result = sobol_analysis(
            ishigami_space(), ishigami, 2**12, seed=6, n_bootstrap=200
        )
        assert result.first_order_ci is not None
        assert np.all(result.first_order_ci[:, 0] <= result.first_order_ci[:, 1])
        assert np.all(result.total_order_ci[:, 0] <= result.total_order_ci[:, 1])

    def test_no_intervals_when_bootstrap_disabled(self) -> None:
        result = sobol_analysis(ishigami_space(), ishigami, 2**10, seed=1)
        assert result.first_order_ci is None
        assert result.total_order_ci is None

    def test_rejects_impossible_confidence_level(self) -> None:
        design = saltelli_design(ishigami_space(), 2**6, seed=1)
        outputs = ishigami(design.design_matrix())
        f_a, f_b, f_ab = design.split_outputs(outputs)
        with pytest.raises(ValueError, match="confidence_level"):
            sobol_indices_from_outputs(
                f_a, f_b, f_ab, names=("x1", "x2", "x3"), confidence_level=1.5
            )


class TestScipyParity:
    """Cross-check against ``scipy.stats.sobol_indices`` on identical samples.

    SciPy ships the same estimator pair, so feeding it our own evaluations is
    a direct check that the sampling and bookkeeping are right rather than
    merely self-consistent.
    """

    @pytest.mark.scientific
    def test_matches_scipy_on_the_same_evaluations(self) -> None:
        scipy_stats = pytest.importorskip("scipy.stats")
        if not hasattr(scipy_stats, "sobol_indices"):
            pytest.skip("scipy.stats.sobol_indices requires SciPy >= 1.11")

        design = saltelli_design(ishigami_space(), 2**12, seed=1234)
        outputs = ishigami(design.design_matrix())
        f_a, f_b, f_ab = design.split_outputs(outputs)
        ours = sobol_indices_from_outputs(f_a, f_b, f_ab, names=ishigami_space().names)

        theirs = scipy_stats.sobol_indices(
            func={
                "f_A": f_a[np.newaxis, :],
                "f_B": f_b[np.newaxis, :],
                "f_AB": f_ab[:, np.newaxis, :],
            },
            n=2**12,
        )
        np.testing.assert_allclose(
            ours.first_order, np.ravel(theirs.first_order), atol=1e-12
        )
        np.testing.assert_allclose(
            ours.total_order, np.ravel(theirs.total_order), atol=1e-12
        )


class TestSaltelliDesign:
    """Structure of the cross-sampling plan."""

    def test_cross_samples_replace_exactly_one_column(self) -> None:
        design = saltelli_design(ishigami_space(), 2**5, seed=8)
        for i in range(3):
            block = design.ab_unit[i]
            np.testing.assert_array_equal(block[:, i], design.b_unit[:, i])
            others = [j for j in range(3) if j != i]
            np.testing.assert_array_equal(block[:, others], design.a_unit[:, others])

    def test_a_and_b_are_distinct_samples(self) -> None:
        design = saltelli_design(ishigami_space(), 2**6, seed=8)
        assert not np.allclose(design.a_unit, design.b_unit)

    def test_design_matrix_row_blocks_line_up_with_split(self) -> None:
        design = saltelli_design(ishigami_space(), 2**5, seed=8)
        matrix = design.design_matrix()
        assert matrix.shape == (2**5 * 5, 3)
        outputs = np.arange(matrix.shape[0], dtype=float)
        f_a, f_b, f_ab = design.split_outputs(outputs)
        np.testing.assert_array_equal(f_a, outputs[: 2**5])
        np.testing.assert_array_equal(f_b, outputs[2**5 : 2**6])
        assert f_ab.shape == (3, 2**5)

    def test_same_seed_reproduces_the_design(self) -> None:
        first = saltelli_design(ishigami_space(), 2**6, seed=555)
        second = saltelli_design(ishigami_space(), 2**6, seed=555)
        np.testing.assert_array_equal(first.a_unit, second.a_unit)
        np.testing.assert_array_equal(first.ab_unit, second.ab_unit)

    def test_rejects_non_power_of_two_base_size(self) -> None:
        with pytest.raises(ValueError, match="power of two"):
            saltelli_design(ishigami_space(), 1000, seed=1)

    def test_split_rejects_wrong_length(self) -> None:
        design = saltelli_design(ishigami_space(), 2**4, seed=1)
        with pytest.raises(ValueError, match="expected"):
            design.split_outputs(np.zeros(7))


class TestFailureModes:
    """Bad inputs must raise rather than produce plausible numbers."""

    def test_nan_outputs_raise(self) -> None:
        design = saltelli_design(ishigami_space(), 2**4, seed=1)
        outputs = ishigami(design.design_matrix())
        outputs[3] = np.nan
        f_a, f_b, f_ab = design.split_outputs(outputs)
        with pytest.raises(ValueError, match="NaN"):
            sobol_indices_from_outputs(f_a, f_b, f_ab, names=("x1", "x2", "x3"))

    def test_constant_output_raises(self) -> None:
        n = 2**4
        with pytest.raises(ValueError, match="variance is zero"):
            sobol_indices_from_outputs(
                np.ones(n),
                np.ones(n),
                np.ones((2, n)),
                names=("a", "b"),
            )

    def test_mismatched_shapes_raise(self) -> None:
        with pytest.raises(ValueError, match="f_ab must have shape"):
            sobol_indices_from_outputs(
                np.zeros(8), np.ones(8), np.zeros((3, 8)), names=("a", "b")
            )

    def test_model_returning_wrong_length_raises(self) -> None:
        space = DesignSpace.from_bounds({"a": (0.0, 1.0), "b": (0.0, 1.0)})
        with pytest.raises(ValueError, match="model returned"):
            sobol_analysis(space, lambda x: np.zeros(3), 2**4, seed=1)

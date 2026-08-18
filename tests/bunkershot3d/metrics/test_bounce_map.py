"""Bounce utilisation of the sole, against hand arithmetic (issue #8614, W7).

The four-element fixture holds constant loads ``[100, 0, 50, 0] N`` for 10 ms
over areas ``[1, 2, 1, 2] cm^2``, so:

* impulses          ``[1.0, 0, 0.5, 0] N.s``
* impulse densities ``[1.0e4, 0, 5.0e3, 0] Pa.s``
* loaded at 1 %     elements 0 and 2
* utilised area     ``1e-4 + 1e-4 = 2e-4 m^2`` of ``6e-4 m^2``, i.e. one third
* removable area    ``4e-4 m^2`` -- the free material
"""

from __future__ import annotations

import numpy as np
import pytest

from bunkershot3d.metrics import SoleLoadTrace, bounce_utilisation

from .conftest import build_sole_load_trace

pytestmark = pytest.mark.unit


@pytest.fixture
def load() -> SoleLoadTrace:
    """The four-element reference load trace."""
    return build_sole_load_trace()


@pytest.fixture
def utilisation(load):
    """Bounce utilisation of the reference load trace."""
    return bounce_utilisation(load)


class TestSoleLoadTrace:
    """The input contract a solver has to satisfy to get this metric."""

    def test_a_negative_element_load_is_refused(self) -> None:
        """Sand pushes; a negative normal load is a sign error, not suction."""
        with pytest.raises(ValueError, match="non-negative"):
            build_sole_load_trace(forces_N=np.array([100.0, -1.0, 50.0, 0.0]))

    def test_a_zero_area_element_is_refused(self) -> None:
        """A zero area makes the impulse density infinite."""
        with pytest.raises(ValueError, match="strictly positive"):
            build_sole_load_trace(areas_m2=np.array([1e-4, 0.0, 1e-4, 2e-4]))

    def test_mismatched_force_and_element_counts_are_refused(self) -> None:
        """The (T, E) block has to agree with the element arrays."""
        with pytest.raises(ValueError, match="element_normal_force_N must have shape"):
            SoleLoadTrace(
                time_s=np.array([0.0, 1.0, 2.0]),
                element_centroid_body_m=np.zeros((2, 3)),
                element_area_m2=np.array([1e-4, 1e-4]),
                element_normal_force_N=np.zeros((3, 3)),
            )


class TestBounceUtilisation:
    """Which sole area carried the load, and how much is removable."""

    def test_element_impulses_are_force_times_duration(self, utilisation) -> None:
        """100 N and 50 N held for 10 ms are 1.0 and 0.5 N.s."""
        np.testing.assert_allclose(
            utilisation.element_impulse_Ns, [1.0, 0.0, 0.5, 0.0], atol=1e-12
        )
        assert utilisation.total_impulse_Ns == pytest.approx(1.5, rel=1e-12)

    def test_impulse_density_is_per_unit_area(self, utilisation) -> None:
        """1.0 N.s over 1e-4 m^2 is 1e4 Pa.s; 0.5 over 1e-4 is 5e3."""
        np.testing.assert_allclose(
            utilisation.element_impulse_density_Pa_s,
            [1.0e4, 0.0, 5.0e3, 0.0],
            atol=1e-9,
        )

    def test_utilised_and_removable_areas(self, utilisation) -> None:
        """Two of the four elements carry load: 2e-4 m^2 of 6e-4 m^2."""
        assert utilisation.total_area_m2 == pytest.approx(6.0e-4, rel=1e-12)
        assert utilisation.utilised_area_m2 == pytest.approx(2.0e-4, rel=1e-12)
        assert utilisation.utilisation_fraction == pytest.approx(1.0 / 3.0, rel=1e-12)
        assert utilisation.removable_area_m2 == pytest.approx(4.0e-4, rel=1e-12)
        np.testing.assert_array_equal(
            utilisation.loaded_mask, [True, False, True, False]
        )

    def test_the_threshold_is_relative_to_the_peak_density(self, load) -> None:
        """At a 60 % threshold only the 1e4 Pa.s element counts as loaded."""
        utilisation = bounce_utilisation(load, threshold_fraction=0.6)

        np.testing.assert_array_equal(
            utilisation.loaded_mask, [True, False, False, False]
        )
        assert utilisation.utilised_area_m2 == pytest.approx(1.0e-4, rel=1e-12)

    def test_centre_of_pressure_is_the_impulse_weighted_centroid(
        self, load, utilisation
    ) -> None:
        """(1.0 * c0 + 0.5 * c2) / 1.5, with c0 and c2 the loaded centroids."""
        expected = (
            1.0 * load.element_centroid_body_m[0]
            + 0.5 * load.element_centroid_body_m[2]
        ) / 1.5

        np.testing.assert_allclose(
            utilisation.centre_of_pressure_body_m, expected, atol=1e-12
        )

    def test_an_unloaded_sole_is_refused_rather_than_divided_by_zero(self) -> None:
        """No load means no utilisation to report."""
        with pytest.raises(ValueError, match="carried no load"):
            bounce_utilisation(build_sole_load_trace(forces_N=np.zeros(4)))

    def test_the_threshold_must_be_a_fraction(self, load) -> None:
        """A threshold above 1 could never be met; below 0 is meaningless."""
        with pytest.raises(ValueError, match="threshold_fraction must be in"):
            bounce_utilisation(load, threshold_fraction=1.5)


class TestLoadProfile:
    """The one-dimensional grind chart."""

    def test_binning_heel_to_toe_splits_the_impulse(self, load, utilisation) -> None:
        """The heel-side pair carries 1.0 N.s, the toe-side pair 0.5 N.s.

        Element centroids run -0.030 to +0.030 m along body axis 1, so two equal
        bins split at 0: elements 0 and 1 fall in the first, 2 and 3 in the
        second.
        """
        profile = utilisation.profile(load, axis_index=1, n_bins=2)

        np.testing.assert_allclose(profile.impulse_Ns, [1.0, 0.5], atol=1e-12)
        np.testing.assert_allclose(profile.area_m2, [3.0e-4, 3.0e-4], atol=1e-12)
        np.testing.assert_allclose(
            profile.impulse_fraction, [2.0 / 3.0, 1.0 / 3.0], rtol=1e-12
        )

    def test_every_bin_of_a_profile_sums_to_the_total_impulse(
        self, load, utilisation
    ) -> None:
        """Binning re-partitions the impulse; it does not create or destroy it."""
        profile = utilisation.profile(load, axis_index=0, n_bins=3)

        assert profile.impulse_Ns.sum() == pytest.approx(
            utilisation.total_impulse_Ns, rel=1e-12
        )
        assert profile.area_m2.sum() == pytest.approx(
            utilisation.total_area_m2, rel=1e-12
        )

    def test_a_bad_axis_or_bin_count_is_refused(self, load, utilisation) -> None:
        """There is no fourth body axis and no zero-bin histogram."""
        with pytest.raises(ValueError, match="axis_index must be"):
            utilisation.profile(load, axis_index=3, n_bins=2)
        with pytest.raises(ValueError, match="n_bins must be at least 1"):
            utilisation.profile(load, axis_index=0, n_bins=0)

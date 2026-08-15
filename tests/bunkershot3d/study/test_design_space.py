"""Design-space construction, scaling and low-discrepancy sampling (#8615)."""

from __future__ import annotations

import numpy as np
import pytest
from bunkershot3d.study import DesignParameter, DesignSpace, is_power_of_two
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st

pytestmark = pytest.mark.unit

WEDGE_BOUNDS = {
    "sole_width_mm": (10.0, 22.0),
    "bounce_deg": (4.0, 14.0),
    "leading_edge_radius_mm": (1.0, 8.0),
}
WEDGE_UNITS = {
    "sole_width_mm": "mm",
    "bounce_deg": "deg",
    "leading_edge_radius_mm": "mm",
}


def wedge_space() -> DesignSpace:
    """Build a small, realistic wedge-sole design space.

    Returns:
        A three-parameter space using the Acushnet sole vocabulary.
    """
    return DesignSpace.from_bounds(WEDGE_BOUNDS, WEDGE_UNITS)


class TestDesignParameter:
    """Validation of a single design parameter."""

    def test_records_name_bounds_and_units(self) -> None:
        parameter = DesignParameter("bounce_deg", 4.0, 14.0, "deg")
        assert parameter.name == "bounce_deg"
        assert parameter.units == "deg"
        assert parameter.span == pytest.approx(10.0)

    @pytest.mark.parametrize(
        ("lower", "upper"),
        [(1.0, 1.0), (2.0, 1.0), (np.nan, 1.0), (0.0, np.inf)],
    )
    def test_rejects_degenerate_bounds(self, lower: float, upper: float) -> None:
        with pytest.raises(ValueError):
            DesignParameter("x", lower, upper)

    def test_rejects_empty_name(self) -> None:
        with pytest.raises(ValueError, match="non-empty"):
            DesignParameter("", 0.0, 1.0)


class TestDesignSpace:
    """Structure and bookkeeping of a design space."""

    def test_exposes_names_units_and_bounds_in_column_order(self) -> None:
        space = wedge_space()
        assert space.names == tuple(WEDGE_BOUNDS)
        assert space.units == ("mm", "deg", "mm")
        assert space.dimension == 3
        np.testing.assert_allclose(space.lower, [10.0, 4.0, 1.0])
        np.testing.assert_allclose(space.upper, [22.0, 14.0, 8.0])

    def test_rejects_duplicate_names(self) -> None:
        with pytest.raises(ValueError, match="duplicate"):
            DesignSpace(
                (
                    DesignParameter("x", 0.0, 1.0),
                    DesignParameter("x", 0.0, 2.0),
                )
            )

    def test_rejects_empty_space(self) -> None:
        with pytest.raises(ValueError, match="at least one"):
            DesignSpace(())

    def test_index_of_rejects_unknown_parameter(self) -> None:
        with pytest.raises(KeyError, match="unknown parameter"):
            wedge_space().index_of("loft_deg")

    def test_contains_flags_out_of_bounds_points(self) -> None:
        space = wedge_space()
        assert bool(space.contains(np.array([16.0, 10.0, 4.0])))
        assert not bool(space.contains(np.array([16.0, 100.0, 4.0])))

    def test_to_unit_cube_rejects_out_of_bounds(self) -> None:
        with pytest.raises(ValueError, match="outside the space bounds"):
            wedge_space().to_unit_cube(np.array([[16.0, 40.0, 4.0]]))

    def test_to_physical_rejects_values_outside_the_cube(self) -> None:
        with pytest.raises(ValueError, match=r"\[0, 1\]"):
            wedge_space().to_physical(np.array([[0.5, 1.5, 0.5]]))

    def test_to_physical_rejects_wrong_width(self) -> None:
        with pytest.raises(ValueError, match="columns"):
            wedge_space().to_physical(np.zeros((4, 2)))


class TestUnitCubeRoundTrip:
    """The physical <-> unit-cube mapping must be an exact bijection."""

    def test_corners_map_to_bounds(self) -> None:
        space = wedge_space()
        np.testing.assert_allclose(space.to_physical(np.zeros(3)), space.lower)
        np.testing.assert_allclose(space.to_physical(np.ones(3)), space.upper)

    @settings(deadline=None, max_examples=200)
    @given(
        unit=st.lists(
            st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
            min_size=3,
            max_size=3,
        )
    )
    def test_round_trips_unit_cube_to_physical_and_back(
        self, unit: list[float]
    ) -> None:
        space = wedge_space()
        point = np.asarray(unit)
        physical = space.to_physical(point)
        assert np.all(physical >= space.lower - 1e-12)
        assert np.all(physical <= space.upper + 1e-12)
        np.testing.assert_allclose(space.to_unit_cube(physical), point, atol=1e-12)

    @settings(deadline=None, max_examples=200)
    @given(
        lower=st.floats(-1e3, 1e3, allow_nan=False, allow_infinity=False),
        width=st.floats(1e-3, 1e3, allow_nan=False, allow_infinity=False),
        fraction=st.floats(0.0, 1.0, allow_nan=False, allow_infinity=False),
    )
    def test_round_trips_for_arbitrary_bounds(
        self, lower: float, width: float, fraction: float
    ) -> None:
        space = DesignSpace.from_bounds({"p": (lower, lower + width)})
        physical = space.to_physical(np.array([fraction]))
        np.testing.assert_allclose(
            space.to_unit_cube(physical), np.array([fraction]), atol=1e-9
        )


class TestSampling:
    """Low-discrepancy sampling and its guard rails."""

    @pytest.mark.parametrize("method", ["sobol", "lhs", "halton", "random"])
    def test_samples_lie_inside_the_bounds(self, method: str) -> None:
        space = wedge_space()
        sample = space.sample(64, method, seed=17)
        assert sample.values.shape == (64, 3)
        assert np.all(space.contains(sample.values))
        assert np.all(sample.unit_cube >= 0.0)
        assert np.all(sample.unit_cube <= 1.0)

    @settings(
        deadline=None,
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        power=st.integers(min_value=1, max_value=6),
        method=st.sampled_from(["sobol", "lhs", "halton", "random"]),
        seed=st.integers(min_value=0, max_value=2**32 - 1),
    )
    def test_property_every_sample_is_inside_the_space(
        self, power: int, method: str, seed: int
    ) -> None:
        space = wedge_space()
        sample = space.sample(2**power, method, seed=seed)
        assert np.all(space.contains(sample.values))
        np.testing.assert_allclose(
            space.to_physical(sample.unit_cube), sample.values, atol=1e-12
        )

    def test_orthogonal_array_lhs_accepts_prime_squares(self) -> None:
        sample = wedge_space().sample(49, "lhs_oa", seed=5)
        assert sample.values.shape == (49, 3)
        assert np.all(wedge_space().contains(sample.values))

    def test_rejects_non_power_of_two_sobol_size(self) -> None:
        with pytest.raises(ValueError, match="power-of-two"):
            wedge_space().sample(100, "sobol", seed=1)

    def test_rejects_non_prime_square_orthogonal_lhs_size(self) -> None:
        with pytest.raises(ValueError, match="square of a prime"):
            wedge_space().sample(64, "lhs_oa", seed=1)

    def test_orthogonal_lhs_rejects_too_many_parameters(self) -> None:
        space = DesignSpace.from_bounds({f"x{i}": (0.0, 1.0) for i in range(5)})
        with pytest.raises(ValueError, match="at most"):
            space.sample(4, "lhs_oa", seed=1)

    def test_rejects_unknown_sampler(self) -> None:
        with pytest.raises(ValueError, match="unknown sampler"):
            wedge_space().sample(8, "latin-supercube", seed=1)  # type: ignore[arg-type]

    def test_rejects_non_positive_size(self) -> None:
        with pytest.raises(ValueError, match="must be positive"):
            wedge_space().sample(0, "sobol", seed=1)

    def test_column_lookup_matches_matrix_column(self) -> None:
        sample = wedge_space().sample(16, "sobol", seed=2)
        np.testing.assert_array_equal(sample.column("bounce_deg"), sample.values[:, 1])
        assert sample.n_samples == 16


class TestReproducibility:
    """Seeds are recorded and replaying one reproduces the matrix exactly."""

    @pytest.mark.parametrize("method", ["sobol", "lhs", "halton", "random"])
    def test_same_seed_gives_identical_sample(self, method: str) -> None:
        first = wedge_space().sample(32, method, seed=20260813)
        second = wedge_space().sample(32, method, seed=20260813)
        np.testing.assert_array_equal(first.values, second.values)

    def test_different_seeds_give_different_samples(self) -> None:
        first = wedge_space().sample(32, "sobol", seed=1)
        second = wedge_space().sample(32, "sobol", seed=2)
        assert not np.allclose(first.values, second.values)

    def test_manifest_replays_the_sample(self) -> None:
        original = wedge_space().sample(32, "halton", seed=None)
        manifest = original.manifest
        replay = wedge_space().sample(
            manifest.n_samples, "halton", seed=manifest.seed.entropy
        )
        np.testing.assert_array_equal(original.values, replay.values)

    def test_manifest_records_entropy_and_library_versions(self) -> None:
        sample = wedge_space().sample(8, "sobol", seed=None)
        manifest = sample.manifest
        assert 0 <= manifest.seed.entropy < 2**128
        assert manifest.seed.numpy_version == np.__version__
        assert manifest.parameter_names == wedge_space().names
        assert manifest.method == "sobol"
        assert manifest.scipy_version


def test_is_power_of_two_matches_definition() -> None:
    assert [n for n in range(1, 20) if is_power_of_two(n)] == [1, 2, 4, 8, 16]
    assert not is_power_of_two(0)
    assert not is_power_of_two(-8)

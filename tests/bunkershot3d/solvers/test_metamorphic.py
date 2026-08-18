"""Metamorphic relations for the F0 solver (issue #8611).

Why the transforms are chosen to be bit-exact
---------------------------------------------

A metamorphic test is only as sharp as its tolerance.  Every transform
used here is exact in IEEE-754 double precision:

* the body's vertices are integer multiples of ``2 ** -10 m``, and every
  translation is a power of two, so ``vertex + offset`` is representable
  with no rounding and every edge difference, normal and area comes out
  bit-identical;
* the rotations are quarter turns about the vertical, whose matrices
  have entries in ``{0, +1, -1}``, so the matrix product is a
  permutation with sign flips and introduces no error at all;
* the reflections are coordinate mirrors, with the same property.

So the only error left in the comparison is the solver's own arithmetic,
and the assertions can be made at 1e-14 rather than at a tolerance
tuned until the test passed.

The relations
-------------

For a transform ``x -> M x + t`` with ``M`` orthogonal:

===================  ====================================================
Quantity             Image
===================  ====================================================
force                ``M F``
torque about origin  ``det(M) M T``   (a cross product is a pseudo-vector)
===================  ====================================================

Quarter turns about the vertical have ``det = +1``; the mirrors have
``det = -1``, which is what makes the reflection relation a *sign flip*
rather than a rotation.  Only rotations about the vertical and
horizontal translations are invariances of the *problem*: the bed is a
horizontal half space, so a vertical translation is only an invariance
when the free surface moves with the body, and that is how it is
applied here.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from numpy.typing import NDArray

from bunkershot3d.solvers import DRFTSolver, IntrusionState, SurfaceElements, Wrench

pytestmark = pytest.mark.unit

_TOLERANCE = 1e-14
_DYADIC_OFFSETS_M = (-4.0, -0.5, -0.0625, 0.0, 0.0625, 0.25, 1.0, 4.0)
"""Powers of two (and zero). Exact on a 2^-10 vertex grid."""

_IDENTITY = np.eye(3, dtype=np.float64)
_QUARTER_TURN_Z = np.array(
    [[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]], dtype=np.float64
)
_MIRROR_X = np.diag([-1.0, 1.0, 1.0]).astype(np.float64)
_MIRROR_Y = np.diag([1.0, -1.0, 1.0]).astype(np.float64)

# Entry velocity: a 25 m/s descending blow, with dyadic components so
# the velocity is exact under the same transforms.
_ENTRY_VELOCITY_M_S = np.array([16.0, 4.0, -2.0], dtype=np.float64)
_FREE_SURFACE_M = 0.0


def _solve(
    solver: DRFTSolver,
    elements: SurfaceElements,
    velocity_m_s: NDArray[np.float64],
    *,
    reference_point_m: NDArray[np.float64],
    free_surface_height_m: float,
) -> Wrench:
    """Solve one query and return its wrench."""
    return solver.solve(
        IntrusionState(
            elements,
            velocity_m_s,
            reference_point_m=reference_point_m,
            free_surface_height_m=free_surface_height_m,
        )
    ).wrench


def _transform(
    elements: SurfaceElements,
    rotation: NDArray[np.float64],
    translation: NDArray[np.float64],
) -> SurfaceElements:
    """Apply ``x -> M x + t`` to the element arrays directly.

    Normals are mapped by ``M`` rather than re-derived from a mirrored
    triangle, because a mirror has determinant -1 and re-deriving would
    silently flip every outward normal inward.
    """
    return SurfaceElements(
        elements.centroids_m @ rotation.T + translation,
        elements.normals @ rotation.T,
        elements.areas_m2,
    )


def _assert_close(
    actual: NDArray[np.float64], expected: NDArray[np.float64], what: str
) -> None:
    """Compare to 1e-14 relative, scaled by the magnitude in play."""
    scale = max(float(np.abs(expected).max()), 1e-30)
    np.testing.assert_allclose(
        actual, expected, rtol=_TOLERANCE, atol=_TOLERANCE * scale, err_msg=what
    )


class TestTranslationInvariance:
    """Moving the whole problem cannot change the answer."""

    @pytest.mark.parametrize("offset", _DYADIC_OFFSETS_M)
    def test_horizontal_translation_leaves_the_wrench_untouched(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements, offset: float
    ) -> None:
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        shift = np.array([offset, -offset, 0.0])
        moved = _solve(
            solver,
            dyadic_elements.translated(shift),
            _ENTRY_VELOCITY_M_S,
            reference_point_m=shift,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        _assert_close(moved.force_n, base.force_n, "force under translation")
        _assert_close(moved.torque_n_m, base.torque_n_m, "torque under translation")

    @pytest.mark.parametrize("offset", _DYADIC_OFFSETS_M)
    def test_vertical_translation_with_the_free_surface_is_an_invariance(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements, offset: float
    ) -> None:
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        shift = np.array([0.0, 0.0, offset])
        moved = _solve(
            solver,
            dyadic_elements.translated(shift),
            _ENTRY_VELOCITY_M_S,
            reference_point_m=shift,
            free_surface_height_m=_FREE_SURFACE_M + offset,
        )
        _assert_close(moved.force_n, base.force_n, "force under vertical translation")
        _assert_close(
            moved.torque_n_m, base.torque_n_m, "torque under vertical translation"
        )

    def test_moving_the_body_without_the_free_surface_is_not_an_invariance(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements
    ) -> None:
        # The bed is a half space, so a purely vertical move is a change
        # of depth. If this ever stopped failing, the depth term would
        # have gone missing.
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        shift = np.array([0.0, 0.0, 0.0625])
        raised = _solve(
            solver,
            dyadic_elements.translated(shift),
            _ENTRY_VELOCITY_M_S,
            reference_point_m=shift,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        assert raised.force_magnitude_n != pytest.approx(
            base.force_magnitude_n, rel=1e-6
        )


class TestRotationEquivariance:
    """A quarter turn about the vertical rotates the wrench with it."""

    @pytest.mark.parametrize("turns", [1, 2, 3, 4])
    def test_resultant_wrench_rotates_with_the_body(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements, turns: int
    ) -> None:
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        rotation = np.linalg.matrix_power(_QUARTER_TURN_Z, turns)
        turned = _solve(
            solver,
            _transform(dyadic_elements, rotation, origin),
            rotation @ _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        _assert_close(turned.force_n, rotation @ base.force_n, "force under rotation")
        _assert_close(
            turned.torque_n_m, rotation @ base.torque_n_m, "torque under rotation"
        )

    def test_four_quarter_turns_return_the_original_answer(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements
    ) -> None:
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        full = np.linalg.matrix_power(_QUARTER_TURN_Z, 4)
        assert np.array_equal(full, _IDENTITY)
        returned = _solve(
            solver,
            _transform(dyadic_elements, full, origin),
            full @ _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        _assert_close(returned.force_n, base.force_n, "force after a full turn")


class TestReflectionAntisymmetry:
    """A mirror flips the wrench, and the torque flips the other way."""

    @pytest.mark.parametrize(("mirror", "name"), [(_MIRROR_X, "x"), (_MIRROR_Y, "y")])
    def test_force_mirrors_and_torque_mirrors_with_a_sign_change(
        self,
        solver: DRFTSolver,
        dyadic_elements: SurfaceElements,
        mirror: NDArray[np.float64],
        name: str,
    ) -> None:
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        mirrored = _solve(
            solver,
            _transform(dyadic_elements, mirror, origin),
            mirror @ _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        _assert_close(
            mirrored.force_n, mirror @ base.force_n, f"force mirrored in {name}"
        )
        _assert_close(
            mirrored.torque_n_m,
            -(mirror @ base.torque_n_m),
            f"torque mirrored in {name}",
        )

    def test_a_symmetric_body_struck_squarely_has_no_lateral_force(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements
    ) -> None:
        # The plate is symmetric about y = 0 and the velocity lies in
        # that plane, so the reflection relation forces F_y = 0 exactly.
        wrench = _solve(
            solver,
            dyadic_elements,
            np.array([16.0, 0.0, -2.0]),
            reference_point_m=np.zeros(3),
            free_surface_height_m=_FREE_SURFACE_M,
        )
        assert abs(wrench.force_n[1]) < _TOLERANCE * wrench.force_magnitude_n


class TestElementPermutationInvariance:
    """The answer is an integral, so element order must not matter."""

    @settings(
        deadline=None,
        max_examples=25,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(seed=st.integers(min_value=0, max_value=2**31 - 1))
    def test_shuffling_the_element_arrays_changes_nothing(
        self, solver: DRFTSolver, dyadic_elements: SurfaceElements, seed: int
    ) -> None:
        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        order = np.random.default_rng(seed).permutation(len(dyadic_elements))
        shuffled = SurfaceElements(
            dyadic_elements.centroids_m[order],
            dyadic_elements.normals[order],
            dyadic_elements.areas_m2[order],
        )
        permuted = _solve(
            solver,
            shuffled,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        _assert_close(permuted.force_n, base.force_n, "force under permutation")
        _assert_close(permuted.torque_n_m, base.torque_n_m, "torque under permutation")


_TRANSFORM_STEPS = st.sampled_from(
    [
        ("rotate", _QUARTER_TURN_Z),
        ("mirror", _MIRROR_X),
        ("mirror", _MIRROR_Y),
        ("identity", _IDENTITY),
    ]
)


class TestComposedTransforms:
    """Hypothesis composes the relations and checks them together.

    A single relation can be satisfied by a bug that happens to be
    symmetric. Composing rotations, mirrors and translations in random
    order and checking one law -- ``F -> M F``, ``T -> det(M) M T`` --
    is much harder to satisfy by accident.
    """

    @settings(
        deadline=None,
        max_examples=60,
        suppress_health_check=[HealthCheck.function_scoped_fixture],
    )
    @given(
        steps=st.lists(_TRANSFORM_STEPS, min_size=0, max_size=5),
        offset_x=st.sampled_from(_DYADIC_OFFSETS_M),
        offset_y=st.sampled_from(_DYADIC_OFFSETS_M),
        offset_z=st.sampled_from(_DYADIC_OFFSETS_M),
    )
    def test_composed_isometry_maps_the_wrench_by_the_same_isometry(
        self,
        solver: DRFTSolver,
        dyadic_elements: SurfaceElements,
        steps: list[tuple[str, NDArray[np.float64]]],
        offset_x: float,
        offset_y: float,
        offset_z: float,
    ) -> None:
        matrix = _IDENTITY.copy()
        for _kind, step in steps:
            matrix = step @ matrix
        translation = np.array([offset_x, offset_y, offset_z], dtype=np.float64)
        determinant = float(np.linalg.det(matrix))

        origin = np.zeros(3)
        base = _solve(
            solver,
            dyadic_elements,
            _ENTRY_VELOCITY_M_S,
            reference_point_m=origin,
            free_surface_height_m=_FREE_SURFACE_M,
        )
        transformed = _solve(
            solver,
            _transform(dyadic_elements, matrix, translation),
            matrix @ _ENTRY_VELOCITY_M_S,
            reference_point_m=translation,
            free_surface_height_m=_FREE_SURFACE_M + offset_z,
        )
        _assert_close(
            transformed.force_n, matrix @ base.force_n, "force under composed isometry"
        )
        _assert_close(
            transformed.torque_n_m,
            determinant * (matrix @ base.torque_n_m),
            "torque under composed isometry",
        )

"""Structural DRY and frozen numerical parity for the impulse kernel.

Reference values came from pre-refactor source SHA-256
69e782795d60b54360efaf2f27e910b7f3eeb56697673b344bbd7f2a56bdb9c5.
"""

from __future__ import annotations

import ast
import inspect
import textwrap

import numpy as np
import pytest

from src.shared.python.physics import incremental_contact_kernel as kernel

pytestmark = pytest.mark.unit


def test_trial_call_arguments_have_one_shared_numerical_helper() -> None:
    """Four integration sites must not repeat the full trial call signature."""
    tree = ast.parse(textwrap.dedent(inspect.getsource(kernel._integrate_source)))
    direct_calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "trial_solver"
    ]
    assert not direct_calls
    assert hasattr(kernel, "_trial_at_step_source")


def test_kernel_stages_fit_architecture_limits() -> None:
    """Shared inputs and staged integration remove four temporary exceptions."""
    for name in ("_trial_source", "_integrate_source", "integrate_impulse"):
        node = ast.parse(
            textwrap.dedent(inspect.getsource(getattr(kernel, name)))
        ).body[0]
        assert isinstance(node, ast.FunctionDef)
        assert len(node.args.args) + len(node.args.kwonlyargs) <= 8
    source = inspect.getsource(kernel._integrate_source)
    assert len(source.splitlines()) <= 100


@pytest.mark.parametrize(
    ("relative", "friction", "restitution", "steps", "expected"),
    [
        (
            [45.0, 3.0, -1.0],
            0.0,
            0.0,
            32,
            ([1.6071428571428572, 0.0, 0.0], 36.1607142857143, 0.0, True),
        ),
        (
            [45.0, 3.0, -1.0],
            0.2,
            0.8,
            64,
            (
                [2.89160915841199, -0.05389924480187901, 0.03355794142996635],
                36.01861483308658,
                23.05191349317542,
                True,
            ),
        ),
        (
            [45.0, 3.0, -1.0],
            1.0,
            1.0,
            32,
            (
                [3.212223951094436, -0.07322525293985252, 0.04098465618219551],
                35.98302313339841,
                35.98302313339842,
                True,
            ),
        ),
        (
            [45.0, 0.0, 0.0],
            0.4,
            0.8,
            32,
            (
                [2.8928571428571446, 0.0, 0.0],
                36.1607142857143,
                23.142857142857164,
                False,
            ),
        ),
    ],
)
@pytest.mark.parametrize("backend", ["python_reference", "numba"])
def test_frozen_pre_refactor_kernel_results(
    relative, friction, restitution, steps, expected, backend
) -> None:
    if backend == "numba" and kernel.numba is None:
        pytest.skip("Numba unavailable")
    normal = np.array([1.0, 0.0, 0.0])
    tangent = np.array([[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    mass = np.array([[28.0, 1.5, -0.7], [1.5, 25.0, 0.3], [-0.7, 0.3, 31.0]])
    if not any(relative[1:]):
        mass = np.diag([28.0, 25.0, 31.0])
    tangent_mass = tangent.T @ mass @ tangent
    eigenvalues, eigenvectors = np.linalg.eigh(tangent_mass)
    inputs = kernel.ContactKernelInput(
        relative_initial=np.array(relative),
        normal=normal,
        inverse_mass=mass,
        tangent=tangent,
        tangent_mass=tangent_mass,
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        friction=friction,
        energetic_restitution=restitution,
        steps_per_compression=steps,
    )
    impulse, compression, rebound, slipped = kernel.integrate_impulse(
        inputs,
        backend=backend,
    )
    np.testing.assert_allclose(impulse, expected[0], rtol=0, atol=1e-10)
    assert compression == pytest.approx(expected[1], abs=1e-10)
    assert rebound == pytest.approx(expected[2], abs=1e-10)
    assert slipped is expected[3]

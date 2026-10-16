"""OpenSim muscle frame for the co-contraction experiment (NSX-94, #12723).

Builds a :class:`MuscleFrame` from a model and a state: the moment-arm matrix
``R = -dL_MT/dq`` (``computeMomentArm``), the equilibrated tendon tension and the
fibre / tendon stiffnesses at a grid of activations, and optionally the
moment-arm Jacobian.  Millard muscles have a compliant tendon, so the series
stiffness ``k_m k_T / (k_m + k_T)`` has both terms from the engine; the muscles
with ``ignore_tendon_compliance`` make the frame ``RIGID``.

The state is copied; the caller's state is never modified.  At each grid
activation the muscle is set to that activation and re-equilibrated
(``equilibrateMuscles``), so the tabulated tension is the equilibrium tension
along the tendon, not an instantaneous one.  That force is *nonlinear* in the
activation for a compliant tendon, which is why the frame keeps the table and
:func:`~src.shared.python.loop_nullspace.muscle_frame.qualify_frame` measures the
secant-law residual instead of assuming linearity.

Non-claims: the generic model's force-length-velocity curves are not a golfer's
muscles; the frame is a software mechanics object, not a physiological
measurement.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np

from src.shared.python.loop_nullspace.evidence import (
    Array,
    InputProvenance,
    ModelProvenance,
)
from src.shared.python.loop_nullspace.muscle_frame import MuscleFrame, TendonKind

__all__ = ["opensim_muscle_frame"]

DEFAULT_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
GEOMETRY_STEP = 1e-5
"""Central-difference step (rad or m) for ``dR/dq``; truncation ``h^2 R'''/6`` and
round-off ``eps R / h`` are both ~1e-11 for ``|R| ~ 0.05``, far below the
``1e-6`` the jacobian is used at."""


def _muscles(model: Any, names: Sequence[str] | None) -> list[Any]:
    muscle_set = model.getMuscles()
    every = [muscle_set.get(i) for i in range(muscle_set.getSize())]
    if names is None:
        return every
    by_name = {mu.getName(): mu for mu in every}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"muscles not in the model: {missing}")
    return [by_name[n] for n in names]


def _coordinates(model: Any, names: Sequence[str] | None) -> list[Any]:
    coord_set = model.getCoordinateSet()
    every = [coord_set.get(i) for i in range(coord_set.getSize())]
    if names is None:
        return every
    by_name = {c.getName(): c for c in every}
    missing = [n for n in names if n not in by_name]
    if missing:
        raise ValueError(f"coordinates not in the model: {missing}")
    return [by_name[n] for n in names]


def _arms(model: Any, state: Any, muscles: list[Any], coords: list[Any]) -> Array:
    model.realizePosition(state)
    return np.array(
        [[mu.computeMomentArm(state, c) for c in coords] for mu in muscles],
        dtype=np.float64,
    )


def _arm_jacobian(
    model: Any, state: Any, muscles: list[Any], coords: list[Any]
) -> Array:
    """``dR[k, i] / dq_j`` by central differences; the state is restored."""
    out = np.zeros((len(muscles), len(coords), len(coords)))
    for j, coord in enumerate(coords):
        q0 = coord.getValue(state)
        columns = []
        for sign in (1.0, -1.0):
            coord.setValue(state, q0 + sign * GEOMETRY_STEP)
            columns.append(_arms(model, state, muscles, coords))
        coord.setValue(state, q0)
        out[:, :, j] = (columns[0] - columns[1]) / (2.0 * GEOMETRY_STEP)
    model.realizePosition(state)
    return out


def _tabulate(
    model: Any, state: Any, muscles: list[Any], grid: Sequence[float]
) -> tuple[Array, Array, Array]:
    """Equilibrated tension, fibre and tendon stiffness per grid activation."""
    force = np.zeros((len(grid), len(muscles)))
    k_fib = np.zeros_like(force)
    k_ten = np.zeros_like(force)
    for g, act in enumerate(grid):
        for mu in muscles:
            mu.setActivation(state, float(act))
        model.equilibrateMuscles(state)
        model.realizeDynamics(state)
        for k, mu in enumerate(muscles):
            force[g, k] = max(float(mu.getTendonForce(state)), 0.0)
            k_fib[g, k] = float(mu.getFiberStiffnessAlongTendon(state))
            k_ten[g, k] = float(mu.getTendonStiffness(state))
    return force, k_fib, k_ten


def opensim_muscle_frame(
    model: Any,
    state: Any,
    *,
    muscles: Sequence[str] | None = None,
    coordinates: Sequence[str] | None = None,
    grid: Sequence[float] = DEFAULT_GRID,
    with_geometry: bool = False,
    source: str = "opensim-state",
) -> MuscleFrame:
    """The :class:`MuscleFrame` of ``model`` at ``state`` (see module doc).

    Preconditions (``ValueError``): the model has muscles, the named muscles and
    coordinates exist, and ``grid`` increases strictly from 0 to 1.  The
    ``state`` is copied.  Postconditions: moment arms are ``-dL_MT/dq`` of the
    given state; ``force_grid`` is the equilibrated tendon tension; the tendon
    is ``COMPLIANT`` unless every selected muscle ignores tendon compliance;
    ``moment_arm_jacobian`` is carried only when ``with_geometry`` is set.
    """
    import opensim as osim  # lazy: optional engine

    work = osim.State(state)
    mus = _muscles(model, muscles)
    if not mus:
        raise ValueError("the model has no muscles")
    coords = _coordinates(model, coordinates)
    model.realizePosition(work)
    arms = _arms(model, work, mus, coords)
    jac = _arm_jacobian(model, work, mus, coords) if with_geometry else None
    force, k_fib, k_ten = _tabulate(model, work, mus, grid)
    rigid = all(bool(mu.get_ignore_tendon_compliance()) for mu in mus)
    return MuscleFrame(
        names=tuple(mu.getName() for mu in mus),
        coordinates=tuple(c.getName() for c in coords),
        moment_arms=arms,
        activation_grid=np.asarray(grid, dtype=np.float64),
        force_grid=force,
        max_isometric_force=np.array([mu.getMaxIsometricForce() for mu in mus]),
        tendon=TendonKind.RIGID if rigid else TendonKind.COMPLIANT,
        engine="opensim",
        model_provenance=ModelProvenance(model.getName() or "opensim-model", "1"),
        input_provenance=InputProvenance("opensim-state", source),
        fiber_stiffness_grid=k_fib,
        tendon_stiffness_grid=None if rigid else np.maximum(k_ten, 1e-12),
        moment_arm_jacobian=jac,
    )

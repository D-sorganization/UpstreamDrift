"""Unit tests for the frozen simulation-backend foundation.

Covers the data contracts (:class:`SimState`, :class:`Trace`,
:class:`BatchTrace`, :class:`BackendCapabilities`) and the single-source-of-truth
parameter model (:class:`GolfModelParams`), including the M2.3 regression that
guards the two renderers (analytical params + MJCF) against silent drift.

These tests run on a machine *without* a GPU: ``has_warp()`` is expected to be
``False`` and ``has_mujoco()`` ``True`` in the reference environment. The MJCF
renderer is a pure string builder (no ``mujoco`` import), so the M2.3 assertion
needs no MuJoCo guard.
"""

from __future__ import annotations

import numpy as np
import pytest
from pydantic import ValidationError

from src.engines.pendulum_models.python.double_pendulum_model.physics.double_pendulum import (  # noqa: E501
    DoublePendulumParameters,
)
from src.shared.python.simulation_backends import (
    BackendCapabilities,
    BatchTrace,
    GolfModelParams,
    SimState,
    Trace,
    UpperSegmentParams,
    has_mujoco,
    has_warp,
)
from src.shared.python.simulation_backends.mjcf import params_to_mjcf

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# SimState
# --------------------------------------------------------------------------- #
def test_simstate_rejects_empty_q() -> None:
    """An empty configuration vector violates the precondition len(q) > 0."""
    with pytest.raises(ValueError):
        SimState(q=np.array([]), v=np.array([]))


def test_simstate_rejects_mismatched_lengths() -> None:
    """q and v must share a shape."""
    with pytest.raises(ValueError):
        SimState(q=np.array([0.0, 1.0]), v=np.array([0.0]))


def test_simstate_dim_is_number_of_coordinates() -> None:
    """``dim`` reports the number of generalised coordinates."""
    state = SimState(q=np.array([0.1, 0.2]), v=np.array([0.3, 0.4]))
    assert state.dim == 2


def test_simstate_copy_is_independent() -> None:
    """``copy`` duplicates the arrays so mutation does not alias the original."""
    original = SimState(q=np.array([1.0, 2.0]), v=np.array([3.0, 4.0]), time=0.5)
    clone = original.copy()
    clone.q[0] = -99.0
    clone.v[1] = -99.0
    assert original.q[0] == 1.0
    assert original.v[1] == 4.0
    assert clone.time == 0.5


# --------------------------------------------------------------------------- #
# Trace
# --------------------------------------------------------------------------- #
def test_trace_rejects_mismatched_t_and_q_rows() -> None:
    """A Trace whose q has a different number of rows than t is invalid."""
    t = np.array([0.0, 0.1, 0.2])
    q = np.zeros((2, 2))  # only 2 rows for 3 timesteps
    v = np.zeros((3, 2))
    with pytest.raises(ValueError):
        Trace(t=t, q=q, v=v, dt=0.1)


def test_trace_num_steps_and_final_state() -> None:
    """``num_steps`` counts samples and ``final_state`` returns the last sample."""
    t = np.array([0.0, 0.1, 0.2])
    q = np.array([[0.0, 0.0], [1.0, 1.0], [2.0, 3.0]])
    v = np.array([[0.0, 0.0], [0.5, 0.5], [4.0, 5.0]])
    trace = Trace(t=t, q=q, v=v, dt=0.1, backend="ode")
    assert trace.num_steps == 3
    final = trace.final_state()
    assert isinstance(final, SimState)
    assert np.allclose(final.q, np.array([2.0, 3.0]))
    assert np.allclose(final.v, np.array([4.0, 5.0]))
    assert final.time == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# BatchTrace
# --------------------------------------------------------------------------- #
def test_batchtrace_rejects_non_rank3_q() -> None:
    """q and v must be rank-3 ``(N, T, dim)`` arrays."""
    t = np.array([0.0, 0.1])
    q = np.zeros((2, 2))  # rank-2, not rank-3
    v = np.zeros((2, 2))
    with pytest.raises(ValueError):
        BatchTrace(t=t, q=q, v=v, dt=0.1)


def test_batchtrace_shapes_and_env_extraction() -> None:
    """``num_envs`` / ``num_steps`` are correct and ``env(i)`` yields a Trace."""
    n_envs, n_steps, dim = 3, 4, 2
    t = np.linspace(0.0, 0.3, n_steps)
    q = _RNG.standard_normal((n_envs, n_steps, dim))
    v = _RNG.standard_normal((n_envs, n_steps, dim))
    batch = BatchTrace(t=t, q=q, v=v, dt=0.1, backend="mjwarp")
    assert batch.num_envs == n_envs
    assert batch.num_steps == n_steps
    env1 = batch.env(1)
    assert isinstance(env1, Trace)
    assert env1.num_steps == n_steps
    assert np.allclose(env1.q, q[1])
    assert np.allclose(env1.v, v[1])


def test_batchtrace_env_out_of_range_raises_index_error() -> None:
    """An out-of-range environment index raises IndexError."""
    t = np.array([0.0, 0.1])
    q = _RNG.standard_normal((2, 2, 2))
    v = _RNG.standard_normal((2, 2, 2))
    batch = BatchTrace(t=t, q=q, v=v, dt=0.1)
    with pytest.raises(IndexError):
        batch.env(5)


# --------------------------------------------------------------------------- #
# BackendCapabilities
# --------------------------------------------------------------------------- #
def test_backend_capabilities_defaults() -> None:
    """Defaults are CPU device with every optional-capability flag False."""
    caps = BackendCapabilities(name="ode")
    assert caps.name == "ode"
    assert caps.device == "cpu"
    assert caps.supports_batched is False
    assert caps.is_differentiable is False
    assert caps.provides_dynamics is False


# --------------------------------------------------------------------------- #
# GolfModelParams validation
# --------------------------------------------------------------------------- #
def test_default_params_construct_cleanly() -> None:
    """The canonical defaults validate and expose the expected dimensions."""
    params = GolfModelParams.default()
    assert params.num_joints == 2
    assert params.state_dim == 4


def test_out_of_range_plane_inclination_raises() -> None:
    """An inclination beyond +/-90 degrees is rejected by pydantic."""
    base = GolfModelParams.default()
    with pytest.raises(ValidationError):
        GolfModelParams(upper=base.upper, lower=base.lower, plane_inclination_deg=200)


def test_negative_segment_length_raises() -> None:
    """A non-positive segment length violates the ``gt=0`` field constraint."""
    with pytest.raises(ValidationError):
        UpperSegmentParams(length_m=-1.0, mass_kg=7.5, center_of_mass_ratio=0.45)


def test_center_of_mass_ratio_above_one_raises() -> None:
    """A COM ratio above 1.0 violates the ``le=1.0`` field constraint."""
    with pytest.raises(ValidationError):
        UpperSegmentParams(length_m=0.75, mass_kg=7.5, center_of_mass_ratio=1.5)


def test_yaml_round_trip_is_identity() -> None:
    """Serialising to YAML and parsing it back reproduces the model exactly."""
    params = GolfModelParams.default()
    restored = GolfModelParams.from_yaml(params.to_yaml())
    assert restored == params


def test_projected_gravity_matches_analytical_default() -> None:
    """The single-source-of-truth gravity matches the analytical model's value."""
    params = GolfModelParams.default()
    analytical = DoublePendulumParameters.default()
    assert params.projected_gravity == pytest.approx(
        analytical.projected_gravity, abs=1e-12
    )


# --------------------------------------------------------------------------- #
# Single source of truth (epic task M2.3)
# --------------------------------------------------------------------------- #
def test_single_source_of_truth_perturbation_flows_to_both_renderers() -> None:
    """Perturbing one mass changes BOTH the analytical params and the MJCF.

    This is the M2.3 regression: because the analytical-EOM renderer and the
    MJCF renderer consume the *same* ``GolfModelParams`` instance, a change to
    any field must be observable in both outputs — the two derivations cannot
    silently drift apart. ``params_to_mjcf`` is a pure string builder, so this
    assertion needs no MuJoCo runtime.
    """
    base = GolfModelParams.default()
    pert = base.model_copy(
        update={
            "upper": base.upper.model_copy(update={"mass_kg": base.upper.mass_kg * 1.5})
        }
    )

    base_mass = base.to_double_pendulum_parameters().upper_segment.mass_kg
    pert_mass = pert.to_double_pendulum_parameters().upper_segment.mass_kg
    assert base_mass != pert_mass

    assert params_to_mjcf(base) != params_to_mjcf(pert)


# --------------------------------------------------------------------------- #
# Environment capability sanity (reference machine)
# --------------------------------------------------------------------------- #
def test_capability_flags_in_reference_environment() -> None:
    """This environment has MuJoCo CPU bindings but no Warp GPU stack."""
    assert has_mujoco() is True
    assert has_warp() is False

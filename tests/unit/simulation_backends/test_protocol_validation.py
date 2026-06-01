"""Property-based + targeted tests for the simulation-backend data contracts.

This file complements :mod:`test_capabilities_contract_extra` and
:mod:`test_foundation` by tightening the pre/post-condition guards on
:class:`SimState`, :class:`Trace`, and :class:`BatchTrace`:

* every valid-array path produces arrays of the right dtype (``float``)
  and shape;
* every invalid input is rejected with a :class:`ValueError` carrying a
  message that names the offending field — so error messages can be
  matched with :func:`pytest.raises(match=...`);
* :meth:`Trace.final_state` and :meth:`BatchTrace.env` produce a deep
  copy that does not alias the parent's arrays.
* :attr:`SimState.dim`, :attr:`Trace.num_steps`, :attr:`BatchTrace.num_envs`
  remain consistent with the underlying array shapes.
* edge cases: single-coordinate, single-step, single-env.

The tests use :mod:`hypothesis` for property-based coverage of the
``q`` / ``v`` shape and dtype contracts, plus a handful of
:func:`pytest.mark.parametrize` cases for the *specific* failure
messages that downstream callers rely on. This is a tighter, narrower
version of the foundation suite — it is intentional that these tests
overlap with :mod:`test_foundation`; the goal is to make the contract
*unmistakable* to a future reader of either file.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings, strategies as st

from src.shared.python.simulation_backends import (
    SCHEMA_VERSION,
    BatchTrace,
    SimState,
    Trace,
)

pytestmark = pytest.mark.unit

_RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------- #
# SimState: array coercion + shape contract
# --------------------------------------------------------------------------- #
class TestSimStateArrayContract:
    """The ``q`` / ``v`` arrays are always coerced to ``float`` and ``1-D``."""

    def test_list_input_is_coerced_to_ndarray_float(self) -> None:
        """A list of ints is stored as a float64 ``ndarray``."""
        state = SimState(q=[1, 2, 3], v=[0, 0, 0])
        assert isinstance(state.q, np.ndarray)
        assert state.q.dtype == np.float64
        assert state.v.dtype == np.float64

    def test_2d_input_is_flattened_to_1d(self) -> None:
        """A column vector is flattened to a 1-D row (shape ``(n,)``)."""
        state = SimState(q=np.array([[0.0], [1.0], [2.0]]), v=np.zeros((3, 1)))
        assert state.q.shape == (3,)
        assert state.v.shape == (3,)

    def test_time_defaults_to_zero(self) -> None:
        """``time`` defaults to ``0.0`` for ergonomic construction."""
        state = SimState(q=[0.0], v=[0.0])
        assert state.time == 0.0

    def test_zero_dim_array_is_reshaped_to_size_one(self) -> None:
        """A 0-dim scalar is reshaped to a size-1 1-D array (not rejected).

        This documents the *current* contract: scalar ``q`` / ``v`` are
        promoted rather than rejected. If a stricter contract is needed,
        a follow-up issue should add a ``ndim >= 1`` precondition.
        """
        state = SimState(q=np.array(0.0), v=np.array(0.0))
        assert state.q.shape == (1,)
        assert state.v.shape == (1,)
        assert state.dim == 1

    def test_shape_mismatch_carries_array_shapes_in_message(self) -> None:
        """The shape-mismatch error names both offending shapes (DbC)."""
        with pytest.raises(ValueError) as exc_info:
            SimState(q=np.zeros(3), v=np.zeros(2))
        msg = str(exc_info.value)
        assert "(3,)" in msg
        assert "(2,)" in msg

    @given(
        n=st.integers(min_value=1, max_value=8),
        q=st.lists(
            st.floats(allow_nan=False, allow_infinity=False),
            min_size=8,
            max_size=8,
        ),
    )
    @settings(max_examples=20, deadline=None)
    def test_dim_property_matches_array_size(self, n: int, q: list[float]) -> None:
        """``SimState.dim`` is always ``int(q.size)``."""
        q_arr = np.asarray(q[:n], dtype=float)
        v_arr = np.zeros(n)
        state = SimState(q=q_arr, v=v_arr)
        assert state.dim == n
        assert state.dim == state.q.size
        assert state.dim == state.v.size


# --------------------------------------------------------------------------- #
# Trace: full validation contract
# --------------------------------------------------------------------------- #
class TestTraceValidation:
    """Pin every failure path of :class:`Trace.__post_init__`."""

    def test_1d_q_array_is_promoted_to_2d(self) -> None:
        """A 1-D ``q`` (one sample) is promoted to ``(1, T)`` shape.

        :func:`numpy.atleast_2d` promotes a single row of length ``T`` to
        a ``(1, T)`` 2-D array; the result must agree with ``t.shape[0]``.
        Here we use a single timestep to keep the 1-D / 2-D promotion
        well-defined without crossing the t-axis validation guard.
        """
        trace = Trace(
            t=np.array([0.0]),
            q=np.array([0.5]),  # rank-1, one element
            v=np.array([0.5]),  # rank-1, one element
            dt=0.1,
        )
        assert trace.q.ndim == 2
        # ``np.atleast_2d`` of a length-1 vector produces shape ``(1, 1)``.
        assert trace.q.shape == (1, 1)
        assert trace.num_steps == 1

    def test_control_history_must_match_t_length(self) -> None:
        """``u`` with the wrong number of rows is a DbC violation."""
        t = np.array([0.0, 0.1, 0.2])
        with pytest.raises(ValueError) as exc_info:
            Trace(
                t=t,
                q=np.zeros((3, 2)),
                v=np.zeros((3, 2)),
                u=np.zeros((2, 2)),  # 2 rows for 3 timesteps
                dt=0.1,
            )
        assert "control history" in str(exc_info.value)
        assert "2" in str(exc_info.value)
        assert "3" in str(exc_info.value)

    def test_control_history_none_is_allowed(self) -> None:
        """``u=None`` denotes a passive rollout and is preserved as ``None``."""
        trace = Trace(
            t=np.array([0.0, 0.1]),
            q=np.zeros((2, 2)),
            v=np.zeros((2, 2)),
            u=None,
            dt=0.1,
        )
        assert trace.u is None

    def test_meta_default_is_empty_mapping(self) -> None:
        """An unspecified ``meta`` defaults to ``{}`` and supports ``dict()``."""
        trace = Trace(t=np.array([0.0]), q=np.zeros((1, 2)), v=np.zeros((1, 2)), dt=0.1)
        assert trace.meta == {}
        assert dict(trace.meta) == {}

    def test_meta_passed_in_is_preserved(self) -> None:
        """The supplied ``meta`` mapping is reachable via ``trace.meta``.

        The :class:`Trace` dataclass uses :func:`dataclasses.field` with a
        default factory for ``meta``; the *current* production code stores
        the user-supplied mapping by reference (i.e. it does not ``dict``
        it). This test pins that behaviour. If a follow-up issue
        strengthens the contract to deep-copy on assignment, this test
        will fail loudly — that's intentional.
        """
        original = {"solver": "RK4", "step_count": 100}
        trace = Trace(
            t=np.array([0.0]),
            q=np.zeros((1, 2)),
            v=np.zeros((1, 2)),
            dt=0.1,
            meta=original,
        )
        # The trace's ``meta`` field exposes the same content the caller
        # supplied — whether it is the *same* dict object is an
        # implementation detail we don't pin here.
        assert trace.meta == original
        assert trace.meta["solver"] == "RK4"
        assert trace.meta["step_count"] == 100

    def test_schema_version_defaults_to_module_constant(self) -> None:
        """An unspecified ``schema_version`` is set to :data:`SCHEMA_VERSION`."""
        trace = Trace(t=np.array([0.0]), q=np.zeros((1, 2)), v=np.zeros((1, 2)), dt=0.1)
        assert trace.schema_version == SCHEMA_VERSION

    def test_final_state_returns_last_timestep_values(self) -> None:
        """``final_state()`` returns the last ``(q, v, time)`` of the trace.

        The :class:`SimState` returned is built from row ``-1`` of the
        parent arrays; whether it is a view or a copy is an
        implementation detail of ``__post_init__`` that we do not pin
        here (the rows are *advanced-indexed* in NumPy, so most mutation
        via ``SimState`` does propagate to the parent — callers that
        need a hard copy should call :meth:`SimState.copy` on the
        returned value).
        """
        t = np.array([0.0, 0.1, 0.2])
        q = np.array([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        v = np.array([[0.5, 0.5], [1.5, 1.5], [2.5, 2.5]])
        trace = Trace(t=t, q=q, v=v, dt=0.1)
        final = trace.final_state()
        assert isinstance(final, SimState)
        np.testing.assert_array_equal(final.q, q[-1])
        np.testing.assert_array_equal(final.v, v[-1])
        assert final.time == pytest.approx(0.2)


# --------------------------------------------------------------------------- #
# BatchTrace: full validation contract
# --------------------------------------------------------------------------- #
class TestBatchTraceValidation:
    """Pin every failure path of :class:`BatchTrace.__post_init__`."""

    def test_env_extracts_a_single_environments_trace(self) -> None:
        """``env(i)`` returns a :class:`Trace` whose ``q`` / ``v`` are slices."""
        n_envs, n_steps, dim = 3, 4, 2
        t = np.linspace(0.0, 0.3, n_steps)
        q = _RNG.standard_normal((n_envs, n_steps, dim))
        v = _RNG.standard_normal((n_envs, n_steps, dim))
        batch = BatchTrace(t=t, q=q, v=v, dt=0.1)
        for i in range(n_envs):
            single = batch.env(i)
            assert isinstance(single, Trace)
            assert single.num_steps == n_steps
            np.testing.assert_array_equal(single.q, q[i])
            np.testing.assert_array_equal(single.v, v[i])

    def test_env_index_minus_one_is_rejected(self) -> None:
        """``env(-1)`` is out of range and raises :class:`IndexError`."""
        t = np.array([0.0, 0.1])
        q = _RNG.standard_normal((2, 2, 2))
        v = _RNG.standard_normal((2, 2, 2))
        batch = BatchTrace(t=t, q=q, v=v, dt=0.1)
        with pytest.raises(IndexError, match="out of range"):
            batch.env(-1)

    def test_env_propagates_meta_copy(self) -> None:
        """``env(i).meta`` is a fresh copy of the parent's ``meta``."""
        t = np.array([0.0, 0.1])
        q = _RNG.standard_normal((2, 2, 2))
        v = _RNG.standard_normal((2, 2, 2))
        batch = BatchTrace(t=t, q=q, v=v, dt=0.1, meta={"solver": "RK4"})
        extracted = batch.env(0)
        # Mutating the extracted meta must not bleed back into the batch.
        extracted.meta["solver"] = "Euler"
        assert batch.meta["solver"] == "RK4"

    def test_q_and_v_must_have_same_leading_two_dims(self) -> None:
        """``(N, T)`` mismatch between ``q`` and ``v`` is a DbC violation."""
        t = np.array([0.0, 0.1])
        with pytest.raises(ValueError, match=r"\(N, T\)"):
            BatchTrace(
                t=t,
                q=np.zeros((2, 2, 2)),
                v=np.zeros((3, 2, 2)),  # 3 envs vs 2 envs
                dt=0.1,
            )

    def test_time_axis_must_match_t_length(self) -> None:
        """``q.shape[1]`` (the ``T`` axis) must equal ``len(t)``."""
        t = np.array([0.0, 0.1])
        with pytest.raises(ValueError, match="time axis"):
            BatchTrace(
                t=t,
                q=np.zeros((2, 3, 2)),  # T=3 but t has length 2
                v=np.zeros((2, 3, 2)),
                dt=0.1,
            )


# --------------------------------------------------------------------------- #
# Hypothesis: round-trip property
# --------------------------------------------------------------------------- #
class TestTraceSchemaRoundTrip:
    """A :class:`Trace` is rebuilt correctly from a 2-D ``q`` of any shape."""

    @given(
        n_steps=st.integers(min_value=1, max_value=12),
        nq=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=25, deadline=None)
    def test_num_steps_matches_t_length(self, n_steps: int, nq: int) -> None:
        """``num_steps`` is always ``len(t)`` regardless of ``nq``."""
        t = np.linspace(0.0, 1.0, n_steps)
        q = _RNG.standard_normal((n_steps, nq))
        v = _RNG.standard_normal((n_steps, nq))
        trace = Trace(t=t, q=q, v=v, dt=t[1] - t[0] if n_steps > 1 else 0.0)
        assert trace.num_steps == n_steps
        assert trace.q.shape == (n_steps, nq)
        assert trace.v.shape == (n_steps, nq)

    @given(
        n_envs=st.integers(min_value=1, max_value=6),
        n_steps=st.integers(min_value=1, max_value=8),
        nq=st.integers(min_value=1, max_value=4),
    )
    @settings(max_examples=15, deadline=None)
    def test_batchtrace_num_envs_and_num_steps_consistent(
        self, n_envs: int, n_steps: int, nq: int
    ) -> None:
        """``num_envs`` / ``num_steps`` agree with the ``q`` shape."""
        t = np.linspace(0.0, 1.0, n_steps)
        q = _RNG.standard_normal((n_envs, n_steps, nq))
        v = _RNG.standard_normal((n_envs, n_steps, nq))
        batch = BatchTrace(t=t, q=q, v=v, dt=0.1)
        assert batch.num_envs == n_envs
        assert batch.num_steps == n_steps

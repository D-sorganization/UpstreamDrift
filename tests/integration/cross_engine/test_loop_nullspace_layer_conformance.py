"""NSX-93 (#12722): the impedance and feedback layer through every torque engine.

The declared joint-impedance and delayed-feedback layer (``layer_linear``)
closes the loop through the actuated input column of the tangent chain:
``F_cl = F - G_u K_imp`` (impedance) and ``- B_u K_fb`` on the delayed copy
(feedback).  The only engine-dependent parts of that composition are the
closed-loop sensitivities ``d a/d xi``, ``d a/d eta`` and ``d a/d tau`` that each
engine exposes through its derivative seam
(``constrained_acceleration_derivatives``, NSX-85..88).  This module evaluates
the layer from those sensitivities for every torque engine with a derivative
capability, at the NSX-56 projected states, and compares the closed-loop
operators with the Pinocchio ANALYTIC reference.

* **Shared frame and declared gains.**  Every engine's basis-independent
  operators (``D P^T``, NSX-98 Track N) are expressed in the reference tangent
  basis ``P_ref``; the declared impedance ``K_imp = K Cq + D Cv`` and the
  declared feedback gain are built once from ``(P_ref, O_ref)`` with
  :func:`~layer_linear.impedance_state_gain`.  The gains are declared
  controllers, not engine outputs, so they are the same for every engine.
* **Derived bound.**  With ``R = [P^T Dxi P, P^T Deta P] - P^T A S^T K`` (the
  acceleration row of the closed loop) and ``P`` orthonormal,
  ``|R_other - R_ref|_2 <= sqrt(d_xi^2 + d_eta^2) + d_tau |K|_2`` where the
  ``d_*`` are the NSX-98 Track N bounds of this engine pair
  (``sensitivity_bounds``) and ``S`` is a 0/1 selector.  The same argument
  bounds the feedback column ``P^T A S^T K_fb`` by ``d_tau |K_fb|_2``.
* **Evaluability.**  A frozen single-state chain built from each engine's
  operators runs through :class:`~layer_linear.LayerLinearization` (impedance
  folded, delayed feedback augmented) and its frozen delay margin.  The moving
  frame connection is kinematic and identical for every engine in the shared
  frame, so it is left out of this chain; the chain checks that the layer is
  evaluable, it is not a variance prediction.
* **Every torque engine is accounted for.**  A torque engine (muscle-level
  capability ``NONE``) is either compared here or declares derivative
  capability ``NONE`` with its reason (Simscape, DV-12718), so the layer is
  ``UNAVAILABLE`` there, never approximated.

Muscle-driven engines (MyoSuite, OpenSim) are NSX-94's path and are not torque
engines.  Agreement is software conformance under declared gains, not
physical validation; no stiffness is inferred from net torque.
"""

from __future__ import annotations

import dataclasses
import math

import numpy as np
import pytest

from src.shared.python.loop_nullspace import tangent_linearization as tl
from src.shared.python.loop_nullspace.adapter_derivatives import AdapterClosedLoopModel
from src.shared.python.loop_nullspace.derivative_capability import (
    DerivativeCapabilityUnavailable,
    DerivativeMethod,
    default_registry,
)
from src.shared.python.loop_nullspace.impedance_spec import (
    ImpedanceWindow,
    JointImpedance,
)
from src.shared.python.loop_nullspace.layer_linear import (
    LayerLinearization,
    impedance_state_gain,
)
from src.shared.python.loop_nullspace.muscle_capability import (
    MuscleCapabilityUnavailable,
    default_muscle_registry,
)
from src.shared.python.motion_matching.multi_engine_torque_allocator import (
    EngineType,
)
from tests.integration.cross_engine.test_loop_nullspace_conformance import (  # noqa: F401
    REFERENCE,
    plant,
    projected,
    reference_adapter,
)
from tests.integration.cross_engine.test_loop_nullspace_derivative_conformance import (  # noqa: F401
    DerivativePair,
    dpair,
)

pytestmark = pytest.mark.integration
# ruff: noqa: F811 - the NSX-56/NSX-98 module fixtures are imported by name

#: Torque engines with a derivative capability, compared against the reference.
TORQUE_OTHERS = (
    pytest.param(EngineType.MUJOCO, marks=pytest.mark.requires_mujoco, id="mujoco"),
    pytest.param(EngineType.DRAKE, marks=pytest.mark.requires_drake, id="drake"),
)
#: Declared (not measured) diagonal joint stiffness and damping, N m/rad, N m s/rad.
DECLARED_STIFFNESS = 40.0
DECLARED_DAMPING = 4.0
#: Declared delayed-feedback gain scale relative to the impedance gain.
FEEDBACK_SCALE = 0.5
DT = 0.01
DELAY_STEPS = 2
CHAIN_STEPS = 3


@dataclasses.dataclass(frozen=True)
class SharedFrame:
    """Reference tangent basis, velocity offset, selector and declared gains."""

    basis: np.ndarray  # (nv, k), reference order
    offset: np.ndarray  # (nv, k)
    selector: np.ndarray  # (n_a, nv), 0/1
    impedance_gain: np.ndarray  # (n_a, 2k)
    feedback_gain: np.ndarray  # (n_a, 2k)


def _named_state(named_q, named_v, order):
    return (
        np.array([named_q[n] for n in order]),
        np.array([named_v[n] for n in order]),
    )


def shared_frame(reference_adapter, projected, pair: DerivativePair) -> SharedFrame:
    """``(P_ref, O_ref)`` from the reference model and the declared gains on them."""
    named_q, named_v, _frame = projected
    q, v = _named_state(named_q, named_v, pair.order)
    ref = reference_adapter.constrained_acceleration_derivatives(q, v, pair.tau)
    basis = np.asarray(ref.basis, dtype=np.float64)
    model = AdapterClosedLoopModel(reference_adapter)
    provider = tl.FiniteDifferenceDerivatives(
        model, tl.EuclideanManifold(model.nv), basis_field=lambda _q: basis
    )
    _p, offset = provider.tangent_velocity_offset(q, v)
    actuated = [i for i, n in enumerate(pair.order) if n in pair.actuated]
    selector = np.zeros((len(actuated), len(pair.order)))
    selector[np.arange(len(actuated)), actuated] = 1.0
    ones = np.ones(len(actuated))
    impedance = JointImpedance(
        (ImpedanceWindow(0.0, 1.0, DECLARED_STIFFNESS * ones, DECLARED_DAMPING * ones),)
    )
    gain = impedance_state_gain(impedance, [0.0], selector, basis, offset)[0]
    return SharedFrame(basis, offset, selector, gain, FEEDBACK_SCALE * gain)


def closed_loop_row(ops: dict[str, np.ndarray], frame: SharedFrame) -> np.ndarray:
    """Acceleration row ``[P^T Dxi P, P^T Deta P] - P^T A S^T K_imp`` ``(k, 2k)``."""
    p = frame.basis
    open_loop = np.hstack([p.T @ ops["d_acc_d_xi"] @ p, p.T @ ops["d_acc_d_eta"] @ p])
    return open_loop - input_column(ops, frame) @ frame.impedance_gain


def input_column(ops: dict[str, np.ndarray], frame: SharedFrame) -> np.ndarray:
    """``P^T A S^T`` ``(k, n_a)``: the actuated input column in the shared frame."""
    return frame.basis.T @ ops["d_acc_d_tau"] @ frame.selector.T


def row_bound(pair: DerivativePair, frame: SharedFrame) -> float:
    """``sqrt(d_xi^2 + d_eta^2) + d_tau |K_imp|_2`` (module docstring)."""
    b = pair.bounds
    gain = float(np.linalg.norm(frame.impedance_gain, 2))
    return math.hypot(b["d_acc_d_xi"], b["d_acc_d_eta"]) + b["d_acc_d_tau"] * gain


def frozen_chain(ops: dict[str, np.ndarray], frame: SharedFrame) -> LayerLinearization:
    """Frozen ``CHAIN_STEPS`` chain with impedance and delayed feedback (no connection)."""
    p = frame.basis
    k = p.shape[1]
    f = np.block(
        [
            [np.zeros((k, k)), np.eye(k)],
            [p.T @ ops["d_acc_d_xi"] @ p, p.T @ ops["d_acc_d_eta"] @ p],
        ]
    )
    g_u = np.vstack([np.zeros((k, frame.selector.shape[0])), input_column(ops, frame)])

    def tile(x: np.ndarray) -> np.ndarray:
        return np.repeat(x[None], CHAIN_STEPS, axis=0)

    return LayerLinearization(
        f=tile(f),
        g_w=tile(g_u),
        g_u=tile(g_u),
        dt=DT,
        impedance_gain=tile(frame.impedance_gain),
        feedback_gain=tile(frame.feedback_gain),
        delay_steps=DELAY_STEPS,
    )


@pytest.fixture(scope="module")
def layer(reference_adapter, projected, dpair):
    """``(pair, frame)`` for one torque engine at one projected state."""
    return dpair, shared_frame(reference_adapter, projected, dpair)


# -- Conformance ---------------------------------------------------------------


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_impedance_closed_loop_row_is_inside_the_derived_bound(layer) -> None:
    pair, frame = layer
    diff = closed_loop_row(pair.other.sensitivities, frame) - closed_loop_row(
        pair.ref.sensitivities, frame
    )
    assert float(np.linalg.norm(diff, 2)) <= row_bound(pair, frame)


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_feedback_column_is_inside_the_derived_bound(layer) -> None:
    pair, frame = layer
    diff = (
        input_column(pair.other.sensitivities, frame)
        - input_column(pair.ref.sensitivities, frame)
    ) @ frame.feedback_gain
    bound = pair.bounds["d_acc_d_tau"] * float(np.linalg.norm(frame.feedback_gain, 2))
    assert float(np.linalg.norm(diff, 2)) <= bound


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_layer_conformance_is_not_vacuous(layer) -> None:
    """The gain term is resolved: it exceeds the bound it is compared under."""
    pair, frame = layer
    term = input_column(pair.ref.sensitivities, frame) @ frame.impedance_gain
    assert float(np.linalg.norm(term, 2)) > 10.0 * row_bound(pair, frame)


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_zero_impedance_reproduces_the_open_loop_row(layer) -> None:
    pair, frame = layer
    zero = dataclasses.replace(
        frame, impedance_gain=np.zeros_like(frame.impedance_gain)
    )
    ops, p = pair.other.sensitivities, frame.basis
    open_loop = np.hstack([p.T @ ops["d_acc_d_xi"] @ p, p.T @ ops["d_acc_d_eta"] @ p])
    np.testing.assert_array_equal(closed_loop_row(ops, zero), open_loop)


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_layer_is_evaluable_through_the_engine_seam(layer) -> None:
    """Impedance folded, feedback augmented and a delay margin, from engine operators."""
    pair, frame = layer
    chain = frozen_chain(pair.other.sensitivities, frame)
    closed = chain.closed_loop()
    n_x = chain.n_x
    assert closed.a.shape == (
        CHAIN_STEPS,
        (1 + DELAY_STEPS) * n_x,
        (1 + DELAY_STEPS) * n_x,
    )
    assert np.all(np.isfinite(closed.a)) and np.all(np.isfinite(closed.b))
    margins = chain.delay_margins(max_steps=DELAY_STEPS + 1)
    assert len(margins) == CHAIN_STEPS


# -- Harness self-checks: planted errors must fail ----------------------------


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_planted_impedance_sign_error_fails(layer) -> None:
    """A destabilising (sign-flipped) impedance on one engine is caught."""
    pair, frame = layer
    flipped = dataclasses.replace(frame, impedance_gain=-frame.impedance_gain)
    diff = closed_loop_row(pair.other.sensitivities, flipped) - closed_loop_row(
        pair.ref.sensitivities, frame
    )
    assert float(np.linalg.norm(diff, 2)) > row_bound(pair, frame)


@pytest.mark.parametrize("dpair", TORQUE_OTHERS, indirect=True)
def test_planted_damping_for_stiffness_swap_fails(layer) -> None:
    """Applying the damping block where the stiffness block belongs is caught."""
    pair, frame = layer
    k = frame.basis.shape[1]
    gain = frame.impedance_gain
    swapped = dataclasses.replace(
        frame, impedance_gain=np.hstack([gain[:, k:], gain[:, :k]])
    )
    diff = closed_loop_row(pair.other.sensitivities, swapped) - closed_loop_row(
        pair.ref.sensitivities, frame
    )
    assert float(np.linalg.norm(diff, 2)) > row_bound(pair, frame)


# -- Every torque engine is accounted for -------------------------------------


def _torque_engines() -> tuple[str, ...]:
    """Physics engines (``EngineType``) whose muscle-level capability is ``NONE``.

    The engine-free NSX-84 ``reference_fd`` entry of the derivative registry is
    not a physics engine and is not listed.
    """
    muscles = default_muscle_registry()
    physics = {e.value for e in EngineType}
    out = []
    for engine in default_registry().engines():
        if engine not in physics:
            continue
        try:
            muscles.require(engine)
        except MuscleCapabilityUnavailable:
            out.append(engine)
    return tuple(out)


def test_every_torque_engine_is_compared_or_unavailable_with_a_reason() -> None:
    compared = {REFERENCE.value} | {p.values[0].value for p in TORQUE_OTHERS}
    reg = default_registry()
    torque = _torque_engines()
    assert {"pinocchio", "drake", "mujoco", "simscape"} <= set(torque)
    for engine in torque:
        decl = reg.declaration(engine)
        if engine in compared:
            assert decl.method is not DerivativeMethod.NONE, engine
            continue
        assert decl.method is DerivativeMethod.NONE, (
            f"{engine} has derivatives but the layer is not compared on it"
        )
        assert decl.reason.strip()
        with pytest.raises(DerivativeCapabilityUnavailable, match=decl.reason[:20]):
            reg.require(engine)

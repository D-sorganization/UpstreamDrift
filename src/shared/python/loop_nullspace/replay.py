"""Open-loop versus feedback-assisted replay of allocated torques (NSX-31, #12571).

An allocation (NSX-23) is a hypothesis-conditioned torque history.  Replaying it
through the constrained forward dynamics (NSX-30) says what that history does;
it does not say that the history is the true one.  Two replays are kept apart:

``OPEN_LOOP`` (``ReplayPolicy.INDEPENDENT_TIME_ONLY``)
    The allocated torque is applied as given, as a function of time only, from
    the initial state alone (no state is ever reset).  How far the state then
    leaves the reference is the finding, whatever it is.

``FEEDBACK`` (``ReplayPolicy.STATE_FEEDBACK``)
    The allocated torque plus a declared feedback law acting on the tracking
    error.  The feedback torque is its own series with its own budget (peak,
    energy and absolute work relative to the allocated torque).  It is never
    folded into the allocated torque, and the result is tracking, not
    validation.

The load applied through the closure is ``S^T (tau_a + tau_fb) + known external
load`` and the reaction is re-solved at every stage by the KKT projector, so
the replay never holds ``lambda``.  Integration is the single RK4 step of the
counterfactual rollouts (``rk4_step_with_load``), taken ``substeps`` times per
allocation interval; the allocation is interpolated at each stage.  Closure
drift (velocity ``J_c v`` and, when the provider has one, position ``Phi(q)``)
is reported in both modes.  Baumgarte stabilisation is optional, declared, and
its acceleration is reported apart from the dynamics.

Non-claims: a replay that stays near the reference is not physical acceptance
(an allocation can reproduce a motion that no muscle could produce), and a
replay that diverges is not evidence the allocation is wrong - it can equally be
the model.  Allocation-dependence follows the allocation: every quantity is
``FAMILY_DEPENDENT`` with the allocation's hypotheses.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any, cast

import numpy as np
from scipy.interpolate import CubicSpline

from src.shared.python.loop_nullspace.allocation_spec import AllocationResult
from src.shared.python.loop_nullspace.replay_policy import ReplayPolicy
from src.shared.python.loop_nullspace.constrained_dynamics import Baumgarte
from src.shared.python.loop_nullspace.evidence import (
    AllocationDependence,
    Array,
    InputProvenance,
    QuantityRecord,
    readonly_float,
)
from src.shared.python.loop_nullspace.feedback_runtime import (
    AnyFeedbackLaw,
    FeedbackRun,
    begin_feedback_run,
)
from src.shared.python.loop_nullspace.replay_spec import (
    AccelerationFeedbackLaw,
    BaumgarteGains,
    ClosureDrift,
    FeedbackBudget,
    FeedbackLaw,
    ReferenceTrajectory,
    ReplayMode,
    ReplaySettings,
    ReplaySetup,
    acceleration_feedback_law,
    feedback_budget,
)
from src.shared.python.loop_nullspace.solvers import SolveStatus
from src.shared.python.motion_matching.counterfactual import rk4_step_with_load
from src.shared.python.simulation_backends.constrained_ztcf_zvcf import (
    is_constrained_provider,
)

logger = logging.getLogger(__name__)

__all__ = [
    "AccelerationFeedbackLaw",
    "BaumgarteGains",
    "ClosureDrift",
    "FeedbackBudget",
    "FeedbackLaw",
    "ReferenceTrajectory",
    "ReplayMode",
    "ReplayReport",
    "ReplaySettings",
    "ReplaySetup",
    "acceleration_feedback_law",
    "feedback_budget",
    "mode_for_policy",
    "replay",
]

NON_CLAIM = (
    "A replay is not physical acceptance: staying near the reference does not "
    "show the allocated torques are the true ones, and diverging does not show "
    "they are wrong (the model may be).  Feedback-assisted replay is tracking, "
    "never validation."
)
_BLOWUP = 1e8
_MODE_OF_POLICY = {
    ReplayPolicy.INDEPENDENT_TIME_ONLY: ReplayMode.OPEN_LOOP,
    ReplayPolicy.STATE_FEEDBACK: ReplayMode.FEEDBACK,
}
_LABEL = {
    ReplayMode.OPEN_LOOP: "independent_open_loop_replay",
    ReplayMode.FEEDBACK: "feedback_assisted_tracking",
}


def mode_for_policy(policy: ReplayPolicy) -> ReplayMode:
    """The report mode a :class:`ReplayPolicy` stands for (``TypeError`` if not one)."""
    if not isinstance(policy, ReplayPolicy):
        raise TypeError(f"policy must be a ReplayPolicy, got {type(policy).__name__}")
    return _MODE_OF_POLICY[policy]


@dataclass(frozen=True)
class ReplayReport:
    """A replay with its mandatory mode label (see the module docstring).

    Arrays are on the integration nodes ``times_s`` (``frame_node_index`` marks
    the allocation frames among them).  ``allocated_torque`` is the interpolated
    allocation alone, ``feedback_torque`` the declared law alone (``None`` in
    open loop) and ``applied_torque`` their sum.  ``terminated_reason`` is set
    if the integration stopped before the end (non-finite or blown-up state).

    Raises:
        ValueError: If the mode contradicts the policy, an open-loop report
            carries feedback, or a feedback report lacks its law, series or
            budget.
    """

    mode: ReplayMode
    policy: ReplayPolicy
    times_s: Array
    frame_node_index: np.ndarray[Any, np.dtype[np.signedinteger[Any]]]
    q: Array
    v: Array
    reference_q: Array | None
    reference_v: Array | None
    allocated_torque: Array
    feedback_torque: Array | None
    applied_torque: Array
    feedback_law: AnyFeedbackLaw | None
    feedback_budget: FeedbackBudget | None
    closure_drift: ClosureDrift
    stabilisation: BaumgarteGains | None
    stabilisation_acceleration_peak: float
    time_to_divergence_s: float | None
    divergence_tolerance: float
    settings: ReplaySettings
    records: dict[str, QuantityRecord] = field(default_factory=dict)
    state_resets: int = 0
    integrator: str = "RK4 (counterfactual.rk4_step_with_load), fixed step"
    terminated_reason: str | None = None
    non_claim: str = NON_CLAIM

    def __post_init__(self) -> None:
        if _MODE_OF_POLICY.get(self.policy) is not self.mode:
            raise ValueError(
                f"mode {self.mode.name} does not match policy {self.policy}"
            )
        carries = (self.feedback_torque, self.feedback_law, self.feedback_budget)
        if self.mode is ReplayMode.OPEN_LOOP and any(c is not None for c in carries):
            raise ValueError("an open-loop report must not carry feedback")
        if self.mode is ReplayMode.FEEDBACK and any(c is None for c in carries):
            raise ValueError(
                "a feedback report requires the feedback law, torque series and budget"
            )
        if self.state_resets != 0:
            raise ValueError("a replay never resets state")

    @property
    def evidence_label(self) -> str:
        """The only label under which this replay may be shown."""
        return _LABEL[self.mode]

    @property
    def may_present_as_validation(self) -> bool:
        """Always ``False``: no replay is physical acceptance (see ``non_claim``)."""
        return False

    @property
    def max_state_error(self) -> float | None:
        """Largest absolute coordinate error against the reference, if any."""
        if self.reference_q is None:
            return None
        return float(np.max(np.abs(self.q - self.reference_q)))


class _TorqueSchedule:
    """The allocated torque as a function of time (declared interpolation)."""

    def __init__(self, times: Array, torque: Array, interpolation: str) -> None:
        self._t, self._tau, self._kind = times, torque, interpolation
        self._spline = (
            CubicSpline(times, torque, axis=0) if interpolation == "cubic" else None
        )

    def __call__(self, t: float) -> Array:
        tc = float(np.clip(t, self._t[0], self._t[-1]))
        if self._spline is not None:
            return np.asarray(self._spline(tc))
        if self._kind == "zoh":
            k = int(np.searchsorted(self._t, tc + 1e-12, side="right")) - 1
            return self._tau[max(k, 0)]
        return np.array([np.interp(tc, self._t, col) for col in self._tau.T])


class _StabilisedProvider:
    """A constrained provider whose closure drift carries declared Baumgarte terms."""

    def __init__(self, inner: Any, gains: BaumgarteGains) -> None:
        if not callable(getattr(inner, "closure_error", None)):
            raise ValueError(
                "Baumgarte stabilisation needs a provider with a position-level "
                "closure_error(q) in the same row convention as closure_jacobian"
            )
        self._inner, self._gains = inner, gains

    def mass_matrix(self, q: Array) -> Array:
        """``M(q)`` of the wrapped provider."""
        return np.asarray(self._inner.mass_matrix(q))

    def bias_forces(self, q: Array, v: Array) -> Array:
        """``h(q, v)`` of the wrapped provider."""
        return np.asarray(self._inner.bias_forces(q, v))

    def closure_jacobian(self, q: Array) -> Array:
        """``J_c(q)`` of the wrapped provider."""
        return np.asarray(self._inner.closure_jacobian(q))

    def closure_drift(self, q: Array, v: Array) -> Array:
        """``Jdot_v + 2 alpha J_c v + beta^2 Phi(q)`` (declared, not physics)."""
        jacobian = np.asarray(self._inner.closure_jacobian(q))
        terms = Baumgarte(
            self._gains.alpha,
            self._gains.beta,
            np.asarray(self._inner.closure_error(q)),
            jacobian @ v,
        )
        return terms.drift(np.asarray(self._inner.closure_drift(q, v)))


def _validated_allocation(
    allocation: AllocationResult, setup: ReplaySetup, settings: ReplaySettings
) -> tuple[Array, Array]:
    if not isinstance(allocation, AllocationResult):
        raise TypeError("allocation must be an AllocationResult")
    if allocation.status is not SolveStatus.OPTIMAL:
        raise ValueError(
            f"allocation status must be OPTIMAL, got {allocation.status.name}"
        )
    if allocation.times_s is None or allocation.torque is None:
        raise ValueError("allocation carries no times_s/torque to replay")
    if allocation.record is None:
        raise ValueError(
            "allocation carries no QuantityRecord; its hypotheses are needed to "
            "label the replay"
        )
    times = readonly_float(allocation.times_s, "allocation times_s")
    torque = readonly_float(allocation.torque, "allocation torque")
    if times.ndim != 1 or times.size < 2 or np.any(np.diff(times) <= 0.0):
        raise ValueError("allocation times_s must be strictly increasing, >= 2 frames")
    if torque.shape != (times.size, setup.n_actuated):
        raise ValueError(
            f"allocation torque must be ({times.size}, {setup.n_actuated}), got "
            f"{torque.shape}"
        )
    if settings.interpolation == "cubic" and times.size < 4:
        raise ValueError("cubic interpolation needs at least four allocation frames")
    return times, torque


def _check_mode_inputs(
    mode: ReplayMode,
    feedback: AnyFeedbackLaw | None,
    setup: ReplaySetup,
    window: tuple[float, float],
) -> None:
    if mode is ReplayMode.OPEN_LOOP and feedback is not None:
        raise ValueError(
            "an open-loop replay (INDEPENDENT_TIME_ONLY) takes no feedback law"
        )
    if mode is ReplayMode.FEEDBACK:
        if feedback is None:
            raise ValueError("a feedback replay (STATE_FEEDBACK) needs a feedback law")
        if setup.reference is None:
            raise ValueError("a feedback replay needs a reference trajectory to track")
        if feedback.n_actuated != setup.n_actuated:
            raise ValueError(
                f"feedback gains must act on every actuated joint "
                f"({setup.n_actuated}), got "
                f"{feedback.kp.shape if isinstance(feedback, FeedbackLaw) else feedback.n_actuated}"
            )
    ref = setup.reference
    if ref is not None and not ref.covers(*window):
        raise ValueError(
            f"reference must cover the replay window {window}, it spans "
            f"[{ref.times_s[0]}, {ref.times_s[-1]}]"
        )


def _node_times(times: Array, substeps: int) -> Array:
    pieces = [
        np.linspace(times[k], times[k + 1], substeps + 1)[:-1]
        for k in range(times.size - 1)
    ]
    return np.concatenate([*pieces, times[-1:]])


@dataclass
class _Trace:
    """Replayed nodes accumulated until the integration ends."""

    q: list[Array]
    v: list[Array]
    reason: str | None = None


def _integrate(
    provider: Any,
    nodes: Array,
    load: Any,
    setup: ReplaySetup,
    run: FeedbackRun | None = None,
) -> _Trace:
    trace = _Trace([setup.q0.copy()], [setup.v0.copy()])
    for i in range(nodes.size - 1):
        try:
            q_next, v_next = rk4_step_with_load(
                provider,
                trace.q[-1],
                trace.v[-1],
                load,
                float(nodes[i]),
                float(nodes[i + 1] - nodes[i]),
            )
        except ValueError as exc:
            trace.reason = f"dynamics failed at t={nodes[i + 1]:.6g}: {exc}"
            break
        if not (np.isfinite(q_next).all() and np.isfinite(v_next).all()):
            trace.reason = f"non-finite state at t={nodes[i + 1]:.6g}"
            break
        if max(np.max(np.abs(q_next)), np.max(np.abs(v_next))) > _BLOWUP:
            trace.reason = f"state exceeded {_BLOWUP:g} at t={nodes[i + 1]:.6g}"
            break
        trace.q.append(q_next)
        trace.v.append(v_next)
        if run is not None:
            run.observe(float(nodes[i + 1]), q_next, v_next)
    if trace.reason:
        logger.warning("replay terminated early: %s", trace.reason)
    return trace


def _closure_drift(provider: Any, q: Array, v: Array) -> ClosureDrift:
    velocity = np.array(
        [
            float(np.max(np.abs(provider.closure_jacobian(qi) @ vi), initial=0.0))
            for qi, vi in zip(q, v, strict=True)
        ]
    )
    error = getattr(provider, "closure_error", None)
    if not callable(error):
        return ClosureDrift(velocity, None)
    position = np.array(
        [float(np.max(np.abs(error(qi)), initial=0.0)) for qi in q], dtype=np.float64
    )
    return ClosureDrift(velocity, position)


def _stabilisation_peak(
    provider: Any, gains: BaumgarteGains | None, q: Array, v: Array
) -> float:
    """Largest ``|2 alpha J_c v + beta^2 Phi|`` the declared terms added."""
    if gains is None:
        return 0.0
    return float(
        max(
            np.max(
                np.abs(
                    2.0 * gains.alpha * (provider.closure_jacobian(qi) @ vi)
                    + gains.beta**2 * np.asarray(provider.closure_error(qi))
                ),
                initial=0.0,
            )
            for qi, vi in zip(q, v, strict=True)
        )
    )


def _make_load(
    sched: _TorqueSchedule,
    setup: ReplaySetup,
    feedback: AnyFeedbackLaw | None,
    run: FeedbackRun | None = None,
):
    selection, ref = setup.actuation, setup.reference

    def feedback_torque(t: float, q: Array, v: Array) -> Array:
        if feedback is None or ref is None:
            return np.zeros(setup.n_actuated)
        if run is not None:
            return run.torque(t, q, v)
        q_ref, v_ref = ref.state_at(t)
        law = cast("FeedbackLaw | AccelerationFeedbackLaw", feedback)  # no run
        return law.torque(selection @ (q_ref - q), selection @ (v_ref - v), q)

    def load(t: float, q: Array, v: Array) -> Array:
        tau = sched(t) + feedback_torque(t, q, v)
        generalised = selection.T @ tau
        if setup.external_load is not None:
            generalised = generalised + np.asarray(setup.external_load(t, q))
        return np.asarray(generalised)

    return load, feedback_torque


def _first_exceedance(error: Array, tolerance: float, times: Array) -> float | None:
    over = np.max(np.abs(error), axis=1) > tolerance
    return float(times[int(np.argmax(over))]) if over.any() else None


def _records(
    allocation: AllocationResult,
    mode: ReplayMode,
    series: dict[str, tuple[Array, str]],
    settings: ReplaySettings,
) -> dict[str, QuantityRecord]:
    base = allocation.record
    assert base is not None  # validated by _validated_allocation
    provenance = InputProvenance(
        "replay",
        f"{mode.value}, {settings.interpolation} interpolation",
        f"substeps={settings.substeps}; allocation-conditioned (NSX-31)",
    )
    return {
        name: QuantityRecord(
            values=values,
            units=units,
            allocation_dependence=AllocationDependence.FAMILY_DEPENDENT,
            model_provenance=base.model_provenance,
            input_provenance=provenance,
            hypotheses=base.hypotheses,
        )
        for name, (values, units) in series.items()
    }


def _feedback_series(
    feedback: Any, feedback_at: Any, nodes: Array, q: Array, v: Array
) -> Array | None:
    """Feedback torque at every replayed node (``None`` in open loop)."""
    if feedback is None:
        return None
    return np.array(
        [feedback_at(t, qi, vi) for t, qi, vi in zip(nodes, q, v, strict=True)]
    )


def _record_series(
    drift: ClosureDrift, q: Array, ref_q: Array | None, fb_series: Array | None
) -> dict[str, tuple[Array, str]]:
    """Series that get a ``QuantityRecord`` (feedback kept as its own entry)."""
    series: dict[str, tuple[Array, str]] = {
        "closure_velocity_drift": (drift.velocity, "m/s|rad/s")
    }
    if ref_q is not None:
        series["state_error"] = (np.max(np.abs(q - ref_q), axis=1), "m|rad")
    if fb_series is not None:
        series["feedback_torque"] = (fb_series, "N m")
    return series


def replay(
    allocation: AllocationResult,
    provider: Any,
    *,
    policy: ReplayPolicy,
    setup: ReplaySetup,
    feedback: AnyFeedbackLaw | None = None,
    settings: ReplaySettings | None = None,
) -> ReplayReport:
    """Replay ``allocation`` through ``provider`` in the mode ``policy`` names.

    Args:
        allocation: An ``OPTIMAL`` NSX-23 result with its record (torque
            ``(frames, n_a)`` on strictly increasing ``times_s``).
        provider: A ``ConstrainedEquationsOfMotionProvider`` (closure rows);
            a ``closure_error(q)`` method, when present, adds the position drift.
        policy: ``INDEPENDENT_TIME_ONLY`` (open loop; no feedback law allowed)
            or ``STATE_FEEDBACK`` (needs ``feedback`` and a reference).
        setup: Actuation selection, initial state, optional reference and known
            external load.
        feedback: The declared feedback law (feedback mode only).
        settings: Interpolation, sub-stepping, divergence tolerance and the
            optional declared Baumgarte terms.

    Returns:
        A :class:`ReplayReport` labelled with its mode.  The integration starts
        at ``(q0, v0)`` and never resets state.  ``feedback_budget`` and the
        feedback series are separate from the allocated torque.

    Raises:
        TypeError: ``policy`` is not a ``ReplayPolicy``.
        ValueError: A precondition above fails (non-OPTIMAL allocation, shape or
            time mismatch, missing law/reference, reference not covering the
            window, Baumgarte without a position closure error).
    """
    cfg = settings if settings is not None else ReplaySettings()
    mode = mode_for_policy(policy)
    if not is_constrained_provider(provider):
        raise TypeError(
            "provider must supply closure_jacobian and closure_drift (a "
            f"ConstrainedEquationsOfMotionProvider); got {type(provider).__name__}"
        )
    times, torque = _validated_allocation(allocation, setup, cfg)
    _check_mode_inputs(mode, feedback, setup, (float(times[0]), float(times[-1])))
    dynamics = (
        provider
        if cfg.baumgarte is None
        else _StabilisedProvider(provider, cfg.baumgarte)
    )
    sched = _TorqueSchedule(times, torque, cfg.interpolation)
    nodes = _node_times(times, cfg.substeps)
    run = begin_feedback_run(feedback, setup, nodes)
    load, feedback_at = _make_load(sched, setup, feedback, run)
    trace = _integrate(dynamics, nodes, load, setup, run)
    nodes, q, v = nodes[: len(trace.q)], np.array(trace.q), np.array(trace.v)
    allocated = np.array([sched(t) for t in nodes])
    fb_series = _feedback_series(feedback, feedback_at, nodes, q, v)
    applied = allocated if fb_series is None else allocated + fb_series
    ref_q = ref_v = None
    if setup.reference is not None:
        ref_q, ref_v = setup.reference.states_at(nodes)
    drift = _closure_drift(provider, q, v)
    budget = None
    if fb_series is not None:
        if feedback is None:
            raise ValueError("a feedback series requires a feedback law")
        budget = feedback_budget(
            nodes, allocated, fb_series, v @ setup.actuation.T, feedback.torque_limit
        )
    series = _record_series(drift, q, ref_q, fb_series)
    frame_index = np.arange(times.size) * cfg.substeps
    return ReplayReport(
        mode=mode,
        policy=policy,
        times_s=nodes,
        frame_node_index=frame_index[frame_index < nodes.size],
        q=q,
        v=v,
        reference_q=ref_q,
        reference_v=ref_v,
        allocated_torque=allocated,
        feedback_torque=fb_series,
        applied_torque=applied,
        feedback_law=feedback,
        feedback_budget=budget,
        closure_drift=drift,
        stabilisation=cfg.baumgarte,
        stabilisation_acceleration_peak=_stabilisation_peak(
            provider, cfg.baumgarte, q, v
        ),
        time_to_divergence_s=(
            None
            if ref_q is None
            else _first_exceedance(q - ref_q, cfg.divergence_tolerance, nodes)
        ),
        divergence_tolerance=cfg.divergence_tolerance,
        settings=cfg,
        records=_records(allocation, mode, series, cfg),
        terminated_reason=trace.reason,
    )

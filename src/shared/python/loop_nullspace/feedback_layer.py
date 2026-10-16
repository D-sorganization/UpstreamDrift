"""Declared impedance plus fixed-delay state feedback inside the replay (NSX-93).

A :class:`FeedbackLayer` is the law handed to :func:`replay.replay`.  It sums two
declared pieces whose torques stay separately reportable:

* the instantaneous joint impedance (:class:`~impedance_spec.JointImpedance`,
  ``H-DECLARED-JOINT-IMPEDANCE``), which acts on the *current* error because it
  is a property of the actuator, not of a neural loop;
* a :class:`DelayedFeedback` (``H-SENSORIMOTOR-DELAY``): a time-scheduled
  full-state gain acting on the error ``delay_s`` ago.

Delay semantics (declared).  The controller sees the delayed *error signal*
``e(s) = x_ref(s) - x(s)`` at ``s = t - delay_s`` and applies the present gain
``K(t)`` to it, ``tau_fb(t) = Kq(t) e_q(s) + Kv(t) e_v(s)`` -- the delayed-error
form of the discrete model ``u_t = -K_t dx_{t-m}`` that the linear variance
prediction uses.  Before the replay starts the state history is the initial
state held constant (``q(s) = q0``, ``v(s) = v0`` for ``s < t0``); the
reference is clipped to its sampled window as everywhere else.  The history
between integration nodes is the cubic Hermite interpolant through the nodes'
positions and velocities (exact for cubics, fourth order otherwise).

A delay is only meaningful with ``delay_s >= `` the largest integration step:
the fixed-step RK4 evaluates the law at the end of each step, and the state
``delay_s`` earlier must already have been integrated.  A shorter delay is
rejected, not approximated by extrapolation.

The law is a frozen value; the state history lives in the run it creates
(:mod:`feedback_runtime`), so one law serves many Monte Carlo trials.

Non-claim: this is a feedback *model* with declared gains and delay.  A replay
that holds the reference with it is tracking, never validation, and no gain or
delay is a measurement of a golfer.  Pure numpy.
"""

from __future__ import annotations

import bisect
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import ArrayLike

from src.shared.python.loop_nullspace.evidence import Array, readonly_float
from src.shared.python.loop_nullspace.feedback_runtime import FeedbackRun
from src.shared.python.loop_nullspace.impedance_spec import JointImpedance
from src.shared.python.loop_nullspace.replay_spec import ReplaySetup

__all__ = [
    "DELAY_HYPOTHESIS_ID",
    "DelayedFeedback",
    "FeedbackLayer",
    "GainSchedule",
    "StateHistory",
]

DELAY_HYPOTHESIS_ID = "H-SENSORIMOTOR-DELAY"
_STEP_RTOL = 1e-9


@dataclass(frozen=True)
class GainSchedule:
    """Piecewise-constant full-state gains ``tau = Kq e_q + Kv e_v``.

    ``times_s`` are the ``T + 1`` interval edges and ``kq``, ``kv`` the ``T``
    gain matrices ``(T, n_a, nv)`` held on ``[times_s[k], times_s[k+1])`` (the last
    gain also holds at and after the final edge).  ``e = ref - state`` so a
    positive ``Kq`` is restoring.

    Raises:
        ValueError: Non-increasing times, non-finite gains or mismatched shapes.
    """

    times_s: Array
    kq: Array
    kv: Array

    def __post_init__(self) -> None:
        t = readonly_float(self.times_s, "times_s")
        kq, kv = readonly_float(self.kq, "kq"), readonly_float(self.kv, "kv")
        if t.ndim != 1 or t.size < 2 or np.any(np.diff(t) <= 0.0):
            raise ValueError("times_s must be strictly increasing with >= 2 edges")
        if kq.ndim != 3 or kq.shape != kv.shape:
            raise ValueError(f"kq and kv must share a (T, n_a, nv) shape: {kq.shape}")
        if kq.shape[0] != t.size - 1:
            raise ValueError(f"kq needs {t.size - 1} gains, got {kq.shape[0]}")
        object.__setattr__(self, "times_s", t)
        object.__setattr__(self, "kq", kq)
        object.__setattr__(self, "kv", kv)

    @property
    def n_actuated(self) -> int:
        """Number of commanded torques."""
        return int(self.kq.shape[1])

    @property
    def n_coordinates(self) -> int:
        """Number of coordinates the gains read."""
        return int(self.kq.shape[2])

    def at(self, t: float) -> tuple[Array, Array]:
        """``(Kq, Kv)`` in force at time ``t`` (clipped to the first/last)."""
        k = int(np.searchsorted(self.times_s, t, side="right")) - 1
        k = min(max(k, 0), self.kq.shape[0] - 1)
        return self.kq[k], self.kv[k]


@dataclass(frozen=True)
class DelayedFeedback:
    """A :class:`GainSchedule` acting on the error ``delay_s`` seconds ago."""

    schedule: GainSchedule
    delay_s: float
    name: str = "delayed full-state feedback"

    def __post_init__(self) -> None:
        if not (math.isfinite(self.delay_s) and self.delay_s >= 0.0):
            raise ValueError(f"delay_s must be finite and >= 0, got {self.delay_s}")

    @property
    def n_actuated(self) -> int:
        """Number of commanded torques."""
        return self.schedule.n_actuated

    @property
    def hypotheses(self) -> tuple[str, ...]:
        """Registry hypotheses: the delay is declared even when it is zero."""
        return (DELAY_HYPOTHESIS_ID,)


class StateHistory:
    """Accepted integration nodes with cubic Hermite lookup of ``(q, v)``.

    ``at(s)`` for ``s`` before the first node returns the initial state held
    constant; for ``s`` after the last node it raises ``ValueError`` (no
    extrapolation), within ``1e-9`` relative of the last node it clamps.
    """

    def __init__(self, t0: float, q0: ArrayLike, v0: ArrayLike) -> None:
        self._t = [float(t0)]
        self._q = [np.array(q0, dtype=np.float64)]
        self._v = [np.array(v0, dtype=np.float64)]

    def append(self, t: float, q: ArrayLike, v: ArrayLike) -> None:
        """Record a node; times must strictly increase."""
        if not t > self._t[-1]:
            raise ValueError(f"history times must increase: {t} <= {self._t[-1]}")
        self._t.append(float(t))
        self._q.append(np.array(q, dtype=np.float64))
        self._v.append(np.array(v, dtype=np.float64))

    @property
    def t_last(self) -> float:
        """Time of the latest node."""
        return self._t[-1]

    def at(self, s: float) -> tuple[Array, Array]:
        """``(q(s), v(s))`` by Hermite interpolation of the recorded nodes."""
        if s <= self._t[0]:
            return self._q[0], self._v[0]
        if s > self._t[-1] + _STEP_RTOL * max(1.0, abs(self._t[-1])):
            raise ValueError(
                f"history reaches t={self._t[-1]:.9g}, cannot read t={s:.9g}: "
                "the delay is shorter than the integration step"
            )
        k = min(max(bisect.bisect_right(self._t, s) - 1, 0), len(self._t) - 2)
        h = self._t[k + 1] - self._t[k]
        x = min((s - self._t[k]) / h, 1.0)
        q0, q1, v0, v1 = self._q[k], self._q[k + 1], self._v[k], self._v[k + 1]
        h00, h10 = 2 * x**3 - 3 * x**2 + 1, x**3 - 2 * x**2 + x
        h01, h11 = -2 * x**3 + 3 * x**2, x**3 - x**2
        q = h00 * q0 + h10 * h * v0 + h01 * q1 + h11 * h * v1
        dh00, dh10 = 6 * x**2 - 6 * x, 3 * x**2 - 4 * x + 1
        dh01, dh11 = -6 * x**2 + 6 * x, 3 * x**2 - 2 * x
        v = (dh00 * q0 + dh01 * q1) / h + dh10 * v0 + dh11 * v1
        return q, v


@dataclass(frozen=True)
class FeedbackLayer:
    """The law given to ``replay``: declared impedance plus delayed feedback.

    ``torque_limit`` (``N m`` per joint) is a declared saturation of the *sum*;
    the saturated fraction is reported by the replay's feedback budget.  The
    separate impedance and feedback torques are reported by
    :mod:`layer_accounting`.

    Raises:
        ValueError: Neither piece declared, or a piece acting on a different
            number of joints than ``n_actuated``.
    """

    n_actuated: int
    impedance: JointImpedance | None = None
    feedback: DelayedFeedback | None = None
    torque_limit: Array | None = None
    name: str = "declared impedance + delayed feedback"

    def __post_init__(self) -> None:
        if self.impedance is None and self.feedback is None:
            raise ValueError("a FeedbackLayer needs at least one of impedance/feedback")
        if self.impedance is not None and self.impedance.n_actuated != self.n_actuated:
            raise ValueError(
                f"impedance acts on {self.impedance.n_actuated} joints, "
                f"n_actuated is {self.n_actuated}"
            )
        if self.feedback is not None and self.feedback.n_actuated != self.n_actuated:
            raise ValueError(
                f"feedback commands {self.feedback.n_actuated} torques, "
                f"n_actuated is {self.n_actuated}"
            )
        if self.torque_limit is not None:
            lim = readonly_float(self.torque_limit, "torque_limit")
            if lim.shape != (self.n_actuated,) or np.any(lim <= 0.0):
                raise ValueError("torque_limit must be positive, one per joint")
            object.__setattr__(self, "torque_limit", lim)

    @property
    def hypotheses(self) -> tuple[str, ...]:
        """Registry hypotheses of the declared pieces (impedance, then delay)."""
        out: tuple[str, ...] = ()
        if self.impedance is not None:
            out += self.impedance.hypotheses
        if self.feedback is not None:
            out += self.feedback.hypotheses
        return out

    def begin_run(
        self, setup: ReplaySetup, t_start: float, max_step: float
    ) -> FeedbackRun:
        """A fresh run for one replay (see :mod:`feedback_runtime`).

        Raises:
            ValueError: No reference to track, gains reading a different number
                of coordinates than the model has, or ``0 < delay_s <`` the
                largest integration step.
        """
        if setup.reference is None:
            raise ValueError("a FeedbackLayer tracks a reference trajectory")
        fb = self.feedback
        if fb is not None:
            if fb.schedule.n_coordinates != setup.n:
                raise ValueError(
                    f"feedback gains read {fb.schedule.n_coordinates} coordinates, "
                    f"the model has {setup.n}"
                )
            if 0.0 < fb.delay_s < max_step * (1.0 - _STEP_RTOL):
                raise ValueError(
                    f"delay_s={fb.delay_s:g} is shorter than the integration step "
                    f"{max_step:g}; use a smaller step or a longer delay"
                )
        return _LayerRun(self, setup, t_start)

    def component_torques(
        self, setup: ReplaySetup, times: Array, q: Array, v: Array
    ) -> tuple[Array, Array]:
        """Impedance and feedback torques at the replayed nodes, ``(T, n_a)`` each.

        Recomputed from the replayed states through a fresh run; the sum equals
        the replay's unsaturated feedback series.
        """
        run = self.begin_run(setup, float(times[0]), float(np.max(np.diff(times))))
        imp, fb = [], []
        for k, (t, qi, vi) in enumerate(zip(times, q, v, strict=True)):
            if k > 0:
                run.observe(float(t), qi, vi)
            a, b = run.components(float(t), qi, vi)
            imp.append(a)
            fb.append(b)
        return np.array(imp), np.array(fb)


class _LayerRun:
    """One replay's history and evaluation of the layer."""

    def __init__(self, layer: FeedbackLayer, setup: ReplaySetup, t0: float) -> None:
        self._layer, self._setup = layer, setup
        self._hist = StateHistory(t0, setup.q0, setup.v0)

    def observe(self, t: float, q: Array, v: Array) -> None:
        self._hist.append(t, q, v)

    def components(self, t: float, q: Array, v: Array) -> tuple[Array, Array]:
        """Unsaturated ``(impedance, feedback)`` torques at ``(t, q, v)``."""
        setup, layer = self._setup, self._layer
        ref = setup.reference
        assert ref is not None  # checked in begin_run
        n_a = setup.n_actuated
        imp = np.zeros(n_a)
        fb = np.zeros(n_a)
        q_ref, v_ref = ref.state_at(t)
        if layer.impedance is not None:
            imp = layer.impedance.torque(
                setup.actuation @ (q_ref - q), setup.actuation @ (v_ref - v), t
            )
        if layer.feedback is not None:
            fb = self._feedback(layer.feedback, t, q, v, q_ref, v_ref)
        return imp, fb

    def _feedback(
        self, law: DelayedFeedback, t: float, q: Any, v: Any, q_ref: Any, v_ref: Any
    ) -> Array:
        if law.delay_s > 0.0:
            s = t - law.delay_s
            q, v = self._hist.at(s)
            ref = self._setup.reference
            assert ref is not None
            q_ref, v_ref = ref.state_at(s)
        kq, kv = law.schedule.at(t)
        return np.asarray(kq @ (q_ref - q) + kv @ (v_ref - v))

    def torque(self, t: float, q: Array, v: Array) -> Array:
        imp, fb = self.components(t, q, v)
        total = imp + fb
        lim = self._layer.torque_limit
        return total if lim is None else np.asarray(np.clip(total, -lim, lim))

"""Run-scoped feedback laws for the replay (NSX-93, #12722).

:class:`~replay_spec.FeedbackLaw` and :class:`~replay_spec.AccelerationFeedbackLaw`
are stateless: the torque is a function of the current error.  A delayed law
needs the replayed state history, and an impedance layer needs the time, so a
law may instead offer ``begin_run(setup, t_start, max_step)``, which returns a
fresh :class:`FeedbackRun` for *one* replay.  The frozen law stays reusable
across the trials of a Monte Carlo campaign (NSX-91) because the history lives in
the run, never in the law.

Contract of a run: ``observe(t, q, v)`` is called once for every accepted
integration node after the initial state (RK4 stage states are never observed),
and ``torque(t, q, v)`` may then be called at any time ``t`` for which every
observation it needs has been made.  Pure Python; no engine imports.
"""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.shared.python.loop_nullspace.evidence import Array
from src.shared.python.loop_nullspace.replay_spec import (
    AccelerationFeedbackLaw,
    FeedbackLaw,
    ReplaySetup,
)

__all__ = ["AnyFeedbackLaw", "FeedbackRun", "RunFeedbackLaw", "begin_feedback_run"]


class FeedbackRun(Protocol):
    """State of one replay of a run-scoped feedback law."""

    def torque(self, t: float, q: Array, v: Array) -> Array:
        """Feedback torque ``(n_a,)`` at time ``t`` and the current state."""

    def observe(self, t: float, q: Array, v: Array) -> None:
        """Record the accepted integration node ``(t, q, v)``."""


@runtime_checkable
class RunFeedbackLaw(Protocol):
    """A declared feedback law that needs per-replay state (history, time)."""

    n_actuated: int
    torque_limit: Array | None
    name: str

    def begin_run(
        self, setup: ReplaySetup, t_start: float, max_step: float
    ) -> FeedbackRun:
        """A fresh run for one replay (``ValueError`` if the law cannot run)."""


AnyFeedbackLaw = FeedbackLaw | AccelerationFeedbackLaw | RunFeedbackLaw


def begin_feedback_run(
    feedback: AnyFeedbackLaw | None, setup: ReplaySetup, nodes: Array
) -> FeedbackRun | None:
    """The run of ``feedback`` over the integration ``nodes`` (``None`` if stateless).

    ``max_step`` handed to the law is the largest node spacing, which a delayed
    law needs to guarantee that the state it reads has already been integrated.
    """
    if feedback is None or not isinstance(feedback, RunFeedbackLaw):
        return None
    return feedback.begin_run(setup, float(nodes[0]), float(np.max(np.diff(nodes))))

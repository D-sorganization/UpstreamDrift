"""Session-level train/test split and train-only noise fit (NSX-97).

Noise amplitudes are fitted on **training sessions only**.  Every fit and
every consistency check raises :class:`LeakageError` if a test-session swing
reaches the fit.  The split is by session, never by swing, so a session
offset cannot leak across the boundary.  Software correctness only.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

Array = NDArray[np.float64]


class LeakageError(ValueError):
    """A test-session swing reached a quantity fitted on training sessions."""


@dataclass(frozen=True)
class SwingSample:
    """One swing: identifier, owning session and impact-state vector."""

    swing_id: str
    session_id: str
    state: Array

    def __post_init__(self) -> None:
        if not self.swing_id or not self.session_id:
            raise ValueError("SwingSample requires non-empty swing_id and session_id")
        state = np.array(self.state, dtype=np.float64, copy=True)
        if state.ndim != 1 or state.size == 0 or not np.all(np.isfinite(state)):
            raise ValueError("SwingSample.state must be a finite non-empty vector")
        state.setflags(write=False)
        object.__setattr__(self, "state", state)


@dataclass(frozen=True)
class SessionSplit:
    """Disjoint train and test session sets with the seed that produced them."""

    train_sessions: frozenset[str]
    test_sessions: frozenset[str]
    seed: int

    def __post_init__(self) -> None:
        if not self.train_sessions or not self.test_sessions:
            raise ValueError(
                "SessionSplit needs at least one train and one test session"
            )
        overlap = self.train_sessions & self.test_sessions
        if overlap:
            raise ValueError(f"train and test sessions overlap: {sorted(overlap)}")

    @classmethod
    def from_sessions(
        cls, session_ids: Iterable[str], test_fraction: float, seed: int
    ) -> SessionSplit:
        """Seeded random session-level split.

        Raises:
            ValueError: If fewer than two distinct sessions exist or
                ``test_fraction`` is not in (0, 1).
        """
        sessions = sorted(set(session_ids))
        if len(sessions) < 2:
            raise ValueError("a session-level split needs at least two sessions")
        if not 0.0 < test_fraction < 1.0:
            raise ValueError("test_fraction must be in (0, 1)")
        n_test = min(len(sessions) - 1, max(1, round(test_fraction * len(sessions))))
        order = np.random.default_rng(seed).permutation(len(sessions))
        test = frozenset(sessions[i] for i in order[:n_test])
        return cls(frozenset(sessions) - test, test, seed)


@dataclass(frozen=True)
class NoiseFit:
    """Pooled within-session state covariance fitted on training swings."""

    amplitudes: Array
    covariance: Array
    train_sessions: frozenset[str]
    swing_ids: tuple[str, ...]


def fit_noise_amplitudes(
    samples: Sequence[SwingSample], split: SessionSplit
) -> NoiseFit:
    """Fit per-dimension noise amplitudes from training sessions only.

    The covariance is pooled within-session (each session is demeaned), so
    session offsets are excluded from the noise.

    Raises:
        LeakageError: If any swing belongs to a test session or to a session
            outside the split.
        ValueError: If a session has fewer than two swings or states differ in
            dimension.
    """
    for s in samples:
        if s.session_id in split.test_sessions:
            raise LeakageError(
                f"test swing {s.swing_id!r} (session {s.session_id!r}) in noise fit"
            )
        if s.session_id not in split.train_sessions:
            raise LeakageError(
                f"swing {s.swing_id!r} belongs to session {s.session_id!r}, "
                "which is not a training session of this split"
            )
    if not samples:
        raise ValueError("no training swings supplied")
    dim = samples[0].state.size
    if any(s.state.size != dim for s in samples):
        raise ValueError("all swing states must have the same dimension")
    scatter = np.zeros((dim, dim))
    dof = 0
    for sid in sorted({s.session_id for s in samples}):
        x = np.stack([s.state for s in samples if s.session_id == sid])
        if x.shape[0] < 2:
            raise ValueError(f"session {sid!r} has fewer than two swings")
        xc = x - x.mean(axis=0)
        scatter += xc.T @ xc
        dof += x.shape[0] - 1
    cov = scatter / dof
    return NoiseFit(
        amplitudes=np.sqrt(np.diag(cov)),
        covariance=cov,
        train_sessions=frozenset(s.session_id for s in samples),
        swing_ids=tuple(s.swing_id for s in samples),
    )


def assert_fit_not_leaked(
    fit: NoiseFit, split: SessionSplit, test_swing_ids: Iterable[str]
) -> None:
    """Require the fit saw no test swing and only training sessions.

    Raises:
        LeakageError: On any overlap or session-set mismatch.
    """
    leaked = set(fit.swing_ids) & set(test_swing_ids)
    if leaked:
        raise LeakageError(f"test swings used in noise fit: {sorted(leaked)}")
    if not fit.train_sessions <= split.train_sessions:
        raise LeakageError("noise fit used sessions outside the training set")

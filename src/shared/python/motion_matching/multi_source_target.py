"""Multi-source motion target container.

Issue #4480: the starting-pose matcher must be able to drive its view from
**any combination** of three independent target sources:

* a 6-DOF club trajectory (``ClubTarget``)
* a club trajectory plus a ball-impact boundary condition (``ClubBallTarget``)
* a full-body anatomical-marker target (``BodyTarget``)

``MultiSourceTarget`` is the small, frozen dataclass that downstream
code (cost functions, animation playback, diagnostics) consumes. It
holds optional slots for the club-side and body-side targets and
exposes ``shared_time()`` plus ``has_*()`` predicates so callers can
dispatch on whichever subset is active.

The dependent target types (``ClubBallTarget`` from issue #4479,
``BodyTarget`` from issue #4476) may not have landed at the time this
module is imported. We therefore type ``club`` and ``body`` against a
duck-typed ``Any`` plus a runtime check that the supplied object has a
``time`` attribute. This keeps the module importable on ``main`` and
the validation contract identical once the dependencies merge.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Any

import numpy as np

from .club_target import TIME_EPS, ClubTarget

if TYPE_CHECKING:  # pragma: no cover — type-only imports
    # These dependencies (issues #4476 and #4479) are not yet on main.
    # Imports are guarded so this module remains importable today.
    try:
        from .club_ball_target import ClubBallTarget  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        ClubBallTarget = Any  # type: ignore[misc,assignment]
    try:
        from .body_target import BodyTarget  # type: ignore[import-not-found]
    except ImportError:  # pragma: no cover
        BodyTarget = Any  # type: ignore[misc,assignment]


__all__ = ["MultiSourceTarget"]


def _has_time_array(obj: Any) -> bool:
    """Return True iff ``obj`` quacks like a target (has a 1-D time array)."""
    if obj is None:
        return False
    time = getattr(obj, "time", None)
    return isinstance(time, np.ndarray) and time.ndim == 1


@dataclass(frozen=True)
class MultiSourceTarget:
    """Container for one or more motion-matching targets.

    Attributes:
        club: Either a ``ClubTarget`` or a ``ClubBallTarget`` (the latter
            adds a ball-impact boundary condition), or ``None`` when the
            user did not load a club source.
        body: A ``BodyTarget`` (full-body anatomical-marker trajectory),
            or ``None`` when the user did not load body markers.

    Validation (run at construction):
        * At least one of ``club`` / ``body`` MUST be non-None.
        * Each non-None slot MUST expose a 1-D ``time`` ndarray.
        * If both slots are present, their ``time`` arrays MUST match
          element-wise within ``TIME_EPS``.

    A frozen dataclass makes this safe to share between threads (Qt
    main thread vs. matplotlib animation worker).
    """

    club: Any  # ClubTarget | ClubBallTarget | None
    body: Any  # BodyTarget | None

    def __post_init__(self) -> None:
        if self.club is None and self.body is None:
            raise ValueError(
                "MultiSourceTarget requires at least one non-None slot (club or body)."
            )
        if self.club is not None and not _has_time_array(self.club):
            raise TypeError(
                "MultiSourceTarget.club must expose a 1-D 'time' ndarray; "
                f"got {type(self.club).__name__!r}."
            )
        if self.body is not None and not _has_time_array(self.body):
            raise TypeError(
                "MultiSourceTarget.body must expose a 1-D 'time' ndarray; "
                f"got {type(self.body).__name__!r}."
            )
        if self.club is not None and self.body is not None:
            t_club = self.club.time
            t_body = self.body.time
            if t_club.shape != t_body.shape or not np.allclose(
                t_club, t_body, atol=TIME_EPS, rtol=0.0
            ):
                raise ValueError(
                    "MultiSourceTarget timegrid mismatch: club has shape "
                    f"{t_club.shape}, body has shape {t_body.shape}. "
                    "Both sources must share the same resampled timegrid "
                    "(plumb the club ClubTarget through as impact_source= "
                    "when loading the body target)."
                )

    def has_club(self) -> bool:
        """Return True iff a club-side target is present."""
        return self.club is not None

    def has_body(self) -> bool:
        """Return True iff a body-side target is present."""
        return self.body is not None

    def shared_time(self) -> np.ndarray:
        """Return the timegrid common to all loaded targets.

        Both slots are required to share a timegrid (validated in
        ``__post_init__``), so returning the time array of any present
        slot is sufficient. We prefer the club slot when present.
        """
        if self.club is not None:
            return self.club.time  # type: ignore[no-any-return]
        if self.body is not None:
            return self.body.time  # type: ignore[no-any-return]
        # Unreachable — __post_init__ enforces at least one slot.
        raise RuntimeError("MultiSourceTarget has no slots")  # pragma: no cover

    def is_club_ball(self) -> bool:
        """Return True iff the club slot is a ``ClubBallTarget`` (i.e. has a
        ball-impact boundary condition).

        Detected via duck-typing on the ``ball`` attribute so this works
        before the ``ClubBallTarget`` dependency lands.
        """
        return self.club is not None and hasattr(self.club, "ball")

    def is_plain_club(self) -> bool:
        """Return True iff the club slot is a plain ``ClubTarget``."""
        return isinstance(self.club, ClubTarget) and not self.is_club_ball()

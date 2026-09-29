"""Canonical ``ClubBallTarget`` dataclass and ball-impact boundary state.

Mirrors the style of :mod:`club_target`: frozen dataclasses, validated at
construction, descriptive ``ValueError``/``TypeError`` raised on contract
violation. The ball-impact boundary condition lets motion matching score
fits not just by club kinematics but also by alignment with the ball at
impact (position, launch direction, launch speed).

The default extractor :func:`extract_ball_impact_from_clubtarget`
approximates the ball state from the club state alone — adequate for
visualisation and as a stand-in until a real launch-monitor feed is wired
in by a separate loader path. The approximations are documented in the
extractor docstring.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .club_target import MAX_POSITION_NORM_M, ClubTarget

# --- Validation tolerances and bounds -------------------------------------

# Unit-norm tolerance for the launch-direction vector. Loose enough for
# float32-then-cast inputs; tight enough to catch un-normalised data.
LAUNCH_DIR_NORM_TOL = 1.0e-6

# Plausible upper bound on launch speed (m/s) for any human-driven club.
MAX_LAUNCH_SPEED_MPS = 100.0

# Plausible upper bound on ball spin (rpm). 15k covers extreme wedge spin.
MAX_SPIN_RPM = 15_000.0

# Elasticity factor used by the default extractor to estimate launch speed
# from clubhead speed. Documented as a stand-in pending real launch-monitor
# data; clamped to ``MAX_LAUNCH_SPEED_MPS``.
DEFAULT_ELASTICITY_FACTOR = 1.5

# Schema version for forward-compatible serialisation.
CLUB_BALL_TARGET_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class BallImpactState:
    """Ball state at the moment of impact (boundary condition).

    Attributes:
        position_at_impact_m: World-frame Z-up position (3,) in metres.
            All components must be finite and ``|r| < MAX_POSITION_NORM_M``.
        launch_direction:     Unit vector (3,) in the world frame. Either
            fully finite (and unit-norm to ``LAUNCH_DIR_NORM_TOL``) or
            fully NaN (unknown). Partial unknowns are rejected.
        launch_speed_mps:     Launch speed in m/s. Either NaN (unknown) or
            finite in ``[0, MAX_LAUNCH_SPEED_MPS]``.
        spin_rpm:             Ball spin in rpm. Either NaN (unknown) or
            finite in ``[0, MAX_SPIN_RPM]``.
    """

    position_at_impact_m: np.ndarray
    launch_direction: np.ndarray
    launch_speed_mps: float
    spin_rpm: float

    def __post_init__(self) -> None:
        """Validate the boundary state at construction."""
        _validate_ball_impact_state(self)


@dataclass(frozen=True)
class ClubBallTarget:
    """Composite target combining a ``ClubTarget`` with a ball boundary state.

    Used by motion matching when the user wants the ball impact state to
    contribute to the cost (e.g. clubface aligned with launch direction at
    impact). The ``time`` and ``impact_idx`` properties delegate to the
    underlying club target so callers can treat a ``ClubBallTarget`` as a
    drop-in for ``ClubTarget`` in code that only needs those fields.
    """

    club: ClubTarget
    ball_impact: BallImpactState

    def __post_init__(self) -> None:
        """Type-check the wrapped components."""
        if not isinstance(self.club, ClubTarget):
            raise TypeError("club must be a ClubTarget instance")
        if not isinstance(self.ball_impact, BallImpactState):
            raise TypeError("ball_impact must be a BallImpactState instance")

    @property
    def time(self) -> np.ndarray:
        """Delegate: the resampled time grid from the club target."""
        return self.club.time

    @property
    def impact_idx(self) -> int:
        """Delegate: the impact index (1-based) from the club target."""
        return self.club.impact_idx


# --- Validation helpers ---------------------------------------------------


def _validate_position(pos: np.ndarray) -> None:
    """Reject malformed, NaN/Inf, or implausibly large impact positions."""
    if not isinstance(pos, np.ndarray):
        raise TypeError("position_at_impact_m must be a numpy.ndarray")
    if pos.shape != (3,):
        raise ValueError(f"position_at_impact_m must have shape (3,), got {pos.shape}")
    if not np.all(np.isfinite(pos)):
        raise ValueError("position_at_impact_m contains NaN or Inf")
    norm = float(np.linalg.norm(pos))
    if norm >= MAX_POSITION_NORM_M:
        raise ValueError(
            f"position_at_impact_m has |r| >= {MAX_POSITION_NORM_M} m (got {norm:.3f})"
        )


def _validate_launch_direction(direction: np.ndarray) -> None:
    """Allow either fully-NaN or fully-finite unit-norm launch directions."""
    if not isinstance(direction, np.ndarray):
        raise TypeError("launch_direction must be a numpy.ndarray")
    if direction.shape != (3,):
        raise ValueError(
            f"launch_direction must have shape (3,), got {direction.shape}"
        )
    nan_mask = np.isnan(direction)
    if np.any(nan_mask) and not np.all(nan_mask):
        raise ValueError(
            "launch_direction must be fully NaN or fully finite (no partial unknowns)"
        )
    if np.all(nan_mask):
        return
    if not np.all(np.isfinite(direction)):
        raise ValueError("launch_direction contains Inf")
    norm = float(np.linalg.norm(direction))
    if abs(norm - 1.0) > LAUNCH_DIR_NORM_TOL:
        raise ValueError(
            "launch_direction must be unit-norm to within "
            f"{LAUNCH_DIR_NORM_TOL} (got {norm:.6f})"
        )


def _validate_scalar_in_range(value: float, name: str, lo: float, hi: float) -> None:
    """Validate a finite-or-NaN scalar against ``[lo, hi]`` when finite."""
    fv = float(value)
    if np.isnan(fv):
        return
    if not np.isfinite(fv):
        raise ValueError(f"{name} must be finite or NaN (got {fv!r})")
    if not (lo <= fv <= hi):
        raise ValueError(f"{name} must be in [{lo}, {hi}] (got {fv})")


def _validate_ball_impact_state(s: BallImpactState) -> None:
    """Enforce the ``BallImpactState`` validation rules."""
    _validate_position(s.position_at_impact_m)
    _validate_launch_direction(s.launch_direction)
    _validate_scalar_in_range(
        s.launch_speed_mps, "launch_speed_mps", 0.0, MAX_LAUNCH_SPEED_MPS
    )
    _validate_scalar_in_range(s.spin_rpm, "spin_rpm", 0.0, MAX_SPIN_RPM)


# --- Default extractor ----------------------------------------------------


def _clubhead_velocity_at_impact(target: ClubTarget) -> np.ndarray:
    """Numerical clubhead velocity at the impact frame (1-based ``impact_idx``).

    Uses a centred difference where possible, falling back to one-sided
    differences at the array endpoints. Returns a (3,) vector in m/s.
    """
    k = int(target.impact_idx) - 1  # 1-based -> 0-based array index
    n = target.time.shape[0]
    if not 0 <= k < n:
        raise ValueError(f"impact_idx {target.impact_idx} maps outside range for N={n}")
    if n < 2:
        raise ValueError("ClubTarget must have at least 2 samples for velocity")
    if k == 0:
        dt = float(target.time[1] - target.time[0])
        return (target.clubhead[1] - target.clubhead[0]) / dt
    if k == n - 1:
        dt = float(target.time[k] - target.time[k - 1])
        return (target.clubhead[k] - target.clubhead[k - 1]) / dt
    dt = float(target.time[k + 1] - target.time[k - 1])
    return (target.clubhead[k + 1] - target.clubhead[k - 1]) / dt


def extract_ball_impact_from_clubtarget(
    target: ClubTarget,
    *,
    elasticity_factor: float = DEFAULT_ELASTICITY_FACTOR,
) -> BallImpactState:
    """Approximate a :class:`BallImpactState` from a :class:`ClubTarget`.

    This is a stand-in for a real launch-monitor feed and is intended for
    visualisation and for the ball-aware cost term in the absence of
    measured ball data. The approximations are:

    * ``position_at_impact_m`` := clubhead position at the impact frame.
      Strictly speaking the ball is offset by the clubhead radius and the
      impact location on the face; using the clubhead centre is good
      enough for visual overlay and within the validation envelope.
    * ``launch_direction`` := unit vector of the centred-difference
      clubhead velocity at impact. If the velocity is degenerate
      (norm below numerical noise) the direction is reported as NaN to
      signal "unknown" rather than fabricate a value.
    * ``launch_speed_mps`` := ``|v_clubhead at impact| * elasticity_factor``,
      clamped to ``MAX_LAUNCH_SPEED_MPS``. The default factor of 1.5 is a
      stand-in pending real launch-monitor data.
    * ``spin_rpm`` := ``NaN`` (no acceptable approximation from club state).

    A real launch-monitor loader can construct a :class:`BallImpactState`
    directly without going through this extractor.

    Args:
        target: A validated :class:`ClubTarget`.
        elasticity_factor: Stand-in factor relating clubhead speed at
            impact to ball launch speed. Must be finite and >= 0.

    Returns:
        A validated :class:`BallImpactState`.

    Raises:
        TypeError:  If ``target`` is not a :class:`ClubTarget`.
        ValueError: If ``elasticity_factor`` is non-finite or negative.
    """
    if not isinstance(target, ClubTarget):
        raise TypeError("target must be a ClubTarget instance")
    if not np.isfinite(elasticity_factor) or elasticity_factor < 0.0:
        raise ValueError(
            f"elasticity_factor must be finite and >= 0 (got {elasticity_factor!r})"
        )

    k = int(target.impact_idx) - 1
    position = np.asarray(target.clubhead[k], dtype=float).copy()

    velocity = _clubhead_velocity_at_impact(target)
    speed = float(np.linalg.norm(velocity))
    if speed > LAUNCH_DIR_NORM_TOL:
        direction = (velocity / speed).astype(float)
    else:
        direction = np.full(3, np.nan, dtype=float)

    launch_speed = min(speed * float(elasticity_factor), MAX_LAUNCH_SPEED_MPS)

    return BallImpactState(
        position_at_impact_m=position,
        launch_direction=direction,
        launch_speed_mps=float(launch_speed),
        spin_rpm=float("nan"),
    )

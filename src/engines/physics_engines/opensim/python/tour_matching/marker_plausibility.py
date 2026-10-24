"""Plausibility screening gates for tour-matched OpenSim models (#13434).

These are DECLARED SCREENING BOUNDS for the total torso-in-pelvis rotation
magnitude, chosen to reject gross registration failures (the unregistered
OS-8 model gave 65/113/120 deg). They are not physiological acceptance
limits; a literature citation is pending owner review (#13434).
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
import logging

import numpy as np

from src.engines.physics_engines.opensim.python.tour_matching.anatomical_marker_priors import (
    AnatomicalPrior,
    lateral_side_ok,
)
from src.shared.python.contracts import ensure, require

logger = logging.getLogger(__name__)

TRUNK_ROTATION_LIMITS_DEG: dict[str, float] = {
    "address": 30.0,
    "top": 75.0,
    "impact": 60.0,
}


class ImplausibleInputError(ValueError):
    """Raised when marker placement or trunk kinematics fail plausibility gates."""


@dataclass(frozen=True)
class MarkerCheck:
    """Individual marker placement plausibility check against anatomical prior."""

    name: str
    body: str
    deviation_m: float
    radius_m: float
    side_ok: bool
    passed: bool


@dataclass(frozen=True)
class TrunkCheck:
    """Trunk rotation plausibility check against declared screening limits."""

    event: str
    rotation_deg: float
    limit_deg: float
    passed: bool


@dataclass(frozen=True)
class PlausibilityReport:
    """Consolidated plausibility report across markers and trunk rotation."""

    checks: tuple[MarkerCheck, ...] = ()
    trunk: tuple[TrunkCheck, ...] = ()

    def __post_init__(self) -> None:
        object.__setattr__(self, "checks", tuple(self.checks))
        object.__setattr__(self, "trunk", tuple(self.trunk))

    @property
    def passed(self) -> bool:
        """True if every marker check and trunk check passed."""
        return all(c.passed for c in self.checks) and all(t.passed for t in self.trunk)

    def failures(self) -> tuple[str, ...]:
        """Return the names of all failed items (markers and trunk events)."""
        failed = [c.name for c in self.checks if not c.passed]
        failed.extend(t.event for t in self.trunk if not t.passed)
        return tuple(failed)


def assess_marker_placements(
    offsets: Mapping[str, tuple[str, Sequence[float] | np.ndarray]],
    priors: Mapping[str, AnatomicalPrior],
) -> tuple[MarkerCheck, ...]:
    """Assess marker offsets against anatomical priors."""
    require(isinstance(offsets, Mapping), "offsets must be a Mapping")
    require(isinstance(priors, Mapping), "priors must be a Mapping")

    checks: list[MarkerCheck] = []
    for name, prior in priors.items():
        if name not in offsets:
            checks.append(
                MarkerCheck(
                    name, prior.body, float("inf"), float(prior.radius_m), False, False
                )
            )
            continue

        body, offset = offsets[name]
        offset_vec = (float(offset[0]), float(offset[1]), float(offset[2]))
        diff = np.asarray(offset_vec, dtype=float) - np.asarray(
            prior.offset_m, dtype=float
        )
        dev = float(np.linalg.norm(diff))
        side_ok = bool(lateral_side_ok(name, offset_vec))
        body_ok = body == prior.body or body.rsplit("/", 1)[-1] == prior.body
        passed = bool(dev <= prior.radius_m + 1e-9 and side_ok and body_ok)
        checks.append(
            MarkerCheck(name, body, dev, float(prior.radius_m), side_ok, passed)
        )

    for c in checks:
        if not c.passed:
            logger.warning(
                "Marker %s failed plausibility: dev=%.4f m (radius=%.4f m), side_ok=%s, body=%s",
                c.name,
                c.deviation_m,
                c.radius_m,
                c.side_ok,
                c.body,
            )

    ensure(len(checks) == len(priors), "expected one check per prior")
    return tuple(checks)


def assess_trunk_rotation(
    rotation_deg_by_event: Mapping[str, float],
    limits: Mapping[str, float] = TRUNK_ROTATION_LIMITS_DEG,
) -> tuple[TrunkCheck, ...]:
    """Assess trunk rotation magnitude against declared screening limits."""
    require(
        isinstance(rotation_deg_by_event, Mapping),
        "rotation_deg_by_event must be a Mapping",
    )
    require(isinstance(limits, Mapping), "limits must be a Mapping")

    checks: list[TrunkCheck] = []
    for event, limit_deg in limits.items():
        require(
            event in rotation_deg_by_event, f"Required trunk event {event!r} missing"
        )
        rot = float(rotation_deg_by_event[event])
        require(
            bool(np.isfinite(rot)), f"Rotation for {event!r} must be finite (got {rot})"
        )
        passed = bool(abs(rot) <= float(limit_deg) + 1e-9)
        checks.append(TrunkCheck(event, rot, float(limit_deg), passed))

    for t in checks:
        if not t.passed:
            logger.warning(
                "Trunk rotation for %s exceeds screening bound: %.2f deg > %.2f deg",
                t.event,
                t.rotation_deg,
                t.limit_deg,
            )

    ensure(len(checks) == len(limits), "expected one check per limited event")
    return tuple(checks)


def require_plausible(report: PlausibilityReport) -> None:
    """Raise ImplausibleInputError listing every failure if any check failed."""
    require(
        isinstance(report, PlausibilityReport), "report must be a PlausibilityReport"
    )
    if not report.passed:
        failures = report.failures()
        msg = f"Marker/trunk plausibility checks failed: {', '.join(failures)}"
        logger.error(msg)
        raise ImplausibleInputError(msg)

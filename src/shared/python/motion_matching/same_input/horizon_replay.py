"""Same-input open-loop replay for the impact horizon search (#15150 D2).

:func:`~src.shared.python.motion_matching.open_loop_horizon.search_horizon`
takes an abstract replay callable.  :class:`BundleReplay` builds it from a
same-input bundle: a candidate start re-anchors to the bundle's tracked state
at that step (as :func:`~.scoring.segmented_replay` does), replays the
recorded efforts open loop with the bundle's own step policy up to the last
step at or before contact, and measures the club delivery there.

Delivery is measured identically for the reference and every replay, on the
rendered face (``Clubhead`` frame, :func:`face_axis_in_frame` and
:func:`face_centre_in_frame`):

* face angle: :func:`horizontal_face_angle_deg` of the world face normal;
* path: the same open-positive horizontal heading of the face-centre velocity
  (positive toward native -X, in-to-out for a right-handed golfer);
* head speed: the face-centre speed over the last pre-contact step.

The replay stops before the ball window opens, so the recorded ball force
never enters it.  Every number comes from the bundle or the horizon settings.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.model_appearance.club_face import horizontal_face_angle_deg
from src.shared.python.motion_matching.club_face_target import (
    FACE_FRAME,
    face_axis_in_frame,
    face_centre_in_frame,
)
from src.shared.python.motion_matching.open_loop_horizon import (
    HorizonResult,
    HorizonSettings,
    ImpactDelivery,
    search_horizon,
)
from src.shared.python.motion_matching.same_input.integrator import open_loop

Array = NDArray[np.float64]
# Fraction of a bundle step accepted as round-off when snapping times to it.
_GRID_SLACK = 1e-6

__all__ = [
    "BundleReplay",
    "HeadGeometry",
    "bundle_impact_s",
    "delivery_from_poses",
    "search_bundle_horizon",
]


@dataclass(frozen=True)
class HeadGeometry:
    """Rendered face normal (unit) and face centre [m] in the head frame."""

    face_axis: Array
    face_centre: Array

    def __post_init__(self) -> None:
        axis = np.asarray(self.face_axis, dtype=float)
        centre = np.asarray(self.face_centre, dtype=float)
        if axis.shape != (3,) or not np.isfinite(axis).all():
            raise ValueError("face_axis must be a finite 3-vector")
        if not math.isclose(float(np.linalg.norm(axis)), 1.0):
            raise ValueError("face_axis must be a unit vector")
        if centre.shape != (3,) or not np.isfinite(centre).all():
            raise ValueError("face_centre must be a finite 3-vector")

    @classmethod
    def from_spec(cls, spec_bytes: bytes) -> HeadGeometry:
        """Face geometry of the spec's club in its ``Clubhead`` frame."""
        document = json.loads(spec_bytes)
        return cls(
            face_axis_in_frame(document, FACE_FRAME),
            face_centre_in_frame(document, FACE_FRAME),
        )

    def centre(self, pose: Array) -> Array:
        """World face centre of a 4x4 head pose."""
        return np.asarray(pose[:3, 3] + pose[:3, :3] @ self.face_centre)

    def normal(self, pose: Array) -> Array:
        """World face normal of a 4x4 head pose."""
        return np.asarray(pose[:3, :3] @ self.face_axis)


def delivery_from_poses(
    before: Array, at: Array, dt_s: float, head: HeadGeometry
) -> ImpactDelivery:
    """Club delivery at pose ``at``, with velocity over the step from ``before``.

    Preconditions: both poses are finite 4x4 arrays and ``dt_s`` is positive.
    """
    if not (math.isfinite(dt_s) and dt_s > 0.0):
        raise ValueError("dt_s must be positive and finite")
    poses = [np.asarray(pose, dtype=float) for pose in (before, at)]
    if any(p.shape != (4, 4) or not np.isfinite(p).all() for p in poses):
        raise ValueError("head poses must be finite 4x4 arrays")
    velocity = (head.centre(poses[1]) - head.centre(poses[0])) / dt_s
    return ImpactDelivery(
        face_angle_deg=horizontal_face_angle_deg(head.normal(poses[1])),
        path_deg=horizontal_face_angle_deg(velocity),
        head_speed_mps=float(np.linalg.norm(velocity)),
    )


def bundle_impact_s(bundle: Any) -> float:
    """Contact time on the bundle clock: the recorded ball window start, else
    the reference impact split.  ``ValueError`` when the bundle has neither."""
    impact = bundle.ball_impact()
    if impact is not None:
        return float(impact.t_start_s)
    split = bundle.provenance.get("impact_split_time_s")
    if split is None:
        raise ValueError("bundle records no impact; pass impact_s explicitly")
    return float(split)


class BundleReplay:
    """Replay callable ``start_s -> ImpactDelivery`` over one bundle.

    ``impact_s`` (default :func:`bundle_impact_s`) is snapped down to the last
    bundle step at or before it, which becomes :attr:`impact_s`.
    """

    def __init__(self, bundle: Any, plant: Any, impact_s: float | None = None):
        contact = bundle_impact_s(bundle) if impact_s is None else float(impact_s)
        if not math.isfinite(contact):
            raise ValueError("impact_s must be finite")
        step = math.floor(contact / bundle.dt_s + _GRID_SLACK)
        if not 1 <= step <= bundle.steps:
            raise ValueError(
                f"impact_s {contact} must lie after the first bundle step and "
                f"within the bundle ({bundle.steps * bundle.dt_s} s)"
            )
        self._bundle = bundle
        self._plant = plant
        self._step = step
        self._head = HeadGeometry.from_spec(bundle.spec_bytes)

    @property
    def impact_s(self) -> float:
        """Grid time of the last step at or before contact."""
        return float(self._step * self._bundle.dt_s)

    @property
    def dt_s(self) -> float:
        """Bundle step [s]."""
        return float(self._bundle.dt_s)

    def grid_steps(self, time_s: float, name: str) -> int:
        """Whole bundle steps in ``time_s``; ``ValueError`` when off the grid."""
        ratio = float(time_s) / self.dt_s
        steps = round(ratio)
        if abs(ratio - steps) > _GRID_SLACK:
            raise ValueError(f"{name} {time_s} is not on the {self.dt_s} s step grid")
        return int(steps)

    def _delivery(self, q_before: Array, q_at: Array) -> ImpactDelivery:
        poses = [self._plant.kinematic_frame(q, FACE_FRAME) for q in (q_before, q_at)]
        return delivery_from_poses(poses[0], poses[1], self.dt_s, self._head)

    def reference(self) -> ImpactDelivery:
        """Delivery of the bundle's tracked reference at :attr:`impact_s`."""
        states = self._bundle.reference_q
        return self._delivery(states[self._step - 1], states[self._step])

    def __call__(self, start_s: float) -> ImpactDelivery:
        """Open-loop delivery from the tracked state at ``start_s``.

        Preconditions: ``start_s`` is on the step grid and before
        :attr:`impact_s`.  A replay that diverges raises ``ValueError``.
        """
        start = self.grid_steps(start_s, "start_s")
        if not 0 <= start < self._step:
            raise ValueError(
                f"start_s {start_s} must be at or after 0 and before impact"
            )
        bundle = self._bundle
        rollout = open_loop(
            self._plant,
            bundle.reference_q[start],
            bundle.reference_v[start],
            bundle.replay_efforts(self._step)[start * bundle.effort_holds :],
            dt_s=self.dt_s,
            policy=bundle.step_policy(stop_on_failure=True),
        )
        if rollout.failure is not None:
            raise ValueError(f"replay from {start_s} s diverged: {rollout.failure}")
        return self._delivery(rollout.q[-2], rollout.q[-1])


def search_bundle_horizon(
    replay: BundleReplay, *, settings: HorizonSettings
) -> HorizonResult:
    """Run :func:`search_horizon` on ``replay`` from its impact to the bundle start.

    Precondition: ``settings.step_s`` is a whole number of bundle steps.
    """
    if not isinstance(replay, BundleReplay):
        raise TypeError("replay must be a BundleReplay")
    if replay.grid_steps(settings.step_s, "step_s") < 1:
        raise ValueError("step_s must span at least one bundle step")
    return search_horizon(
        replay,
        replay.reference(),
        impact_s=replay.impact_s,
        earliest_s=0.0,
        settings=settings,
    )

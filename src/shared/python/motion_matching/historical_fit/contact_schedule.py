"""Authored, capture-evidence-bound normal contact hypotheses on source PTS."""

from __future__ import annotations
from collections.abc import Mapping
from dataclasses import dataclass, replace
from fractions import Fraction
import math
import re
from typing import Any, Literal
from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.contact_law import GroundPlane


def _pts(value: tuple[int, int]) -> Fraction:
    if (
        not isinstance(value, tuple)
        or len(value) != 2
        or any(type(x) is not int for x in value)
        or value[1] <= 0
    ):
        raise ValueError(
            "Source PTS must be an integer rational with positive denominator"
        )
    return Fraction(*value)


def _hash(value: str) -> None:
    if (
        not isinstance(value, str)
        or re.fullmatch(r"sha256:[0-9a-f]{64}", value) is None
    ):
        raise ValueError("Evidence hash must be a lowercase sha256 identity")


@dataclass(frozen=True)
class ContactPinPhase:
    """Half-open authored phase; final schedule endpoint is inclusive."""

    start_pts: tuple[int, int]
    end_pts: tuple[int, int]
    pinned_spheres: tuple[str, ...]
    review_frame_sha256: tuple[str, ...]

    def __post_init__(self) -> None:
        if _pts(self.start_pts) >= _pts(self.end_pts):
            raise ValueError("Contact phase source interval must increase")
        names = self.pinned_spheres
        if (
            not isinstance(names, tuple)
            or any(not isinstance(x, str) or not x.strip() for x in names)
            or len(set(names)) != len(names)
        ):
            raise ValueError("Pinned sphere identities must be immutable and unique")
        hashes = self.review_frame_sha256
        if (
            not isinstance(hashes, tuple)
            or not hashes
            or any(not isinstance(value, str) for value in hashes)
            or len(set(hashes)) != len(hashes)
        ):
            raise ValueError(
                "Contact phases require unique immutable review frame hashes"
            )
        for value in hashes:
            _hash(value)


@dataclass(frozen=True)
class ContactPinSchedule:
    """Authored evidence reference; external capture membership must be verified."""

    capture_id: str
    capture_sha256: str
    phases: tuple[ContactPinPhase, ...]
    status: Literal["authored_contact_hypothesis"] = "authored_contact_hypothesis"

    def __post_init__(self) -> None:
        if not isinstance(self.capture_id, str) or not self.capture_id.strip():
            raise ValueError("Contact schedule requires capture identity")
        _hash(self.capture_sha256)
        if self.status != "authored_contact_hypothesis":
            raise ValueError("Contact schedule must remain an authored hypothesis")
        if (
            not isinstance(self.phases, tuple)
            or not self.phases
            or any(not isinstance(p, ContactPinPhase) for p in self.phases)
        ):
            raise ValueError("Contact phases must be nonempty immutable typed records")
        for left, right in zip(self.phases[:-1], self.phases[1:], strict=True):
            if _pts(left.end_pts) != _pts(right.start_pts):
                raise ValueError(
                    "Contact phases must cover interval without gaps or overlap"
                )
        if any(float(_pts(p.start_pts)) >= float(_pts(p.end_pts)) for p in self.phases):
            raise ValueError(
                "Contact phase boundaries must remain distinct as numerical source times"
            )

    def pins_at(self, source_time: float) -> tuple[str, ...]:
        if (
            isinstance(source_time, bool)
            or not isinstance(source_time, (int, float))
            or not math.isfinite(source_time)
        ):
            raise ValueError("Contact query requires finite source time")
        for index, phase in enumerate(self.phases):
            left, right = float(_pts(phase.start_pts)), float(_pts(phase.end_pts))
            if left <= source_time < right or (
                index == len(self.phases) - 1 and source_time == right
            ):
                return phase.pinned_spheres
        raise ValueError("Contact query lies outside reviewed source interval")

    def boundary_times(self) -> tuple[float, ...]:
        return tuple(float(_pts(p.start_pts)) for p in self.phases) + (
            float(_pts(self.phases[-1].end_pts)),
        )


@dataclass(frozen=True)
class ScheduledConstraintOptions:
    """Resolve normal pins only; base nonpenetration remains active throughout."""

    base: ConstraintOptions
    schedule: ContactPinSchedule

    def __post_init__(self) -> None:
        if not isinstance(self.base, ConstraintOptions) or not isinstance(
            self.schedule, ContactPinSchedule
        ):
            raise ValueError("Scheduled constraints require typed base and schedule")

    def resolve(self, source_time: float) -> ConstraintOptions:
        return replace(self.base, pinned_spheres=self.schedule.pins_at(source_time))


def constraint_options_from_record(
    record: Mapping[str, Any],
) -> ConstraintOptions | ScheduledConstraintOptions:
    """Decode legacy or scheduled JSON without introducing a provider dependency."""
    if not isinstance(record, Mapping):
        raise ValueError("Constraint configuration must be an object")
    try:
        if "schedule" in record:
            if set(record) != {"base", "schedule"}:
                raise ValueError(
                    "Scheduled constraint record requires only base and schedule"
                )
            base = constraint_options_from_record(record["base"])
            if not isinstance(base, ConstraintOptions):
                raise ValueError("Scheduled constraint base cannot be nested")
            data = dict(record["schedule"])
            if not isinstance(data["phases"], (tuple, list)):
                raise ValueError("Contact phases must be an array")
            phases = []
            for value in data["phases"]:
                item = dict(value)
                for name in (
                    "start_pts",
                    "end_pts",
                    "pinned_spheres",
                    "review_frame_sha256",
                ):
                    if not isinstance(item[name], (tuple, list)):
                        raise ValueError("Contact phase fields must be arrays")
                    item[name] = tuple(item[name])
                phases.append(ContactPinPhase(**item))
            data["phases"] = tuple(phases)
            return ScheduledConstraintOptions(base, ContactPinSchedule(**data))
        data = dict(record)
        if not isinstance(data.get("ground"), Mapping):
            raise ValueError(
                "Fit constraint configuration must contain a ground object"
            )
        if not isinstance(data.get("pinned_spheres", ()), (tuple, list)):
            raise ValueError("Pinned sphere names must be an array")
        data["ground"] = GroundPlane(**dict(data["ground"]))
        data["pinned_spheres"] = tuple(data.get("pinned_spheres", ()))
        return ConstraintOptions(**data)
    except (TypeError, KeyError) as exc:
        raise ValueError("Malformed contact configuration") from exc

"""A contact hypothesis must bind to exact reviewed capture evidence."""

from dataclasses import replace
from types import SimpleNamespace

import pytest

from src.shared.python.motion_matching.constraint_kinematics import ConstraintOptions
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.historical_fit.contact_schedule import (
    ContactPinPhase,
    ContactPinSchedule,
    ScheduledConstraintOptions,
)
from src.shared.python.workspace.necromatcher_contacts import contact_schedule_binding

pytestmark = pytest.mark.unit


def _options():
    return ConstraintOptions(GroundPlane((0, 0, 1), 0), 1, 1, 1, 1, 1, 1)


class _Review:
    capture_id = "capture"
    frame_count = 3

    def __init__(self):
        self.frames = [
            {
                "asset_id": "video",
                "shot_id": "shot",
                "camera_id": "camera",
                "swing_id": "swing",
                "pts_ticks": i + 10,
                "timebase_numerator": 1,
                "timebase_denominator": 3,
                "frame_sha256": str(i) * 64,
            }
            for i in range(self.frame_count)
        ]

    def frame(self, index):
        return {"frame": self.frames[index]}


def _inputs():
    review = _Review()
    source = {
        "capture_id": "capture",
        "capture_hash": "sha256:" + "a" * 64,
        "frames": [review.frames[0], review.frames[-1]],
    }
    phases = (
        ContactPinPhase(
            (10, 3),
            (11, 3),
            ("heel_r",),
            ("sha256:" + review.frames[0]["frame_sha256"],),
        ),
        ContactPinPhase(
            (11, 3), (4, 1), (), ("sha256:" + review.frames[2]["frame_sha256"],)
        ),
    )
    schedule = ContactPinSchedule("capture", source["capture_hash"], phases)
    config = SimpleNamespace(
        constraint_options=ScheduledConstraintOptions(_options(), schedule)
    )
    return config, source, review


def test_legacy_constraints_need_no_contact_schedule_binding():
    assert (
        contact_schedule_binding(SimpleNamespace(constraint_options=None), {}, None)
        is None
    )
    assert (
        contact_schedule_binding(
            SimpleNamespace(constraint_options=_options()), {}, None
        )
        is None
    )


def test_schedule_records_exact_bound_review_and_boundary_frames():
    config, source, review = _inputs()
    receipt = contact_schedule_binding(config, source, review)
    assert receipt["capture_id"] == source["capture_id"]
    assert receipt["capture_hash"] == source["capture_hash"]
    assert receipt["status"] == "authored_contact_hypothesis"
    assert receipt["phases"][0]["boundary_frame_indices"] == [0, 1]
    assert receipt["phases"][1]["review_frame_indices"] == [2]
    assert receipt["continuous_certified"] is False


@pytest.mark.parametrize(
    "field,value", [("capture_id", "other"), ("capture_sha256", "sha256:" + "b" * 64)]
)
def test_schedule_rejects_other_capture_version(field, value):
    config, source, review = _inputs()
    config.constraint_options = replace(
        config.constraint_options,
        schedule=replace(config.constraint_options.schedule, **{field: value}),
    )
    with pytest.raises(ValueError, match="capture"):
        contact_schedule_binding(config, source, review)


def test_schedule_rejects_hash_from_a_different_phase():
    config, source, review = _inputs()
    phases = config.constraint_options.schedule.phases
    bad = replace(
        phases[0], review_frame_sha256=("sha256:" + review.frames[2]["frame_sha256"],)
    )
    config.constraint_options = replace(
        config.constraint_options,
        schedule=replace(config.constraint_options.schedule, phases=(bad, phases[1])),
    )
    with pytest.raises(ValueError, match="review"):
        contact_schedule_binding(config, source, review)


def test_schedule_rejects_boundary_outside_exact_source_clock():
    config, source, review = _inputs()
    phases = config.constraint_options.schedule.phases
    split = (7, 2)
    config.constraint_options = replace(
        config.constraint_options,
        schedule=replace(
            config.constraint_options.schedule,
            phases=(
                replace(phases[0], end_pts=split),
                replace(phases[1], start_pts=split),
            ),
        ),
    )
    with pytest.raises(ValueError, match="boundary"):
        contact_schedule_binding(config, source, review)


def test_schedule_rejects_review_from_another_camera():
    config, source, review = _inputs()
    review.frames[1] = {**review.frames[1], "camera_id": "other"}
    with pytest.raises(ValueError, match="source"):
        contact_schedule_binding(config, source, review)


def test_schedule_rejects_ambiguous_duplicate_review_image():
    config, source, review = _inputs()
    review.frames[1] = {
        **review.frames[1],
        "frame_sha256": review.frames[0]["frame_sha256"],
    }
    with pytest.raises(ValueError, match="one frame"):
        contact_schedule_binding(config, source, review)

"""Adverse timing and sampling fixtures for reference overlays (#9881)."""

from uuid import uuid4

import numpy as np
import pytest

from src.motion_capture.reference.model import ReferenceMotion
from src.motion_capture.reference.registration import (
    EventAnchors,
    ReferenceRegistration,
    TimeMapping,
    sample_reference_motion,
)

pytestmark = pytest.mark.unit


@pytest.mark.parametrize(
    "reference,scene",
    [
        ({"a": 0, "b": 0}, {"a": 0, "b": 1}),
        ({"a": 0, "b": 1}, {"a": 1, "b": 0}),
        ({"a": 0, "b": 1}, {"a": 1, "b": 1}),
        ({"a": 0, "b": 1}, {"a": 0, "b": 10}),
        ({"a": 0, "b": 1}, {"a": 0}),
    ],
)
def test_invalid_event_pairs_are_rejected(reference, scene) -> None:
    with pytest.raises(ValueError):
        EventAnchors(reference=reference, scene=scene)


def test_one_anchor_aligns_event_and_preserves_rate() -> None:
    mapping = TimeMapping(
        rate_scale=1.5,
        event_anchors=EventAnchors(
            reference={"impact": 2},
            scene={"impact": 4},
        ),
    )
    assert mapping.reference_to_scene(2.0) == 4.0
    assert mapping.reference_to_scene(3.0) == 5.5
    np.testing.assert_allclose(mapping.scene_to_reference(np.array([4.0, 5.5])), [2, 3])


def test_anchor_maps_cannot_mutate_after_validation() -> None:
    anchors = EventAnchors(reference={"impact": 2}, scene={"impact": 4})
    with pytest.raises(TypeError):
        anchors.scene["impact"] = -100
    assert EventAnchors.model_validate_json(anchors.model_dump_json()) == anchors


def motion(times: tuple[float, ...]) -> ReferenceMotion:
    return ReferenceMotion.model_validate(
        {
            "id": str(uuid4()),
            "title": "Timing fixture",
            "source": {
                "path": "fixture.json",
                "sha256": "0" * 64,
                "format": "body_target_json_v1",
            },
            "source_units": "m",
            "source_axes": ("+X", "+Y", "+Z"),
            "source_names": ("joint",),
            "joint_names": ("joint",),
            "time_s": times,
            "points_m": tuple(((float(i), 0, 0),) for i in range(len(times))),
        }
    )


def test_large_gaps_stay_masked_but_exact_origin_remains_valid() -> None:
    asset = motion((0.0, 2.0))
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="fixture")
    points, valid = sample_reference_motion(asset, reg, np.array([0.0, 1.0, 2.0]))
    assert valid[:, 0].tolist() == [True, False, True]
    np.testing.assert_equal(points[0, 0], [0, 0, 0])


def test_large_absolute_clock_does_not_snap_to_wrong_frame() -> None:
    asset = motion((100_000.0, 100_000.1))
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="fixture")
    points, valid = sample_reference_motion(asset, reg, np.array([100_000.05]))
    assert valid[0, 0]
    assert points[0, 0, 0] == pytest.approx(0.5)


def test_sampling_never_transforms_the_full_source(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.motion_capture.reference import registration as module

    asset = motion(tuple(i / 100 for i in range(10_000)))
    reg = ReferenceRegistration(reference_id=asset.id, calibration_id="fixture")

    def forbidden(*args):
        raise AssertionError("Sampling transformed the entire source")

    monkeypatch.setattr(module, "transform_reference_motion", forbidden)
    points, valid = sample_reference_motion(asset, reg, np.array([0.005]))
    assert valid[0, 0]
    assert points[0, 0, 0] == pytest.approx(0.5)


@pytest.mark.parametrize("rate", [0.1, 10.0])
def test_extreme_playback_rates_are_rejected(rate: float) -> None:
    with pytest.raises(ValueError):
        TimeMapping(rate_scale=rate)


def test_piecewise_mapping_roundtrip_including_extrapolation() -> None:
    mapping = TimeMapping(
        event_anchors=EventAnchors(
            reference={"address": 0, "top": 1, "impact": 2},
            scene={"address": 3, "top": 3.5, "impact": 5.5},
        )
    )
    values = np.array([-1, 0, 0.25, 1, 1.5, 2, 3])
    np.testing.assert_allclose(
        mapping.scene_to_reference(mapping.reference_to_scene(values)), values
    )

"""Display geometry remains reversible and separate from measured/fitted data."""

import numpy as np
import pytest

from src.motion_capture.reference.comparison import ComparisonLayer
from src.motion_capture.reference.registration import (
    ReferenceRegistration,
    ReferenceTransform,
    sample_reference_motion,
    transform_reference_motion,
)
from tests.motion_capture.test_reference_registration import sample_motion

pytestmark = pytest.mark.unit


def test_old_asset_fingerprint_survives_default_appearance_fields() -> None:
    from src.motion_capture.reference.evidence import asset_identity, fingerprint

    motion = sample_motion()
    old = motion.model_dump(
        mode="json", exclude={"title", "notes", "archived", "club_edges"}
    )
    assert asset_identity(motion) == fingerprint(old)


def test_mirror_precedes_placement_and_preserves_source_and_gaps() -> None:
    motion = sample_motion()
    motion = motion.changed(
        points_m=tuple(
            tuple(None if p is None else (p[0], p[1] + 0.3, p[2]) for p in row)
            for row in motion.points_m
        )
    )
    original = motion.model_dump_json()
    transform = ReferenceTransform(scale=2, translation_m=(1, 2, 3))
    reg = ReferenceRegistration(
        reference_id=motion.id,
        calibration_id="test",
        mirror_lateral=True,
        transform=transform,
    )
    times, world, valid = transform_reference_motion(motion, reg)
    assert world[0, 0] == pytest.approx((1, 4, 3.6))
    sampled, mask = sample_reference_motion(motion, reg, times)
    np.testing.assert_allclose(sampled, world)
    np.testing.assert_array_equal(mask, valid)
    assert not mask[2, 1]
    assert motion.model_dump_json() == original
    restored = ReferenceRegistration.model_validate(
        reg.model_dump() | {"mirror_lateral": False}
    )
    _, unmirrored, _ = transform_reference_motion(motion, restored)
    assert unmirrored[0, 0] == pytest.approx((1, 4, 2.4))


def test_appearance_round_trip_and_invalid_dimensions() -> None:
    layer = ComparisonLayer(
        draw_club=False,
        draw_ellipsoids=True,
        ellipsoid_opacity=0.35,
        segment_radius_ratio=0.12,
    )
    assert ComparisonLayer.model_validate_json(layer.model_dump_json()) == layer
    for values in (
        {"ellipsoid_opacity": 1.1},
        {"segment_radius_ratio": 0},
        {"segment_radius_ratio": float("nan")},
    ):
        with pytest.raises(ValueError):
            ComparisonLayer(**values)


def test_club_edge_contract_rejects_non_edges() -> None:
    motion = sample_motion()
    with pytest.raises(ValueError, match="Club edges"):
        motion.changed(club_edges=((0, 2),))
    assert motion.changed(club_edges=((1, 2),)).club_edges == ((1, 2),)

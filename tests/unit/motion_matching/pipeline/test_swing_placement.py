"""Whole-swing marker placement as a stage of the matching pipeline (#15274).

``swing_placement`` reads each segment's pose from the pipeline's own marker
kinematics, re-places the markers with
``swing_marker_placement.place_markers_over_swing`` and returns new
attachments with a report for the receipt.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

pytest.importorskip("mujoco")

from src.shared.python.motion_matching.pipeline.plant import get_plant  # noqa: E402
from src.shared.python.motion_matching.pipeline.swing_placement import (  # noqa: E402
    add_placement_arguments,
    limit_total_moves,
    placement_settings,
    replaced_attachments,
    segment_poses,
)
from src.shared.python.motion_matching.swing_marker_placement import (  # noqa: E402
    PlacementSettings,
)

pytestmark = [pytest.mark.unit, pytest.mark.timeout(300)]
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver_v2.json"
FRAMES = 24


@pytest.fixture(scope="module")
def spec() -> dict[str, Any]:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def truth(spec: dict[str, Any]) -> dict[str, tuple[str, tuple[float, float, float]]]:
    return {
        label: (entry["body"], tuple(entry["offset_m"]))
        for label, entry in spec["marker_attachments"].items()
        if entry["offset_m"] is not None
    }


@pytest.fixture(scope="module")
def plant(spec: dict[str, Any]) -> Any:
    return get_plant("mujoco", spec)


@pytest.fixture(scope="module")
def swing(plant: Any, truth: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Poses spread about zero and the marker positions the truth gives."""
    kin = plant.create_ik(truth, ik_backend="lm")
    rng = np.random.default_rng(15274)
    q = rng.uniform(-0.3, 0.3, size=(FRAMES, len(kin.coordinate_order)))
    points = np.stack([kin.marker_positions(row) for row in q])
    return q, points


def _shifted(truth: dict[str, Any], by_m: float) -> dict[str, Any]:
    rng = np.random.default_rng(7)
    out = {}
    for label, (body, offset) in truth.items():
        step = rng.normal(size=3)
        out[label] = (
            body,
            tuple(np.asarray(offset) + by_m * step / np.linalg.norm(step)),
        )
    return out


def test_segment_poses_carry_a_marker_to_where_the_model_has_it(
    plant, truth, swing
) -> None:
    q, points = swing
    bodies = sorted({body for body, _ in truth.values()})
    rotations, origins = segment_poses(plant.create_ik, bodies, q)
    for body in bodies:
        turn = rotations[body]
        assert turn.shape == (FRAMES, 3, 3)
        np.testing.assert_allclose(
            np.einsum("fij,fkj->fik", turn, turn),
            np.broadcast_to(np.eye(3), turn.shape),
            atol=1e-9,
        )
    for column, (body, offset) in enumerate(truth.values()):
        world = origins[body] + np.einsum("fij,j->fi", rotations[body], offset)
        np.testing.assert_allclose(world, points[:, column], atol=1e-9)


def test_misplaced_markers_return_to_their_segments(plant, truth, swing) -> None:
    q, points = swing
    start = _shifted(truth, 0.02)
    before = json.dumps(start, sort_keys=True)
    placed, report = replaced_attachments(
        plant.create_ik,
        points,
        np.ones(points.shape[:2], dtype=bool),
        tuple(truth),
        start,
        q,
        PlacementSettings(),
        keep_bodies=(),
    )
    assert json.dumps(start, sort_keys=True) == before
    for label, (body, offset) in truth.items():
        assert placed[label][0] == body
        np.testing.assert_allclose(placed[label][1], offset, atol=1e-8)
    assert report["rms_before_m"] == pytest.approx(0.02, rel=1e-6)
    assert report["rms_after_m"] < 1e-8
    assert report["moved"] == len(truth)
    assert report["markers"]["BackTop"]["moved_m"] == pytest.approx(0.02, rel=1e-6)
    json.dumps(report)


def test_markers_on_kept_segments_stay(plant, truth, swing) -> None:
    q, points = swing
    start = _shifted(truth, 0.02)
    placed, report = replaced_attachments(
        plant.create_ik,
        points,
        np.ones(points.shape[:2], dtype=bool),
        tuple(truth),
        start,
        q,
        PlacementSettings(),
        keep_bodies=("Clubhead",),
    )
    club = [label for label, (body, _) in truth.items() if body == "Clubhead"]
    assert club
    for label in club:
        assert placed[label] == start[label]
        assert label not in report["markers"]
    assert report["moved"] == len(truth) - len(club)
    assert report["kept_bodies"] == ["Clubhead"]


def test_bad_poses_are_refused(plant, truth, swing) -> None:
    q, points = swing
    with pytest.raises(ValueError, match="frames"):
        replaced_attachments(
            plant.create_ik,
            points,
            np.ones(points.shape[:2], dtype=bool),
            tuple(truth),
            truth,
            q[:-1],
            PlacementSettings(),
        )


def test_the_options_are_off_by_default_and_checked() -> None:
    parser = argparse.ArgumentParser()
    add_placement_arguments(parser)
    args = parser.parse_args([])
    assert args.placement_passes == 0
    assert placement_settings(args).max_move_m == pytest.approx(0.05)
    args = parser.parse_args(
        ["--placement-passes", "3", "--placement-max-move-m", "0.03"]
    )
    assert args.placement_passes == 3
    assert placement_settings(args).max_move_m == pytest.approx(0.03)
    assert args.placement_keep_bodies == ["Clubhead"]
    assert args.placement_max_total_move_m == pytest.approx(0.05)
    with pytest.raises(SystemExit):
        parser.parse_args(["--placement-max-total-move-m", "-0.01"])
    with pytest.raises(SystemExit):
        parser.parse_args(["--placement-passes", "-1"])


def test_a_marker_never_ends_further_from_its_first_placement_than_the_total_limit(
    truth,
) -> None:
    first = dict(truth)
    drifted = _shifted(truth, 0.08)
    near = _shifted(truth, 0.01)
    before = json.dumps(drifted, sort_keys=True)
    held, report = limit_total_moves(drifted, first, 0.05)
    assert json.dumps(drifted, sort_keys=True) == before
    for label, (body, offset) in first.items():
        assert held[label][0] == body
        moved = np.linalg.norm(np.subtract(held[label][1], offset))
        assert moved == pytest.approx(0.05, rel=1e-9)
        # Cut along its own direction, not moved somewhere else.
        along = np.subtract(drifted[label][1], offset) / 0.08
        np.testing.assert_allclose(
            np.subtract(held[label][1], offset) / 0.05, along, atol=1e-9
        )
        assert report["markers"][label] == pytest.approx(0.05, rel=1e-9)
    assert report["limited"] == sorted(first)
    assert report["max_total_move_m"] == pytest.approx(0.05)
    kept, report = limit_total_moves(near, first, 0.05)
    assert kept == {k: (b, tuple(float(x) for x in o)) for k, (b, o) in near.items()}
    assert report["limited"] == []
    free, report = limit_total_moves(drifted, first, None)
    assert report["limited"] == []
    assert report["markers"]["BackTop"] == pytest.approx(0.08, rel=1e-9)
    json.dumps(report)


def test_total_limit_refuses_a_marker_that_changed_segment(truth) -> None:
    label = next(iter(truth))
    moved = {**truth, label: ("NotThatSegment", truth[label][1])}
    with pytest.raises(ValueError, match="segment"):
        limit_total_moves(moved, truth, 0.05)
    with pytest.raises(ValueError, match="non-negative"):
        limit_total_moves(truth, truth, -1.0)

"""OSV-10 (#11759): the face-orientation residual reaches every shared IK solve."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching.pipeline.constants import (
    LEG_SEEDS,
    capture_path,
)
from src.shared.python.motion_matching.pipeline.lane import Lane, configure_lane
from src.shared.python.motion_matching.pipeline.reference import consistency_resolve

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    return json.loads(SPEC.read_text(encoding="utf-8"))


@pytest.fixture(scope="module")
def attachments(spec: dict) -> dict:
    return {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }


@pytest.fixture()
def lane(spec: dict, attachments: dict) -> Lane:
    out = Lane(tuple({**attachments, **LEG_SEEDS}), capture_path("driver"))
    configure_lane(out, spec)
    return out


def test_face_targets_cover_every_observed_capture_frame(
    lane: Lane, spec: dict, attachments: dict
) -> None:
    lane.set_face_targets(attachments, spec, 2.0)
    assert lane.face_targets is not None and len(lane.face_targets) == lane.frames
    cols = [lane.labels.index(m) for m in cft.HEAD_TRIAD_LABELS]
    observed = lane.valid[:, cols].all(axis=1)
    targeted = np.array([t is not None for t in lane.face_targets])
    np.testing.assert_array_equal(targeted, observed)
    first = next(t for t in lane.face_targets if t is not None)
    assert first[cft.FACE_FRAME][2] == 2.0
    assert lane.face_weight == 2.0
    assert lane.face_unavailable_reason is None


def test_mismatched_head_triad_leaves_face_and_impact_unavailable() -> None:
    """#12030: the driver triad offsets on the 7-iron capture fail closed."""
    iron_spec = json.loads(
        (
            ROOT / "docs/development/full_body_models/full_body_spec_anthro_iron7.json"
        ).read_text(encoding="utf-8")
    )
    iron_att = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in iron_spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    lane = Lane(tuple({**iron_att, **LEG_SEEDS}), capture_path("iron"))
    configure_lane(lane, iron_spec)
    lane.set_face_targets(iron_att, iron_spec, 3.0)
    assert lane.face_targets is None
    assert lane.face_weight == 3.0
    assert "do not match" in (lane.face_unavailable_reason or "")
    lane.set_impact_split(iron_att, iron_spec)
    assert lane.impact_index is None
    assert lane.impact_split_reason.startswith("unavailable")
    with pytest.raises(ValueError, match="face weight"):
        lane.set_face_targets(iron_att, iron_spec, -1.0)


def test_attach_face_report_records_the_unavailable_reason() -> None:
    from types import SimpleNamespace

    from src.shared.python.motion_matching.pipeline.cli import _attach_face_report

    lane = SimpleNamespace(
        impact_split_report=dict,
        face_targets=None,
        face_unavailable_reason="triad mismatch",
    )
    report: dict = {}
    _attach_face_report(report, lane, None, (np.zeros(1), np.zeros(1)), 3.0)
    block = report["face_orientation"]
    assert block["available"] is False and block["reason"] == "triad mismatch"
    assert block["weight"] == 3.0 and block["targeted_frames"] == 0


def test_zero_weight_keeps_the_marker_only_fit(
    lane: Lane, spec: dict, attachments: dict
) -> None:
    lane.set_face_targets(attachments, spec, 0.0)
    assert lane.face_targets is None
    with pytest.raises(ValueError, match="face weight"):
        lane.set_face_targets(attachments, spec, -0.5)


def test_trajectory_merges_face_and_elbow_pit_targets(
    lane: Lane, spec: dict, attachments: dict
) -> None:
    lane.set_face_targets(attachments, spec, 1.0)
    kin = MagicMock()
    kin.solve_trajectory.return_value = (np.zeros((1, 3)), [])
    lane.trajectory(kin, np.zeros(3))
    merged = kin.solve_trajectory.call_args.kwargs["axis_targets_per_frame"]
    assert len(merged) == lane.frames
    names = set().union(*(set(m) for m in merged if m))
    assert cft.FACE_FRAME in names
    assert lane.anthropometric and names & {"LS", "RS"}


def test_consistency_resolve_keeps_the_face_targets() -> None:
    lane = MagicMock()
    lane.frames = 2
    lane.face_targets = [None, {"Clubhead": ((1, 0, 0), (0, 1, 0), 1.0)}]
    lane.gaze_axis_targets_cache = None
    kin = MagicMock()
    kin.solve_trajectory.return_value = (np.zeros((2, 4)), [])
    consistency_resolve(lane, kin, np.zeros((2, 4)))
    kwargs = kin.solve_trajectory.call_args.kwargs
    assert kwargs["axis_targets_per_frame"] == lane.face_targets


def test_consistency_resolve_unions_the_gaze_and_face_targets() -> None:
    head = {"Head": ((0, 0, 1), (0, 0, 1), 1.0)}
    lane = MagicMock()
    lane.frames = 2
    lane.face_targets = [None, {"Clubhead": ((1, 0, 0), (0, 1, 0), 1.0)}]
    lane.gaze_axis_targets_cache = [head, head]
    kin = MagicMock()
    kin.solve_trajectory.return_value = (np.zeros((2, 4)), [])
    consistency_resolve(lane, kin, np.zeros((2, 4)))
    kwargs = kin.solve_trajectory.call_args.kwargs
    assert kwargs["axis_targets_per_frame"] == [head, {**head, **lane.face_targets[1]}]


def test_cli_exposes_the_face_weight_with_the_shared_default() -> None:
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    assert build_parser().parse_args([]).face_weight == cft.FACE_ORIENTATION_WEIGHT
    args = build_parser().parse_args(["--face-weight", "0"])
    assert args.face_weight == 0.0


def test_shooting_refit_keeps_the_face_targets(monkeypatch: pytest.MonkeyPatch) -> None:
    import logging
    from types import SimpleNamespace

    from src.shared.python.motion_matching import full_body_forward_dynamics
    from src.shared.python.motion_matching.pipeline import dynamics
    from src.shared.python.motion_matching.pipeline.constants import SHOOTING_LOCKED

    frames, n_q = 600, 8
    q = np.zeros((frames, n_q))
    lane = SimpleNamespace(
        points=np.zeros((frames, 2, 3)),
        valid=np.ones((frames, 2), dtype=bool),
        times=np.arange(frames) / 360.0,
        rate_hz=360.0,
        ground=None,
        labels=("WaistLeft", "Other"),
        stance=[()] * frames,
        bounds={},
        face_targets=[None] * frames,
    )
    kin = MagicMock()
    kin.coordinate_order = (*SHOOTING_LOCKED, *(f"q{i}" for i in range(n_q)))[:n_q]
    kin.solve_trajectory.return_value = (q, [])
    record = SimpleNamespace(time_s=lane.times, weight_fraction=np.full(frames, 0.9))
    monkeypatch.setattr(dynamics, "replay", lambda *a, **k: (record, q))
    monkeypatch.setattr(
        dynamics, "marker_errors", lambda *a: np.full((frames, 2), 0.01)
    )
    zmp = {"outside_m": np.zeros(frames), "unloaded": np.zeros(frames, dtype=bool)}
    monkeypatch.setattr(
        full_body_forward_dynamics, "reference_zmp", lambda *a, **k: zmp
    )
    dynamics.shooting_fit(
        lane,
        kin,
        object(),
        q,
        q,
        logging.getLogger("test"),
        dynamics.ShootingFitConfig(iterations=1, gain=0.5, tracking_backend="kkt"),
    )
    kwargs = kin.solve_trajectory.call_args.kwargs
    assert kwargs["axis_targets_per_frame"] is lane.face_targets

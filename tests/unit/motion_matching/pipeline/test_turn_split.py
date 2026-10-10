"""#12042 slice 7: the thorax / shoulder-girdle turn split in the trajectory IK."""

from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.shared.python.motion_matching.pipeline import turn_split as ts
from src.shared.python.motion_matching.pipeline.reference import consistency_resolve

pytestmark = pytest.mark.unit
ROOT = Path(__file__).resolve().parents[4]
RUN = ROOT / "docs/development/full_body_models/evidence/ground_support"
DRIVER = RUN / "anthro_driver_seeds"

LABELS = ("BackLeft", "BackRight", "LShoulderBack", "RShoulderBack", "WaistLeft")
ATTACH = {
    "BackLeft": ("Spine", (-0.1, 0.09, 0.3)),
    "BackRight": ("Spine", (-0.1, -0.09, 0.3)),
}


def _points(frames: int = 3) -> tuple[np.ndarray, np.ndarray]:
    pts = np.zeros((frames, len(LABELS), 3))
    for f in range(frames):
        angle = np.radians(30.0 * f)
        pts[f, 0] = (0.09 * np.sin(angle), 0.09 * np.cos(angle), 1.3)
        pts[f, 1] = -pts[f, 0] + (0.0, 0.0, 2.6)
    return pts, np.ones((frames, len(LABELS)), dtype=bool)


def test_body_axis_is_the_unit_attachment_difference() -> None:
    np.testing.assert_allclose(ts.thorax_body_axis(ATTACH), (0.0, 1.0, 0.0))


@pytest.mark.parametrize(
    ("attachments", "match"),
    [
        ({"BackLeft": ATTACH["BackLeft"]}, "missing"),
        ({**ATTACH, "BackRight": ("Hip", (0, -0.09, 0.3))}, "thorax frame"),
        ({**ATTACH, "BackRight": ATTACH["BackLeft"]}, "coincide"),
    ],
)
def test_body_axis_rejects_bad_attachments(attachments: dict, match: str) -> None:
    with pytest.raises(ValueError, match=match):
        ts.thorax_body_axis(attachments)


def test_axis_targets_follow_the_capture_line_and_skip_gaps() -> None:
    pts, valid = _points()
    valid[1, 1] = False
    targets = ts.thorax_axis_targets(pts, valid, LABELS, ATTACH, 0.3)
    assert targets is not None and len(targets) == 3 and targets[1] is None
    body, world, weight = targets[2][ts.THORAX_FRAME]
    assert weight == 0.3
    np.testing.assert_allclose(body, (0.0, 1.0, 0.0))
    expected = pts[2, 0] - pts[2, 1]
    np.testing.assert_allclose(world, expected / np.linalg.norm(expected))


def test_axis_targets_off_at_zero_weight_and_validate_inputs() -> None:
    pts, valid = _points()
    assert ts.thorax_axis_targets(pts, valid, LABELS, ATTACH, 0.0) is None
    with pytest.raises(ValueError, match=">= 0"):
        ts.thorax_axis_targets(pts, valid, LABELS, ATTACH, -1.0)
    with pytest.raises(TypeError):
        ts.thorax_axis_targets(pts, valid, LABELS, ATTACH, "0.3")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="labels"):
        ts.thorax_axis_targets(pts, valid, LABELS[:2], ATTACH, 0.3)


def test_degenerate_capture_line_gives_no_target() -> None:
    pts, valid = _points(1)
    pts[0, 1] = pts[0, 0]
    assert ts.thorax_axis_targets(pts, valid, LABELS, ATTACH, 0.3) == [None]


def test_shoulder_girdle_weights() -> None:
    assert ts.shoulder_girdle_weights(LABELS, 5.0) == {
        "LShoulderBack": 5.0,
        "RShoulderBack": 5.0,
    }
    assert ts.shoulder_girdle_weights(LABELS, 1.0) == {}
    assert ts.shoulder_girdle_weights(("BackLeft",), 5.0) == {}
    with pytest.raises(ValueError):
        ts.shoulder_girdle_weights(LABELS, -2.0)


def test_lane_helpers_ignore_stand_in_attributes() -> None:
    assert ts.lane_axis_targets(MagicMock()) is None
    assert ts.lane_split_weights(MagicMock()) is None
    face = [None, {"Clubhead": ((1, 0, 0), (0, 1, 0), 1.0)}]
    thorax = [{"Spine": ((0, 1, 0), (0, 1, 0), 0.3)}, None]
    lane = SimpleNamespace(
        face_targets=face, thorax_targets=thorax, split_marker_weights={"A": 5.0}
    )
    assert ts.lane_axis_targets(lane) == [thorax[0], face[1]]
    face_only = SimpleNamespace(face_targets=face, thorax_targets=None)
    assert ts.lane_axis_targets(face_only) is face
    assert ts.lane_split_weights(lane) == {"A": 5.0}


def test_consistency_resolve_keeps_the_split() -> None:
    lane = MagicMock()
    lane.frames = 2
    lane.face_targets = None
    lane.gaze_axis_targets_cache = None
    lane.thorax_targets = [{"Spine": ((0, 1, 0), (0, 1, 0), 0.3)}, None]
    lane.split_marker_weights = {"LShoulderBack": 5.0}
    kin = MagicMock()
    kin.solve_trajectory.return_value = (np.zeros((2, 4)), [])
    consistency_resolve(lane, kin, np.zeros((2, 4)))
    kwargs = kin.solve_trajectory.call_args.kwargs
    assert kwargs["axis_targets_per_frame"] == lane.thorax_targets
    assert kwargs["marker_weights"] == {"LShoulderBack": 5.0}


def test_cli_split_is_opt_in_and_reported_only_when_active() -> None:
    """The IK-passing split regresses forward dynamics on both captures, so
    the pipeline default stays marker-only and the receipts are unchanged."""
    from src.shared.python.motion_matching.pipeline.cli import build_parser

    args = build_parser().parse_args([])
    assert (args.thorax_weight, args.shoulder_girdle_weight) == (0.0, 1.0)
    assert ts.DEFAULT_THORAX_WEIGHT == 0.0
    assert ts.DEFAULT_SHOULDER_GIRDLE_WEIGHT == 1.0
    opted = build_parser().parse_args(
        ["--thorax-weight", "0.3", "--shoulder-girdle-weight", "5"]
    )
    assert (opted.thorax_weight, opted.shoulder_girdle_weight) == (
        ts.THORAX_AXIS_WEIGHT,
        ts.SHOULDER_GIRDLE_MARKER_WEIGHT,
    )
    with pytest.raises(SystemExit):
        build_parser().parse_args(["--thorax-weight", "-1"])
    off = SimpleNamespace(thorax_targets=None, split_marker_weights={})
    assert not ts.turn_split_active(off)
    lane = SimpleNamespace(
        thorax_targets=[None, {"Spine": 1}],
        thorax_weight=0.3,
        shoulder_girdle_weight=5.0,
        split_marker_weights={"LShoulderBack": 5.0},
    )
    assert ts.turn_split_active(lane)
    report = ts.turn_split_report(lane)
    assert report["thorax_targeted_frames"] == 1
    assert report["shoulder_girdle_weight"] == 5.0


# --- Synthetic IK: the split keeps the thorax on its measured line ----------


def _yaw(points: np.ndarray, labels: list[str], pair: tuple[str, str]) -> float:
    v = points[labels.index(pair[0])] - points[labels.index(pair[1])]
    return float(np.degrees(np.arctan2(v[1], v[0])))


def test_split_ik_follows_a_known_thorax_scapula_partition() -> None:
    """Markers from a pose with a known partition (thorax -50 deg, scapulae
    -10 deg each), then the arm and club markers dragged 12 deg further about
    the vertical, the way the arms pulled the thorax at the top. Marker-only
    IK puts part of the drag into the thorax; the split keeps the thorax on
    its BackLeft/BackRight line and moves the shoulder line closer."""
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.pipeline.lane import document_seed
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    spec = json.loads((DRIVER / "full_body_spec_hipcal_scaled.json").read_text())
    receipt = json.loads((DRIVER / "receipt.json").read_text())
    attachments = {
        k: (v["body"], tuple(v["offset_m"]))
        for k, v in receipt["ik"]["attachments_m"].items()
    }
    plant = get_plant("mujoco", spec)
    kin = plant.create_ik(attachments)
    labels, names = list(kin.labels), list(kin.coordinate_order)
    truth = document_seed(spec, kin)
    for name, delta in (
        ("TorsoInput", -50),
        ("LScapInputY", -10),
        ("RScapInputY", -10),
    ):
        truth[names.index(name)] += np.radians(delta)
    kin._set(truth)
    exact = kin._positions().copy()
    mid = 0.5 * (
        exact[labels.index("LShoulderBack")] + exact[labels.index("RShoulderBack")]
    )
    c, s = np.cos(np.radians(-12.0)), np.sin(np.radians(-12.0))
    drag = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])
    biased = exact.copy()
    dragged = ("LU", "RU", "LE", "RE", "LW", "RW", "Marker", "LShoulderTop")
    for i, label in enumerate(labels):
        if label.startswith(dragged) or label == "RShoulderTop":
            biased[i] = mid + drag @ (exact[i] - mid)
    line = biased[labels.index("BackLeft")] - biased[labels.index("BackRight")]
    split = {
        "marker_weights": ts.shoulder_girdle_weights(labels),
        "axis_targets": {
            ts.THORAX_FRAME: (
                ts.thorax_body_axis(attachments),
                tuple(line / np.linalg.norm(line)),
                ts.THORAX_AXIS_WEIGHT,
            )
        },
    }
    errors = {}
    for name, extra in (("plain", {}), ("split", split)):
        fit = kin.solve_pose(
            biased,
            np.ones(len(labels), dtype=bool),
            truth,
            ground=plant.ground_plane,
            closure_weight=0.0,
            ground_weight=0.0,
            **extra,
        )
        kin._set(fit.q)
        model = kin._positions()
        errors[name] = {
            key: _yaw(model, labels, pair) - _yaw(exact, labels, pair)
            for key, pair in (
                ("trunk", ts.THORAX_LINE),
                ("girdle", ts.SHOULDER_GIRDLE_MARKERS),
            )
        }
    assert abs(errors["plain"]["trunk"]) > 3.0
    assert abs(errors["split"]["trunk"]) < 1.5
    assert abs(errors["split"]["girdle"]) < abs(errors["plain"]["girdle"])

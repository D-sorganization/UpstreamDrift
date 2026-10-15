"""R5 (#12682): the overlay impact detector measures at the face centre.

The exporter's impact time is the shared ``ball_passage`` of the face centre
carried by the ``Clubhead`` frame (``club_face_target.face_centre_path``), the
point the receipt and the fixture speed timing use, never the frame origin.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.model_appearance.club_face import ball_passage
from src.shared.python.motion_matching import club_face_target as cft

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[4]
SPEC_DIR = ROOT / "docs/development/full_body_models"
FIXTURES = ROOT / "tests/fixtures/club_face"
FIXTURE_DT_S = 0.002  # every second sample of the 1 kHz reference
CLUBS = ("driver", "iron7")


def _spec_bytes(club: str) -> bytes:
    path = SPEC_DIR / f"full_body_spec_anthro_{club}.json"
    if not path.exists():
        pytest.skip(f"{club} specification not present")
    return path.read_bytes()


def _closing_swing(t: np.ndarray) -> np.ndarray:
    """World 4x4 ``Clubhead`` poses: the arm swings about +x through the ball
    at t = 1.4 while the head rolls about the shaft (the face closes)."""
    theta = np.where(
        t < 1.0,
        np.pi / 2.0 * (1.0 - np.cos(np.pi * t)),
        np.pi * (1.0 - ((t - 1.0) / 0.4) ** 2),
    )
    roll = 6.0 * (t - 1.4)  # rad: about 340 deg/s of face closure
    poses = np.repeat(np.eye(4)[None], len(t), axis=0)
    for k, (a, b) in enumerate(zip(theta, roll, strict=True)):
        ca, sa, cb, sb = np.cos(a), np.sin(a), np.cos(b), np.sin(b)
        swing = np.array([[1.0, 0.0, 0.0], [0.0, ca, -sa], [0.0, sa, ca]])
        about_shaft = np.array([[cb, -sb, 0.0], [sb, cb, 0.0], [0.0, 0.0, 1.0]])
        poses[k, :3, :3] = swing @ about_shaft
        poses[k, :3, 3] = np.array([0.0, 0.0, 1.1]) + swing @ [0.0, 0.0, -1.0]
    return poses


def _patch_source(monkeypatch: pytest.MonkeyPatch, poses: np.ndarray) -> None:
    class _Source:
        def __init__(self, _spec: bytes) -> None:
            self.i = -1

        def frame_poses(self, _coords: dict) -> dict:
            self.i += 1
            return {"Clubhead": poses[self.i]}

    mod = types.ModuleType("overlay_source_stub")
    mod.MujocoOverlaySource = _Source  # type: ignore[attr-defined]
    monkeypatch.setitem(
        sys.modules, "src.engines.physics_engines.mujoco.python.overlay_source", mod
    )


def test_overlay_impact_is_the_face_centre_passage_not_the_frame_origin(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tools.native_viewer_export import overlay

    spec_bytes = _spec_bytes("driver")
    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    poses = _closing_swing(t)
    centre = cft.face_centre_in_frame(json.loads(spec_bytes))
    face = cft.face_centre_path(poses[:, :3, :3], poses[:, :3, 3], centre)
    at_face, _, _ = ball_passage(t, face)
    at_origin, _, _ = ball_passage(t, poses[:, :3, 3])
    assert abs(at_face - at_origin) > 2e-4  # the two points disagree here
    _patch_source(monkeypatch, poses)
    swing = SimpleNamespace(
        bundle=SimpleNamespace(coordinate_order=("a",), spec_bytes=spec_bytes),
        q=np.zeros((len(t), 1)),
        source_times_s=tuple(t),
    )
    assert overlay.detect_impact_time_s(swing) == pytest.approx(at_face, abs=1e-9)


def test_overlay_impact_rejects_a_specification_without_a_club(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    from src.tools.native_viewer_export import overlay

    t = np.arange(0.0, 1.6, 1.0 / 360.0)
    _patch_source(monkeypatch, _closing_swing(t))
    swing = SimpleNamespace(
        bundle=SimpleNamespace(coordinate_order=("a",), spec_bytes=b'{"bodies": []}'),
        q=np.zeros((len(t), 1)),
        source_times_s=tuple(t),
    )
    with pytest.raises(ValueError):
        overlay.detect_impact_time_s(swing)


@pytest.mark.parametrize("club", CLUBS)
def test_overlay_impact_matches_the_fixture_impact_within_one_sample(club: str) -> None:
    """Committed fixtures: the exporter's impact time and the fixture's
    recorded face-centre ball passage (provenance) agree within one sample."""
    pytest.importorskip("mujoco")
    from src.tools.native_viewer_export import overlay

    spec_bytes = _spec_bytes(club)
    q = np.load(FIXTURES / f"swing_q_{club}.npz")["q"].astype(float)
    order = json.loads(spec_bytes)["coordinate_order"]
    swing = SimpleNamespace(
        bundle=SimpleNamespace(coordinate_order=tuple(order), spec_bytes=spec_bytes),
        q=q,
        source_times_s=tuple(np.arange(len(q)) * FIXTURE_DT_S),
    )
    provenance = json.loads((FIXTURES / "provenance.json").read_text("utf-8"))
    recorded = provenance["clubs"][club]["after"]["speed_timing"]["model"]
    got = overlay.detect_impact_time_s(swing)
    assert abs(got - recorded["impact_time_s"]) <= FIXTURE_DT_S

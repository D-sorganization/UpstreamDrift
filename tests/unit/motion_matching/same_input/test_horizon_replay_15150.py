"""Same-input open-loop replay adapter for the impact horizon search (#15150 D2).

The delivery measurement is checked on hand-built head poses with known face,
path and speed.  The MuJoCo adapter is checked on a 30 ms reference bundle of
the committed ground-support seed: MuJoCo replays its own efforts bit for bit
from any tracked state, so every candidate start agrees exactly and the search
must accept the whole window.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.motion_matching import open_loop_horizon as olh
from src.shared.python.motion_matching.same_input import horizon_replay as hr

EVIDENCE = (
    Path(__file__).resolve().parents[4]
    / "docs/development/full_body_models/evidence/ground_support"
)
# Binary-exact grid so candidate starts land on bundle steps without round-off.
SETTINGS = olh.HorizonSettings(
    face_tolerance_deg=1.0,
    path_tolerance_deg=1.0,
    head_speed_tolerance_mps=0.5,
    step_s=0.005,
    max_horizon_s=0.015,
)
IMPACT_S = 0.02

pytestmark = pytest.mark.unit


def _pose(yaw_rad: float, origin: np.ndarray) -> np.ndarray:
    """Head pose turned ``yaw_rad`` about world +Z, at ``origin``."""
    c, s = math.cos(yaw_rad), math.sin(yaw_rad)
    pose = np.eye(4)
    pose[:2, :2], pose[:3, 3] = [[c, -s], [s, c]], origin
    return pose


# --- delivery measurement (pure) ----------------------------------------------


def test_delivery_reads_face_path_and_speed_from_two_poses() -> None:
    head = hr.HeadGeometry(
        face_axis=np.array([0.0, -1.0, 0.0]), face_centre=np.zeros(3)
    )
    # Face turned 10 deg open (toward native -X); head moving along the target
    # axis (-Y) and 5 deg toward -X at 40 m/s over a 1 ms step.
    yaw = math.radians(-10.0)  # +Z yaw turns the -Y normal toward +X (closed)
    heading = math.radians(5.0)
    step = 0.04 * np.array([-math.sin(heading), -math.cos(heading), 0.0])
    before = _pose(0.0, np.zeros(3))
    at = _pose(yaw, step)
    delivery = hr.delivery_from_poses(before, at, dt_s=0.001, head=head)
    assert delivery.face_angle_deg == pytest.approx(10.0)
    assert delivery.path_deg == pytest.approx(5.0)
    assert delivery.head_speed_mps == pytest.approx(40.0)


def test_delivery_tracks_the_face_centre_not_the_frame_origin() -> None:
    # A pure spin about the frame origin moves an offset face centre.
    head = hr.HeadGeometry(
        face_axis=np.array([0.0, -1.0, 0.0]), face_centre=np.array([0.1, 0.0, 0.0])
    )
    angle = 0.01
    delivery = hr.delivery_from_poses(
        _pose(0.0, np.zeros(3)), _pose(angle, np.zeros(3)), 0.001, head
    )
    chord = 2.0 * 0.1 * math.sin(angle / 2.0)
    assert delivery.head_speed_mps == pytest.approx(chord / 0.001)


def test_delivery_rejects_bad_inputs() -> None:
    head = hr.HeadGeometry(
        face_axis=np.array([0.0, -1.0, 0.0]), face_centre=np.zeros(3)
    )
    pose = np.eye(4)
    with pytest.raises(ValueError, match="dt_s"):
        hr.delivery_from_poses(pose, pose, 0.0, head)
    with pytest.raises(ValueError, match="4x4"):
        hr.delivery_from_poses(np.eye(3), pose, 0.001, head)
    with pytest.raises(ValueError, match="face_axis"):
        hr.HeadGeometry(face_axis=np.zeros(3), face_centre=np.zeros(3))


# --- MuJoCo adapter on a synthetic seed bundle --------------------------------


@pytest.fixture(scope="module")
def bundle():
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.same_input import (
        generate_reference_bundle,
    )

    spec = (EVIDENCE / "full_body_spec_hipcal_scaled.json").read_bytes()
    with np.load(EVIDENCE / "dynamics_record.npz") as record:
        return generate_reference_bundle(
            spec, record["time_s"], record["q"], duration_s=0.03
        )


@pytest.fixture(scope="module")
def replay(bundle):
    from src.shared.python.motion_matching.same_input import VectorPlant

    return hr.BundleReplay(bundle, VectorPlant("mujoco", bundle.spec_bytes), IMPACT_S)


@pytest.mark.requires_mujoco
def test_replay_from_a_tracked_state_reproduces_the_reference(replay) -> None:
    reference = replay.reference()
    assert reference.head_speed_mps > 0.0
    assert replay(IMPACT_S - SETTINGS.step_s) == reference


@pytest.mark.requires_mujoco
def test_search_accepts_the_whole_window_of_a_self_replay(bundle, replay) -> None:
    result = hr.search_bundle_horizon(replay, settings=SETTINGS)
    assert result.status == "accepted"
    assert result.horizon_s == pytest.approx(SETTINGS.max_horizon_s)
    assert result.earliest_start_s == pytest.approx(IMPACT_S - SETTINGS.max_horizon_s)
    assert len(result.candidates) == 3
    assert all(c.face_error_deg == 0.0 for c in result.candidates)
    json.dumps(result.summary(), allow_nan=False)


@pytest.mark.requires_mujoco
def test_replay_rejects_off_grid_and_out_of_range_times(bundle, replay) -> None:
    from src.shared.python.motion_matching.same_input import VectorPlant

    plant = VectorPlant("mujoco", bundle.spec_bytes)
    with pytest.raises(ValueError, match="grid"):
        replay(IMPACT_S - 0.0005 * 1.5)
    with pytest.raises(ValueError, match="before impact"):
        replay(IMPACT_S)
    with pytest.raises(ValueError, match="impact_s"):
        hr.BundleReplay(bundle, plant, bundle.steps * bundle.dt_s * 2.0)
    with pytest.raises(ValueError, match="impact_s"):
        hr.BundleReplay(bundle, plant, 0.0)
    coarse = olh.HorizonSettings(1.0, 1.0, 0.5, step_s=0.0015, max_horizon_s=0.015)
    with pytest.raises(ValueError, match="step_s"):
        hr.search_bundle_horizon(replay, settings=coarse)


def test_impact_time_needs_a_recorded_contact() -> None:
    class _NoImpact:
        provenance: dict = {"impact_split_time_s": None}

        def ball_impact(self):
            return None

    with pytest.raises(ValueError, match="impact"):
        hr.bundle_impact_s(_NoImpact())

    class _Split(_NoImpact):
        provenance = {"impact_split_time_s": 0.25}

    assert hr.bundle_impact_s(_Split()) == 0.25


# --- CLI ----------------------------------------------------------------------


def test_horizon_subcommand_defaults_to_cross_engine_and_the_settings_file() -> None:
    from scripts.same_input_bundle import build_parser

    args = build_parser().parse_args(["horizon", "--bundle", "b.npz"])
    assert args.engine is None
    assert args.settings == olh.SETTINGS_PATH
    assert args.impact_s is None and args.out is None


@pytest.mark.requires_mujoco
def test_horizon_cli_writes_the_result_json(bundle, tmp_path) -> None:
    from scripts.same_input_bundle import horizon_receipt

    path = tmp_path / "bundle.npz"
    bundle.save(path)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(vars(SETTINGS)))
    receipt = horizon_receipt(path, "mujoco", settings, impact_s=IMPACT_S)
    assert receipt["schema"] == "same-input-horizon/v1"
    assert receipt["replay_engine"] == "mujoco"
    assert receipt["reference_engine"] == "mujoco"
    assert receipt["self_replay"] is True
    assert receipt["note"] == hr.SELF_REPLAY_NOTE
    assert receipt["horizon"]["status"] == "accepted"
    assert receipt["reference"]["head_speed_mps"] > 0.0
    json.dumps(receipt, allow_nan=False)


# --- divergence and cross-engine replay ---------------------------------------


@pytest.mark.requires_mujoco
def test_a_diverged_rollout_raises_replay_diverged(replay, monkeypatch) -> None:
    from types import SimpleNamespace

    def failed(*_args, **_kwargs):
        return SimpleNamespace(failure="step 3: nonfinite state", q=np.zeros((2, 1)))

    monkeypatch.setattr(hr, "open_loop", failed)
    with pytest.raises(olh.ReplayDiverged, match="nonfinite"):
        replay(IMPACT_S - SETTINGS.step_s)
    result = hr.search_bundle_horizon(replay, settings=SETTINGS)
    assert result.status == "none_accepted"
    assert all(c.failure and "diverged" in c.failure for c in result.candidates)


def test_default_replay_engine_is_another_parity_engine() -> None:
    from src.shared.python.motion_matching.same_input import PARITY_ENGINES

    for reference in PARITY_ENGINES:
        engine = hr.cross_engine(reference)
        assert engine in PARITY_ENGINES and engine != reference
    assert hr.cross_engine("opensim") == PARITY_ENGINES[0]


def test_replay_provenance_flags_self_replay() -> None:
    same = hr.replay_provenance("mujoco", "mujoco")
    assert same["self_replay"] is True and same["note"] == hr.SELF_REPLAY_NOTE
    cross = hr.replay_provenance("mujoco", "drake")
    assert cross["self_replay"] is False and cross["note"] is None
    with pytest.raises(ValueError, match="engine"):
        hr.replay_provenance("mujoco", "")


@pytest.mark.requires_mujoco
@pytest.mark.requires_pinocchio
def test_cross_engine_horizon_is_not_exact_by_construction(bundle, tmp_path) -> None:
    pytest.importorskip("pinocchio")
    from scripts.same_input_bundle import horizon_receipt

    path = tmp_path / "bundle.npz"
    bundle.save(path)
    settings = tmp_path / "settings.json"
    settings.write_text(json.dumps(vars(SETTINGS)))
    receipt = horizon_receipt(path, "pinocchio", settings, impact_s=IMPACT_S)
    assert receipt["reference_engine"] == "mujoco"
    assert receipt["replay_engine"] == "pinocchio"
    assert receipt["self_replay"] is False and receipt["note"] is None
    candidates = receipt["horizon"]["candidates"]
    assert any(c["path_error_deg"] > 0.0 for c in candidates)
    json.dumps(receipt, allow_nan=False)

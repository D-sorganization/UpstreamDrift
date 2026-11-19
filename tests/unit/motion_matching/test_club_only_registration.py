"""Body-independent club marker registration from the club markers alone (#15296)."""

from __future__ import annotations

from dataclasses import asdict, replace
import json
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from src.shared.python.model_appearance.hand_backends.registration import (
    kabsch_rigid_transform,
)
from src.shared.python.motion_matching.club_models import DRIVER
from src.shared.python.motion_matching.execution import assets
from src.shared.python.motion_matching.execution.club_markers import load_club_markers
from src.shared.python.motion_matching.execution.club_only_registration import (
    ClubOnlySettings,
    axis_marker_distances_m,
    club_only_markers_from_capture,
    derive_club_only_offsets,
    head_support_m,
)

from src.shared.python.motion_matching.ground_support import capture_to_native_world
from src.shared.python.motion_matching.tour_capture_contract import load_tour_capture

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[3]
DOCS = REPO_ROOT / "docs" / "development" / "full_body_models"
DRIVER_MARKERS = DOCS / "evidence" / "club_markers" / "driver.json"
CLUB_ONLY_MARKERS = DOCS / "evidence" / "club_markers" / "driver_club_only.json"
CAPTURE = REPO_ROOT / "data" / "C3D_TA_Driver.c3d"
SETTINGS = ClubOnlySettings()
LABELS = (*SETTINGS.head_labels, "Marker_3:3:1", "Marker_3:3:2", "Marker_3:3:3")
RECOVERY_TOLERANCE_M = 1.0e-3  # for 0.1 mm marker noise over the address window
ADDRESS_FRAME, IMPACT_FRAME = 0, 476
_LEVEL = Rotation.from_euler("zy", [-20.0, -45.0], degrees=True)  # shaft horizontal


def _true_offsets() -> np.ndarray:
    """Clubhead-frame offsets that satisfy the rule: axis pair on x, set azimuth."""
    azimuth = np.radians(SETTINGS.head_centroid_azimuth_deg)
    spread = np.array([[-0.03, 0.0, 0.01], [0.02, 0.02, -0.03], [0.01, -0.02, 0.02]])
    head = spread + [0.05, 0.09 * np.cos(azimuth), 0.09 * np.sin(azimuth)]
    shaft = [[-0.67, 0.0, 0.0], [-0.74, 0.02, 0.026], [-0.79, 0.0, 0.0]]
    return np.vstack([head, shaft])


def _trajectory(noise_m: float = 1.0e-4, frames: int = 60) -> np.ndarray:
    """Soled address (first frames at rest), then a rigid swing; z up, ground z=0."""
    offsets = _true_offsets()
    address = Rotation.from_euler("yz", [45.0, 20.0], degrees=True)
    origin = np.array([0.3, -0.1, head_support_m(address.as_matrix(), SETTINGS)])
    rng = np.random.default_rng(15296)
    points = np.empty((frames, len(LABELS), 3))
    for k in range(frames):
        moved = max(0, k - SETTINGS.address_frames)
        swing = Rotation.from_euler("x", 2.0 * moved, degrees=True) * address
        points[k] = swing.apply(offsets) + origin + [0.0, 0.0, 0.004 * moved]
    return points + rng.normal(0.0, noise_m, points.shape)


def test_recovers_known_offsets_from_noisy_markers() -> None:
    derived = derive_club_only_offsets(LABELS, _trajectory(), SETTINGS)
    recovered = np.array([derived[label] for label in LABELS])
    assert np.abs(recovered - _true_offsets()).max() < RECOVERY_TOLERANCE_M
    # Postcondition: the axis pair lies on the Clubhead x axis.
    assert max(axis_marker_distances_m(derived, SETTINGS).values()) < 1e-12


def test_hover_setting_moves_the_club_along_the_shaft() -> None:
    hover = replace(SETTINGS, hover_m=0.010)
    soled = derive_club_only_offsets(LABELS, _trajectory(0.0), SETTINGS)
    lifted = derive_club_only_offsets(LABELS, _trajectory(0.0), hover)
    shift = np.subtract(lifted[LABELS[0]], soled[LABELS[0]])
    # 45 deg shaft: 10 mm of hover moves the origin 10 / sin(45 deg) up the shaft.
    assert shift == pytest.approx([0.010 / np.sin(np.radians(45.0)), 0, 0], abs=1e-9)


def _nan_at_address(points: np.ndarray) -> np.ndarray:
    points[1, 2, 0] = np.nan
    return points


def _bent(points: np.ndarray) -> np.ndarray:
    points[40:, 0] += [0.0, 0.05, 0.0]
    return points


def _moving_address(points: np.ndarray) -> np.ndarray:
    points[: SETTINGS.address_frames] += np.linspace(0, 0.02, SETTINGS.address_frames)[
        :, None, None
    ]
    return points


@pytest.mark.parametrize(
    ("damage", "message"),
    [
        (_nan_at_address, "address window"),
        (_bent, "not rigid"),
        (_moving_address, "at rest"),
        (lambda p: p[: SETTINGS.min_frames - 1], "complete frames"),
        (lambda p: _LEVEL.apply(p.reshape(-1, 3)).reshape(p.shape), "elevation"),
        (lambda p: np.where(np.arange(len(p))[:, None, None] > 30, np.inf, p), "inf"),
    ],
)
def test_rejects_unusable_marker_data(damage, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        derive_club_only_offsets(LABELS, damage(_trajectory()), SETTINGS)


def test_rejects_missing_labels_and_bad_settings() -> None:
    with pytest.raises(ValueError, match="missing"):
        derive_club_only_offsets(LABELS[:-1], _trajectory()[:, :-1], SETTINGS)
    with pytest.raises(ValueError, match="tolerance"):
        replace(SETTINGS, rigid_rms_tolerance_m=0.0)
    with pytest.raises(ValueError, match="orthonormal"):
        replace(SETTINGS, head_axes=((1, 0, 0), (1, 0, 0), (0, 0, 1)))


def test_default_head_is_the_model_driver_head_in_the_clubhead_frame() -> None:
    document = json.loads(
        (DOCS / "full_body_spec_anthro_driver.json").read_text(encoding="utf-8")
    )
    frame = next(f for f in document["frames"] if f["name"] == "Clubhead")
    assert SETTINGS.head_half_size_m == DRIVER.head_half_size_m
    assert np.array(SETTINGS.head_axes) == pytest.approx(
        np.array(frame["placement"])[:3, :3], abs=1e-12
    )


def test_club_only_file_has_its_shaft_markers_on_the_club_axis() -> None:
    limit = SETTINGS.max_axis_marker_distance_m
    club_only = load_club_markers(CLUB_ONLY_MARKERS)
    assert club_only.club == "driver"
    assert club_only.source["method"].startswith("club markers alone")
    assert club_only.source["settings"] == json.loads(json.dumps(asdict(SETTINGS)))
    assert max(axis_marker_distances_m(club_only.offsets_m, SETTINGS).values()) < limit
    # The reason for the alternative: the body-driven static-seed registration
    # holds the same two shaft markers 71-78 mm off the model shaft axis.
    body_driven = load_club_markers(DRIVER_MARKERS)
    off_axis = axis_marker_distances_m(body_driven.offsets_m, SETTINGS)
    assert all(0.070 < distance < 0.079 for distance in off_axis.values())
    assert min(off_axis.values()) > limit
    # Same physical cluster: the head triad keeps its shape to the capture noise.
    assert club_only.head_triad_edges_m == pytest.approx(
        body_driven.head_triad_edges_m, abs=5e-4
    )


def test_registration_is_selectable_by_name_and_the_default_is_unchanged() -> None:
    assert assets.get_club_markers("driver") == DRIVER_MARKERS
    assert assets.get_club_markers("driver", DRIVER_MARKERS) == DRIVER_MARKERS
    assert assets.get_club_markers("driver", "club_only") == CLUB_ONLY_MARKERS
    with pytest.raises(FileNotFoundError):
        assets.get_club_markers("iron7", "club_only")


def _head_lowest_m(offsets: dict, labels: list[str], world: np.ndarray) -> float:
    """Lowest point of the model head with the club placed on its markers."""
    template = np.array([offsets[label] for label in LABELS])
    pose = kabsch_rigid_transform(template, world[[labels.index(k) for k in LABELS]])
    return float(pose[2, 3] - head_support_m(pose[:3, :3], SETTINGS))


def test_committed_file_is_the_capture_derivation_and_soles_the_head() -> None:
    pytest.importorskip("ezc3d")
    committed = load_club_markers(CLUB_ONLY_MARKERS)
    derived = club_only_markers_from_capture(CAPTURE, club="driver", settings=SETTINGS)
    assert derived.source == committed.source
    for label, offset in committed.offsets_m.items():
        assert derived.offsets_m[label] == pytest.approx(offset, abs=1e-9)
    capture = load_tour_capture(CAPTURE)
    labels = list(capture.labels)
    world = capture_to_native_world(np.asarray(capture.points_m))
    body_driven = load_club_markers(DRIVER_MARKERS).offsets_m
    lowest = {
        (name, frame): _head_lowest_m(offsets, labels, world[frame])
        for name, offsets in (("old", body_driven), ("new", committed.offsets_m))
        for frame in (ADDRESS_FRAME, IMPACT_FRAME)
    }
    assert abs(lowest["new", ADDRESS_FRAME]) < 1.0e-3  # soled, by the datum rule
    assert 0.040 < lowest["old", ADDRESS_FRAME] < 0.048  # measured 43.7 mm
    assert 0.020 < lowest["new", IMPACT_FRAME] < 0.028  # measured, not tuned

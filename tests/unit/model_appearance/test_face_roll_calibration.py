"""Head roll about the shaft from a stated calibration rule (#15150)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import face_roll_calibration as frc
from src.shared.python.model_appearance.club_head_mesh import (
    club_face_normal,
    load_club_head,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"


def _address_body(lie_deg: float, lean_deg: float) -> np.ndarray:
    """Club body in a world with +Z up, target -Y: shaft at ``lie_deg`` to the
    ground, the grip leaned ``lean_deg`` toward the target."""
    lie, lean = math.radians(lie_deg), math.radians(lean_deg)
    grip = np.array([math.cos(lie), -math.sin(lean), math.sin(lie)])
    y = -grip / np.linalg.norm(grip)
    x = np.cross(y, [0.0, 0.0, 1.0])
    x = -x / np.linalg.norm(x) if x[1] > 0 else x / np.linalg.norm(x)
    body = np.eye(4)
    body[:3, :3] = np.column_stack([x, y, np.cross(x, y)])
    return body


def _loft_deg(body: np.ndarray, alias: str, roll_deg: float) -> float:
    normal = body[:3, :3] @ club_face_normal(load_club_head(alias), roll_deg)
    return math.degrees(math.asin(normal[2]))


def test_default_mode_is_address_square_and_changes_nothing() -> None:
    settings = frc.load_face_roll_settings()
    assert settings.mode == "address_square"
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    out = frc.spec_with_face_roll_calibration(spec, _address_body(45.0, 0.0))
    assert ca.assembly_from_spec(out) == ca.assembly_from_spec(spec)
    assert out["club"]["face_roll_calibration"] == "address_square"
    assert "face_roll_calibration" not in spec["club"]  # the input is not edited


@pytest.mark.parametrize("alias", ["driver", "iron7"])
@pytest.mark.parametrize(("lie", "lean"), [(42.0, -1.7), (50.0, 0.0), (60.0, 6.0)])
def test_design_loft_mode_gives_the_design_loft_at_address(
    alias: str, lie: float, lean: float
) -> None:
    body = _address_body(lie, lean)
    head = load_club_head(alias)
    near = ca.ADDRESS_SQUARE_FACE_ROLL_DEG[alias]
    roll = frc.design_loft_face_roll_deg(
        head, frc.up_in_club_frame(body), near_deg=near
    )
    assert _loft_deg(body, alias, roll) == pytest.approx(head.loft_deg, abs=1e-9)
    assert -180.0 < roll <= 180.0


def test_design_loft_mode_takes_the_solution_nearest_the_reference_roll() -> None:
    body = _address_body(45.0, 0.0)
    head = load_club_head("driver")
    up = frc.up_in_club_frame(body)
    one = frc.design_loft_face_roll_deg(head, up, near_deg=0.0)
    other = frc.design_loft_face_roll_deg(head, up, near_deg=one + 170.0)
    assert abs(one) < 90.0
    assert abs(other - one) > 1.0  # the far branch faces away from the target
    assert _loft_deg(body, "driver", other) == pytest.approx(head.loft_deg, abs=1e-9)


def test_design_loft_can_be_overridden_and_must_be_reachable() -> None:
    body = _address_body(45.0, 0.0)
    head = load_club_head("driver")
    up = frc.up_in_club_frame(body)
    roll = frc.design_loft_face_roll_deg(head, up, near_deg=0.0, loft_deg=9.0)
    assert _loft_deg(body, "driver", roll) == pytest.approx(9.0, abs=1e-9)
    with pytest.raises(ValueError, match="no head roll"):
        frc.design_loft_face_roll_deg(head, up, near_deg=0.0, loft_deg=80.0)
    with pytest.raises(ValueError, match="loft_deg"):
        frc.design_loft_face_roll_deg(head, up, near_deg=0.0, loft_deg=math.nan)
    with pytest.raises(ValueError, match="near_deg"):
        frc.design_loft_face_roll_deg(head, up, near_deg=math.inf)


def test_a_vertical_shaft_has_no_solution() -> None:
    head = load_club_head("driver")
    with pytest.raises(ValueError, match="no head roll"):
        frc.design_loft_face_roll_deg(head, [0.0, -1.0, 0.0], near_deg=0.0)


@pytest.mark.parametrize("bad", [[0.0, 0.0], [0.0, 0.0, 0.0], [math.nan, 0.0, 1.0]])
def test_up_vector_is_validated(bad: list[float]) -> None:
    with pytest.raises(ValueError, match="up_club"):
        frc.design_loft_face_roll_deg(load_club_head("driver"), bad, near_deg=0.0)


def test_up_in_club_frame_validates_the_pose() -> None:
    body = _address_body(42.0, 0.0)
    up_world = body[:3, :3] @ frc.up_in_club_frame(body)
    np.testing.assert_allclose(up_world, [0.0, 0.0, 1.0], atol=1e-12)
    with pytest.raises(ValueError, match="4x4"):
        frc.up_in_club_frame(np.eye(3))
    with pytest.raises(ValueError, match="4x4"):
        frc.up_in_club_frame(np.full((4, 4), np.nan))


def test_spec_carries_the_solved_roll_and_the_mode() -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    body = _address_body(42.0, -1.7)
    out = frc.spec_with_face_roll_calibration(spec, body, "design_loft_at_address")
    club = ca.assembly_from_spec(out)
    assert out["club"]["face_roll_calibration"] == "design_loft_at_address"
    assert club.face_roll_deg != ca.ADDRESS_SQUARE_FACE_ROLL_DEG["driver"]
    world_normal = body[:3, :3] @ ca.clubface_vector(club)
    assert math.degrees(math.asin(world_normal[2])) == pytest.approx(10.5, abs=1e-6)
    out["club"]["design_loft_deg"] = 9.0
    again = frc.spec_with_face_roll_calibration(out, body, "design_loft_at_address")
    world_normal = body[:3, :3] @ ca.clubface_vector(ca.assembly_from_spec(again))
    assert math.degrees(math.asin(world_normal[2])) == pytest.approx(9.0, abs=1e-6)


def test_mode_and_spec_are_validated(tmp_path: Path) -> None:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    with pytest.raises(ValueError, match="mode"):
        frc.spec_with_face_roll_calibration(spec, np.eye(4), "square_at_impact")
    with pytest.raises(TypeError, match="mode"):
        frc.spec_with_face_roll_calibration(spec, np.eye(4), 3)  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="club"):
        frc.spec_with_face_roll_calibration({"bodies": []}, np.eye(4))
    bad = tmp_path / "settings.json"
    bad.write_text(json.dumps({"mode": "open"}), encoding="utf-8")
    with pytest.raises(ValueError, match="mode"):
        frc.load_face_roll_settings(bad)


def test_settings_file_documents_every_key() -> None:
    raw = json.loads(frc.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}


def test_replayed_address_pose_gets_the_design_loft() -> None:
    pytest.importorskip("mujoco")
    from src.tools.native_viewer_export.club_pose import (
        ClubPoseModel,
        face_roll_calibrated_spec_bytes,
    )

    poses = json.loads(
        (ROOT / "tests/fixtures/club_face/address_poses.json").read_text("utf-8")
    )["poses"]["driver"]
    coords = dict(zip(poses["coordinate_order"], poses["q_rad"], strict=True))
    raw = SPEC.read_bytes()
    assert face_roll_calibrated_spec_bytes(raw, coords, "address_square") == raw
    calibrated = face_roll_calibrated_spec_bytes(raw, coords, "design_loft_at_address")
    normal = ClubPoseModel(calibrated).face_world(coords).normal
    assert math.degrees(math.asin(normal[2])) == pytest.approx(10.5, abs=1e-6)
    before = ClubPoseModel(raw).body_world(coords)
    np.testing.assert_array_equal(ClubPoseModel(calibrated).body_world(coords), before)

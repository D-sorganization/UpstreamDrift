"""#15296: the address pin is the document's ``club.sole_point_m`` (the sole in
the preset's address lie), not the club-body origin; without it, the origin."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import club_face_target as cft
from src.shared.python.motion_matching import club_models
from src.shared.python.motion_matching.contact_law import GroundPlane
from src.shared.python.motion_matching.execution import driver, spec_builder
from src.shared.python.motion_matching.execution import subject_preset as presets
from src.shared.python.motion_matching.pipeline import address_club
from src.shared.python.motion_matching.pipeline.constants import (
    ADDRESS_CLUB_SOLE_WEIGHT,
    CLUB_SOLE_GROUND_TOLERANCE_M,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
EVIDENCE = ROOT / "docs/development/full_body_models/evidence"
GROUND = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=0.0)
HEAD_CENTRE = np.array([0.0, 0.0, club_models.HEAD_HEIGHT_M])
PRESET = presets.load_subject_preset("tour_average_driver")
LIE = (PRESET.club_shaft_elevation_deg, PRESET.club_head_roll_deg)
E, R = (math.radians(angle) for angle in LIE)  # UP: ground normal, club frame
UP = np.array([math.cos(E) * math.sin(R), -math.sin(E), math.cos(E) * math.cos(R)])


@pytest.fixture(scope="module")
def document() -> dict[str, Any]:
    built = spec_builder.compose_anthropometric_document(**PRESET.builder_kwargs())
    return built.document


class _LieKinematics:
    """Club body with ``up`` (club frame) vertical and the head centre at a height."""

    def __init__(self, up: np.ndarray, centre_height_m: float) -> None:
        x = np.cross([0.0, 1.0, 0.0], up)
        x /= np.linalg.norm(x)
        self.rotation = np.vstack([x, np.cross(up, x), up])  # world from club
        centre_world = np.array([0.3, -0.2, centre_height_m])
        self.origin = centre_world - self.rotation @ HEAD_CENTRE

    def body_poses(self, q: Any, bodies: Any) -> dict[str, Any]:
        return dict.fromkeys(bodies, (self.rotation, self.origin))


def test_the_sole_point_is_the_lowest_point_of_the_head() -> None:
    driver_club = club_models.DRIVER
    half = np.asarray(driver_club.head_half_size_m)
    point = np.asarray(club_models.sole_point_m(driver_club, *LIE))
    drop = float((HEAD_CENTRE - point) @ UP)  # head centre above the sole
    along_shaft = club_models.sole_offset_m(driver_club, *LIE)
    assert drop == pytest.approx(along_shaft * math.sin(E))
    assert drop == pytest.approx(0.0454, abs=2e-4)
    scaled = (point - HEAD_CENTRE) / half
    assert float(scaled @ scaled) == pytest.approx(1.0)  # on the head ellipsoid
    normal = scaled / half  # outward surface normal there is straight down
    assert normal / np.linalg.norm(normal) == pytest.approx(-UP)
    # The club-body origin is above the head centre in this lie, not under it.
    assert float(-HEAD_CENTRE @ UP) > 0.02
    for elevation in (0.0, 91.0, float("nan")):
        with pytest.raises(ValueError, match="elevation"):
            club_models.sole_point_m(driver_club, elevation, 0.0)


def test_the_preset_document_pins_the_sole_under_the_head(
    document: dict[str, Any],
) -> None:
    expected = club_models.sole_point_m(club_models.DRIVER, *LIE)
    assert document["club"]["sole_point_m"] == pytest.approx(list(expected))
    assert document["club"]["sole_point_source"] == club_models.SOLE_POINT_SOURCE
    kin = _LieKinematics(UP, centre_height_m=0.0454)
    q = np.zeros(len(document["coordinate_order"]))
    report = address_club.club_sole_report(kin, q, document, GROUND)
    assert report["measurement_point"] == "club_sole_point"
    assert report["sole_point_m"] == document["club"]["sole_point_m"]
    assert report["height_above_ground_m"] == pytest.approx(0.0, abs=2e-4)
    assert report["head_centre_height_above_ground_m"] == pytest.approx(0.0454)
    assert report["club_body_origin_height_above_ground_m"] > 0.06
    address_club.assert_club_sole_at_ground(kin, q, document, GROUND)


def test_a_document_without_the_setting_keeps_the_club_body_origin(
    document: dict[str, Any],
) -> None:
    legacy = json.loads(json.dumps(document))
    del legacy["club"]["sole_point_m"], legacy["club"]["sole_point_source"]
    kin = _LieKinematics(np.array([0.0, 0.0, 1.0]), centre_height_m=0.07)
    q = np.zeros(len(legacy["coordinate_order"]))
    sole = cft.model_club_soles(kin, q[None, :], legacy)[0]
    assert np.array_equal(sole, kin.origin)
    assert address_club.club_sole_report(kin, q, legacy, GROUND) == {
        "available": True,
        "measurement_point": "club_body_origin",
        "height_above_ground_m": float(kin.origin[2]),
        "tolerance_m": 0.01,
        "within_tolerance": True,
    }
    plain = spec_builder.compose_anthropometric_document().document
    assert not {"sole_point_m", "sole_point_source"} & set(plain["club"])
    committed = sorted(EVIDENCE.rglob("full_body_spec*.json"))
    assert committed
    for path in committed:
        spec = json.loads(path.read_text(encoding="utf-8"))
        assert club_models.document_sole_point_m(spec) is None, path


@pytest.mark.parametrize(
    ("value", "message"),
    [
        ([0.0, 0.0], "3-vector"),
        ([True, 0.0, 0.0], "finite numbers"),
        ([0.0, float("nan"), 0.0], "finite numbers"),
        ([0.0, 0.5, 0.064], "of the head centre"),
    ],
)
def test_a_bad_sole_point_is_refused(value: Any, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        club_models.document_sole_point_m({"club": {"sole_point_m": value}})
    if len(value) == 3 and value[1] == 0.5:  # the builder refuses it as well
        with pytest.raises(ValueError, match=message):
            spec_builder.compose_anthropometric_document(club_sole_point_m=value)


def _mujoco(document: dict[str, Any], attachments: dict[str, Any]) -> Any:
    pytest.importorskip("mujoco")
    from src.engines.physics_engines.mujoco.python import full_body_ik, overlay_source

    adapter = overlay_source.MujocoOverlaySource(json.dumps(document).encode("utf-8"))
    return full_body_ik.FullBodyMarkerKinematics(adapter, attachments)


def _drake(document: dict[str, Any], attachments: dict[str, Any]) -> Any:
    drake_all = pytest.importorskip("pydrake.all")
    if not hasattr(drake_all, "MultibodyPlant"):
        pytest.skip("real pydrake runtime required")
    from src.engines.physics_engines.drake.python.full_body_ik import DrakeFullBodyIK

    return DrakeFullBodyIK(document, attachments=attachments)


@pytest.mark.parametrize(
    "make",
    [
        pytest.param(_mujoco, marks=pytest.mark.requires_mujoco),
        pytest.param(_drake, marks=pytest.mark.requires_drake),
    ],
)
def test_the_address_solve_pins_the_stated_sole(
    make: Any, document: dict[str, Any]
) -> None:
    attachments = {
        label: (entry["body"], tuple(entry["offset_m"]))
        for label, entry in document["marker_attachments"].items()
        if entry.get("offset_m") is not None
    }
    kin = make(document, attachments)
    q0 = np.zeros(len(kin.coordinate_order))
    sole0 = cft.model_club_soles(kin, q0[None, :], document)[0]
    ground = GroundPlane(normal=(0.0, 0.0, 1.0), height_m=float(sole0[2]) - 0.05)
    rows, jacs = list[Any](), list[Any]()
    kin._set(q0)
    kin._append_club_sole(rows, jacs, ground, 1.0)
    assert rows[0][0] == pytest.approx(0.05)  # the stated point, not the origin
    column = int(np.argmax(np.abs(jacs[0][0])))
    moved = q0 + 1e-6 * np.eye(len(q0))[column]
    height = cft.model_club_soles(kin, moved[None, :], document)[0][2]
    assert abs(jacs[0][0][column]) > 1e-3
    assert jacs[0][0][column] == pytest.approx((height - sole0[2]) / 1e-6, abs=1e-4)
    targets, weight = kin.marker_positions(q0), ADDRESS_CLUB_SOLE_WEIGHT
    options = {"ground": ground, "iterations": 80, "club_sole_weight": weight}
    solved = kin.solve_pose(targets, np.ones(len(targets), bool), q0, **options)
    report = address_club.club_sole_report(kin, solved.q, document, ground)
    assert report["height_above_ground_m"] <= CLUB_SOLE_GROUND_TOLERANCE_M


def test_the_driver_honours_address_only(monkeypatch: pytest.MonkeyPatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(driver, "run_address_stage", lambda a: calls.append("address"))
    monkeypatch.setattr(driver, "run_pipeline", lambda a: calls.append("pipeline"))
    assert driver.main(["--address-only"]) == 0
    assert driver.main([]) == 0
    assert calls == ["address", "pipeline"]

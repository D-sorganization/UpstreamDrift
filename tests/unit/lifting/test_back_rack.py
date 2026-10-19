"""Back-squat bar on a cited back rack, window started settled (OSL-17, #13131).

Red tests, written before the regeneration.  Diagnosis (PR #13130): the
back-squat bundle's bar sat on a shelf fitted to the parity standard's
start-pose bar position, 0.157 m in front of and 0.159 m above the
shoulder-centre midpoint, i.e. in front of the face.  And the window's first
state was still settling onto its contact (feet 1347.2 N of 1768.3 N), so the
OpenSim Hicks gate failed at that state only.

* **Rack.**  The high-bar rack is the bar "across the top of the trapezius,
  just below the process of the C7 vertebra" (Glassbrook et al. 2019, citing
  Wretenberg et al. 1996): one shaft radius behind the lift body's C7
  surface landmark (``BackTop``), read from the spec, never copied here.
* **Behind the neck.**  In the trunk frame the bar centre lies behind the
  neck joint and the shoulder centres, between their levels.
* **Settled start.**  The source bundle's tracking run starts before the
  window (``provenance.lead_in``); the bundle starts on the settled state.
* **Hicks.**  The committed OpenSim squat receipt meets the marker gate and
  the Hicks et al. (2015) residual gate on every state.
"""

from __future__ import annotations

import json

import numpy as np
import pytest

from src.shared.python.lifting.back_bar import BAR_CENTRE_FRAME, BackBarGeometry
from src.shared.python.lifting.back_rack import (
    HIGH_BAR,
    RACK_SOURCES,
    TRUNK_POSTERIOR_LOCAL,
    BackRack,
    rack_offset,
)
from src.shared.python.lifting.barbell_bushing import Barbell
from src.shared.python.lifting.bundles import (
    BACK_BAR_REACH_TOLERANCE_M,
    BACK_SQUAT_BARBELL_BUNDLE,
    BACK_SQUAT_BUNDLE,
    LIFT_SPEC,
    load_seed_pose,
    solve_back_bar_standing,
)

pytestmark = pytest.mark.unit

HEIGHT_M = 1.78
#: Frames of the lift body the rack is measured against.
TRUNK_FRAME = "Spine"
NECK_FRAME = "Head"
SHOULDER_FRAMES = ("LS", "RS")
#: How far the committed bundles' bar centre may sit from the rack point,
#: metres: one shaft radius, so the bar never reaches the C7 skin landmark.
#: The arm solve trades the bar target against the grip weld, the bar axis
#: and the arm bounds; measured 8.0 mm at most over both bundles (4-5 mm
#: anterior, 6 mm below), so the bar centre stays 9-10 mm behind the skin.
RACK_SEAT_TOLERANCE_M = HIGH_BAR.clearance_m
#: Steps the committed source tracks before its window (declared before the
#: measurement: the contact transient decays within tens of milliseconds).
MIN_LEAD_IN_STEPS = 50
RECEIPTS = BACK_SQUAT_BUNDLE.parents[1] / "opensim_ik_id" / "receipts.json"


@pytest.fixture(scope="module")
def spec() -> dict:
    loaded: dict = json.loads(LIFT_SPEC.read_bytes())
    return loaded


# -- the rack definition --------------------------------------------------------


def test_high_bar_is_the_cited_c7_rack() -> None:
    assert HIGH_BAR.name == "high_bar"
    assert HIGH_BAR.landmark == "BackTop"
    assert "C7" in HIGH_BAR.description
    assert any("Glassbrook" in s for s in RACK_SOURCES)
    assert any("Wretenberg" in s for s in RACK_SOURCES)
    assert HIGH_BAR.to_record()["sources"] == list(RACK_SOURCES)


def test_high_bar_clears_the_skin_by_the_shaft_radius() -> None:
    assert HIGH_BAR.clearance_m == Barbell(100.0).shaft_radius_m


def test_rack_offset_is_the_landmark_plus_the_clearance_behind_it(spec) -> None:
    marker = spec["marker_attachments"][HIGH_BAR.landmark]
    frame, offset = rack_offset(HIGH_BAR, spec)
    expected = np.asarray(marker["offset_m"]) + HIGH_BAR.clearance_m * np.asarray(
        TRUNK_POSTERIOR_LOCAL
    )
    assert frame == marker["body"] == TRUNK_FRAME
    assert np.allclose(offset, expected, atol=1e-15)
    assert rack_offset(HIGH_BAR, LIFT_SPEC.read_bytes()) == (frame, offset)


def test_rack_offset_rejects_a_spec_without_the_landmark() -> None:
    with pytest.raises(ValueError, match="BackTop"):
        rack_offset(HIGH_BAR, {"marker_attachments": {}})
    with pytest.raises(TypeError):
        rack_offset("high_bar", {})  # type: ignore[arg-type]


@pytest.mark.parametrize("clearance", [-1e-3, float("nan")])
def test_rack_rejects_a_bad_clearance(clearance: float) -> None:
    with pytest.raises(ValueError):
        BackRack("x", "BackTop", "", clearance_m=clearance)


def test_geometry_carries_the_rack_and_rejects_a_non_rack() -> None:
    geometry = BackBarGeometry.from_standard("squat", height_m=HEIGHT_M, rack=HIGH_BAR)
    assert geometry.rack is HIGH_BAR
    assert geometry.to_record()["rack"]["name"] == "high_bar"
    assert BackBarGeometry.from_standard("squat", height_m=HEIGHT_M).rack is None
    with pytest.raises(TypeError):
        BackBarGeometry.from_standard(
            "squat",
            height_m=HEIGHT_M,
            rack="high_bar",  # type: ignore[arg-type]
        )


# -- the bar behind the neck (real MuJoCo kinematics) ---------------------------


def _trunk_geometry(spec_bytes: bytes, q: np.ndarray) -> dict[str, np.ndarray]:
    """Bar centre, neck joint and shoulder centres in the trunk frame."""
    from src.shared.python.lifting.bundles import back_bar_kinematics

    kin = back_bar_kinematics(spec_bytes)
    names = [TRUNK_FRAME, NECK_FRAME, BAR_CENTRE_FRAME, *SHOULDER_FRAMES]
    poses = kin.body_poses(np.asarray(q, dtype=float), names)
    rotation, origin = poses[TRUNK_FRAME]

    def local(name: str) -> np.ndarray:
        return np.asarray(rotation.T @ (poses[name][1] - origin))

    return {
        "bar": local(BAR_CENTRE_FRAME),
        "neck": local(NECK_FRAME),
        "shoulders": 0.5 * (local(SHOULDER_FRAMES[0]) + local(SHOULDER_FRAMES[1])),
    }


def assert_behind_the_neck(points: dict[str, np.ndarray]) -> None:
    """Posterior to neck and shoulders by a shaft radius, between their levels."""
    posterior = np.asarray(TRUNK_POSTERIOR_LOCAL)
    bar, neck, shoulders = points["bar"], points["neck"], points["shoulders"]
    assert (bar - neck) @ posterior > HIGH_BAR.clearance_m
    assert (bar - shoulders) @ posterior > HIGH_BAR.clearance_m
    assert shoulders[2] <= bar[2] <= neck[2]
    assert abs(bar[1]) < 0.01  # on the sagittal midline


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_standing_pose_seats_the_bar_on_the_high_bar_rack(spec) -> None:
    pytest.importorskip("mujoco")
    geometry = BackBarGeometry.from_standard("squat", height_m=HEIGHT_M, rack=HIGH_BAR)
    start = solve_back_bar_standing(LIFT_SPEC.read_bytes(), load_seed_pose(), geometry)
    points = _trunk_geometry(start.spec_bytes, start.q)
    _, offset = rack_offset(HIGH_BAR, spec)
    assert np.linalg.norm(points["bar"] - offset) <= BACK_BAR_REACH_TOLERANCE_M
    assert_behind_the_neck(points)


@pytest.fixture(scope="module")
def source():
    from src.shared.python.motion_matching.same_input import InputBundle

    return InputBundle.load(BACK_SQUAT_BUNDLE)


@pytest.mark.integration
@pytest.mark.requires_mujoco
def test_committed_bundle_is_racked_high_bar(source, spec) -> None:
    pytest.importorskip("mujoco")
    back_bar = source.provenance["ik"]["back_bar"]
    assert back_bar["geometry"]["rack"]["name"] == HIGH_BAR.name
    _, offset = rack_offset(HIGH_BAR, spec)
    standing = np.asarray(source.provenance["ik"]["q_standing"], dtype=float)
    for q in [standing, *source.reference_q[::20]]:
        points = _trunk_geometry(source.spec_bytes, q)
        assert np.linalg.norm(points["bar"] - offset) <= RACK_SEAT_TOLERANCE_M


@pytest.mark.integration
@pytest.mark.requires_mujoco
@pytest.mark.parametrize("path", [BACK_SQUAT_BUNDLE, BACK_SQUAT_BARBELL_BUNDLE])
def test_committed_bundles_keep_the_bar_behind_the_neck(path) -> None:
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.same_input import InputBundle

    bundle = InputBundle.load(path)
    for q in bundle.reference_q[::20]:
        assert_behind_the_neck(_trunk_geometry(bundle.spec_bytes, q))


# -- the settled window start ---------------------------------------------------


def test_committed_source_starts_after_a_lead_in(source) -> None:
    lead_in = source.provenance["lead_in"]
    assert lead_in["steps"] >= MIN_LEAD_IN_STEPS
    assert len(lead_in["reference_q"]) == lead_in["steps"]
    assert lead_in["duration_s"] == pytest.approx(lead_in["steps"] * source.dt_s)
    # The window still starts where the recipe puts it.
    assert source.provenance["window"]["duration_s"] == pytest.approx(
        source.steps * source.dt_s
    )


def _committed_squat_receipt() -> dict:
    return json.loads(RECEIPTS.read_text())["lifts"]["squat"]["ik_id"]


def test_committed_receipt_meets_the_marker_gate() -> None:
    squat = _committed_squat_receipt()
    markers = squat["ik"]["marker_error"]
    assert markers["rms_m"] < 0.02 and markers["max_m"] < 0.04


def test_committed_receipt_meets_the_hicks_gate() -> None:
    squat = _committed_squat_receipt()
    assert squat["status"] == "complete"
    gate = squat["id"]["residual_gate"]
    assert gate["forces_within"] is True
    assert gate["moments_within"] is True
    # Every state of the window is in the gate: no state is excluded.
    assert squat["ik"]["frames"] == squat["bundle"]["frames"]


def test_committed_retrack_leads_in_with_the_source(source) -> None:
    from src.shared.python.motion_matching.same_input import InputBundle

    barbell = InputBundle.load(BACK_SQUAT_BARBELL_BUNDLE)
    retrack = barbell.provenance["barbell"]["lead_in"]
    assert retrack["steps"] == source.provenance["lead_in"]["steps"]
    assert barbell.steps == source.steps

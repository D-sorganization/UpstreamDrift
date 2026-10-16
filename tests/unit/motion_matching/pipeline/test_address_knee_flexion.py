"""Address knee flexion: owner prior, capture override, no hyperextension (R4, #12684).

Owner ruling on #12684 (2026-10-15): knee flexion at address is a prior of
about 15 deg with a +-5 deg band for both knees; a capture that determines the
knee flexion reliably wins; hyperextension is forbidden as a hard bound, never
a soft penalty; each knee's target carries its source into the receipt with
R3's ``target_source`` vocabulary.
"""

from __future__ import annotations

import argparse
import json
from types import SimpleNamespace

import numpy as np
import pytest

from src.shared.python.motion_matching import address_knee
from src.shared.python.motion_matching.address_knee import (
    CANONICAL_KNEE_FLEXION,
    KNEE_FLEX_BAND_DEG,
    KNEE_FLEX_PRIOR_DEG,
    KNEE_TARGET_SOURCES,
    MODES,
    KneeFlexion,
    KneeTargets,
    knee_flexion_deg,
    knee_flexion_from_markers,
    resolve_knee_target,
)
from src.shared.python.motion_matching.foot_progression import (
    FOOT_TARGET_SOURCES,
    TARGET_SOURCES,
)
from src.shared.python.motion_matching.pipeline.constants import LEG_SEEDS

pytestmark = [pytest.mark.unit]

_CANONICAL = {
    "driver": "full_body_spec_anthro_driver.json",
    "iron": "full_body_spec_anthro_iron7.json",
}
_SIDES = ("left", "right")


# ------------------------------------------------------------ owner constants


def test_owner_prior_and_band_are_single_sourced() -> None:
    assert KNEE_FLEX_PRIOR_DEG == 15.0
    assert KNEE_FLEX_BAND_DEG == 5.0
    assert MODES == ("off", "prior", "capture")
    assert CANONICAL_KNEE_FLEXION in MODES
    assert CANONICAL_KNEE_FLEXION != "off"


def test_knee_sources_reuse_the_shared_target_source_vocabulary() -> None:
    assert set(KNEE_TARGET_SOURCES) == {"capture", "prior"}
    assert set(KNEE_TARGET_SOURCES) <= set(TARGET_SOURCES)
    # the foot vocabulary is unchanged by the knee prior (R3 receipts stay valid)
    assert set(FOOT_TARGET_SOURCES) == {"capture", "default", "style"}
    assert set(TARGET_SOURCES) == set(FOOT_TARGET_SOURCES) | {"prior"}


# ------------------------------------------------------ geometric measurement


def _leg(flexion_deg: float) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Hip, knee and ankle centres with the shank bent ``flexion_deg`` forward."""
    hip = np.array([0.1, -0.2, 0.95])
    knee = hip + np.array([0.0, 0.0, -0.42])
    t = np.radians(flexion_deg)
    ankle = knee + 0.40 * np.array([-np.sin(t), 0.0, -np.cos(t)])
    return hip, knee, ankle


@pytest.mark.parametrize("flexion", [0.0, 15.0, 90.0])
def test_knee_flexion_of_known_joint_centres(flexion: float) -> None:
    assert knee_flexion_deg(*_leg(flexion)) == pytest.approx(flexion, abs=1e-9)


def test_collinear_leg_reads_zero_flexion() -> None:
    hip = np.zeros(3)
    assert knee_flexion_deg(hip, hip + [0, 0, -0.4], hip + [0, 0, -0.8]) == 0.0


@pytest.mark.parametrize(
    "points",
    [
        (np.zeros(3), np.zeros(3), np.array([0.0, 0.0, -0.4])),  # zero thigh
        (np.zeros(3), np.array([0.0, 0.0, -0.4]), np.array([0.0, 0.0, -0.4])),
        (np.zeros(3), np.array([np.nan, 0.0, -0.4]), np.array([0.0, 0.0, -0.8])),
        (np.zeros(2), np.zeros(2), np.zeros(2)),
    ],
)
def test_degenerate_or_non_finite_legs_raise(points) -> None:
    with pytest.raises(ValueError):
        knee_flexion_deg(*points)


@pytest.mark.parametrize("side", ["left", "right"])
@pytest.mark.parametrize("flexion", [12.0, 25.0])
def test_lateral_markers_recover_the_centre_flexion(side: str, flexion: float) -> None:
    """Lateral epicondyle/malleolus markers, moved back onto the centres."""
    hip, knee, ankle = _leg(flexion)
    lateral_dir = np.array([0.0, 1.0 if side == "right" else -1.0, 0.0])
    w_knee, w_ankle = address_knee.LATERAL_MARKER_OFFSETS_M
    measured = knee_flexion_from_markers(
        hip, knee + w_knee * lateral_dir, ankle + w_ankle * lateral_dir, side=side
    )
    assert measured == pytest.approx(flexion, abs=0.05)


# --------------------------------------------------------- target resolution


def _measured(angle: float, *, reliable: bool = True) -> KneeFlexion:
    return KneeFlexion(
        side="right",
        role="trail",
        angle_deg=angle,
        frames=90,
        spread_deg=0.4 if reliable else 9.0,
        reliable=reliable,
        reason="test",
    )


def test_prior_mode_ignores_the_capture() -> None:
    assert resolve_knee_target(_measured(26.0), mode="prior") == (15.0, "prior")
    assert resolve_knee_target(None, mode="prior") == (15.0, "prior")


def test_capture_overrides_the_prior_when_reliable() -> None:
    assert resolve_knee_target(_measured(26.2), mode="capture") == (26.2, "capture")


def test_capture_mode_falls_back_to_the_prior_when_unreliable() -> None:
    assert resolve_knee_target(_measured(40.0, reliable=False), mode="capture") == (
        15.0,
        "prior",
    )
    assert resolve_knee_target(None, mode="capture") == (15.0, "prior")


@pytest.mark.parametrize("mode", ["off", "bogus"])
def test_resolve_rejects_modes_without_a_target(mode: str) -> None:
    with pytest.raises(ValueError, match="mode"):
        resolve_knee_target(None, mode=mode)


def test_a_hyperextended_capture_never_becomes_a_target() -> None:
    with pytest.raises(ValueError, match="flexion"):
        resolve_knee_target(_measured(-3.0), mode="capture")


def test_knee_targets_validate_sources_and_angles() -> None:
    ok = KneeTargets(
        {"left": 15.0, "right": 26.0}, {"left": "prior", "right": "capture"}, "capture"
    )
    assert ok.weight > 0
    with pytest.raises(ValueError, match="sources"):
        KneeTargets({"left": 15.0, "right": 15.0}, {"left": "style", "right": "prior"})
    with pytest.raises(ValueError, match="hyperextension"):
        KneeTargets({"left": -1.0, "right": 15.0}, {"left": "prior", "right": "prior"})
    with pytest.raises(ValueError, match="weight"):
        KneeTargets(
            {"left": 15.0, "right": 15.0},
            {"left": "prior", "right": "prior"},
            weight=0.0,
        )


# ----------------------------------------------------- capture-derived knees


def _capture_inputs(capture: str):
    from src.shared.python.motion_matching.pipeline.constants import (
        REPO_ROOT,
        capture_path,
    )
    from src.shared.python.motion_matching.pipeline.lane import Lane

    path = REPO_ROOT / "docs/development/full_body_models" / _CANONICAL[capture]
    spec = json.loads(path.read_text(encoding="utf-8"))
    upper = {
        label: a["body"]
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    return spec, Lane(tuple({**upper, **LEG_SEEDS}), capture_path(capture))


#: Capture-derived address knee flexion (median over the address window),
#: measured by this change: lateral markers moved onto the joint centres, hip
#: centre from the functional (sphere-fit) hip calibration.
_CAPTURE_KNEES = {
    "driver": {"right": 26.2, "left": 28.6},
    "iron": {"right": 24.6, "left": 26.3},
}


@pytest.mark.parametrize("capture", ["driver", "iron"])
def test_capture_knees_are_measured_reliably_on_both_captures(capture: str) -> None:
    from src.shared.python.motion_matching.pipeline.address import (
        pelvis_marker_offsets,
    )

    spec, lane = _capture_inputs(capture)
    knees = address_knee.capture_knee_flexion(
        lane.points, lane.valid, lane.labels, pelvis_marker_offsets(spec)
    )
    for side in _SIDES:
        entry = knees[side]
        assert entry.reliable, entry.reason
        assert entry.role == ("lead" if side == "left" else "trail")
        assert entry.angle_deg == pytest.approx(_CAPTURE_KNEES[capture][side], abs=0.1)
        assert entry.to_receipt()["angle_deg"] == pytest.approx(entry.angle_deg)


@pytest.mark.parametrize("capture", ["driver", "iron"])
def test_build_knee_targets_per_mode(capture: str) -> None:
    from src.shared.python.motion_matching.pipeline.address import build_knee_targets

    spec, lane = _capture_inputs(capture)
    assert build_knee_targets(lane, spec, "off") is None
    prior = build_knee_targets(lane, spec, "prior")
    assert dict(prior.target_deg) == {"left": 15.0, "right": 15.0}
    assert dict(prior.sources) == {"left": "prior", "right": "prior"}
    chosen = build_knee_targets(lane, spec, "capture")
    assert dict(chosen.sources) == {"left": "capture", "right": "capture"}
    for side in _SIDES:
        assert chosen.target_deg[side] == pytest.approx(
            _CAPTURE_KNEES[capture][side], abs=0.1
        )
    with pytest.raises(ValueError, match="knee_flexion"):
        build_knee_targets(lane, spec, "bogus")


def test_build_knee_targets_without_waist_markers_uses_the_prior() -> None:
    from src.shared.python.motion_matching.pipeline.address import build_knee_targets

    lane = SimpleNamespace(
        points=np.zeros((10, 1, 3)), valid=np.ones((10, 1), bool), labels=("X",)
    )
    spec = {"marker_attachments": {}}
    targets = build_knee_targets(lane, spec, "capture")
    assert dict(targets.sources) == {"left": "prior", "right": "prior"}
    assert "unusable" in targets.notes


# ------------------------------------------------------------- model side


def _spec_kin(capture: str = "driver"):
    pytest.importorskip("mujoco")
    from src.shared.python.motion_matching.pipeline.constants import REPO_ROOT
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    path = REPO_ROOT / "docs/development/full_body_models" / _CANONICAL[capture]
    document = json.loads(path.read_text(encoding="utf-8"))
    return document, get_plant("mujoco", document).create_ik(dict(LEG_SEEDS))


def _prior_targets(**kw) -> KneeTargets:
    return KneeTargets(
        {"left": 15.0, "right": 15.0}, {"left": "prior", "right": "prior"}, **kw
    )


def test_model_knees_flex_with_a_negative_coordinate() -> None:
    from src.shared.python.motion_matching.pipeline.lane import document_seed

    document, kin = _spec_kin()
    q = document_seed(document, kin)
    for side in _SIDES:
        assert address_knee.flexion_sign(kin, q, side) == -1.0


@pytest.mark.parametrize("target", [5.0, 15.0, 28.6])
def test_coordinate_for_flexion_round_trips_through_forward_kinematics(
    target: float,
) -> None:
    from src.shared.python.motion_matching.pipeline.lane import document_seed

    document, kin = _spec_kin()
    q = document_seed(document, kin)
    names = list(kin.coordinate_order)
    for side in _SIDES:
        coord = address_knee.knee_coordinate_for_flexion(kin, q, side, target)
        assert coord <= 0.0
        moved = q.copy()
        moved[names.index(f"knee_angle_{side[0]}")] = coord
        model = address_knee.model_knee_flexion_deg(kin, moved)[side]
        assert model == pytest.approx(target, abs=0.05)


def test_hyperextension_bound_only_ever_tightens() -> None:
    from src.shared.python.motion_matching.pipeline.lane import document_seed

    document, kin = _spec_kin()
    q = document_seed(document, kin)
    wide = {"knee_angle_r": (-2.0, 0.17), "knee_angle_l": (-2.0, 0.17), "x": (0, 1)}
    tight = address_knee.no_hyperextension_bounds(kin, q, wide)
    for sfx in "rl":
        assert tight[f"knee_angle_{sfx}"] == (-2.0, 0.0)
    assert tight["x"] == (0, 1)
    already = {"knee_angle_r": (-2.0, -0.1), "knee_angle_l": (-1.0, -0.2)}
    assert address_knee.no_hyperextension_bounds(kin, q, already) == already


def test_hyperextension_is_never_produced_even_when_markers_ask_for_it() -> None:
    """Markers placed on a 9 deg hyperextended pose: the solve stops at straight."""
    from src.shared.python.motion_matching.pipeline.lane import document_seed
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    document, kin = _spec_kin()
    q = document_seed(document, kin)
    names = list(kin.coordinate_order)
    for sfx in "rl":
        q[names.index(f"knee_angle_{sfx}")] = np.radians(9.0)
    markers = kin.marker_positions(q)
    ground = get_plant("mujoco", document).ground_plane
    bounds = {
        f"knee_angle_{sfx}": (np.radians(-120.0), np.radians(10.0)) for sfx in "rl"
    }
    fit = kin.solve_pose(
        markers,
        np.ones(len(markers), bool),
        q,
        ground=ground,
        bounds=address_knee.no_hyperextension_bounds(kin, q, bounds),
        iterations=30,
    )
    for sfx in "rl":
        assert fit.q[names.index(f"knee_angle_{sfx}")] <= 1e-12
    address_knee.assert_no_hyperextension(kin, fit.q)
    with pytest.raises(ValueError, match="hyperextended"):
        address_knee.assert_no_hyperextension(kin, q)


def test_posture_prior_targets_the_coordinate_of_the_requested_flexion() -> None:
    from src.shared.python.motion_matching.pipeline.lane import document_seed

    document, kin = _spec_kin()
    q = document_seed(document, kin)
    prior = address_knee.knee_posture_prior(kin, q, _prior_targets(weight=2.0))
    assert set(prior) == {"knee_angle_l", "knee_angle_r"}
    for name, (coord, weight) in prior.items():
        assert weight == 2.0
        moved = q.copy()
        moved[list(kin.coordinate_order).index(name)] = coord
        side = "left" if name.endswith("_l") else "right"
        flex = address_knee.model_knee_flexion_deg(kin, moved)[side]
        assert flex == pytest.approx(15.0, abs=0.05)


# ------------------------------------------------- address solve (real lane)


@pytest.mark.parametrize("capture", ["driver", "iron"])
def test_prior_mode_address_knees_land_in_the_owner_band(
    capture: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """One address solve on the canonical capture frame 0 in ``prior`` mode.

    The same single-seed solve with ``knees`` off lands the trail knee on the
    legacy +10 deg hyperextension bound (8.7 deg geometric, both captures).
    """
    from src.shared.python.motion_matching.pipeline import address as address_mod
    from src.shared.python.motion_matching.pipeline.constants import (
        ADDRESS_SEEDS_DEG,
    )
    from src.shared.python.motion_matching.pipeline.lane import (
        configure_lane,
        document_seed,
    )
    from src.shared.python.motion_matching.pipeline.plant import get_plant

    pytest.importorskip("mujoco")
    spec, lane = _capture_inputs(capture)
    configure_lane(lane, spec)
    spec = address_mod._calibrate_ground_height(spec, lane.ground.height_m)
    upper = {
        label: (a["body"], tuple(a["offset_m"]))
        for label, a in spec["marker_attachments"].items()
        if a["offset_m"] is not None
    }
    lane.plant = get_plant("mujoco", spec)
    _, kin = lane.kinematics(spec, {**upper, **lane.leg_seeds()})
    lane.knees = address_mod.build_knee_targets(lane, spec, "prior")
    monkeypatch.setattr(address_mod, "ADDRESS_SEEDS_DEG", ADDRESS_SEEDS_DEG[3:4])
    monkeypatch.setattr(address_mod, "ADDRESS_RESTARTS", 0)
    fit = address_mod.best_address(lane, kin, document_seed(spec, kin))
    knees = address_knee.model_knee_flexion_deg(kin, fit.q)
    for side in _SIDES:
        assert abs(knees[side] - KNEE_FLEX_PRIOR_DEG) <= KNEE_FLEX_BAND_DEG, knees
    address_knee.assert_no_hyperextension(kin, fit.q)


# ---------------------------------------------- legacy modes stay unchanged


class _RecordingKin:
    """Records the keyword arguments of every ``solve_pose`` call."""

    coordinate_order = ("a", "b", "c", "d", "e", "f", "g")

    def __init__(self) -> None:
        self.calls: list[dict] = []

    def solve_pose(self, *args, **kwargs):  # noqa: ANN002, ANN003 - test double
        self.calls.append(kwargs)
        return SimpleNamespace(q=np.zeros(7), marker_rms_m=0.01)


def test_off_mode_address_solve_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    """``knees is None`` passes exactly the legacy solve arguments (no prior, no
    tightened bounds), so ``off`` and every pre-R4 path stay bit-identical."""
    from src.shared.python.motion_matching.pipeline import address as address_mod

    lane = SimpleNamespace(
        bounds={"knee_angle_r": (-2.0, 0.17)},
        anthropometric=False,
        points=np.zeros((1, 1, 3)),
        valid=np.ones((1, 1), bool),
        ground=None,
        stance=[()],
        marker_weights={},
        prior_weights={},
        feet=None,
        knees=None,
    )
    monkeypatch.setattr(address_mod, "ADDRESS_RESTARTS", 0)
    kin = _RecordingKin()
    address_mod.best_address(lane, kin, np.zeros(7))
    assert kin.calls
    for call in kin.calls:
        assert "posture_prior" not in call
        assert call["bounds"] == {"knee_angle_r": (-2.0, 0.17)}
    legacy = SimpleNamespace(**{k: v for k, v in vars(lane).items() if k != "knees"})
    kin2 = _RecordingKin()
    address_mod.best_address(legacy, kin2, np.zeros(7))
    assert kin2.calls == kin.calls


# ------------------------------------------------------------------ receipt


def test_report_records_model_target_and_source_per_knee() -> None:
    from src.shared.python.motion_matching.pipeline.lane import document_seed

    document, kin = _spec_kin()
    q = document_seed(document, kin)
    names = list(kin.coordinate_order)
    q[names.index("knee_angle_r")] = address_knee.knee_coordinate_for_flexion(
        kin, q, "right", 17.0
    )
    q[names.index("knee_angle_l")] = address_knee.knee_coordinate_for_flexion(
        kin, q, "left", 26.0
    )
    targets = KneeTargets(
        {"left": 15.0, "right": 26.2},
        {"left": "prior", "right": "capture"},
        "capture",
        weight=3.0,
    )
    block = address_knee.knee_flexion_report(kin, q, targets)
    assert block["enabled"] is True
    assert block["mode"] == "capture"
    assert block["prior_deg"] == KNEE_FLEX_PRIOR_DEG
    assert block["band_deg"] == KNEE_FLEX_BAND_DEG
    assert block["weight"] == 3.0
    right, left = block["knees"]["right"], block["knees"]["left"]
    assert right["target_source"] == "capture" and left["target_source"] == "prior"
    assert right["role"] == "trail" and left["role"] == "lead"
    assert right["model_deg"] == pytest.approx(17.0, abs=0.05)
    assert right["error_deg"] == pytest.approx(17.0 - 26.2, abs=0.05)
    assert right["within_band"] is False
    assert left["within_band"] is False  # 26 vs the 15 prior: 11 deg out
    assert right["hyperextended"] is False
    assert address_knee.knee_flexion_report(kin, q, None) == {"enabled": False}


def test_record_knee_flexion_only_writes_when_enabled() -> None:
    report: dict = {}
    address_knee.record_knee_flexion(report, None, None, None)
    assert report == {}


# ---------------------------------------------------------------------- CLI


def test_cli_knee_flexion_option_defaults_to_the_canonical_mode() -> None:
    from src.shared.python.motion_matching.pipeline import cli

    args = cli.build_parser().parse_args([])
    assert args.knee_flexion == CANONICAL_KNEE_FLEXION
    action = next(a for a in cli.build_parser()._actions if a.dest == "knee_flexion")
    assert tuple(action.choices) == MODES
    # a programmatic Namespace without the attribute keeps the legacy path
    assert address_knee.requested_mode(argparse.Namespace()) == "off"

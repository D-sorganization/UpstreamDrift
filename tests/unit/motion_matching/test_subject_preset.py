"""#15296: named subject presets for the anthropometric spec builder."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching import anthropometric_geometry as geometry
from src.shared.python.motion_matching import club_models
from src.shared.python.motion_matching.execution import assets, spec_builder
from src.shared.python.motion_matching.execution import marker_seed_overrides as seeds
from src.shared.python.motion_matching.execution import subject_preset as module

pytestmark = pytest.mark.unit

NAME = "tour_average_driver"


def _raw() -> dict[str, Any]:
    return json.loads(module.subject_preset_path(NAME).read_text(encoding="utf-8"))


def _joint(document: dict[str, Any], child: str) -> np.ndarray:
    joint = next(j for j in document["joints"] if j["child"].endswith("/" + child))
    return np.asarray(joint["parent_to_base"], dtype=float)[:3, 3]


@pytest.fixture(scope="module")
def built() -> spec_builder.AnthropometricDocument:
    kwargs = module.load_subject_preset(NAME).builder_kwargs()
    return spec_builder.compose_anthropometric_document(**kwargs)


def test_the_preset_states_every_setting_with_its_provenance() -> None:
    preset = module.load_subject_preset(NAME)
    assert (preset.name, preset.stature_m, preset.club) == (NAME, 1.82, "driver")
    assert preset.club_butt_to_sole_m == pytest.approx(45 * 0.0254)
    assert set(preset.provenance) == set(module.SETTINGS)
    assert all(text.strip() for text in preset.provenance.values())
    assert module.load_subject_preset(module.subject_preset_path(NAME)) == preset
    with pytest.raises(FileNotFoundError, match="no_such_preset"):
        module.load_subject_preset("no_such_preset")


def _set(**values: Any) -> Any:
    return lambda raw: raw["settings"].update(values)


def _seed(label: str, **values: Any) -> Any:
    entry = {"body": "Head", "offset_m": [0.0, 0.0, 0.1], "provenance": "test"}
    return lambda raw: raw["marker_seeds"].update({label: {**entry, **values}})


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (_set(stature_m=-1.0), "stature_m"),
        (_set(stature_m="tall"), "stature_m"),
        (_set(mass_kg=float("nan")), "mass_kg"),
        (_set(arm_scale=0.0), "arm_scale"),
        (_set(shoulder_below_cervicale_m=-0.01), "shoulder_below_cervicale_m"),
        (_set(club="putter"), "club"),
        (_set(club_butt_to_sole_m=0.01), "butt to sole"),
        (_set(club_shaft_elevation_deg=95.0), "club_shaft_elevation_deg"),
        (_set(club_markers=3), "club_markers"),
        (_set(hat_size=1), "unknown.*hat_size"),
        (lambda raw: raw["settings"].pop("mass_kg"), "missing.*mass_kg"),
        (lambda raw: raw.update(schema="subject-preset/0"), "schema"),
        (lambda raw: raw["provenance"].update(mass_kg=" "), "provenance.*mass_kg"),
        (_seed("Nose"), "unknown marker seed.*Nose"),
        (_seed("Marker_2:2:1"), "unknown marker seed.*Marker_2:2:1"),
        (_seed("HeadTop", offset_m=[0.0, float("nan"), 0.1]), "offset_m"),
        (_seed("HeadTop", offset_m=[0.0, 0.1]), "offset_m"),
        (_seed("HeadTop", provenance=" "), "provenance.*HeadTop"),
        (_seed("HeadTop", anchor=""), "anchor.*HeadTop"),
        (_seed("HeadTop", offset_mm=[1, 2, 3]), "HeadTop.*offset_mm"),
        (lambda raw: raw["dropped_markers"].update(Nose="x"), "dropped.*Nose"),
        (lambda raw: raw["dropped_markers"].update(HeadTop="x"), "both.*HeadTop"),
        (lambda raw: raw["dropped_markers"].update({"Uname*36": ""}), "provenance"),
    ],
)
def test_a_bad_preset_file_is_refused(
    tmp_path: Path, change: Any, message: str
) -> None:
    raw = _raw()
    change(raw)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    with pytest.raises((ValueError, TypeError), match=message):
        module.load_subject_preset(path)


def test_the_preset_body_has_the_stated_joint_to_joint_lengths(built: Any) -> None:
    document = built.document
    upper_arm = -_joint(document, "Spherical Solid")[2]
    forearm = -2 * _joint(document, "LLowerForearm")[2]
    half_spacing = _joint(document, "LUpperArm")[1]
    neck, hub = _joint(document, "Head")[2], _joint(document, "HubtoLS")[2]
    mid_hip_to_neck = _joint(document, "UpperTorsoBase")[2] + neck
    assert upper_arm == pytest.approx(0.297, abs=0.001)
    assert forearm == pytest.approx(0.283, abs=0.001)
    assert 0.178 <= half_spacing <= 0.182
    assert _joint(document, "RUpperArm")[1] == pytest.approx(-half_spacing)
    assert neck - hub == pytest.approx(0.095)
    assert 0.650 <= mid_hip_to_neck <= 0.665 + 1e-4
    subject = document["subject"]
    assert subject["shoulder_below_cervicale_m"] == 0.095
    assert subject["subject_preset"] == NAME
    assert (subject["stature_m"], subject["mass_kg"]) == (1.82, 78.0)


def test_the_preset_seeds_follow_the_placement_review(built: Any) -> None:
    document, raw = built.document, _raw()
    at = {
        label: np.array(a["offset_m"])
        for label, a in document["marker_attachments"].items()
    }
    assert set(raw["marker_seeds"]) == set(seeds.SEEDABLE_LABELS)
    assert at["HeadFront"] - at["HeadTop"] == pytest.approx([0.142, 0.0, -0.065])
    assert at["HeadSide"] - at["HeadTop"] == pytest.approx([-0.009, 0.088, -0.100])
    assert at["WaistLeft"][1] == -at["WaistRight"][1] == 0.1725  # belt, not ASIS
    assert at["WaistLBack"][1] == -at["WaistRBack"][1] == 0.0845
    assert at["WaistLBack"][0] - at["WaistLeft"][0] == pytest.approx(-0.147)
    neck, hub = _joint(document, "Head")[2], _joint(document, "HubtoLS")[2]
    assert at["BackTop"][2] == pytest.approx(neck - 0.030)  # not hub + 0.02
    assert at["BackTop"][2] - (hub + 0.02) == pytest.approx(0.045)
    assert at["BackLeft"] - at["BackTop"] == pytest.approx([-0.010, 0.090, -0.180])
    assert at["BackRight"][1] == -at["BackLeft"][1]
    elbow = _joint(document, "Spherical Solid")
    assert at["LElbowOut"] == pytest.approx(elbow + [0.0, 0.040, 0.0])
    assert at["LUArmHigh"][2] / elbow[2] == pytest.approx(0.60, abs=0.002)
    wrist = _joint(document, "Club/Clubface Vector")
    assert at["LWristTop"] == pytest.approx(wrist + [0.028, 0.0, 0.025])
    assert at["RWristTop"] == pytest.approx(at["LWristTop"])
    assert abs(at["LShoulderTop"][1]) == 0.165 and at["RShoulderBack"][1] == -0.150
    assert at["RToeOut"] == pytest.approx([0.175, 0.010, 0.065])  # on calcn_r
    assert at["RKneeOut"][2] - at["LKneeOut"][2] == pytest.approx(0.110, abs=0.006)
    dropped = ["Marker_0:0:0", "Uname*36", "Uname*37", "Uname*38"]
    assert document["subject"]["dropped_markers"] == dropped
    assert not set(dropped) & set(at)
    preset = module.load_subject_preset(NAME)
    assert all(s["provenance"].strip() for s in preset.marker_seeds.values())


def test_seed_overrides_anchor_to_a_joint_and_drop_markers() -> None:
    base = {
        label: spec_builder.MarkerAttachment("Hip", (0.0, 0.0, 0.0))
        for label in seeds.SEEDABLE_LABELS
    }
    assert seeds.apply_marker_seed_overrides(base, None, None, [], []) is base
    frames = [{"name": "Spine", "body": "rod", "placement": np.eye(4).tolist()}]
    joint = {"parent": "rod", "child": "skull", "parent_to_base": np.eye(4)}
    joint["parent_to_base"][:3, 3] = [0.0, 0.0, 0.5]
    override = {"BackTop": {"body": "Spine", "offset_m": [-0.1, 0, -0.03]}}
    plain = seeds.apply_marker_seed_overrides(base, override, None, [joint], frames)
    assert plain["BackTop"].offset_m == (-0.1, 0.0, -0.03)
    override["BackTop"]["anchor"] = "skull"
    out = seeds.apply_marker_seed_overrides(
        base, override, ["HeadTop", "Uname*36"], [joint], frames
    )
    assert out["BackTop"] == spec_builder.MarkerAttachment("Spine", (-0.1, 0.0, 0.47))
    assert set(base) - set(out) == {"HeadTop"} and base["BackTop"].body == "Hip"
    with pytest.raises(ValueError, match="no joint from 'Spine' to anchor 'skull'"):
        seeds.apply_marker_seed_overrides(base, override, None, [], frames)
    with pytest.raises(ValueError, match="both.*BackTop"):
        seeds.apply_marker_seed_overrides(base, override, ["BackTop"], [], [])
    with pytest.raises(ValueError, match="unknown marker seed"):
        seeds.apply_marker_seed_overrides(base, {"Nose": {}}, None, [], [])


def test_the_preset_club_is_45_inches_butt_to_sole(built: Any) -> None:
    preset = module.load_subject_preset(NAME)
    club = preset.club_spec()
    lie = (preset.club_shaft_elevation_deg, preset.club_head_roll_deg)
    offset = club_models.sole_offset_m(club_models.DRIVER, *lie)
    assert offset == pytest.approx(0.0637, abs=2e-4)  # club-only registration
    assert club.length_m + offset == pytest.approx(1.143)
    assert club.length_m == pytest.approx(1.0793, abs=2e-4)
    recorded = built.document["club"]
    assert recorded["length_m"] == club.length_m
    assert recorded["butt_to_sole_m"] == pytest.approx(1.143)
    assert recorded["wrist_to_head_m"] == pytest.approx(club.length_m - 0.032)
    solids = [s for b in built.document["bodies"] for s in b["solids"]]
    grip = next(s for s in solids if s["name"].endswith("Clubface Vector/Grip"))
    butt_y = np.asarray(grip["placement"])[1, 3] - club.grip_length_m / 2
    assert -butt_y == pytest.approx(club.length_m)  # head centre is the origin
    assert club_models.DRIVER.length_m == 1.156
    assert club == club_models.with_butt_to_sole_length(
        club_models.DRIVER, 1.143, offset
    )
    on_shaft = built.document["marker_attachments"]["Marker_3:3:1"]["offset_m"]
    assert on_shaft[1:] == pytest.approx([0.0, 0.0], abs=1e-12)  # club-only file


def test_the_sole_offset_follows_the_head_ellipsoid() -> None:
    driver = club_models.DRIVER
    half_x, half_y, half_z = driver.head_half_size_m
    assert club_models.sole_offset_m(driver, 90.0, 0.0) == pytest.approx(half_y)
    flat = club_models.sole_offset_m(driver, 30.0, 0.0)
    expected = math.hypot(half_y * 0.5, half_z * math.cos(math.radians(30.0))) / 0.5
    assert flat == pytest.approx(expected)
    for elevation in (0.0, 91.0, float("nan")):
        with pytest.raises(ValueError, match="elevation"):
            club_models.sole_offset_m(driver, elevation, 0.0)
    with pytest.raises(ValueError, match="butt to sole"):
        club_models.with_butt_to_sole_length(driver, 0.2, 0.0637)


def test_builder_settings_are_validated() -> None:
    with pytest.raises(ValueError, match="club_spec"):
        spec_builder.compose_anthropometric_document(
            club="iron7", club_spec=club_models.DRIVER
        )
    native = json.loads(assets.get_native_geometry_spec(None).read_text())
    for drop in (-0.01, float("inf")):
        with pytest.raises(ValueError, match="shoulder_below_cervicale_m"):
            geometry.build_upper_body(
                native, stature_m=1.8, mass_kg=80.0, shoulder_below_cervicale_m=drop
            )


def test_the_default_build_does_not_change() -> None:
    """Explicit defaults equal no settings; no preset key reaches the document."""
    plain = spec_builder.compose_anthropometric_document().document
    explicit = spec_builder.compose_anthropometric_document(
        shoulder_below_cervicale_m=geometry.SHOULDER_BELOW_CERVICALE_M,
        club_spec=club_models.CLUBS["driver"],
        subject_preset=None,
        marker_seed_overrides=None,
        dropped_markers=None,
    ).document
    assert json.dumps(plain, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    assert "butt_to_sole_m" not in plain["club"]
    assert not {"shoulder_below_cervicale_m", "subject_preset"} & set(plain["subject"])
    assert "dropped_markers" not in plain["subject"]
    assert plain["marker_attachments"]["LToeIn"]["offset_m"] is None
    assert geometry.SHOULDER_BELOW_CERVICALE_M == 0.04


def test_the_command_line_builds_from_the_preset(tmp_path: Path) -> None:
    def run(out: Path, *flags: str) -> dict[str, Any]:
        paths = (
            ("--native", assets.get_native_geometry_spec(None)),
            ("--osim", assets.get_opensim_model(None)),
            ("--native-candidate", assets.get_candidate_geometry_spec(None)),
            ("--output", out),
        )
        argv = [str(v) for pair in paths for v in pair]
        assert spec_builder.main([*argv, *flags]) == 0
        return json.loads((out / "full_body_spec_anthro_driver.json").read_text())

    document = run(tmp_path / "a", "--subject-preset", NAME)
    assert document["subject"]["stature_m"] == 1.82
    assert document["club"]["butt_to_sole_m"] == pytest.approx(1.143)
    assert "driver_club_only.json" in document["provenance"]
    override = run(tmp_path / "b", "--subject-preset", NAME, "--stature", "1.75")
    assert override["subject"]["stature_m"] == 1.75  # explicit flags win
    with pytest.raises(SystemExit):  # no preset: stature and mass stay required
        run(tmp_path / "c", "--mass", "80")

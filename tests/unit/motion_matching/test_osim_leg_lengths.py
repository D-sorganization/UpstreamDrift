"""#15296: thigh, shank and hip spacing as settings of the spec builder."""

from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.shared.python.motion_matching.execution import assets, spec_builder
from src.shared.python.motion_matching.execution import osim_leg_lengths as legs
from src.shared.python.motion_matching.execution import subject_preset as presets

pytestmark = pytest.mark.unit

NAME = "tour_average_driver"
SIDES = ("r", "l")


def _raw() -> dict[str, Any]:
    return json.loads(presets.subject_preset_path(NAME).read_text(encoding="utf-8"))


def _load(tmp_path: Path, raw: dict[str, Any]) -> presets.SubjectPreset:
    path = tmp_path / "preset.json"
    path.write_text(json.dumps(raw), encoding="utf-8")
    return presets.load_subject_preset(path)


def _document_lengths(document: dict[str, Any]) -> dict[str, float]:
    """Joint centre distances of a built document, m, by ``<segment>_<side>``."""
    joints = {j["name"]: j for j in document["joints"]}

    def base(name: str) -> np.ndarray:
        return np.asarray(joints[name]["parent_to_base"], dtype=float)[:3, 3]

    def follower(name: str) -> np.ndarray:
        return np.asarray(joints[name]["child_to_follower"], dtype=float)[:3, 3]

    out = {"hip_spacing": float(np.linalg.norm(base("hip_l") - base("hip_r")))}
    for side in SIDES:
        thigh = base(f"knee_{side}") - follower(f"hip_{side}")
        shank = base(f"ankle_{side}") - follower(f"knee_{side}")
        out[f"thigh_{side}"] = float(np.linalg.norm(thigh))
        out[f"shank_{side}"] = float(np.linalg.norm(shank))
    return out


@pytest.fixture(scope="module")
def osim() -> tuple[dict[str, Any], dict[str, Any]]:
    return spec_builder.read_osim(assets.get_opensim_model(None))


def test_the_asset_legs_are_stored_lengths_that_differ_left_to_right(
    osim: Any,
) -> None:
    """Why leg length does not follow stature: it is read from the model file."""
    stored = legs.leg_lengths_m(osim[1])
    assert stored["thigh_r"] == pytest.approx(0.4051, abs=1e-4)
    assert stored["thigh_l"] == pytest.approx(0.3933, abs=1e-4)
    assert stored["shank_r"] == pytest.approx(0.4346, abs=1e-4)
    assert stored["shank_l"] == pytest.approx(0.4306, abs=1e-4)


def test_scaling_gives_both_legs_the_stated_lengths(osim: Any) -> None:
    bodies, joints = osim
    before = copy.deepcopy((bodies, joints))
    stored = legs.leg_lengths_m(joints)
    scaled_bodies, scaled_joints = legs.scale_leg_lengths(
        bodies, joints, thigh_length_m=0.44, shank_length_m=0.43
    )
    after = legs.leg_lengths_m(scaled_joints)
    for side in SIDES:
        assert after[f"thigh_{side}"] == pytest.approx(0.44)
        assert after[f"shank_{side}"] == pytest.approx(0.43)
        for segment, target in (("femur", "thigh"), ("tibia", "shank")):
            name = f"{segment}_{side}"
            factor = after[f"{target}_{side}"] / stored[f"{target}_{side}"]
            old, new = bodies[name], scaled_bodies[name]
            assert new["mass"] == old["mass"]
            assert new["com"] == pytest.approx(np.asarray(old["com"]) * factor)
            assert new["inertia"] == pytest.approx(
                np.asarray(old["inertia"]) * factor**2
            )
        assert scaled_bodies[f"calcn_{side}"] is bodies[f"calcn_{side}"]
    for name in ("mass", "com", "inertia"):  # the inputs are not modified
        assert np.array_equal(bodies["femur_r"][name], before[0]["femur_r"][name])
    assert np.array_equal(
        joints["walker_knee_l"]["parent_translation"],
        before[1]["walker_knee_l"]["parent_translation"],
    )


def test_no_setting_returns_the_stored_legs(osim: Any) -> None:
    bodies, joints = osim
    assert legs.scale_leg_lengths(bodies, joints) == (bodies, joints)
    only_thigh = legs.leg_lengths_m(
        legs.scale_leg_lengths(bodies, joints, thigh_length_m=0.42)[1]
    )
    assert only_thigh["thigh_l"] == pytest.approx(0.42)
    assert only_thigh["shank_l"] == legs.leg_lengths_m(joints)["shank_l"]


@pytest.mark.parametrize("value", [0.0, -0.4, float("nan"), float("inf")])
def test_a_length_must_be_positive_and_finite(osim: Any, value: float) -> None:
    for key in ("thigh_length_m", "shank_length_m"):
        with pytest.raises(ValueError, match=key):
            legs.scale_leg_lengths(*osim, **{key: value})
    with pytest.raises(ValueError, match="hip_half_spacing_m"):
        spec_builder.compose_anthropometric_document(hip_half_spacing_m=value)


@pytest.mark.parametrize("value", ["0.44", True, [0.44]])
def test_a_length_must_be_a_number(osim: Any, value: Any) -> None:
    with pytest.raises(TypeError, match="thigh_length_m"):
        legs.scale_leg_lengths(*osim, thigh_length_m=value)


def test_the_default_build_does_not_change() -> None:
    plain = spec_builder.compose_anthropometric_document().document
    explicit = spec_builder.compose_anthropometric_document(
        thigh_length_m=None, shank_length_m=None, hip_half_spacing_m=None
    ).document
    assert json.dumps(plain, sort_keys=True) == json.dumps(explicit, sort_keys=True)
    lengths = _document_lengths(plain)
    assert lengths["thigh_r"] - lengths["thigh_l"] == pytest.approx(0.0118, abs=1e-4)
    assert "scaled" not in plain["lower_limb_provenance"]


def test_the_preset_legs_are_the_stated_lengths_and_symmetric() -> None:
    preset = presets.load_subject_preset(NAME)
    document = spec_builder.compose_anthropometric_document(
        **preset.builder_kwargs()
    ).document
    lengths = _document_lengths(document)
    assert preset.thigh_length_m is not None and preset.shank_length_m is not None
    assert preset.hip_half_spacing_m is not None
    for side in SIDES:
        assert lengths[f"thigh_{side}"] == pytest.approx(preset.thigh_length_m)
        assert lengths[f"shank_{side}"] == pytest.approx(preset.shank_length_m)
    assert lengths["hip_spacing"] == pytest.approx(2 * preset.hip_half_spacing_m)
    note = f"thigh scaled to {preset.thigh_length_m:.4f} m on both sides"
    assert note in document["lower_limb_provenance"]
    assert set(presets.OPTIONAL) <= set(preset.provenance)


def test_a_preset_may_leave_the_leg_settings_out(tmp_path: Path) -> None:
    raw = _raw()
    for key in presets.OPTIONAL:
        raw["settings"].pop(key, None)
        raw["provenance"].pop(key, None)
    preset = _load(tmp_path, raw)
    kwargs = preset.builder_kwargs()
    assert all(getattr(preset, key) is None for key in presets.OPTIONAL)
    assert all(kwargs[key] is None for key in presets.OPTIONAL)
    assert all(preset.provenance[key] == "not set" for key in presets.OPTIONAL)


@pytest.mark.parametrize(
    ("change", "message"),
    [
        (lambda raw: raw["settings"].update(thigh_length_m=-0.4), "thigh_length_m"),
        (lambda raw: raw["settings"].update(shank_length_m="long"), "shank_length_m"),
        (lambda raw: raw["settings"].update(hip_half_spacing_m=0.0), "hip_half"),
        (lambda raw: raw["provenance"].pop("thigh_length_m"), "provenance.*thigh"),
    ],
)
def test_a_bad_leg_setting_is_refused(
    tmp_path: Path, change: Any, message: str
) -> None:
    raw = _raw()
    change(raw)
    with pytest.raises((ValueError, TypeError), match=message):
        _load(tmp_path, raw)

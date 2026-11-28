"""Sourced current-era golfer appearance catalog contracts."""

import pytest

from src.shared.python.model_appearance.player_models import (
    player_appearance,
    player_profile,
)
from src.shared.python.model_appearance.schema import document_to_dict

pytestmark = [pytest.mark.unit, pytest.mark.timeout(240)]


@pytest.mark.parametrize(
    ("player", "height_m", "mass_kg"),
    [
        ("rory_mcilroy", 1.778, 72.5747792),
        ("jon_rahm", 1.8796, 99.7903214),
        ("dustin_johnson", 1.9304, 86.1825503),
        ("sergio_garcia", 1.778, 81.6466266),
    ],
)
def test_modern_profiles_are_sourced_and_valid(player, height_m, mass_kg):
    """Preserve exact listed-unit conversions and an explicit visual document."""
    profile = player_profile(player)
    assert (profile["stature_m"], profile["mass_kg"]) == (height_m, mass_kg)
    assert any("pgatour.com" in x["url"] for x in profile["sources"])
    assert any("youtube.com" in x["url"] for x in profile["sources"])
    assert profile["limitations"]
    assert (
        document_to_dict(player_appearance(player))["schema_version"] == "appearance-v1"
    )


@pytest.mark.parametrize(
    ("player", "height_m", "mass_kg", "source_values", "raw_scale", "value"),
    [
        (
            "rory_mcilroy",
            1.778,
            72.5747792,
            "5 ft 10 in and 160 lb",
            0.9097919523,
            -0.5011558205,
        ),
        (
            "jon_rahm",
            1.8796,
            99.7903214,
            "6 ft 2 in and 220 lb",
            0.9815059132,
            -0.1027449266,
        ),
        (
            "dustin_johnson",
            1.9304,
            86.1825503,
            "6 ft 4 in and 190 lb",
            0.8763661041,
            -0.6868549774,
        ),
        (
            "sergio_garcia",
            1.778,
            81.6466266,
            "5 ft 10 in and 180 lb",
            0.9649800884,
            -0.1945550643,
        ),
    ],
)
def test_modern_physical_frame_prior_selects_the_dimension_source(
    player, height_m, mass_kg, source_values, raw_scale, value
):
    """Bind a physical frame assumption to the selected, era-qualified source."""
    from src.shared.python.model_appearance.player_models import frame_prior

    profile = player_profile(player)
    assert (profile["stature_m"], profile["mass_kg"]) == (height_m, mass_kg)
    prior = profile["frame_prior"]
    source = profile["sources"][prior["source_index"]]
    assert prior["method"] == "height_mass"
    assert prior["source_index"] == 0
    assert source["url"].startswith("https://www.pgatour.com/player/")
    assert source_values in source["values"]
    assert "measurement date" in source["era"].lower()
    assert any("not measured girth" in item for item in profile["limitations"])
    resolved = frame_prior(profile)
    assert resolved["source"] == source
    assert resolved["raw_transverse_scale"] == pytest.approx(raw_scale)
    assert resolved["value"] == pytest.approx(value)


def test_modern_physical_prior_ignores_visual_build_setting(monkeypatch):
    """A visual body-build choice must not select the segment inertia prior."""
    from copy import deepcopy

    from src.shared.python.model_appearance import player_models
    from src.shared.python.model_appearance import player_stature

    seen = []

    def fake_builder(height, mass, club, *, frame_build):
        """Capture the selected physical frame input without rebuilding anatomy."""
        seen.append((height, mass, frame_build))
        return {"subject": {}}, {}

    monkeypatch.setattr(player_stature, "build_stature_calibrated", fake_builder)
    for player in ("rory_mcilroy", "jon_rahm", "dustin_johnson", "sergio_garcia"):
        profile = deepcopy(player_models.player_profile(player))
        profile["appearance"]["body_build"] = 0.8
        monkeypatch.setattr(
            player_models, "player_profile", lambda _player, profile=profile: profile
        )
        expected = player_models.frame_prior(profile)["value"]
        player_models.build_player_spec(player)
        assert seen[-1] == (
            profile["stature_m"],
            profile["mass_kg"],
            expected,
        )


def test_modern_frame_candidates_preserve_mass_com_and_native_realizability():
    """Keep source-derived inertia candidates distinct and physically admissible."""
    import json

    import numpy as np

    from src.engines.physics_engines.mujoco.python.full_body_mjcf import (
        export_full_body_mjcf,
    )
    from src.shared.python.model_appearance.player_frame_inertia import (
        LIMB_SOLIDS,
        apply_frame_inertia,
    )
    from src.shared.python.model_appearance.player_models import frame_prior
    from src.shared.python.model_appearance.player_stature import (
        build_stature_calibrated,
    )

    mujoco = pytest.importorskip("mujoco")
    base = build_stature_calibrated(1.8, 80.0)[0]
    base_bodies = {body["name"]: body for body in base["bodies"]}
    transformed = []
    for player in ("rory_mcilroy", "jon_rahm", "dustin_johnson", "sergio_garcia"):
        resolved = frame_prior(player_profile(player))
        value = resolved["value"]
        candidate = apply_frame_inertia(base, value)
        bodies = {body["name"]: body for body in candidate["bodies"]}
        for body_name in LIMB_SOLIDS:
            old = base_bodies[body_name]["solids"]
            new = bodies[body_name]["solids"]
            assert len(old) == len(new)
            for before, after in zip(old, new, strict=True):
                assert after["mass_kg"] == before["mass_kg"]
                assert after["com_m"] == before["com_m"]
        inertias = tuple(
            tuple(np.asarray(bodies[name]["solids"][0]["inertia_com_kg_m2"]).flat)
            for name in LIMB_SOLIDS
        )
        transformed.append(inertias)
        model = mujoco.MjModel.from_xml_string(
            export_full_body_mjcf(json.dumps(candidate).encode())[0]
        )
        state = mujoco.MjData(model)
        mujoco.mj_forward(model, state)
        dense = np.empty((model.nv, model.nv))
        mujoco.mj_fullM(model, dense, state.qM)
        assert np.linalg.eigvalsh(dense).min() > 0
        assert resolved["source"] == player_profile(player)["sources"][0]
    assert len(set(transformed)) == 4


def test_modern_appearance_export_keeps_reference_physics_exact(tmp_path, monkeypatch):
    """The sourced frame prior is ignored when exporting a skin on fitted physics."""
    import json

    from src.shared.python.model_appearance import player_models
    from src.shared.python.model_appearance.schema import physics_spec_sha256
    from src.shared.python.motion_matching.execution.golfer_reference_specs import (
        reference_spec,
    )
    from src.engines.physics_engines.mujoco.python import full_body_mjcf

    spec = reference_spec("driver")
    source = tmp_path / "fitted.json"
    source.write_text(json.dumps(spec))
    monkeypatch.setattr(
        full_body_mjcf,
        "export_full_body_mjcf",
        lambda *_args, **_kwargs: ("<mujoco/>", {}),
    )
    player_models.export_player("jon_rahm", tmp_path / "rahm", base_spec=source)
    exported = json.loads((tmp_path / "rahm" / "model.json").read_text())
    receipt = json.loads((tmp_path / "rahm" / "receipt.json").read_text())
    assert exported == spec
    assert receipt["physics_spec_sha256"] == physics_spec_sha256(spec)

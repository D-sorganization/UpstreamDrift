"""Frame priors are explicit model assumptions, independent of skin styling."""

import copy
from typing import Any

import pytest

from src.shared.python.model_appearance import player_models

pytestmark = pytest.mark.unit


def _profile():
    """Create a minimal sourced player profile for prior calculations."""
    return {
        "stature_m": 1.71,
        "mass_kg": 78.0,
        "sources": [{"url": "https://example.org/bio", "era": "undated"}],
        "appearance": {"body_build": 0.8},
    }


def test_unknown_and_geometric_similarity():
    """Distinguish unknown from a specified baseline and preserve cubic scaling."""
    profile = _profile()
    assert player_models.frame_prior(profile)["method"] == "unknown"
    profile["frame_prior"] = {"method": "height_mass", "source_index": 0}
    expected = player_models.frame_prior(profile)
    assert expected["value"] == pytest.approx(0.0)
    assert expected["raw_transverse_scale"] == pytest.approx(1.0)
    assert expected["source"]["url"] == "https://example.org/bio"
    profile["stature_m"] *= 2
    profile["mass_kg"] *= 8
    assert player_models.frame_prior(profile)["value"] == pytest.approx(0.0)


def test_proxy_clamps_and_rejects_invalid_inputs():
    """Clamp finite proxy values and reject invalid inputs and missing provenance."""
    p = _profile()
    p["frame_prior"] = {"method": "height_mass", "source_index": 0}
    p["mass_kg"] = 200
    prior = player_models.frame_prior(p)
    assert prior["value"] == 1
    assert prior["saturated"]
    for bad in (0, -1, float("nan"), float("inf"), True):
        p["mass_kg"] = bad
        with pytest.raises(ValueError):
            player_models.frame_prior(p)
    p = _profile()
    p["frame_prior"] = {"method": "height_mass"}
    with pytest.raises(ValueError):
        player_models.frame_prior(p)


def test_override_requires_reason_and_distinguishes_zero():
    """Treat explicit zero as an override and require a meaningful rationale."""
    p = _profile()
    p["frame_prior"] = {
        "method": "override",
        "value": 0.0,
        "rationale": "reference comparison",
    }
    prior = player_models.frame_prior(p)
    assert prior["method"] == "override" and prior["value"] == 0
    for value, reason in ((1.01, "test"), (True, "test"), (0, "  ")):
        p["frame_prior"] = {"method": "override", "value": value, "rationale": reason}
        with pytest.raises(ValueError):
            player_models.frame_prior(p)


def test_physical_prior_is_independent_of_visual_build(monkeypatch):
    """Changing skin breadth does not change the sized candidate's frame argument."""
    from src.shared.python.model_appearance import player_stature

    profile = _profile()
    values = []
    monkeypatch.setattr(
        player_models, "player_profile", lambda _: copy.deepcopy(profile)
    )

    def fake_builder(height, mass, club, *, frame_build):
        """Capture the physical prior without a heavyweight model build."""
        values.append(frame_build)
        return {"subject": {}}, {}

    monkeypatch.setattr(player_stature, "build_stature_calibrated", fake_builder)
    player_models.build_player_spec("sample")
    profile["appearance"]["body_build"] = -0.7
    player_models.build_player_spec("sample")
    assert values == [0, 0]
    profile["frame_prior"] = {
        "method": "override",
        "value": 0.3,
        "rationale": "test assumption",
    }
    player_models.build_player_spec("sample")
    assert values[-1] == 0.3


def test_sized_receipt_reports_prior_and_reference_skin_does_not(tmp_path, monkeypatch):
    """Record provenance on sized exports without modifying an existing plant."""
    import json

    from src.engines.physics_engines.mujoco.python import full_body_mjcf

    profile = player_models.player_profile("tiger_woods")
    profile["frame_prior"] = {"method": "height_mass", "source_index": 0}
    spec: dict[str, Any] = {"subject": {}, "coordinate_order": []}
    source = tmp_path / "base.json"
    source.write_text(json.dumps(spec))
    monkeypatch.setattr(
        player_models, "player_profile", lambda _: copy.deepcopy(profile)
    )
    monkeypatch.setattr(
        player_models, "build_player_spec", lambda *_: copy.deepcopy(spec)
    )
    monkeypatch.setattr(
        full_body_mjcf,
        "export_full_body_mjcf",
        lambda *args, **kwargs: ("<mujoco/>", {}),
    )
    sized = player_models.export_player("test", tmp_path / "sized")
    sized_receipt = json.loads((sized.parent / "receipt.json").read_text())
    assert sized_receipt["frame_prior"]["method"] == "height_mass"
    assert sized_receipt["frame_prior"]["source"] == profile["sources"][0]
    skin = player_models.export_player("test", tmp_path / "skin", base_spec=source)
    assert json.loads((skin.parent / "model.json").read_text()) == spec
    assert "frame_prior" not in json.loads((skin.parent / "receipt.json").read_text())

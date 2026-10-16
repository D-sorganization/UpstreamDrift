"""Regression for UD#13015: the resolver missed the production full-body golfer."""

from __future__ import annotations

import logging
from pathlib import Path

import mujoco
import pytest
from mujoco_humanoid_golf.biomechanics import BiomechanicalAnalyzer
from mujoco_humanoid_golf.club_body_names import find_club_head_body_id

from src.engines.physics_engines.mujoco.python import full_body_mjcf as exporter
from src.shared.python.club_body_names import (
    CLUB_HEAD_BODY_ALIASES,
    SPEC_CLUB_HEAD_BODY_NAME,
    find_club_head_index,
)

_SPEC = (
    Path(__file__).resolve().parents[3].parent
    / "docs/development/full_body_models/evidence/ground_support/anthro_driver"
    / "full_body_spec_hipcal_scaled.json"
)

_NAMING_SCHEMES = [
    pytest.param("clubhead", id="mujoco-swing"),
    pytest.param("Clubhead", id="opensim-pinocchio"),
    pytest.param(SPEC_CLUB_HEAD_BODY_NAME, id="mujoco-full-body"),
    pytest.param("club_head", id="mujoco-generated"),
]


def _production_model(*, visual: bool) -> mujoco.MjModel:
    xml, _ = exporter.export_full_body_mjcf(_SPEC.read_bytes(), visual=visual)
    return mujoco.MjModel.from_xml_string(xml)


@pytest.mark.unit
@pytest.mark.parametrize("visual", [False, True])
def test_resolver_finds_club_on_production_full_body_mjcf(visual: bool) -> None:
    model = _production_model(visual=visual)
    bid = find_club_head_body_id(model, owner="probe", strict=True)
    assert bid is not None
    assert (
        mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
        == SPEC_CLUB_HEAD_BODY_NAME
    )


@pytest.mark.unit
def test_analyzer_has_club_head_on_production_model() -> None:
    model = _production_model(visual=False)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    analyzer = BiomechanicalAnalyzer(model, data)
    assert analyzer.club_head_id is not None
    pos, _, _ = analyzer.get_club_head_data()
    assert pos is not None


@pytest.mark.unit
@pytest.mark.parametrize("name", _NAMING_SCHEMES)
def test_every_engine_naming_scheme_resolves(name: str) -> None:
    assert find_club_head_index(["world", "arm", name], strict=True) == 2


@pytest.mark.unit
def test_alias_table_contains_spec_name() -> None:
    assert SPEC_CLUB_HEAD_BODY_NAME in CLUB_HEAD_BODY_ALIASES


@pytest.mark.unit
def test_strict_raises_listing_candidates() -> None:
    with pytest.raises(ValueError, match="elbow") as exc:
        find_club_head_index(["world", "elbow"], owner="T", strict=True)
    assert "clubhead" in str(exc.value)


@pytest.mark.unit
def test_non_strict_warns_and_returns_none(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        assert find_club_head_index(["world", "elbow"], owner="T") is None
    assert "no club-head body found" in caplog.text


@pytest.mark.unit
def test_strict_analyzer_raises_without_club() -> None:
    model = mujoco.MjModel.from_xml_string(
        "<mujoco><worldbody><body name='b'><geom size='.1'/></body></worldbody></mujoco>"
    )
    with pytest.raises(ValueError, match="no club-head body found"):
        find_club_head_body_id(model, strict=True)


_CLUBLESS = (
    "<mujoco><worldbody><body name='b'><geom size='.1'/></body></worldbody></mujoco>"
)


@pytest.mark.unit
def test_analyzer_require_club_head_raises_on_clubless_model() -> None:
    model = mujoco.MjModel.from_xml_string(_CLUBLESS)
    data = mujoco.MjData(model)
    with pytest.raises(ValueError, match="no club-head body found"):
        BiomechanicalAnalyzer(model, data, require_club_head=True)
    assert BiomechanicalAnalyzer(model, data).club_head_id is None

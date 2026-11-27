"""Where the drawn club head sits along the shaft (#15150)."""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.model_appearance import club_assembly as ca
from src.shared.python.model_appearance import club_head_seat as seat
from src.shared.python.model_appearance.club_chain_meshes import (
    ChainLink,
    ShaftChainLayout,
    shaft_piece_meshes,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
SPEC = ROOT / "docs/development/full_body_models/full_body_spec_anthro_driver.json"
SOLE_OFFSET_M = 0.0637


def _spec(**club: object) -> dict:
    spec = json.loads(SPEC.read_text(encoding="utf-8"))
    spec["club"] = {**spec["club"], **club}
    return spec


def _stated(spec: dict) -> dict:
    length = float(spec["club"]["length_m"])
    spec["club"]["butt_to_sole_m"] = length + SOLE_OFFSET_M
    return spec


def test_default_seat_is_the_head_centre_and_moves_nothing() -> None:
    assert seat.load_head_seat_settings().seat == "head_centre"
    assembly = ca.assembly_from_spec(_stated(_spec()))
    assert assembly.head_slide_m == 0.0
    plain = ca.ClubAssembly(
        assembly.head_alias,
        assembly.length_m,
        assembly.grip_length_m,
        assembly.shaft_radius_m,
        assembly.axis_offset_m,
        assembly.face_roll_deg,
    )
    assert assembly == plain


def test_sole_seat_slides_the_head_to_the_stated_butt_to_sole_length() -> None:
    spec = _stated(_spec(head_seat="sole_on_shaft_axis"))
    before = ca.assembly_from_spec(_stated(_spec()))
    after = ca.assembly_from_spec(spec)
    assert after.head_slide_m == pytest.approx(SOLE_OFFSET_M, abs=1e-12)
    shift = np.array([0.0, SOLE_OFFSET_M, 0.0])
    np.testing.assert_allclose(
        ca.assembly_meshes(after)["head"].vertices,
        ca.assembly_meshes(before)["head"].vertices + shift,
        atol=1e-12,
    )
    np.testing.assert_allclose(
        ca.clubface_centre(after), ca.clubface_centre(before) + shift, atol=1e-12
    )
    np.testing.assert_allclose(ca.clubface_vector(after), ca.clubface_vector(before))
    np.testing.assert_array_equal(
        ca.assembly_meshes(after)["grip"].vertices,
        ca.assembly_meshes(before)["grip"].vertices,
    )


def test_the_shaft_still_reaches_the_hosel_of_a_slid_head() -> None:
    before = ca.assembly_from_spec(_stated(_spec()))
    after = ca.assembly_from_spec(_stated(_spec(head_seat="sole_on_shaft_axis")))
    low = ca.assembly_meshes(before)["shaft"].vertices[:, 1]
    slid = ca.assembly_meshes(after)["shaft"].vertices[:, 1]
    assert slid.max() == pytest.approx(low.max() + SOLE_OFFSET_M, abs=1e-12)
    assert slid.min() == pytest.approx(low.min(), abs=1e-12)
    layout = ShaftChainLayout("club", (ChainLink("tip", -0.4, 0.0),))
    pieces = shaft_piece_meshes(after, layout)
    assert pieces["tip"].vertices[:, 1].max() == pytest.approx(slid.max(), abs=1e-12)


def test_sole_seat_needs_a_stated_butt_to_sole_length() -> None:
    spec = _spec(head_seat="sole_on_shaft_axis")
    spec["club"].pop("butt_to_sole_m", None)
    with pytest.raises(ValueError, match="butt_to_sole_m"):
        ca.assembly_from_spec(spec)
    spec["club"]["butt_to_sole_m"] = float(spec["club"]["length_m"]) - 0.01
    with pytest.raises(ValueError, match="butt_to_sole_m"):
        ca.assembly_from_spec(spec)
    spec["club"]["butt_to_sole_m"] = math.nan
    with pytest.raises(ValueError, match="butt_to_sole_m"):
        ca.assembly_from_spec(spec)


def test_seat_and_slide_are_validated(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="head_seat"):
        ca.assembly_from_spec(_spec(head_seat="on_the_toe"))
    with pytest.raises(TypeError, match="head_seat"):
        ca.assembly_from_spec(_spec(head_seat=1))
    with pytest.raises(ValueError, match="head_slide_m"):
        ca.ClubAssembly("driver", 1.08, 0.265, 0.0065, head_slide_m=math.inf)
    bad = tmp_path / "settings.json"
    bad.write_text(json.dumps({"seat": "nowhere"}), encoding="utf-8")
    with pytest.raises(ValueError, match="head_seat"):
        seat.load_head_seat_settings(bad)


def test_settings_file_documents_every_key() -> None:
    raw = json.loads(seat.SETTINGS_PATH.read_text(encoding="utf-8"))
    assert set(raw["sources"]) == set(raw) - {"sources"}

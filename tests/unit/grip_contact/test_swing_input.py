"""Shared coordinate mapping of the closure-consistent fitted swings (#11739)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from src.shared.python.grip_contact import (
    CoordinateSwing,
    load_coordinate_swing,
    map_coordinates,
)

pytestmark = pytest.mark.unit

ROOT = Path(__file__).resolve().parents[3]
FIXTURES = ROOT / "tests/fixtures/club_face"
MODELS = ROOT / "docs/development/full_body_models"


def test_map_reorders_columns_by_name() -> None:
    q = np.array([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
    out = map_coordinates(["a", "b", "c"], q, ["c", "a", "b"])
    np.testing.assert_array_equal(out, [[3.0, 1.0, 2.0], [6.0, 4.0, 5.0]])


def test_map_rejects_a_missing_target_coordinate() -> None:
    with pytest.raises(ValueError, match="missing"):
        map_coordinates(["a", "b"], np.zeros((2, 2)), ["a", "z"])


def test_map_rejects_duplicate_names_and_bad_shapes() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        map_coordinates(["a", "a"], np.zeros((2, 2)), ["a"])
    with pytest.raises(ValueError, match="shape"):
        map_coordinates(["a", "b"], np.zeros((2, 3)), ["a"])
    with pytest.raises(ValueError, match="finite"):
        map_coordinates(["a"], np.array([[np.nan]]), ["a"])


def test_map_drops_source_extras_only_when_allowed() -> None:
    q = np.array([[1.0, 2.0]])
    with pytest.raises(ValueError, match="unused"):
        map_coordinates(["a", "b"], q, ["a"])
    out = map_coordinates(["a", "b"], q, ["a"], allow_unused=True)
    np.testing.assert_array_equal(out, [[1.0]])


@pytest.mark.parametrize("club", ["driver", "iron7"])
def test_fixture_swing_maps_onto_the_spec_coordinates(club: str) -> None:
    spec = json.loads((MODELS / f"full_body_spec_anthro_{club}.json").read_bytes())
    swing = load_coordinate_swing(
        FIXTURES / f"swing_q_{club}.npz",
        FIXTURES / "address_poses.json",
        club,
        spec["coordinate_order"],
    )
    assert isinstance(swing, CoordinateSwing)
    assert swing.names == list(spec["coordinate_order"])
    assert swing.q.shape == (swing.time_s.size, len(swing.names))
    assert swing.q.dtype == np.float64
    np.testing.assert_allclose(np.diff(swing.time_s), 0.002)
    assert 1.8 < swing.time_s[-1] < 1.85
    assert swing.sha256 and len(swing.sha256) == 64


def test_loader_rejects_an_unknown_club(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="club"):
        load_coordinate_swing(
            FIXTURES / "swing_q_driver.npz",
            FIXTURES / "address_poses.json",
            "putter",
            ["TranslationInputX"],
        )
    with pytest.raises(ValueError, match="dt_s"):
        load_coordinate_swing(
            FIXTURES / "swing_q_driver.npz",
            FIXTURES / "address_poses.json",
            "driver",
            ["TranslationInputX"],
            dt_s=0.0,
        )

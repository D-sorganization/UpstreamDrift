"""Golf-model joint ranges come from one source and stay physiological (#12315).

The forearm ``pro_sup_*`` range of the Rajagopal base stops at the neutral
forearm (pi/2), so the lead forearm of every golf grip sat on its limit. The
golf models widen it to the AAOS normative pronation. These checks read the
committed ``.osim`` XML directly, so they run without the OpenSim bindings.
"""

from __future__ import annotations

import math
import defusedxml.ElementTree as ET
import pytest

from scripts import build_humanoid_osim as builder
from src.engines.physics_engines.opensim.python import golf_joint_limits as limits
from src.engines.physics_engines.opensim.python import msk_club as mc
from src.engines.physics_engines.opensim.python import msk_club_calibration as cal

pytestmark = pytest.mark.unit

MODELS = ("golf_humanoid", "golf_humanoid_scaled")
FOREARM = ("pro_sup_l", "pro_sup_r")
BOUND_MARGIN = 0.01


def _osim_range(model: str, coord: str) -> tuple[float, float]:
    root = ET.parse(mc.MODELS_DIR / f"{model}.osim").getroot()
    nodes = [c for c in root.iter("Coordinate") if c.get("name") == coord]
    assert len(nodes) == 1, (model, coord)
    low, high = nodes[0].findtext("range").split()
    return float(low), float(high)


def test_forearm_range_is_within_physiological_bounds() -> None:
    for coord in FOREARM:
        low, high = limits.GOLF_COORDINATE_RANGES[coord]
        pronation = math.degrees(high - limits.FOREARM_NEUTRAL_RAD)
        assert low == 0.0
        assert 75.0 <= pronation <= 90.0, coord  # AAOS 80 deg; literature <= 90
        assert limits.ANATOMICAL_BOUNDS[coord][0] <= low
        assert high <= limits.ANATOMICAL_BOUNDS[coord][1]


def test_validator_accepts_the_golf_ranges() -> None:
    limits.validate_coordinate_ranges(limits.GOLF_COORDINATE_RANGES)


@pytest.mark.parametrize(
    "ranges",
    [
        {"pro_sup_l": (1.0, 1.0)},
        {"pro_sup_l": (2.0, 1.0)},
        {"pro_sup_l": (0.0, math.pi + 0.1)},  # pronation past the ceiling
        {"pro_sup_l": (-0.1, 2.0)},  # supination past the ceiling
        {"wrist_dev_l": (0.0, math.nan)},
        {"wrist_dev_l": (-math.inf, 0.0)},
    ],
)
def test_validator_rejects_bad_ranges(ranges: dict) -> None:
    with pytest.raises(ValueError):
        limits.validate_coordinate_ranges(ranges)


def test_builder_consumes_the_shared_ranges() -> None:
    """One source: the builder holds no private copy of the golf ranges."""
    assert builder.GOLF_COORDINATE_RANGES is limits.GOLF_COORDINATE_RANGES


@pytest.mark.parametrize("model", MODELS)
@pytest.mark.parametrize("coord", sorted(limits.GOLF_COORDINATE_RANGES))
def test_model_exposes_the_spec_range(model: str, coord: str) -> None:
    """Parity: every golf model carries the spec's range for each override."""
    low, high = limits.GOLF_COORDINATE_RANGES[coord]
    assert _osim_range(model, coord) == pytest.approx((low, high), abs=1e-12)


@pytest.mark.parametrize("coord", FOREARM)
def test_address_limits_use_the_spec_forearm_range(coord: str) -> None:
    assert cal.ADDRESS_RANGES[coord] == limits.GOLF_COORDINATE_RANGES[coord]


@pytest.mark.parametrize("club", mc.CLUBS)
@pytest.mark.parametrize("model", MODELS)
def test_no_calibrated_forearm_sits_at_a_range_bound(model: str, club: str) -> None:
    recorded = cal.ADDRESS_BOUND_REASONS.get(model, {})
    address_q = mc.load_calibration(model, club).address_q
    for coord in FOREARM:
        if coord in recorded:
            continue
        low, high = _osim_range(model, coord)
        assert low + BOUND_MARGIN < address_q[coord] < high - BOUND_MARGIN, (
            model,
            club,
            coord,
        )

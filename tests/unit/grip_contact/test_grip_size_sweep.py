"""Grip-size sweep keyed to hand length through Golf Pride bins (#13892)."""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_topology import contact_topology
from src.shared.python.grip_contact.grip_geometry import get_grip_preset
from src.shared.python.grip_contact.grip_size_sweep import (
    INCH_M,
    STYLION_TO_CREASE_M,
    golf_pride_bin,
    grip_size_sweep,
    per_hand_bushing_demand,
)
from src.shared.python.grip_contact.pad_layout import (
    peak_bushing_demand,
    required_squeeze_n,
    squeeze_from_bushing_series,
)
from src.shared.python.hand_size import HAND_LENGTH_RANGE_M
from src.shared.python.motion_matching.grip_contract import GripSpec

pytestmark = pytest.mark.unit

MU = 0.9
DEMAND = {"lead": (30.0, 1.2, 50.0), "trail": (20.0, 1.6, 40.0)}


def _stylion_for_crease_in(crease_in: float) -> float:
    return crease_in * INCH_M + STYLION_TO_CREASE_M


def test_ansur_mean_male_is_standard_corrected_and_midsize_uncorrected() -> None:
    size = golf_pride_bin(1.75 * 0.1101)
    assert size.crease_length_in == pytest.approx((1.75 * 0.1101 - 0.0062) / INCH_M)
    assert size.bin == "standard"
    assert size.uncorrected_bin == "midsize"
    assert size.preset == "mens_standard"
    assert size.edge_neighbours is None


@pytest.mark.parametrize(
    ("edge_in", "below", "above"),
    [(6.55, "undersize", "standard"), (7.55, "standard", "midsize")]
    + [(9.05, "midsize", "jumbo")],
)
def test_bin_edges_are_reported_both_ways(
    edge_in: float, below: str, above: str
) -> None:
    lo = golf_pride_bin(_stylion_for_crease_in(edge_in - 0.03))
    hi = golf_pride_bin(_stylion_for_crease_in(edge_in + 0.03))
    assert (lo.bin, hi.bin) == (below, above)
    assert lo.edge_neighbours == hi.edge_neighbours == (below, above)
    assert golf_pride_bin(_stylion_for_crease_in(edge_in - 0.2)).edge_neighbours is None
    assert (hi.preset is None) == (above == "jumbo")  # jumbo is unsourced


@pytest.mark.parametrize("hand_length_m", [0.139, 0.241, math.nan])
def test_hand_length_bounds_are_inclusive_and_enforced(hand_length_m: float) -> None:
    assert [golf_pride_bin(x).hand_length_m for x in HAND_LENGTH_RANGE_M] == list(
        HAND_LENGTH_RANGE_M
    )
    with pytest.raises(ValueError, match="hand_length_m"):
        golf_pride_bin(hand_length_m)


def test_pad_above_the_butt_cap_is_rejected() -> None:
    """``lead_below_butt_m = 0`` puts the lead palm pad at s < 0."""
    with pytest.raises(ValueError, match="outside the grip"):
        contact_topology(
            GripSpec(lead_below_butt_m=0.0), grip_geometry=get_grip_preset("midsize")
        )


def _series() -> tuple[dict, dict, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(13892)
    force = {s: rng.normal(size=(50, 3)) * 40.0 for s in "LR"}
    torque = {s: rng.normal(size=(50, 3)) for s in "LR"}
    rot = np.repeat(np.eye(3)[None], 50, axis=0)
    return force, torque, rot, np.array([0.0, 1.0, 0.0])


def test_per_hand_demand_takes_the_lead_side_from_handedness() -> None:
    force, torque, rot, axis = _series()
    peak = peak_bushing_demand(force, torque, rot, axis)
    assert squeeze_from_bushing_series(
        force, torque, rot, axis, MU, 0.0127
    ) == required_squeeze_n(*peak, MU, 0.0127)
    right = per_hand_bushing_demand(force, torque, rot, axis, "right")
    left = per_hand_bushing_demand(force, torque, rot, axis, "left")
    lead_l = peak_bushing_demand({"L": force["L"]}, {"L": torque["L"]}, rot, axis)
    assert right["lead"] == left["trail"] == lead_l
    assert right["trail"] == left["lead"]


def test_sweep_rows_cover_the_grid_and_use_each_hands_radius() -> None:
    grid = ((0.18, 0.20), ("mens_standard", "midsize"), (0, 4), DEMAND, MU)
    rows = grip_size_sweep(*grid)
    assert len(rows) == 8
    assert rows == grip_size_sweep(*grid)  # deterministic
    for row in rows:
        geom = get_grip_preset(row.preset).with_buildup(row.buildup_layers)
        for hand in ("lead", "trail"):
            assert row.squeeze_n[hand] == required_squeeze_n(
                *DEMAND[hand], MU, row.radius_m[hand]
            )
            assert row.torque_branch_n[hand] == pytest.approx(
                DEMAND[hand][1] / (MU * row.radius_m[hand]), rel=1e-15
            )
            assert row.governing[hand] in {"axial", "torque", "transverse", "combined"}
            assert row.pad_normal_deg[hand]
        assert row.radius_m["trail"] < row.radius_m["lead"] <= geom.radius_at(0.0)
        assert row.size_bin == golf_pride_bin(row.hand_length_m)


def test_buildup_lowers_squeeze_and_a_one_hand_demand_is_rejected() -> None:
    thin, thick = grip_size_sweep((0.19,), ("mens_standard",), (0, 4), DEMAND, MU)
    for hand in ("lead", "trail"):
        assert thick.radius_m[hand] > thin.radius_m[hand]
        assert thick.squeeze_n[hand] <= thin.squeeze_n[hand]
        assert thick.torque_branch_n[hand] < thin.torque_branch_n[hand]
    with pytest.raises(ValueError, match="lead and trail"):
        grip_size_sweep((0.19,), ("midsize",), (0,), {"lead": (1.0, 1.0, 1.0)}, MU)


def test_cli_writes_a_receipt(tmp_path) -> None:
    from scripts.grip import grip_size_sweep as cli

    out = tmp_path / "sweep.json"
    argv = ["--club", "driver", "--hand-lengths", "0.19", "--presets", "midsize"]
    assert cli.main([*argv, "--layers", "0", "--out", str(out)]) == 0
    receipt = json.loads(out.read_text(encoding="utf-8"))
    assert receipt["issue"] == 13892
    assert receipt["stylion_to_crease_m"] == STYLION_TO_CREASE_M
    (row,) = receipt["rows"]
    assert row["preset"] == "midsize"
    assert row["size_bin"]["bin"] == "standard"
    assert row["squeeze_n"]["lead"] > 0.0

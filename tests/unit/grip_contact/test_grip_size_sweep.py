"""Grip-size sweep keyed to hand length (HRG-GD4 PR 2, #13892).

Hand length (ANSUR II ``handlength``, stylion to dactylion) maps to Golf
Pride's grip-size bins (wrist crease to middle-finger tip) after the
stylion-to-crease offset; the sweep reports per-hand radius, squeeze and pad
normals across presets x build-up layers x hand length.
"""

from __future__ import annotations

import json
import math

import numpy as np
import pytest

from src.shared.python.grip_contact.contact_topology import contact_topology
from src.shared.python.grip_contact.grip_geometry import (
    STATION_LOWER_M,
    get_grip_preset,
)
from src.shared.python.grip_contact.grip_size_sweep import (
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

INCH_M = 0.0254
MU = 0.9
DEMAND = {"lead": (30.0, 1.2, 50.0), "trail": (20.0, 1.6, 40.0)}


def _stylion_for_crease_in(crease_in: float) -> float:
    return crease_in * INCH_M + STYLION_TO_CREASE_M


def test_crease_length_subtracts_the_stylion_offset() -> None:
    size = golf_pride_bin(0.19)
    assert size.crease_length_in == pytest.approx((0.19 - 0.0062) / INCH_M)


def test_ansur_mean_male_is_standard_corrected_and_midsize_uncorrected() -> None:
    size = golf_pride_bin(1.75 * 0.1101)
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


def test_jumbo_has_no_sourced_preset() -> None:
    size = golf_pride_bin(_stylion_for_crease_in(9.2))
    assert size.bin == "jumbo"
    assert size.preset is None


@pytest.mark.parametrize("hand_length_m", HAND_LENGTH_RANGE_M)
def test_hand_length_bounds_are_accepted(hand_length_m: float) -> None:
    assert golf_pride_bin(hand_length_m).hand_length_m == hand_length_m


@pytest.mark.parametrize("hand_length_m", [0.139, 0.241, math.nan])
def test_hand_length_outside_bounds_is_rejected(hand_length_m: float) -> None:
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


def test_bushing_squeeze_is_required_squeeze_of_the_peak_demand() -> None:
    force, torque, rot, axis = _series()
    peak = peak_bushing_demand(force, torque, rot, axis)
    assert squeeze_from_bushing_series(
        force, torque, rot, axis, MU, 0.0127
    ) == required_squeeze_n(*peak, MU, 0.0127)


def test_per_hand_demand_takes_the_lead_side_from_handedness() -> None:
    force, torque, rot, axis = _series()
    right = per_hand_bushing_demand(force, torque, rot, axis, "right")
    left = per_hand_bushing_demand(force, torque, rot, axis, "left")
    lead_l = peak_bushing_demand({"L": force["L"]}, {"L": torque["L"]}, rot, axis)
    assert right["lead"] == left["trail"] == lead_l
    assert right["trail"] == left["lead"]


def test_sweep_rows_cover_the_grid_and_use_each_hands_radius() -> None:
    rows = grip_size_sweep((0.18, 0.20), ("mens_standard", "midsize"), (0, 4), DEMAND, MU)
    assert len(rows) == 8
    assert rows == grip_size_sweep(
        (0.18, 0.20), ("mens_standard", "midsize"), (0, 4), DEMAND, MU
    )
    for row in rows:
        geom = get_grip_preset(row.preset).with_buildup(row.buildup_layers)
        for hand in ("lead", "trail"):
            assert row.squeeze_n[hand] == required_squeeze_n(
                *DEMAND[hand], MU, row.radius_m[hand]
            )
            assert row.pad_normal_deg[hand]
        assert row.radius_m["trail"] < row.radius_m["lead"] <= geom.radius_at(0.0)
        assert row.radius_m["trail"] > geom.radius_at(STATION_LOWER_M) - 0.002
        assert row.size_bin == golf_pride_bin(row.hand_length_m)


def test_buildup_raises_the_radius_and_never_raises_squeeze() -> None:
    thin, thick = grip_size_sweep((0.19,), ("mens_standard",), (0, 4), DEMAND, MU)
    for hand in ("lead", "trail"):
        assert thick.radius_m[hand] > thin.radius_m[hand]
        assert thick.squeeze_n[hand] <= thin.squeeze_n[hand]


def test_sweep_rejects_a_demand_without_both_hands() -> None:
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
    assert row["bin"] == "standard"
    assert row["squeeze_n"]["lead"] > 0.0

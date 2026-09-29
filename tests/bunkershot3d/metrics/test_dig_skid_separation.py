"""The dig-versus-skid verdict must separate the shipped design space (#8703).

This is the acceptance test for issue #8703. The verdict it replaced was built
on the *entry slope ratio* -- the penetration slope over the first 10 mm of
travel divided by the delivered path slope -- and across the demo's 77-point
sweep that ratio spanned **0.9987 to 1.0000** and returned ``MARGINAL`` at
every point. 10 mm is 0.4 ms at greenside speed, and a 0.3 kg head under an
order-5 N.s impulse cannot bend measurably in 0.4 ms. Widening the window was
measured and rejected: the spread opens, but the correlation with maximum sole
depth is negative at every informative window, so a resized window would ship
a confidently *inverted* verdict.

The replacement is the **descent-return ratio**: the sole's upward speed as it
leaves the sand over its downward speed as it entered. It is the direct
expression of the physical claim -- a digging head gives its descent to the
sand and stays down; a skidding head bottoms out and is handed its descent
back -- and it has no window parameter to place, because its window is the
divot itself.

What this file pins, all measured rather than asserted from theory:

* the ratio is **not saturated**: it spans more than 0.3 over a sweep the old
  ratio spanned 0.0012 of;
* steeper attack reads as **more dig**, monotonically, in every design;
* more bounce reads as **more skid** where the head actually buries;
* the ratio tracks an *independent* physical measure -- the solver's own
  maximum sole depth -- with a rank correlation below -0.7, and with the
  **right sign**: deeper is more dig; and
* the verdict is still uncalibrated, and says so.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from bunkershot3d.geometry import (
    build_wedge_mesh,
    compute_mass_properties,
    get_preset,
    shaft_axis,
)
from bunkershot3d.metrics import (
    DigSkidVerdict,
    HeadModel,
    StrikeScene,
    StrikeTrace,
    dig_vs_skid,
)
from bunkershot3d.sand import PlayingCondition, playing_condition
from bunkershot3d.solvers import (
    DRFTSolver,
    HeadKinematics,
    MaterialResponse,
    RefusalPolicy,
    SurfaceElements,
    simulate_shot,
)

pytestmark = pytest.mark.integration

#: Delivery speed [m/s]; greenside is 20-27 m/s.
_SPEED_M_S = 25.0

#: The design axis: three shipped presets spanning 5.0 to 14.42 deg of
#: marketed bounce, low bounce first.
_PRESETS = ("tour_shaved_heel_lob", "sm9_58_m", "acushnet_example_3")

#: The delivery axis, as positive descent angles [deg].
_ATTACK_DEG = (2.0, 6.0, 10.0, 14.0)

#: The condition axis.
_CONDITIONS = (PlayingCondition.FIRM, PlayingCondition.FLUFFY)

#: Thresholds used only to show the quantity supports a three-way split with
#: comfortable margins. They are conventions, like the shipped defaults.
_SPLIT_DIG = 0.30
_SPLIT_SKID = 0.70


def _scene() -> StrikeScene:
    """A flat lie with the ball 30 mm down the travel axis."""
    return StrikeScene(
        sand_surface_height_m=0.0,
        ball_position_m=(-0.030, 0.0, 0.0),
        travel_axis=(-1.0, 0.0, 0.0),
    )


def _build(name: str) -> tuple[SurfaceElements, HeadModel]:
    """Loft one preset and describe it for the metrics layer.

    Args:
        name: Preset name.

    Returns:
        The discretised surface and the matching head model, whose sole
        reference is the lowest sole element so the depth the verdict is
        measured on is the depth the solver marched.
    """
    preset = get_preset(name)
    mesh = build_wedge_mesh(preset.geometry, n_profile_points=24, n_stations=11)
    elements = SurfaceElements.from_mesh(mesh)
    mass = compute_mass_properties(mesh, mass_kg=preset.geometry.head_mass_kg)
    _, axis = shaft_axis(preset.geometry)
    lowest = int(np.argmin(elements.centroids_m[:, 2]))
    return elements, HeadModel(
        mass_kg=preset.geometry.head_mass_kg,
        centre_of_mass_body_m=mass.centroid_m,
        sole_reference_body_m=elements.centroids_m[lowest],
        shaft_axis_body=np.asarray(axis, dtype=np.float64),
        inertia_body_kg_m2=mass.inertia_kg_m2,
    )


def _rank(values: np.ndarray) -> np.ndarray:
    """Return the ordinal ranks of ``values``, so a correlation is Spearman's."""
    return np.argsort(np.argsort(values)).astype(float)


def _classify(ratio: float) -> DigSkidVerdict:
    """Classify at the wide split thresholds this file uses for its margins."""
    if ratio <= _SPLIT_DIG:
        return DigSkidVerdict.DIG
    if ratio >= _SPLIT_SKID:
        return DigSkidVerdict.SKID
    return DigSkidVerdict.MARGINAL


@pytest.fixture(scope="module")
def sweep() -> list[dict[str, object]]:
    """Run the design sweep once: 3 presets x 4 attack angles x 2 conditions.

    Returns:
        One row per design point, carrying the design coordinates, the
        discriminator and the solver's own maximum sole depth -- an
        independent measure the verdict is checked against.
    """
    scene = _scene()
    rows: list[dict[str, object]] = []
    for name in _PRESETS:
        elements, head = _build(name)
        bounce_deg = get_preset(name).geometry.marketed_bounce.angle_deg
        for condition in _CONDITIONS:
            solver = DRFTSolver(
                material=MaterialResponse.from_sand_state(playing_condition(condition)),
                refusal_policy=RefusalPolicy.REPORT,
            )
            for attack_deg in _ATTACK_DEG:
                angle = math.radians(attack_deg)
                velocity = _SPEED_M_S * np.array(
                    [-math.cos(angle), 0.0, -math.sin(angle)]
                )
                shot = simulate_shot(
                    solver,
                    elements,
                    head_mass_kg=0.30,
                    kinematics=HeadKinematics(velocity_m_s=velocity),
                )
                result = dig_vs_skid(StrikeTrace.from_shot(shot), head, scene)
                rows.append(
                    {
                        "preset": name,
                        "bounce_deg": bounce_deg,
                        "condition": condition,
                        "attack_deg": attack_deg,
                        "result": result,
                        "ratio": result.descent_return_ratio,
                        "max_depth_m": shot.max_sole_depth_m,
                    }
                )
    return rows


class TestTheRatioIsNotSaturated:
    """The failure #8703 records was a 0.13 %-wide, design-independent signal."""

    def test_the_sweep_spans_more_than_a_third_of_the_scale(self, sweep) -> None:
        """0.0012 was the whole span of the ratio this replaced."""
        ratios = np.array([row["ratio"] for row in sweep], dtype=float)

        if ratios.max() - ratios.min() <= 0.30:
            raise AssertionError(
                f"the descent-return ratio spans only "
                f"{ratios.max() - ratios.min():.4f} over {ratios.size} design "
                "points, which is the saturation issue #8703 was filed for"
            )

    def test_no_two_design_points_share_a_ratio_by_construction(self, sweep) -> None:
        """A saturated metric repeats itself; this one does not."""
        ratios = [round(float(row["ratio"]), 6) for row in sweep]

        if len(set(ratios)) != len(ratios):
            raise AssertionError(
                "design points share a descent-return ratio to 1e-6, which "
                "means the quantity is not resolving the design space"
            )

    def test_the_quantity_supports_a_three_way_split(self, sweep) -> None:
        """Split at 0.30/0.70, well clear of every point the sweep produced."""
        classes = {_classify(float(row["ratio"])) for row in sweep}

        if classes != {
            DigSkidVerdict.DIG,
            DigSkidVerdict.MARGINAL,
            DigSkidVerdict.SKID,
        }:
            raise AssertionError(
                f"the sweep produced {sorted(str(c) for c in classes)}, not all "
                "three verdicts, so the quantity does not separate"
            )


class TestTheOrderingIsPhysicallySensible:
    """Dig and skid are physical claims, so the ordering is checkable."""

    def test_steeper_attack_reads_as_more_dig_in_every_design(self, sweep) -> None:
        """Monotone within each preset and condition, with no exceptions."""
        for preset in _PRESETS:
            for condition in _CONDITIONS:
                series = [
                    float(row["ratio"])
                    for row in sorted(
                        (
                            row
                            for row in sweep
                            if row["preset"] == preset and row["condition"] == condition
                        ),
                        key=lambda row: row["attack_deg"],
                    )
                ]
                pairs = zip(series, series[1:], strict=False)
                if any(later >= earlier for earlier, later in pairs):
                    raise AssertionError(
                        f"{preset} in {condition} gives {series} as the attack "
                        "angle steepens; a steeper blow must read as more dig"
                    )

    def test_more_bounce_reads_as_more_skid_where_the_head_buries(self, sweep) -> None:
        """At -10 deg the 5 deg sole buries and the 14.4 deg sole does not."""
        for condition in _CONDITIONS:
            burying = sorted(
                (
                    row
                    for row in sweep
                    if row["attack_deg"] == 10.0 and row["condition"] == condition
                ),
                key=lambda row: row["bounce_deg"],
            )
            low, high = burying[0], burying[-1]
            if high["ratio"] <= low["ratio"]:
                raise AssertionError(
                    f"in {condition} at -10 deg the {high['bounce_deg']:.2f} deg "
                    f"bounce sole returns {high['ratio']:.4f} of its descent "
                    f"and the {low['bounce_deg']:.2f} deg sole "
                    f"{low['ratio']:.4f}; more bounce must read as more skid"
                )

    def test_the_shallowest_delivery_never_reads_as_a_dig(self, sweep) -> None:
        """A -2 deg blow barely enters; nothing about it is a dig."""
        for row in sweep:
            if row["attack_deg"] == 2.0 and row["result"].verdict is DigSkidVerdict.DIG:
                raise AssertionError(
                    f"{row['preset']} at -2 deg in {row['condition']} was called "
                    f"a dig on a ratio of {row['ratio']:.4f}"
                )

    def test_the_steepest_delivery_always_reads_as_a_dig(self, sweep) -> None:
        """A -14 deg blow buries every one of these soles."""
        for row in sweep:
            if (
                row["attack_deg"] == 14.0
                and row["result"].verdict is not DigSkidVerdict.DIG
            ):
                raise AssertionError(
                    f"{row['preset']} at -14 deg in {row['condition']} was called "
                    f"{row['result'].verdict} on a ratio of {row['ratio']:.4f}"
                )


class TestTheRatioTracksAnIndependentPhysicalMeasure:
    """Sole depth is measured by the solver, not by the discriminator."""

    def test_the_rank_correlation_with_max_sole_depth_is_strongly_negative(
        self, sweep
    ) -> None:
        """Deeper is more dig. The ratio this replaced got this sign wrong."""
        ratios = np.array([row["ratio"] for row in sweep], dtype=float)
        depths = np.array([row["max_depth_m"] for row in sweep], dtype=float)

        spearman = float(np.corrcoef(_rank(ratios), _rank(depths))[0, 1])
        if spearman > -0.70:
            raise AssertionError(
                f"the descent-return ratio ranks against maximum sole depth at "
                f"{spearman:+.3f}; a dig verdict must get deeper as the head "
                "buries, and the slope ratio of #8703 failed on exactly this"
            )

    def test_the_deepest_and_shallowest_shots_land_on_opposite_sides(
        self, sweep
    ) -> None:
        """The extremes of an independent measure must not agree."""
        deepest = max(sweep, key=lambda row: row["max_depth_m"])
        shallowest = min(sweep, key=lambda row: row["max_depth_m"])

        if deepest["ratio"] >= shallowest["ratio"]:
            raise AssertionError(
                f"the deepest shot ({deepest['max_depth_m'] * 1e3:.1f} mm) "
                f"returns {deepest['ratio']:.4f} of its descent and the "
                f"shallowest ({shallowest['max_depth_m'] * 1e3:.1f} mm) "
                f"{shallowest['ratio']:.4f}"
            )


class TestSeparationDoesNotBuyCalibration:
    """A separating verdict is still not a measured one (issue #8703)."""

    def test_every_point_declares_itself_uncalibrated(self, sweep) -> None:
        for row in sweep:
            if row["result"].calibration.calibrated:
                raise AssertionError(
                    f"{row['preset']} reported a calibrated verdict; no "
                    "threshold on this ratio has been measured"
                )

    def test_every_point_refuses_to_be_quoted(self, sweep) -> None:
        for row in sweep:
            with pytest.raises(ValueError, match="not calibrated"):
                row["result"].calibration.require_calibrated()

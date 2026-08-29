"""Rendering the workbench results as text (issue #8618, W11).

Pure functions over the value objects in :mod:`src.tools.bunker_shot_gui.model`.
No Qt, no state, no arithmetic beyond formatting, so the wording a designer
reads can be tested without a display and reused verbatim by another front end.

The first rule of this module is that **the verdict comes first**. A force
that 3D-RFT produced 60x outside its stated Froude limit is not a measurement,
and every report opens by saying so.
"""

from __future__ import annotations

import numpy as np

from bunkershot3d.solvers import EnvelopeStatus, ValidityVerdict

from .field import ContactPatch, LoadComponent, SoleLoadField
from .model import (
    DesignEvaluation,
    PlayabilityOutcome,
    ShotOutcome,
    SoleLoadMap,
    WorkbenchComparison,
)

__all__ = [
    "CARRY_CAVEAT",
    "MAX_REPORTED_REASONS",
    "PATCH_CONFOUND_CAVEAT",
    "SHADE_RAMP",
    "STATUS_COLOUR",
    "STATUS_HEADLINE",
    "comparison_report",
    "evaluation_report",
    "playability_text",
    "shade_grid",
    "shot_report",
    "sole_field_text",
    "sole_map_text",
    "status_colour",
    "status_headline",
    "verdict_report",
]

SHADE_RAMP = " .:-=+*#%@"
"""Ten shading levels. Index 0 is the space reserved for an **empty** cell;
a present but unloaded cell renders as ``.``, so "nothing here" and "nothing
happened here" are never the same character."""

MAX_REPORTED_REASONS = 8
"""Reason lines shown before the remainder is counted rather than listed."""

STATUS_HEADLINE: dict[EnvelopeStatus, str] = {
    EnvelopeStatus.WITHIN: ("WITHIN ENVELOPE - inside 3D-RFT's own published limits"),
    EnvelopeStatus.EXTRAPOLATED: (
        "EXTRAPOLATED - outside the stated limits, inside the published validation set"
    ),
    EnvelopeStatus.BEYOND_VALIDATION: (
        "BEYOND VALIDATION - past every published measurement. These numbers "
        "are an extrapolation, not a measurement"
    ),
    EnvelopeStatus.REFUSED: (
        "REFUSED - 3D-RFT declines this query. No force, depth or carry is "
        "reported, because any number here would be wrong by an order of "
        "magnitude"
    ),
}
"""What each verdict status means, in the words shown to a designer."""

STATUS_COLOUR: dict[EnvelopeStatus, str] = {
    EnvelopeStatus.WITHIN: "#1b7f3b",
    EnvelopeStatus.EXTRAPOLATED: "#b07a00",
    EnvelopeStatus.BEYOND_VALIDATION: "#c05600",
    EnvelopeStatus.REFUSED: "#9b1c1c",
}
"""Banner colour per status. Refusal is the only red in the tool."""

_RULE = "=" * 66
_THIN = "-" * 66


def status_headline(status: EnvelopeStatus) -> str:
    """Return the one-line meaning of a verdict status.

    Args:
        status: The verdict status.

    Returns:
        The headline shown in the verdict banner.
    """
    return STATUS_HEADLINE[EnvelopeStatus(status)]


def status_colour(status: EnvelopeStatus) -> str:
    """Return the banner colour for a verdict status.

    Args:
        status: The verdict status.

    Returns:
        A ``#rrggbb`` colour string.
    """
    return STATUS_COLOUR[EnvelopeStatus(status)]


def verdict_report(
    verdict: ValidityVerdict, *, max_reasons: int = MAX_REPORTED_REASONS
) -> str:
    """Render a validity verdict: status, scales, reasons, caveats.

    A shot's verdict is the union over its timesteps, so the same finding
    recurs once per step with a slightly different number and the raw list
    runs to dozens of lines. The reasons are therefore truncated, with the
    remainder counted rather than dropped silently -- the caveats, which are
    the part a designer must not miss, are never truncated.

    Args:
        verdict: The verdict travelling with a result.
        max_reasons: How many reason lines to show before summarising.

    Returns:
        A multi-line report.

    Raises:
        ValueError: If ``max_reasons`` is negative.
    """
    if max_reasons < 0:
        raise ValueError(f"max_reasons must not be negative, got {max_reasons}")
    status = verdict.status
    lines = [
        status_headline(status),
        "",
        f"Validity: {status.value.upper()}",
        f"  governing scale: {verdict.governing.describe()}",
    ]
    if verdict.clamped_area_fraction > 0.0:
        lines.append(
            f"  {verdict.clamped_area_fraction:.1%} of the active area was clamped "
            "into the fitted orientation domain"
        )
    for reason in verdict.reasons[:max_reasons]:
        lines.append(f"  reason: {reason}")
    remaining = len(verdict.reasons) - max_reasons
    if remaining > 0:
        lines.append(
            f"  ... and {remaining} further finding(s) at other timesteps and scales"
        )
    # The caveat prose lives in a private table in the solvers package, so it
    # is lifted from the verdict's own summary rather than restated here,
    # where a copy would drift the moment a caveat is reworded.
    lines.extend(
        line
        for line in verdict.summary().splitlines()
        if line.lstrip().startswith("caveat:")
    )
    return "\n".join(lines)


_LABEL_WIDTH = 34


def _line(label: str, value: str) -> str:
    """Return one aligned ``label: value`` row."""
    return f"{label:<{_LABEL_WIDTH}}{value}"


def _optional(value: float | None, unit: str, scale: float = 1.0) -> str:
    """Format an optional number, or say it is unavailable."""
    if value is None:
        return "not reported"
    return f"{value * scale:.4g} {unit}"


def _head_load_lines(outcome: ShotOutcome) -> tuple[str, ...]:
    """Head-load section, or nothing when the metrics were not computed."""
    if outcome.loads is None:
        return ()
    loads = outcome.loads
    return (
        "Head loads",
        _THIN,
        _line("Peak deceleration", f"{loads.peak_deceleration_g:.4g} g"),
        _line("Peak moment about the CG", f"{loads.peak_resultant_moment_Nm:.4g} N.m"),
        _line("Mean resultant force", f"{loads.mean_resultant_force_N:.4g} N"),
        "",
    )


def _divot_lines(outcome: ShotOutcome) -> tuple[str, ...]:
    """Divot section, or nothing when no divot profile was produced."""
    if outcome.divot is None:
        return ()
    divot = outcome.divot
    return (
        "Divot",
        _THIN,
        _line(
            "Entry behind ball",
            f"{divot.entry_distance_behind_ball_m * 1e3:.4g} mm",
        ),
        _line("Maximum depth", f"{divot.max_depth_m * 1e3:.4g} mm"),
        _line("Length", f"{divot.length_m * 1e3:.4g} mm"),
        _line("Sand under the sole", f"{divot.mass_kg:.4g} kg"),
        _line("Sand accelerated", divot.accelerated_mass.summary()),
        "",
    )


DIG_SKID_CAVEAT = (
    "the dig-versus-skid verdict is UNCALIBRATED and is not a finding: the "
    "descent-return ratio separates the design space, but no vertical "
    "restitution has been published for a wedge sole leaving bunker sand, so "
    "its DIG and SKID thresholds are conventions; and the F0 model reads more "
    "marketed bounce as more dig in the shallow regime, so this is never a "
    "bounce recommendation (#8703)"
)
"""The sentence a dig-versus-skid verdict is never shown without (#8703)."""


def _dig_skid_lines(outcome: ShotOutcome) -> tuple[str, ...]:
    """Dig-versus-skid section, or nothing when it was not evaluated.

    The verdict never appears without its calibration state. The metric type
    already refuses to hold one without the other, and this is where that
    guarantee reaches the reader -- the same rule :data:`CARRY_CAVEAT` applies
    to carry.
    """
    if outcome.dig_skid is None:
        return ()
    skid = outcome.dig_skid
    dig_skid_verdict = skid.verdict
    state = "calibrated" if skid.calibration.calibrated else "UNCALIBRATED"
    return (
        "Dig versus skid",
        _THIN,
        _line("Verdict", f"{dig_skid_verdict.value.upper()} ({state})"),
        _line("Descent returned", f"{skid.descent_return_ratio:.3f}"),
        _line(
            "Entry descent / exit climb",
            f"{skid.entry_descent_speed_mps:.3g} / {skid.exit_climb_speed_mps:.3g} m/s",
        ),
        _line("Submerged samples", f"{skid.calibration.submerged_samples}"),
        _line("Vertical sand impulse", f"{skid.vertical_sand_impulse_Ns:.4g} N.s"),
        f"  {DIG_SKID_CAVEAT}",
        "",
    )


CARRY_CAVEAT = (
    "carry is derived from the delivered sand impulse and the measured divot "
    "mass through an uncalibrated transfer efficiency; no published "
    "measurement of ball speed or launch angle out of sand exists (#8616)"
)
"""The sentence a carry number is never shown without (issue #8657)."""


def _ball_lines(outcome: ShotOutcome) -> tuple[str, ...]:
    """Ball section, or nothing when no carry was computed.

    The carry never appears without its own validity line: the model type
    already refuses to hold one without the other, and this is where that
    guarantee reaches the reader.
    """
    verdict = outcome.carry_verdict
    if outcome.carry_m is None or verdict is None:
        return ()
    return (
        "Ball",
        _THIN,
        _line("Carry", f"{outcome.carry_m:.4g} m"),
        _line("Carry validity", status_headline(verdict.status)),
        f"  {CARRY_CAVEAT}",
        "",
    )


def shot_report(outcome: ShotOutcome) -> str:
    """Render one shot: verdict first, then whatever numbers exist.

    Args:
        outcome: The shot outcome.

    Returns:
        A multi-line report.
    """
    lines = [
        _RULE,
        f"F0 dynamic RFT ({outcome.fidelity_tier.value})",
        _RULE,
        status_headline(outcome.status),
        "",
    ]
    delivered = outcome.delivered
    bounce = delivered.effective_bounce
    lines.extend(
        (
            "Delivered geometry",
            _THIN,
            _line("Effective loft", f"{delivered.effective_loft_deg:.2f} deg"),
            _line(
                f"Effective bounce ({bounce.convention.value})",
                f"{bounce.angle_deg:.2f} deg",
            ),
            _line("Aim offset", f"{delivered.aim_offset_deg:.2f} deg right"),
            _line(
                "Presentation to path", f"{delivered.presentation_bounce_deg:.2f} deg"
            ),
            "",
        )
    )
    if outcome.refused:
        lines.extend(
            (
                "No numbers are reported for a refused query.",
                "",
                verdict_report(outcome.verdict),
            )
        )
        return "\n".join(lines)

    lines.extend(
        (
            "Sand loads",
            _THIN,
            _line("Peak resultant force", _optional(outcome.peak_force_n, "N")),
            _line("Impulse", _optional(outcome.impulse_n_s, "N.s")),
            _line("Entry speed", _optional(outcome.entry_speed_mps, "m/s")),
            _line("Exit speed", _optional(outcome.exit_speed_mps, "m/s")),
            _line("Maximum depth", _optional(outcome.max_depth_m, "mm", 1e3)),
            _line("Contact duration", _optional(outcome.contact_duration_s, "ms", 1e3)),
            _line(
                "Peak inertial share",
                _optional(outcome.peak_inertial_fraction, "of the force"),
            ),
            _line("Solver runtime", _optional(outcome.runtime_s, "ms", 1e3)),
            "",
        )
    )
    if outcome.sole_field is not None:
        lines.extend(
            sole_field_text(outcome.sole_field, outcome.contact_patch).splitlines()
        )
    lines.extend(_head_load_lines(outcome))
    lines.extend(_divot_lines(outcome))
    lines.extend(_dig_skid_lines(outcome))
    lines.extend(_ball_lines(outcome))
    if outcome.unavailable:
        lines.append("Not computed")
        lines.append(_THIN)
        lines.extend(f"  {reason}" for reason in outcome.unavailable)
        lines.append("")
    lines.extend((verdict_report(outcome.verdict), ""))
    return "\n".join(lines)


def shade_grid(values: np.ndarray, *, peak: float | None = None) -> tuple[str, ...]:
    """Render a 2-D array as shaded rows, lightest character first.

    NaN cells render as a space and every present cell renders as at least
    ``.``, so "no sole element here" is visibly different from "an element
    that carried nothing" -- a distinction that is the whole point of a
    utilisation map.

    Args:
        values: ``(n, m)`` array; NaN marks an empty cell.
        peak: Value mapped to the darkest shade; defaults to the array max.

    Returns:
        One string per row.

    Raises:
        ValueError: If ``values`` is not two-dimensional.
    """
    grid = np.asarray(values, dtype=float)
    if grid.ndim != 2:
        raise ValueError(f"shade_grid needs a 2-D array, got {grid.ndim}D")
    finite = np.isfinite(grid)
    if not finite.any():
        return tuple(" " * grid.shape[1] for _ in range(grid.shape[0]))
    top = float(np.nanmax(grid)) if peak is None else float(peak)
    if top <= 0.0:
        top = 1.0
    levels = len(SHADE_RAMP) - 1
    scaled = 1 + np.clip(
        np.rint(np.where(finite, grid, 0.0) / top * (levels - 1)), 0, levels - 1
    )
    return tuple(
        "".join(
            SHADE_RAMP[int(level)] if flag else " "
            for level, flag in zip(row, mask, strict=True)
        )
        for row, mask in zip(scaled, finite, strict=True)
    )


PATCH_CONFOUND_CAVEAT = (
    "the contact patch on this design is a bounce-AND-camber result, not a "
    "bounce result: at least one spanwise station could not carry the camber "
    "area its relieved sole width implied, so the lofter refitted that station "
    "to its own narrower constructible band (#8698) -- and the stations that "
    "move are the heel and toe, which is where the patch is read, so any patch "
    "trend across a bounce sweep moves both terms at once"
)
"""What a patch series may not be read as, when the camber was substituted.

Issue #8707 asks for this explicitly. The patch shrinking across a bounce
sweep is the mechanism behind more bounce producing *more* depth and force,
but where camber was substituted, "bounce" is not the only thing that changed.

**This keys off the per-station account, not off the declared-versus-effective
aggregate**, because the aggregate is the weaker test and misses the common
case. Heel and toe relief narrows the sole toward the ends, a narrower sole
admits a narrower camber band, and so a declaration can be honoured exactly at
the declared width while the relieved stations are refitted. The shipped
``sm9_58_m`` preset is precisely that: 42.00 mm^2 declared, 42.00 mm^2
effective, inside its (38.70, 42.44) mm^2 band -- and three of its seventeen
stations refitted regardless. Keyed off the aggregate flag this caveat would
stay silent on the default design; keyed off the stations it fires, which is
when a caveat earns its place.
"""


def sole_field_text(field: SoleLoadField, patch: ContactPatch | None = None) -> str:
    """Render the per-element load field and the contact patch as numbers.

    The animated view answers the same questions by eye. This is what can be
    pasted into an issue, diffed between two runs, or read where no display
    exists.

    Args:
        field: The per-element load field.
        patch: The contact-patch series, when there is one.

    Returns:
        A multi-line report, opening with the validity statement the whole
        section must be read under.
    """
    depth, inertial = LoadComponent.DEPTH, LoadComponent.INERTIAL
    lines = [
        "Per-element sole load and contact patch",
        _THIN,
        _line("Validity", status_headline(field.status)),
        _line(
            "Resolution",
            f"{field.n_elements} sole elements x {field.n_frames} samples "
            f"(the 12x12 map above is this, binned and summed)",
        ),
        _line(
            f"Peak {depth.label.lower()}",
            f"{field.peak_resultant_force_N(depth):.4g} N at "
            f"{field.peak_time_s(depth) * 1e3:.2f} ms",
        ),
        f"  {depth.description}",
        _line(
            f"Peak {inertial.label.lower()}",
            f"{field.peak_resultant_force_N(inertial):.4g} N at "
            f"{field.peak_time_s(inertial) * 1e3:.2f} ms",
        ),
        f"  {inertial.description}",
        _line(
            "Inertial share at peak load",
            f"{field.peak_inertial_share:.1%} of the sole's own resultant",
        ),
    ]
    if patch is not None:
        lines.extend(
            (
                _line(
                    "Patch at first contact",
                    f"{patch.initial_area_m2 * 1e4:.4g} cm^2 at "
                    f"{patch.initial_time_s * 1e3:.2f} ms",
                ),
                _line(
                    "Largest patch",
                    f"{patch.peak_area_m2 * 1e4:.4g} cm^2 at "
                    f"{patch.peak_area_time_s * 1e3:.2f} ms",
                ),
                _line(
                    "Closest approach to leading edge",
                    f"{patch.closest_approach_m * 1e3:.2f} mm at "
                    f"{patch.time_of_closest_approach_s * 1e3:.2f} ms",
                ),
                _line(
                    "Sole span, leading to trailing",
                    f"{(patch.trailing_edge_m - patch.leading_edge_m) * 1e3:.2f} mm",
                ),
            )
        )
    lines.append("")
    return "\n".join(lines)


def sole_map_text(sole_load: SoleLoadMap) -> str:
    """Render the bounce-utilisation map as a shaded sole plan.

    Args:
        sole_load: The map.

    Returns:
        A multi-line report: the shaded plan, then the numbers that say where
        to grind.
    """
    utilisation = sole_load.utilisation
    rows = shade_grid(sole_load.density_pa_s, peak=sole_load.peak_density_pa_s)
    body = [
        "Bounce utilisation (impulse density over the sole)",
        _THIN,
        "  heel -> toe across, leading edge -> trailing edge down",
    ]
    body.extend(f"  |{row}|" for row in rows)
    centre = utilisation.centre_of_pressure_body_m
    body.extend(
        (
            "",
            _line("Sole area supplied", f"{utilisation.total_area_m2 * 1e4:.4g} cm^2"),
            _line(
                "Area that carried load",
                f"{utilisation.utilised_area_m2 * 1e4:.4g} cm^2 "
                f"({utilisation.utilisation_fraction:.1%})",
            ),
            _line(
                "Removable for free", f"{utilisation.removable_area_m2 * 1e4:.4g} cm^2"
            ),
            _line("Total sole impulse", f"{utilisation.total_impulse_Ns:.4g} N.s"),
            _line(
                "Centre of pressure",
                f"x {centre[0] * 1e3:.1f} mm, y {centre[1] * 1e3:.1f} mm (body)",
            ),
            "",
        )
    )
    return "\n".join(body)


def playability_text(playability: PlayabilityOutcome) -> str:
    """Render the playability window and the carry grid behind it.

    Args:
        playability: The measured window, or the reason there is not one.

    Returns:
        A multi-line report.
    """
    window = playability.window
    if window is None:
        return "\n".join(
            (
                "Playability window",
                _THIN,
                f"  not measured: {playability.unavailable_reason}",
                "",
            )
        )
    low, high = window.carry_band_m
    grid_verdict = playability.carry_verdict
    lines = [
        "Playability window (attack angle x sand firmness)",
        _THIN,
        _line(
            "Carry validity",
            status_headline(grid_verdict.status) if grid_verdict else "not measured",
        ),
        f"  {CARRY_CAVEAT}",
        _line("Target carry", f"{window.target_carry_m:.4g} m"),
        _line("Acceptance band", f"{low:.4g} to {high:.4g} m"),
        _line("Window area", f"{window.area:.4g} {window.area_unit}"),
        _line("Share of the domain", f"{window.fraction:.1%}"),
        _line("Largest connected region", f"{window.largest_connected_area:.4g}"),
        _line("Refused share", f"{window.refused_fraction:.1%}"),
        _line(
            "Nominal delivery inside",
            "yes" if window.contains_nominal else "no",
        ),
        "",
        "  carry [m], rows = attack angle (deg), columns = firmness (kg/cm^2)",
        "        " + " ".join(f"{v:>6.2f} " for v in playability.firmness_kg_per_cm2),
    ]
    for angle, row, flags in zip(
        playability.attack_angle_deg,
        playability.carry_m,
        window.in_window,
        strict=True,
    ):
        cells = " ".join(
            ("   ---" if not np.isfinite(value) else f"{value:>6.2f}")
            + ("*" if flag else " ")
            for value, flag in zip(row, flags, strict=True)
        )
        lines.append(f"  {angle:>5.1f} {cells}")
    lines.extend(("  (* inside the window, --- refused)", ""))
    return "\n".join(lines)


def _camber_area_text(evaluation: DesignEvaluation) -> str:
    """State the camber area, and every substitution behind it.

    A sole of a given width and bounce can only realise camber areas inside a
    band, so a declared value outside it is built as the nearest one that is
    constructible. Reporting only the declared number would tell the designer
    about a sole that was never built (issue #8698).

    The two substitution scopes are reported separately because they are
    separate facts. The declared number can be honoured exactly while the
    relieved heel and toe stations - narrower, and so admitting narrower
    bands - are refitted. Printing only the aggregate result would render the
    shipped ``sm9_58_m`` preset as a clean "42.0 mm^2" while three of its
    stations carry something else.

    Args:
        evaluation: The evaluated design.

    Returns:
        The rendered value.
    """
    effective_mm2 = evaluation.effective_camber_area_m2 * 1e6
    text = f"{effective_mm2:.1f} mm^2"
    if evaluation.aggregate_camber_was_clamped:
        declared_mm2 = evaluation.geometry.sole_camber_area_m2 * 1e6
        text += (
            f"  (declared {declared_mm2:.1f} mm^2; not constructible at this "
            "sole width and bounce)"
        )
    clamped = evaluation.clamped_camber_stations
    if clamped:
        text += (
            f"  [{len(clamped)} of {len(evaluation.camber_stations)} spanwise "
            "stations refitted to their own narrower bands]"
        )
    return text


def evaluation_report(evaluation: DesignEvaluation) -> str:
    """Render one design end to end.

    Args:
        evaluation: The evaluated design.

    Returns:
        A multi-line report.
    """
    geometry = evaluation.geometry
    design = evaluation.design
    sand = evaluation.sand
    header = [
        _RULE,
        f"Design: {design.name}  (grind {design.grind_preset})",
        f"Sand:   {sand.name}  "
        f"{sand.firmness_kg_per_cm2:.2f} kg/cm^2  "
        f"{sand.firmness_rating.value}",
        _RULE,
        _line("Static loft", f"{geometry.loft_deg:.1f} deg"),
        _line("Marketed bounce", f"{geometry.marketed_bounce.angle_deg:.2f} deg"),
        _line(
            "Geometric bounce (patent)",
            f"{geometry.geometric_bounce.angle_deg:.2f} deg",
        ),
        _line("Sole width", f"{geometry.sole_width_m * 1e3:.2f} mm"),
        _line("Sole entry angle", f"{geometry.sole_entry_angle_deg:.2f} deg"),
        _line("Leading-edge radius", f"{geometry.leading_edge_radius_m * 1e3:.2f} mm"),
        _line("Sole contour ratio", f"{geometry.sole_contour_ratio:.3f}"),
        _line("Camber area", _camber_area_text(evaluation)),
        _line("Head mass", f"{geometry.head_mass_kg * 1e3:.0f} g"),
        "",
    ]
    parts = ["\n".join(header), shot_report(evaluation.shot)]
    if evaluation.shot.sole_load is not None:
        parts.append(sole_map_text(evaluation.shot.sole_load))
    clamped_stations = evaluation.clamped_camber_stations
    if clamped_stations and evaluation.shot.contact_patch is not None:
        # The caveat belongs beside the patch numbers, not only beside the
        # geometry: a designer reading a patch trend across a bounce sweep is
        # exactly the reader who must be told the camber moved with it.
        #
        # Gated on the stations rather than on the aggregate flag: the
        # aggregate compares the declared area against the *declared* width's
        # band and is False on the shipped presets, so gating on it would
        # silence the caveat on exactly the design most people run.
        parts.append(
            f"  {PATCH_CONFOUND_CAVEAT}\n"
            f"  ({len(clamped_stations)} of {len(evaluation.camber_stations)} "
            "spanwise stations refitted)\n"
        )
    parts.append(playability_text(evaluation.playability))
    return "\n".join(parts)


def comparison_report(comparison: WorkbenchComparison) -> str:
    """Render the A/B answer: which sole is better, and how sure we are.

    Args:
        comparison: The two evaluations and their ranking.

    Returns:
        A multi-line report.
    """
    left, right = comparison.left, comparison.right
    lines = [
        _RULE,
        f"A/B: {left.design.name}  vs  {right.design.name}",
        _RULE,
        f"{'Metric':<30}{left.design.name:>18}{right.design.name:>18}",
        _THIN,
    ]
    for label, first, second in _comparison_rows(left, right):
        lines.append(f"{label:<30}{first:>18}{second:>18}")
    lines.append("")
    ranking = comparison.ranking
    if ranking is None:
        lines.extend(
            (
                "Ranking",
                _THIN,
                f"  not available: {comparison.ranking_unavailable_reason}",
                "",
            )
        )
        return "\n".join(lines)
    lines.extend(
        (
            "Ranking on absolute carry error over the shared delivery sweep",
            _THIN,
            _line("Conditions compared", str(comparison.shared_points)),
            _line("Leader", ranking.best),
            _line(
                "Probability it is better",
                f"{float(ranking.probability_best.max()):.0%}",
            ),
            _line(
                "Distinguishable",
                "yes"
                if comparison.separated
                else "NO - the intervals overlap, so this study does not separate them",
            ),
            "",
        )
    )
    for index, name in enumerate(ranking.names):
        lines.append(
            f"  {name:<24}mean {ranking.mean[index]:.3f} m  "
            f"95% CI [{ranking.ci_low[index]:.3f}, {ranking.ci_high[index]:.3f}] m"
        )
    lines.append("")
    return "\n".join(lines)


#: The side-by-side table: label, key understood by :func:`_comparison_cell`.
_COMPARISON_ROWS: tuple[tuple[str, str], ...] = (
    ("Verdict", "status"),
    ("Marketed bounce", "bounce"),
    ("Peak force", "peak_force_n"),
    ("Maximum depth", "max_depth_m"),
    ("Carry", "carry_m"),
    ("Sole utilisation", "utilisation"),
    ("Playability window area", "window_area"),
    ("Window share of domain", "window_fraction"),
)

#: Unit and scale for the plain :class:`ShotOutcome` attributes.
_SHOT_UNITS: dict[str, tuple[str, float]] = {
    "peak_force_n": ("N", 1.0),
    "max_depth_m": ("mm", 1e3),
    "carry_m": ("m", 1.0),
}


def _comparison_rows(
    left: DesignEvaluation, right: DesignEvaluation
) -> tuple[tuple[str, str, str], ...]:
    """Build the side-by-side metric rows for two evaluations."""
    return tuple(
        (label, _comparison_cell(left, key), _comparison_cell(right, key))
        for label, key in _COMPARISON_ROWS
    )


def _comparison_cell(evaluation: DesignEvaluation, key: str) -> str:
    """Format one cell of the side-by-side table."""
    shot = evaluation.shot
    window = evaluation.playability.window
    if key == "status":
        return shot.status.value
    if key == "bounce":
        bounce = evaluation.geometry.marketed_bounce
        return f"{bounce.angle_deg:.2f} deg"
    if key == "utilisation":
        if shot.sole_load is None:
            return "-"
        utilisation = shot.sole_load.utilisation
        return f"{utilisation.utilisation_fraction * 100.0:.1f}%"
    if key == "window_area":
        return "-" if window is None else f"{window.area:.4g}"
    if key == "window_fraction":
        return "-" if window is None else f"{window.fraction * 100.0:.1f}%"
    value = getattr(shot, key)
    if value is None:
        return "-"
    unit, scale = _SHOT_UNITS[key]
    return f"{value * scale:.4g} {unit}"

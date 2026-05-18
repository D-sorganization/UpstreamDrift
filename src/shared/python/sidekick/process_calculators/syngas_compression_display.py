"""Pure display/formatting helpers for syngas compression results.

All functions are free of Qt and self dependencies.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

from .constants import ATOL_ZERO, CELSIUS_TO_KELVIN_OFFSET

if TYPE_CHECKING:
    from matplotlib.figure import Figure


def format_results_text(result: dict[str, Any], analysis: dict[str, Any]) -> str:
    """Format compression results as a human-readable string."""
    if not (result is not None):
        raise ValueError("result must be provided")
    parts = [
        "SYNGAS COMPRESSION CALCULATION RESULTS\n",
        "=" * 50 + "\n\n",
    ]

    mix_props = result["mixture_properties"]
    parts.extend(
        [
            "Mixture Properties:\n",
            f"  Molecular Weight: {mix_props['molecular_weight']:.2f} g/mol\n",
            f"  Critical Temperature: {mix_props['critical_temperature']:.1f} K\n",
            f"  Critical Pressure: {mix_props['critical_pressure']:.1f} bar\n",
            f"  Heat Capacity Ratio (\u03b3): {mix_props['heat_capacity_ratio']:.3f}\n\n",
            "Compression Stages:\n",
            "-" * 30 + "\n",
        ]
    )

    for stage_result in result["stages"]:
        stage_num = stage_result["stage_number"]
        parts.extend(
            [
                f"\nStage {stage_num}:\n",
                f"  Inlet Temperature: {stage_result['inlet_temp']:.1f} K "
                f"({stage_result['inlet_temp'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",
                f"  Outlet Temperature: {stage_result['outlet_temp']:.1f} K "
                f"({stage_result['outlet_temp'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",
                f"  Heat Rise: {stage_result['heat_rise']:.1f} K\n",
                f"  Pressure Ratio: {stage_result['pressure_ratio']:.2f}\n",
                f"  Power Required: {stage_result['power_hp']:.1f} HP\n",
            ]
        )
        water_info = stage_result["water_dropout"]
        if water_info["water_dropout"] > ATOL_ZERO:
            parts.extend(
                [
                    f"  Water Dropout: {water_info['water_dropout']:.3f} mol%\n",
                    f"  Condensation Rate: {water_info['condensation_rate']:.1f}%\n",
                ]
            )

    parts.extend(
        [
            "\nSUMMARY:\n",
            "-" * 20 + "\n",
            f"Total Power Required: {result['total_power_hp']:.1f} HP\n",
            f"Final Temperature: {result['final_temperature']:.1f} K "
            f"({result['final_temperature'] - CELSIUS_TO_KELVIN_OFFSET:.1f} deg C)\n",
            f"Final Pressure: {result['final_pressure']:.1f} bar\n",
            f"Total Water Dropout: {analysis['total_water_dropout']:.3f} mol%\n",
        ]
    )
    if analysis["average_efficiency"]:
        parts.append(
            f"Average Efficiency: {analysis['average_efficiency'] * 100:.1f}%\n"
        )
    return "".join(parts)


def format_analysis_text(analysis: dict[str, Any]) -> str:
    """Format process analysis and concerns as a human-readable string."""
    if not (analysis is not None):
        raise ValueError("analysis must be provided")
    parts = [
        "PROCESS ANALYSIS & CONCERNS\n",
        "=" * 40 + "\n\n",
    ]

    if analysis["warnings"]:
        parts.extend(["⚠️  CRITICAL WARNINGS:\n", "-" * 25 + "\n"])
        for warning in analysis["warnings"]:
            parts.append(f"\u2022 {warning}\n")
        parts.append("\n")

    if analysis["concerns"]:
        parts.extend(["⚠️  CONCERNS:\n", "-" * 15 + "\n"])
        for concern in analysis["concerns"]:
            parts.append(f"\u2022 {concern}\n")
        parts.append("\n")

    if analysis["recommendations"]:
        parts.extend(["💡 RECOMMENDATIONS:\n", "-" * 20 + "\n"])
        for rec in analysis["recommendations"]:
            parts.append(f"\u2022 {rec}\n")
        parts.append("\n")

    if not analysis["warnings"] and not analysis["concerns"]:
        parts.extend(
            [
                "\u2705 No significant concerns detected.\n",
                "Process conditions appear to be within acceptable limits.\n",
            ]
        )

    return "".join(parts)


def render_compression_plots(
    figure: Figure,
    canvas: Any,
    result: dict[str, Any],
) -> None:
    """Render compression stage plots onto *figure* and refresh *canvas*."""
    if not (result is not None):
        raise ValueError("result must be provided")
    figure.clear()

    stages = result["stages"]
    stage_nums = [s["stage_number"] for s in stages]
    temperatures = [s["outlet_temp"] - CELSIUS_TO_KELVIN_OFFSET for s in stages]
    pressures = [s["pressure_ratio"] for s in stages]
    powers = [s["power_hp"] for s in stages]
    water_dropouts = [s["water_dropout"]["water_dropout"] for s in stages]

    ax1 = figure.add_subplot(2, 2, 1)
    ax2 = figure.add_subplot(2, 2, 2)
    ax3 = figure.add_subplot(2, 2, 3)
    ax4 = figure.add_subplot(2, 2, 4)

    ax1.plot(stage_nums, temperatures, "bo-", linewidth=2, markersize=8)
    ax1.set_xlabel("Compression Stage")
    ax1.set_ylabel("Temperature (\u00b0C)")
    ax1.set_title("Temperature Profile")
    ax1.grid(True, alpha=0.3)

    ax2.bar(stage_nums, pressures, alpha=0.7, color="green")
    ax2.set_xlabel("Compression Stage")
    ax2.set_ylabel("Pressure Ratio")
    ax2.set_title("Pressure Ratio per Stage")
    ax2.grid(True, alpha=0.3)

    ax3.bar(stage_nums, powers, alpha=0.7, color="orange")
    ax3.set_xlabel("Compression Stage")
    ax3.set_ylabel("Power (HP)")
    ax3.set_title("Power Requirement per Stage")
    ax3.grid(True, alpha=0.3)

    ax4.bar(stage_nums, water_dropouts, alpha=0.7, color="blue")
    ax4.set_xlabel("Compression Stage")
    ax4.set_ylabel("Water Dropout (mol%)")
    ax4.set_title("Water Dropout per Stage")
    ax4.grid(True, alpha=0.3)

    figure.tight_layout()
    canvas.draw()

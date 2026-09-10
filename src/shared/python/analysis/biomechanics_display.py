"""Renderer-independent display preparation for computed biomechanics results.

No biomechanical algorithm lives here: only unit presentation, vector component
selection and CSV serialization. Missing observations remain explicit gaps.
"""

from __future__ import annotations

import csv
import io
import math
from collections.abc import Mapping, Sequence
from typing import Any


def prepare_biomechanics_plot(
    result: Mapping[str, Any],
    selected: Sequence[str] | None = None,
    angle_unit: str = "deg",
) -> dict[str, Any]:
    """Return scalar display channels without mutating scientific results.

    Preconditions: finite increasing timestamps and aligned scalar/XYZ channels.
    Postconditions: all values are JSON-safe finite numbers or null gaps.
    """
    if angle_unit not in ("rad", "deg"):
        raise ValueError("angle_unit must be rad or deg")
    times = [float(t) for t in result["times"]]
    if any(not math.isfinite(t) for t in times) or any(
        b <= a for a, b in zip(times, times[1:], strict=False)
    ):
        raise ValueError("times must be finite and strictly increasing")
    channels: dict[str, Any] = {}
    for name, channel in result["channels"].items():
        values = channel["values"]
        if len(values) != len(times):
            raise ValueError(f"{name} channel length differs from times")
        unit = channel["unit"]
        factor = 180 / math.pi if angle_unit == "deg" and unit.startswith("rad") else 1
        display_unit = unit.replace("rad", "deg", 1) if factor != 1 else unit
        vector = bool(values) and isinstance(values[0], (list, tuple))
        if vector and any(
            not isinstance(row, (list, tuple)) or len(row) != 3 for row in values
        ):
            raise ValueError(f"{name} vector channel must have three components")
        for component in range(3) if vector else (None,):
            key = f"{name}.{'xyz'[component]}" if component is not None else name
            data = (
                [row[component] for row in values] if component is not None else values
            )
            channels[key] = {
                **channel,
                "unit": display_unit,
                "values": [_display_value(value, factor) for value in data],
            }
    if selected is not None:
        unknown = set(selected) - channels.keys()
        if unknown:
            raise ValueError(f"Unknown channels: {sorted(unknown)}")
        channels = {name: channels[name] for name in selected}
    events = {}
    for name, index in result.get("events", {}).items():
        if (
            not isinstance(index, int)
            or isinstance(index, bool)
            or not 0 <= index < len(times)
        ):
            raise ValueError("Event indices must refer to a recorded sample")
        events[name] = times[index]
    return {
        **result,
        "times": times,
        "channels": channels,
        "events": events,
        "source": result.get(
            "source", result.get("provenance", {}).get("source", "Unspecified")
        ),
    }


def _display_value(value: Any, factor: float) -> float | None:
    if value is None:
        return None
    converted = float(value) * factor
    return converted if math.isfinite(converted) else None


def biomechanics_csv(plot: Mapping[str, Any]) -> str:
    """Serialize prepared scalar channels with units and empty cells for gaps."""
    output = io.StringIO(newline="")
    writer = csv.writer(output)
    channels = plot["channels"]
    writer.writerow(
        ["Time (s)", *[f"{key} ({value['unit']})" for key, value in channels.items()]]
    )
    for index, time in enumerate(plot["times"]):
        writer.writerow(
            [time, *[channel["values"][index] for channel in channels.values()]]
        )
    return output.getvalue()


def render_biomechanics_figure(
    plot: Mapping[str, Any],
    figure: Any,
    colors: Mapping[str, str] | None = None,
    limits: tuple[float, float] | None = None,
) -> None:
    """Render one subplot per unit; preserve gaps and annotate event times.

    The caller owns the Matplotlib figure and can save it as PNG, PDF or SVG.
    """
    if limits is not None and (
        not all(math.isfinite(v) for v in limits) or limits[0] >= limits[1]
    ):
        raise ValueError("Axis limits must be finite and increasing")
    figure.clear()
    units = list(dict.fromkeys(c["unit"] for c in plot["channels"].values()))
    for index, unit in enumerate(units):
        axis = figure.add_subplot(len(units), 1, index + 1)
        for name, channel in plot["channels"].items():
            if channel["unit"] == unit:
                values = [math.nan if v is None else v for v in channel["values"]]
                axis.plot(
                    plot["times"], values, label=name, color=(colors or {}).get(name)
                )
        for label, time in plot.get("events", {}).items():
            axis.axvline(time, linestyle="--", alpha=0.5, label=label)
        axis.set(xlabel="Time (s)", ylabel=unit)
        if limits is not None:
            axis.set_ylim(*limits)
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize="small")
    figure.tight_layout()

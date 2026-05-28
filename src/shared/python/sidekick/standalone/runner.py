"""Headless calculator runner for ``sidekick run``.

Provides a registry-based dispatcher that loads a named calculator, feeds it
JSON inputs, and writes JSON results to stdout (or a file).  All code here is
intentionally free of GUI imports so it works inside CI and PyInstaller smoke
tests without a display.
"""

from __future__ import annotations

import csv
import difflib
import io
import json
import logging
import math
from pathlib import Path
import sys
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps CLI name → callable(inputs: dict) -> dict.
# Add new calculators here; do NOT import GUI modules at module level.
_REGISTRY: dict[str, Any] = {}


def register(name: str):
    """Decorator: register a headless calculation function under *name*."""

    def _decorator(fn):
        _REGISTRY[name] = fn
        return fn

    return _decorator


# ---------------------------------------------------------------------------
# Built-in: wgs_reactor
# ---------------------------------------------------------------------------


@register("wgs_reactor")
def _wgs_reactor(inputs: dict) -> dict:
    """Headless Water-Gas Shift reactor calculation.

    Computes equilibrium composition and key performance metrics without
    any GUI or matplotlib dependencies.

    Args:
        inputs: dict with keys:
            temperature_c  (float) reactor temperature in °C (default 350)
            co_fraction    (float) inlet CO mole fraction (default 0.30)
            h2o_fraction   (float) inlet H2O mole fraction (default 0.40)
            co2_fraction   (float) inlet CO2 mole fraction (default 0.10)
            h2_fraction    (float) inlet H2 mole fraction (default 0.20)
            pressure_bar   (float) reactor pressure in bar (default 20.0)

    Returns:
        dict with equilibrium mole fractions and CO conversion.

    Postcondition:
        All returned mole fractions are in [0, 1] and sum to ≈ 1.
    """
    assert isinstance(inputs, dict), "inputs must be a dict"

    R = 8.314  # J/(mol·K)
    delta_h = -41000.0  # J/mol
    delta_s = -42.0  # J/(mol·K)

    T_c = float(inputs.get("temperature_c", 350.0))
    assert -273.15 < T_c <= 2000.0, f"temperature_c {T_c} out of range"
    T_k = T_c + 273.15

    y_co_in = float(inputs.get("co_fraction", 0.30))
    y_h2o_in = float(inputs.get("h2o_fraction", 0.40))
    y_co2_in = float(inputs.get("co2_fraction", 0.10))
    y_h2_in = float(inputs.get("h2_fraction", 0.20))

    for name, val in [
        ("co_fraction", y_co_in),
        ("h2o_fraction", y_h2o_in),
        ("co2_fraction", y_co2_in),
        ("h2_fraction", y_h2_in),
    ]:
        assert 0.0 <= val <= 1.0, f"{name}={val} must be in [0, 1]"

    # Equilibrium constant K = exp(-ΔG / RT) where ΔG = ΔH - TΔS
    delta_g = delta_h - T_k * delta_s
    K_eq = math.exp(-delta_g / (R * T_k))

    # Solve for extent of reaction ξ using the quadratic form of Kp:
    # K = (y_co2 + ξ)(y_h2 + ξ) / ((y_co - ξ)(y_h2o - ξ))
    # K(y_co - ξ)(y_h2o - ξ) = (y_co2 + ξ)(y_h2 + ξ)
    # (K-1)ξ² - [K(y_co + y_h2o) + y_co2 + y_h2]ξ + K·y_co·y_h2o - y_co2·y_h2 = 0
    a = K_eq - 1.0
    b = -(K_eq * (y_co_in + y_h2o_in) + y_co2_in + y_h2_in)
    c = K_eq * y_co_in * y_h2o_in - y_co2_in * y_h2_in

    if abs(a) < 1e-12:
        xi = -c / b if abs(b) > 1e-12 else 0.0
    else:
        disc = b * b - 4.0 * a * c
        disc = max(disc, 0.0)
        xi_pos = (-b + math.sqrt(disc)) / (2.0 * a)
        xi_neg = (-b - math.sqrt(disc)) / (2.0 * a)
        xi = xi_pos if 0.0 <= xi_pos <= min(y_co_in, y_h2o_in) else xi_neg

    xi = max(0.0, min(xi, min(y_co_in, y_h2o_in)))

    y_co_eq = y_co_in - xi
    y_h2o_eq = y_h2o_in - xi
    y_co2_eq = y_co2_in + xi
    y_h2_eq = y_h2_in + xi

    co_conversion = xi / y_co_in if y_co_in > 0 else 0.0

    equilibrium_composition = {
        "co": y_co_eq,
        "h2o": y_h2o_eq,
        "co2": y_co2_eq,
        "h2": y_h2_eq,
    }

    result = {
        "temperature_c": T_c,
        "equilibrium_constant": K_eq,
        "extent_of_reaction": xi,
        "co_conversion_fraction": co_conversion,
        "equilibrium_composition": equilibrium_composition,
    }

    total = sum(equilibrium_composition.values())
    assert abs(total - 1.0) < 1e-6, f"mole fractions sum to {total}, expected 1.0"
    assert 0.0 <= co_conversion <= 1.0, f"co_conversion={co_conversion} out of [0,1]"

    return result


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def run_calculator(
    calculator: str,
    inputs_path: str,
    output: str = "-",
    format: str = "json",
) -> int:
    """Run *calculator* with inputs from *inputs_path* and write results.

    Args:
        calculator:  Name of the registered calculator (e.g. ``wgs_reactor``).
        inputs_path: Path to a JSON file with calculator inputs.
        output:      Output path; ``"-"`` means stdout.
        format:      Output format: ``"json"`` (default) or ``"csv"``.

    Returns:
        Exit codes:
          0 — success
          1 — I/O error (missing file, JSON parse error, write failure)
          3 — validation or calculation failure
          4 — unknown calculator id

    Precondition:
        *calculator* must be a non-empty string.
        *inputs_path* must point to a readable JSON file.
    """
    assert isinstance(calculator, str) and calculator, (
        "calculator name must be non-empty"
    )
    assert isinstance(inputs_path, str) and inputs_path, "inputs_path must be non-empty"

    if calculator not in _REGISTRY:
        matches = difflib.get_close_matches(
            calculator, sorted(_REGISTRY), n=3, cutoff=0.4
        )
        sys.stderr.write(
            json.dumps(
                {"error": f"Unknown calculator '{calculator}'", "closest": matches}
            )
            + "\n"
        )
        return 4

    path = Path(inputs_path)
    if not path.exists():
        logger.error("Inputs file not found: %s", path)
        sys.stderr.write(json.dumps({"error": f"Inputs file not found: {path}"}) + "\n")
        return 1

    try:
        with open(path, encoding="utf-8") as fh:
            inputs = json.load(fh)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse inputs JSON: %s", exc)
        sys.stderr.write(
            json.dumps({"error": f"Failed to parse inputs JSON: {exc}"}) + "\n"
        )
        return 1

    fn = _REGISTRY[calculator]

    if hasattr(fn, "validate_inputs") and hasattr(fn, "calculate"):
        vr = fn.validate_inputs(inputs)
        if not vr.valid:
            sys.stderr.write(json.dumps({"errors": vr.errors}) + "\n")
            return 3
        calc_result = fn.calculate(inputs)
        values: dict[str, Any] = getattr(calc_result, "values", {})
        units: dict[str, str] = getattr(calc_result, "units", {})
        output_data: Any = {"values": values, "units": units}
        warnings = getattr(calc_result, "warnings", [])
        if warnings:
            output_data["warnings"] = warnings
    else:
        try:
            raw = fn(inputs)
        except (ValueError, AssertionError) as exc:
            logger.error("Calculation failed: %s", exc)
            sys.stderr.write(json.dumps({"errors": [str(exc)]}) + "\n")
            return 3
        values = raw if isinstance(raw, dict) else {}
        units = {}
        output_data = raw

    if format == "csv":
        buf = io.StringIO()
        writer = csv.writer(buf)
        writer.writerow(["metric", "value", "unit"])
        for key, val in values.items():
            writer.writerow([key, val, units.get(key, "")])
        output_str = buf.getvalue()
    else:
        output_str = json.dumps(output_data, indent=2) + "\n"

    if output == "-":
        sys.stdout.write(output_str)
    else:
        out_path = Path(output)
        try:
            out_path.parent.mkdir(parents=True, exist_ok=True)
            out_path.write_text(output_str, encoding="utf-8")
        except OSError as exc:
            logger.error("sidekick run failed: %s", exc)
            sys.stderr.write(json.dumps({"error": f"Write failed: {exc}"}) + "\n")
            return 1
        logger.info("Results written to %s", out_path)

    return 0


def list_calculators() -> list[str]:
    """Return sorted list of registered calculator names."""
    return sorted(_REGISTRY.keys())

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
import sys
from collections.abc import Callable
from pathlib import Path
from typing import Any

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

# Maps CLI name → callable(inputs: dict) -> dict.
#
# Entries are populated lazily by :func:`_ensure_registered` so that this
# module imports with ZERO GUI / matplotlib / scipy dependencies (the
# canonical ``process_calculators`` engines guard their PyQt6 imports but
# still pull in heavy scientific stacks; deferring keeps ``sidekick run``
# importable in headless CI and PyInstaller smoke tests — see
# ``tests/.../test_run.py::test_runner_module_does_not_import_pyqt6``).
_REGISTRY: dict[str, Any] = {}
_REGISTERED = False
_CalculatorFunction = Callable[[dict[str, Any]], Any]


def register(name: str) -> Callable[[_CalculatorFunction], _CalculatorFunction]:
    """Decorator: register a headless calculation function under *name*."""

    def _decorator(fn: _CalculatorFunction) -> _CalculatorFunction:
        _REGISTRY[name] = fn
        return fn

    return _decorator


# ---------------------------------------------------------------------------
# Canonical-engine adapters (#7067)
# ---------------------------------------------------------------------------
#
# Each adapter is a thin ``callable(inputs: dict) -> dict`` that delegates to
# the single canonical engine in ``sidekick.process_calculators`` — the SAME
# code path the GUI calculators use. No physics or constants are
# re-implemented here (DRY): in particular WGS routes through
# ``WGSReactorEngine`` so the equilibrium constant comes from the canonical
# ``WGS_DELTA_H`` / ``WGS_DELTA_S`` and can never silently diverge.


def _adapt_wgs_reactor(inputs: dict) -> dict:
    """Water-Gas Shift equilibrium via the canonical ``WGSReactorEngine``.

    Args:
        inputs: dict with optional keys ``temperature_c`` (°C, default 350),
            ``co_fraction``/``h2o_fraction``/``co2_fraction``/``h2_fraction``
            (inlet mole fractions), ``pressure_bar`` (default 20.0).

    Returns:
        dict with ``co_conversion_fraction`` and an ``equilibrium_composition``
        whose mole fractions sum to ≈ 1.

    Postcondition:
        All returned mole fractions are in [0, 1] and sum to ≈ 1.
    """
    assert isinstance(inputs, dict), "inputs must be a dict"
    from sidekick.process_calculators.constants import CELSIUS_TO_KELVIN_OFFSET
    from sidekick.process_calculators.wgs_reactor_calculator import WGSReactorEngine

    t_c = float(inputs.get("temperature_c", 350.0))
    assert -273.15 < t_c <= 2000.0, f"temperature_c {t_c} out of range"
    inlet = {
        "CO": float(inputs.get("co_fraction", 0.30)),
        "H2O": float(inputs.get("h2o_fraction", 0.40)),
        "CO2": float(inputs.get("co2_fraction", 0.10)),
        "H2": float(inputs.get("h2_fraction", 0.20)),
    }
    for name, val in inlet.items():
        assert 0.0 <= val <= 1.0, f"{name} fraction={val} must be in [0, 1]"
    pressure_bar = float(inputs.get("pressure_bar", 20.0))

    engine = WGSReactorEngine()
    eq = engine.calculate_equilibrium_composition(
        inlet,
        t_c + CELSIUS_TO_KELVIN_OFFSET,
        pressure_bar,
        steam_ratio=0.0,  # inlet already carries the H2O fraction
    )
    # The canonical engine reports composition in mol% and conversion in %.
    composition = {k.lower(): v / 100.0 for k, v in eq["composition"].items()}
    return {
        "temperature_c": t_c,
        "equilibrium_constant": eq["equilibrium_constant"],
        "co_conversion_fraction": eq["conversion"] / 100.0,
        "h2_co_ratio": eq["h2_co_ratio"],
        "heat_released_kj": eq["heat_released"],
        "equilibrium_composition": composition,
    }


def _adapt_water_vapor_pressure(inputs: dict) -> dict:
    """Saturated water-vapour pressure via ``SyngasWaterCalculator``."""
    assert isinstance(inputs, dict), "inputs must be a dict"
    from sidekick.process_calculators.syngas_water_calculator import (
        SyngasWaterCalculator,
    )

    temperature_c = float(inputs.get("temperature_c", 25.0))
    method = str(inputs.get("method", "auto"))
    pressure_pa, method_used = SyngasWaterCalculator().calculate_vapor_pressure(
        temperature_c, method
    )
    return {
        "temperature_c": temperature_c,
        "vapor_pressure_pa": pressure_pa,
        "vapor_pressure_kpa": pressure_pa / 1000.0,
        "method": method_used,
    }


def _adapt_flare(inputs: dict) -> dict:
    """Flare sizing via the canonical ``FlareCalculator``."""
    assert isinstance(inputs, dict), "inputs must be a dict"
    from sidekick.process_calculators.flare_calculator import FlareCalculator

    total_flow = float(inputs.get("total_flow_kg_hr", 1000.0))
    composition = dict(inputs.get("gas_composition", {"CH4": 80.0, "CO2": 20.0}))
    temperature_k = float(inputs.get("temperature_k", 298.15))
    pressure_bar = float(inputs.get("pressure_bar", 1.5))
    design = FlareCalculator().calculate_flare_size(
        total_flow, composition, temperature_k, pressure_bar
    )
    return {
        "height_m": design.height,
        "diameter_m": design.diameter,
        "exit_velocity_m_s": design.exit_velocity,
        "heat_release_kw": design.heat_release,
        "radiation_intensity_kw_m2": design.radiation_intensity,
    }


def _adapt_financial(inputs: dict) -> dict:
    """Plant financial model via ``FinancialModelCalculator``."""
    assert isinstance(inputs, dict), "inputs must be a dict"
    from sidekick.process_calculators.financial_calculator import (
        FinancialModelCalculator,
        FinancialParameters,
    )

    # FinancialParameters is a dataclass with sane zero defaults; only pass
    # through keys it actually declares so unknown inputs are ignored.
    valid = set(FinancialParameters.__dataclass_fields__)
    params = FinancialParameters(**{k: v for k, v in inputs.items() if k in valid})
    results = FinancialModelCalculator().calculate_financial_model(params)
    return {
        "annual_product_tons": results.annual_product_tons,
        "total_revenue": results.total_revenue,
        "total_variable_costs": results.total_variable_costs,
        "net_income": results.net_income,
        "margin_per_ton": results.margin_per_ton,
    }


def _adapt_syngas_water(inputs: dict) -> dict:
    """Water content of syngas via ``SyngasWaterCalculator``.

    Distinct from ``water_vapor_pressure`` (which returns only the saturation
    pressure): this also reports the method-selected saturation pressure at
    the supplied dew-point/temperature for a saturated stream.
    """
    assert isinstance(inputs, dict), "inputs must be a dict"
    from sidekick.process_calculators.syngas_water_calculator import (
        SyngasWaterCalculator,
    )

    temperature_c = float(inputs.get("temperature_c", 40.0))
    total_pressure_pa = float(inputs.get("total_pressure_pa", 101325.0))
    method = str(inputs.get("method", "auto"))
    calc = SyngasWaterCalculator()
    p_sat, method_used = calc.calculate_vapor_pressure(temperature_c, method)
    # Saturated mole fraction of water = P_sat / P_total (Raoult/Dalton).
    mole_fraction = min(p_sat / total_pressure_pa, 1.0) if total_pressure_pa else 0.0
    return {
        "temperature_c": temperature_c,
        "saturation_pressure_pa": p_sat,
        "water_mole_fraction": mole_fraction,
        "method": method_used,
    }


def _ensure_registered() -> None:
    """Populate :data:`_REGISTRY` with the canonical-engine adapters once.

    Idempotent. Called at the top of every public entry point so the heavy
    ``process_calculators`` imports stay out of module import time.
    """
    global _REGISTERED
    if _REGISTERED:
        return
    _REGISTRY.setdefault("wgs_reactor", _adapt_wgs_reactor)
    _REGISTRY.setdefault("water_vapor_pressure", _adapt_water_vapor_pressure)
    _REGISTRY.setdefault("flare", _adapt_flare)
    _REGISTRY.setdefault("financial", _adapt_financial)
    _REGISTRY.setdefault("syngas_water", _adapt_syngas_water)
    _REGISTERED = True


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

    _ensure_registered()
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
        try:
            vr = fn.validate_inputs(inputs)
        except (ValueError, AssertionError) as exc:
            logger.error("Validation failed: %s", exc)
            sys.stderr.write(json.dumps({"errors": [str(exc)]}) + "\n")
            return 3
        if not vr.valid:
            sys.stderr.write(json.dumps({"errors": vr.errors}) + "\n")
            return 3
        try:
            calc_result = fn.calculate(inputs)
        except (ValueError, AssertionError) as exc:
            logger.error("Calculation failed: %s", exc)
            sys.stderr.write(json.dumps({"errors": [str(exc)]}) + "\n")
            return 3
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
            # Preserve the legacy "sidekick run failed" stderr prefix that
            # tests/unit/sidekick/test_cli.py treats as the user-facing
            # error contract, while still emitting the structured JSON
            # body for machine consumers (#6533).
            logger.error("sidekick run failed: %s", exc)
            sys.stderr.write(f"sidekick run failed: {exc}\n")
            sys.stderr.write(json.dumps({"error": f"Write failed: {exc}"}) + "\n")
            return 1
        logger.info("Results written to %s", out_path)

    return 0


def list_calculators() -> list[str]:
    """Return sorted list of registered calculator names."""
    _ensure_registered()
    return sorted(_REGISTRY.keys())


__all__ = [
    "list_calculators",
    "register",
    "run_calculator",
]

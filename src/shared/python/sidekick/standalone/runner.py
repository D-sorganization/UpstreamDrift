"""Headless calculator runner for ``sidekick run``.

Provides a registry-based dispatcher that loads a named calculator, feeds it
JSON or YAML inputs, and writes JSON or CSV results to stdout or a file. All
code here is intentionally free of GUI imports so it works inside CI and
PyInstaller smoke tests without a display.
"""

from __future__ import annotations

import csv
import contextlib
import difflib
import io
import json
import logging
import math
import re
import sys
from collections.abc import Mapping
from pathlib import Path
from typing import Any

from sidekick.protocols import CalculationResult, Calculator, ValidationResult

logger = logging.getLogger(__name__)

EXIT_OK = 0
EXIT_GENERIC = 1
EXIT_VALIDATION = 3
EXIT_UNKNOWN_CALCULATOR = 4

_CALCULATOR_ID_RE = re.compile(r"^[a-z][a-z0-9_]*$")
_OUTPUT_FORMATS = frozenset({"json", "csv"})
_YAML_SUFFIXES = frozenset({".yaml", ".yml"})


# Maps CLI name to a headless-safe Calculator implementation. Add new
# calculators here; do not import GUI modules at module level.
_REGISTRY: dict[str, Calculator] = {}


def register(name: str, calculator: Calculator | None = None):
    """Register a headless calculator under ``name``.

    Can be used as ``register("id", calculator)`` or as a class/function
    decorator for objects that implement the Calculator protocol.
    """
    _validate_calculator_id(name)

    def _decorator(candidate: Calculator):
        _REGISTRY[name] = candidate
        return candidate

    if calculator is not None:
        return _decorator(calculator)
    return _decorator


class WgsReactorCalculator:
    """Headless Water-Gas Shift reactor calculator."""

    @property
    def name(self) -> str:
        return "Water-Gas Shift Reactor"

    @property
    def version(self) -> str:
        return "1.0.0"

    def validate_inputs(self, inputs: dict[str, Any]) -> ValidationResult:
        errors: list[str] = []
        for key in ("co_fraction", "h2o_fraction", "co2_fraction", "h2_fraction"):
            try:
                value = float(inputs.get(key, 0.0))
            except (TypeError, ValueError):
                errors.append(f"{key} must be numeric")
                continue
            if not 0.0 <= value <= 1.0:
                errors.append(f"{key} must be in [0, 1]")
        try:
            temperature_c = float(inputs.get("temperature_c", 350.0))
        except (TypeError, ValueError):
            errors.append("temperature_c must be numeric")
        else:
            if not -273.15 < temperature_c <= 2000.0:
                errors.append("temperature_c must be in (-273.15, 2000]")

        return ValidationResult(valid=not errors, errors=errors)

    def calculate(self, inputs: dict[str, Any]) -> CalculationResult:
        result = _calculate_wgs(inputs)
        return CalculationResult(
            values=result,
            units={
                "temperature_c": "degC",
                "equilibrium_constant": "",
                "extent_of_reaction": "mol_fraction",
                "co_conversion_fraction": "fraction",
                "co_fraction": "fraction",
                "h2o_fraction": "fraction",
                "co2_fraction": "fraction",
                "h2_fraction": "fraction",
            },
            metadata={"calculator": "wgs_reactor", "version": self.version},
        )


def run_calculator(
    calculator: str,
    inputs_path: str,
    output: str | None = "-",
    *,
    fmt: str = "json",
) -> int:
    """Run ``calculator`` with inputs from ``inputs_path``.

    Returns one of ``EXIT_*`` constants. Validation failures are emitted as
    structured JSON on stderr; successful results are emitted on stdout unless
    ``output`` names a destination file.
    """
    try:
        _validate_calculator_id(calculator)
        _validate_format(fmt)
        resolved = _resolve_calculator(calculator)
        inputs = _load_inputs(Path(inputs_path))
        validation = resolved.validate_inputs(inputs)
        if not validation.valid:
            _write_error(
                {
                    "error": "validation_failed",
                    "calculator": calculator,
                    "errors": validation.errors,
                    "warnings": validation.warnings,
                },
                EXIT_VALIDATION,
            )
            return EXIT_VALIDATION

        result = resolved.calculate(inputs)
        rendered = _format_result(result, fmt)
        _write_output(rendered, output)
    except UnknownCalculatorError as exc:
        _write_error(
            {
                "error": "unknown_calculator",
                "calculator": exc.calculator,
                "suggestions": exc.suggestions,
            },
            EXIT_UNKNOWN_CALCULATOR,
        )
        return EXIT_UNKNOWN_CALCULATOR
    except (FileNotFoundError, KeyError, TypeError, ValueError) as exc:
        logger.debug("sidekick run failed: %s", exc)
        _write_error(
            {"error": "sidekick_run_failed", "message": str(exc)}, EXIT_GENERIC
        )
        return EXIT_GENERIC
    return EXIT_OK


def list_calculators() -> list[str]:
    """Return sorted list of registered calculator names."""
    return sorted(_REGISTRY.keys())


def closest_calculator_matches(name: str, *, n: int = 3) -> list[str]:
    """Return the closest registered or catalog-advertised calculator ids."""
    return difflib.get_close_matches(name, _known_calculator_ids(), n=n, cutoff=0.4)


class UnknownCalculatorError(ValueError):
    """Raised when a calculator id cannot be resolved."""

    def __init__(self, calculator: str, suggestions: list[str]) -> None:
        super().__init__(f"unknown calculator: {calculator}")
        self.calculator = calculator
        self.suggestions = suggestions


def _resolve_calculator(calculator: str) -> Calculator:
    resolved = _REGISTRY.get(calculator)
    if resolved is None:
        raise UnknownCalculatorError(calculator, closest_calculator_matches(calculator))
    return resolved


def _known_calculator_ids() -> list[str]:
    ids = set(_REGISTRY)
    ids.update(_catalog_calculator_ids())
    return sorted(ids)


def _catalog_calculator_ids() -> set[str]:
    # Feature discovery may import optional vendor-backed modules that emit
    # diagnostics during import. Keep ``sidekick run`` stderr reserved for the
    # structured JSON payload expected by shell callers.
    with (
        contextlib.redirect_stdout(io.StringIO()),
        contextlib.redirect_stderr(io.StringIO()),
    ):
        try:
            from sidekick.agent.feature_catalog import build_feature_catalog

            catalog = build_feature_catalog()
        except Exception:  # noqa: BLE001 - catalog discovery imports optional trees
            return set()
    ids: set[str] = set()
    for feature_id, entry in catalog.items():
        if entry.kind not in {"calculator", "process_calculator"}:
            continue
        _, _, tail = feature_id.partition(".")
        if tail:
            ids.add(tail)
    return ids


def _validate_calculator_id(calculator: str) -> None:
    if not _CALCULATOR_ID_RE.fullmatch(calculator):
        raise ValueError(
            f"calculator id must match ^[a-z][a-z0-9_]*$ (got {calculator!r})"
        )


def _validate_format(fmt: str) -> None:
    if fmt not in _OUTPUT_FORMATS:
        raise ValueError(f"format must be one of {sorted(_OUTPUT_FORMATS)}")


def _load_inputs(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(path)
    if not path.is_file():
        raise ValueError(f"inputs path is not a file: {path}")

    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in _YAML_SUFFIXES:
        try:
            import yaml
        except ImportError as exc:
            raise ValueError("YAML inputs require PyYAML") from exc
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)
    if not isinstance(payload, dict):
        raise ValueError("inputs payload must be a JSON/YAML object")
    return dict(payload)


def _format_result(result: CalculationResult | Mapping[str, Any], fmt: str) -> str:
    payload = _result_payload(result)
    if fmt == "json":
        return json.dumps(payload, indent=2, sort_keys=True) + "\n"
    if fmt == "csv":
        return _result_csv(payload)
    raise ValueError(f"unsupported output format: {fmt}")


def _result_payload(result: CalculationResult | Mapping[str, Any]) -> dict[str, Any]:
    if isinstance(result, CalculationResult):
        return {
            "values": result.values,
            "units": result.units,
            "warnings": result.warnings,
            "metadata": result.metadata,
        }
    values = dict(result)
    return {"values": values, "units": {}, "warnings": [], "metadata": {}}


def _result_csv(payload: Mapping[str, Any]) -> str:
    values = payload.get("values", {})
    units = payload.get("units", {})
    if not isinstance(values, Mapping):
        raise ValueError("result values must be a mapping")
    if not isinstance(units, Mapping):
        raise ValueError("result units must be a mapping")

    output = io.StringIO()
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(["key", "value"])
    for key, value in values.items():
        writer.writerow([key, value])
    writer.writerow(
        ["units", ";".join(f"{key}={value}" for key, value in units.items())]
    )
    return output.getvalue()


def _write_output(rendered: str, output: str | None) -> None:
    if output in {None, "-"}:
        sys.stdout.write(rendered)
        return
    out_path = Path(output)
    if not out_path.parent.exists():
        raise FileNotFoundError(out_path.parent)
    out_path.write_text(rendered, encoding="utf-8")


def _write_error(payload: Mapping[str, Any], code: int) -> None:
    data = dict(payload)
    data["exit_code"] = code
    sys.stderr.write(json.dumps(data, sort_keys=True) + "\n")


def _calculate_wgs(inputs: dict[str, Any]) -> dict[str, float]:
    r = 8.314
    delta_h = -41000.0
    delta_s = -42.0

    temperature_c = float(inputs.get("temperature_c", 350.0))
    temperature_k = temperature_c + 273.15
    co_in = float(inputs.get("co_fraction", 0.30))
    h2o_in = float(inputs.get("h2o_fraction", 0.40))
    co2_in = float(inputs.get("co2_fraction", 0.10))
    h2_in = float(inputs.get("h2_fraction", 0.20))

    delta_g = delta_h - temperature_k * delta_s
    equilibrium_constant = math.exp(-delta_g / (r * temperature_k))

    a = equilibrium_constant - 1.0
    b = -(equilibrium_constant * (co_in + h2o_in) + co2_in + h2_in)
    c = equilibrium_constant * co_in * h2o_in - co2_in * h2_in

    if abs(a) < 1e-12:
        extent = -c / b if abs(b) > 1e-12 else 0.0
    else:
        discriminant = max(b * b - 4.0 * a * c, 0.0)
        root_a = (-b + math.sqrt(discriminant)) / (2.0 * a)
        root_b = (-b - math.sqrt(discriminant)) / (2.0 * a)
        extent = root_a if 0.0 <= root_a <= min(co_in, h2o_in) else root_b

    extent = max(0.0, min(extent, min(co_in, h2o_in)))
    co_out = co_in - extent
    h2o_out = h2o_in - extent
    co2_out = co2_in + extent
    h2_out = h2_in + extent

    total = math.fsum((co_out, h2o_out, co2_out, h2_out))
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"mole fractions sum to {total}, expected 1.0")

    return {
        "temperature_c": temperature_c,
        "equilibrium_constant": equilibrium_constant,
        "extent_of_reaction": extent,
        "co_conversion_fraction": extent / co_in if co_in > 0.0 else 0.0,
        "co_fraction": co_out,
        "h2o_fraction": h2o_out,
        "co2_fraction": co2_out,
        "h2_fraction": h2_out,
    }


register("wgs_reactor", WgsReactorCalculator())

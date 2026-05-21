# ruff: noqa: E501
"""Specification validation with structured error output.

Design-by-Contract: validate_spec() is the single precondition gate.
All downstream modules assume the spec has passed validation.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from programmatic_pid.geometry import to_float
from programmatic_pid.types import SpecValidationError, ValidationIssue

logger = logging.getLogger(__name__)

_SCHEMA: dict[str, Any] | None = None


def _load_schema() -> dict[str, Any] | None:
    """Load JSON schema from the schema/ directory (best-effort)."""
    global _SCHEMA
    if _SCHEMA is not None:
        return _SCHEMA
    schema_path = (
        Path(__file__).resolve().parents[2] / "schema" / "pid_spec.schema.json"
    )
    if schema_path.exists():
        with open(schema_path, encoding="utf-8") as f:
            _SCHEMA = json.load(f)
    return _SCHEMA


def _equipment_dims(eq: dict[str, Any]) -> tuple[float, float]:
    return (
        to_float(eq.get("w", eq.get("width", 0.0))),
        to_float(eq.get("h", eq.get("height", 0.0))),
    )


def collect_issues(spec: Any) -> list[ValidationIssue]:
    """Return a list of structured validation issues (errors + warnings).

    Agent-friendly: callers can serialise these to JSON for programmatic fixes.
    """
    issues: list[ValidationIssue] = []

    if not isinstance(spec, dict):
        issues.append(ValidationIssue("", "Specification must be a YAML mapping."))
        return issues

    project = spec.get("project", {})
    if not project.get("id"):
        issues.append(ValidationIssue("project.id", "project.id is required"))
    if not (project.get("title") or project.get("document_title")):
        issues.append(
            ValidationIssue(
                "project.title", "project.title or project.document_title is required"
            )
        )

    equipment = spec.get("equipment", [])
    if not equipment:
        issues.append(ValidationIssue("equipment", "equipment list cannot be empty"))

    equipment_ids: set[str] = set()
    for idx, eq in enumerate(equipment):
        eq_id = eq.get("id")
        if not eq_id:
            issues.append(
                ValidationIssue(f"equipment[{idx}].id", "equipment entry missing id")
            )
            continue
        if eq_id in equipment_ids:
            issues.append(
                ValidationIssue(
                    f"equipment[{idx}].id", f"duplicate equipment id: {eq_id}"
                )
            )
        equipment_ids.add(eq_id)

        w, h = _equipment_dims(eq)
        if w <= 0 or h <= 0:
            issues.append(
                ValidationIssue(
                    f"equipment[{idx}]",
                    f"equipment {eq_id} has non-positive width/height",
                )
            )

    instrument_ids: set[str] = set()
    for idx, ins in enumerate(spec.get("instruments", [])):
        ins_id = ins.get("id")
        if not ins_id:
            issues.append(
                ValidationIssue(f"instruments[{idx}].id", "instrument entry missing id")
            )
            continue
        if ins_id in instrument_ids:
            issues.append(
                ValidationIssue(
                    f"instruments[{idx}].id", f"duplicate instrument id: {ins_id}"
                )
            )
        instrument_ids.add(ins_id)

    for idx, stream in enumerate(spec.get("streams", [])):
        sid = stream.get("id", "<unknown>")
        if "from" in stream:
            fr = stream.get("from", {})
            eq = fr.get("equipment")
            if eq and eq not in equipment_ids:
                issues.append(
                    ValidationIssue(
                        f"streams[{idx}].from.equipment",
                        f"stream {sid} references unknown from equipment: {eq}",
                    )
                )
        if "to" in stream:
            to = stream.get("to", {})
            eq = to.get("equipment")
            if eq and eq not in equipment_ids:
                issues.append(
                    ValidationIssue(
                        f"streams[{idx}].to.equipment",
                        f"stream {sid} references unknown to equipment: {eq}",
                    )
                )

    references = equipment_ids | instrument_ids
    for idx, loop in enumerate(spec.get("control_loops", [])):
        lid = loop.get("id", "<unknown>")
        meas = loop.get("measurement")
        final = loop.get("final_element")
        if not meas or not final:
            issues.append(
                ValidationIssue(
                    f"control_loops[{idx}]",
                    f"control loop {lid} missing measurement/final_element",
                )
            )
            continue
        if meas not in references:
            issues.append(
                ValidationIssue(
                    f"control_loops[{idx}].measurement",
                    f"control loop {lid} unknown measurement reference: {meas}",
                )
            )
        if final not in references:
            issues.append(
                ValidationIssue(
                    f"control_loops[{idx}].final_element",
                    f"control loop {lid} unknown final element reference: {final}",
                )
            )

    return issues


def validate_spec(spec: Any) -> None:
    """Validate spec and raise SpecValidationError if any errors found.

    Precondition: spec is a dict loaded from YAML.
    Postcondition: if returns normally, spec satisfies all structural + referential constraints.
    """
    # Optional: schema-based structural validation
    schema = _load_schema()
    if schema is not None:
        try:
            import jsonschema

            jsonschema.validate(spec, schema)
        except ImportError:
            logger.debug("jsonschema not installed; skipping schema validation")
        except Exception as exc:  # noqa: BLE001
            raise SpecValidationError(f"Schema violation: {exc}") from exc

    # Referential integrity checks
    issues = collect_issues(spec)
    errors = [i for i in issues if i.severity == "error"]
    if errors:
        raise SpecValidationError(
            "Invalid spec:\n- " + "\n- ".join(e.message for e in errors)
        )


def validate_spec_json(spec: Any) -> list[dict[str, str]]:
    """Agent-friendly validation that returns structured JSON-serializable issues.

    Never raises; callers inspect the returned list.
    """
    return [issue.to_dict() for issue in collect_issues(spec)]

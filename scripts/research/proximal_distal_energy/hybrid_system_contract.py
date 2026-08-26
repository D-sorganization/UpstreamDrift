"""Versioned hybrid-system contract for the proximal-distal model ladder."""

from __future__ import annotations

from argparse import ArgumentParser
import json
import math
from pathlib import Path, PurePosixPath
from typing import Any

from scripts.research.proximal_distal_energy._hybrid_system_contract_records import (
    AVAILABILITY as _AVAILABILITY,
    AUTHORITY_STATUS as _AUTHORITY_STATUS,
    COMPARISON_STATUS as _COMPARISON_STATUS,
    EXPECTED_TIER_IDS,
    SCHEMA_VERSION,
    ActuatorDynamics,
    AlgebraicConstraint,
    Control,
    EventUncertainty,
    Guard,
    HybridSystemContract,
    HybridTier,
    Impact,
    Mode,
    Observable,
    Reset,
    StateBlock,
)
from scripts.research.proximal_distal_energy._hybrid_system_contract_validation import (
    validate_references,
    validate_unique_ids,
)


def _text(name: str, value: object) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{name} must be non-empty")
    return value.strip()


def _availability(record: dict[str, Any]) -> tuple[str, str]:
    availability = _text("availability", record.get("availability"))
    if availability not in _AVAILABILITY:
        raise ValueError(f"availability must be one of {sorted(_AVAILABILITY)}")
    reason = str(record.get("reason", "")).strip()
    if availability != "implemented" and not reason:
        raise ValueError(
            "reason must be non-empty when availability is not implemented"
        )
    return availability, reason


def _strings(name: str, value: object) -> tuple[str, ...]:
    if not isinstance(value, list) or not value:
        raise ValueError(f"{name} must be a non-empty list")
    result = tuple(_text(name, item) for item in value)
    if len(result) != len(set(result)):
        raise ValueError(f"{name} values must be unique")
    return result


def load_contract_from_json(
    path: str | Path, *, repository_root: str | Path
) -> HybridSystemContract:
    """Load and validate a contract from canonical JSON."""
    return load_contract_from_dict(
        json.loads(Path(path).read_text(encoding="utf-8")),
        repository_root=repository_root,
    )


def load_contract_from_dict(
    record: dict[str, object], *, repository_root: str | Path
) -> HybridSystemContract:
    """Validate a decoded contract and all internal references."""
    if record.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    issue = _text("issue", record.get("issue"))
    boundary = _text("inference_boundary", record.get("inference_boundary"))
    tolerances = _parse_tolerances(record.get("numerical_tolerances"))
    raw_tiers = record.get("tiers")
    if not isinstance(raw_tiers, list):
        raise ValueError("tiers must be a list")
    root = Path(repository_root).resolve()
    tiers = tuple(_parse_tier(item, root) for item in raw_tiers)
    ids = tuple(tier.tier_id for tier in tiers)
    if ids != EXPECTED_TIER_IDS:
        raise ValueError(f"tier order must be {EXPECTED_TIER_IDS}")
    return HybridSystemContract(SCHEMA_VERSION, issue, boundary, tolerances, tiers)


def _parse_tolerances(value: object) -> dict[str, float]:
    if not isinstance(value, dict) or not value:
        raise ValueError("numerical_tolerances must be a non-empty object")
    result: dict[str, float] = {}
    for key, raw in value.items():
        name = _text("tolerance name", key)
        number = float(raw)
        if not math.isfinite(number) or number <= 0.0:
            raise ValueError(f"{name} tolerance must be finite and positive")
        result[name] = number
    return result


def _parse_tier(raw: object, root: Path) -> HybridTier:
    if not isinstance(raw, dict):
        raise ValueError("each tier must be an object")
    tier_id = _text("tier_id", raw.get("tier_id"))
    status = _text("authority_status", raw.get("authority_status"))
    if status not in _AUTHORITY_STATUS:
        raise ValueError(f"authority_status must be one of {sorted(_AUTHORITY_STATUS)}")
    source_paths = _source_paths(raw.get("source_paths"), root, status)
    states = _items(raw, "state_blocks", _parse_state)
    controls = _items(raw, "controls", _parse_control)
    constraints = _items(raw, "algebraic_constraints", _parse_constraint)
    modes = _items(raw, "modes", _parse_mode)
    guards = _items(raw, "guards", _parse_guard)
    resets = _items(raw, "resets", _parse_reset)
    impacts = _items(raw, "impacts", _parse_impact)
    actuators = _items(raw, "actuator_dynamics", _parse_actuator)
    uncertainties = _items(raw, "uncertain_event_surfaces", _parse_uncertainty)
    observables = _items(raw, "observables", _parse_observable)
    validate_unique_ids(states, "block_id")
    validate_unique_ids(controls, "control_id")
    validate_unique_ids(constraints, "constraint_id")
    validate_unique_ids(modes, "mode_id")
    validate_unique_ids(guards, "guard_id")
    validate_unique_ids(resets, "reset_id")
    validate_unique_ids(impacts, "impact_id")
    validate_unique_ids(actuators, "actuator_id")
    validate_unique_ids(uncertainties, "uncertainty_id")
    validate_unique_ids(observables, "observable_id")
    validate_references(
        states, controls, modes, guards, resets, impacts, actuators, uncertainties
    )
    eligibility = _text("comparison_eligibility", raw.get("comparison_eligibility"))
    if eligibility not in _COMPARISON_STATUS:
        raise ValueError(
            f"comparison_eligibility must be one of {sorted(_COMPARISON_STATUS)}"
        )
    blockers = tuple(str(item).strip() for item in raw.get("comparison_blockers", []))
    if any(not item for item in blockers):
        raise ValueError("comparison blockers must be non-empty strings")
    if status == "unavailable" and (eligibility != "unavailable" or not blockers):
        raise ValueError("unavailable tiers require unavailable comparison blockers")
    return HybridTier(
        tier_id=tier_id,
        title=_text("title", raw.get("title")),
        authority_status=status,
        source_paths=source_paths,
        state_blocks=states,
        controls=controls,
        algebraic_constraints=constraints,
        modes=modes,
        guards=guards,
        resets=resets,
        impacts=impacts,
        actuator_dynamics=actuators,
        uncertain_event_surfaces=uncertainties,
        observables=observables,
        limitations=_strings("limitations", raw.get("limitations")),
        falsifiers=_strings("falsifiers", raw.get("falsifiers")),
        comparison_eligibility=eligibility,
        comparison_blockers=blockers,
    )


def _source_paths(value: object, root: Path, status: str) -> tuple[str, ...]:
    if not isinstance(value, list):
        raise ValueError("source_paths must be a list")
    paths = tuple(_text("source path", item).replace("\\", "/") for item in value)
    if len(paths) != len(set(paths)):
        raise ValueError("source_paths must be unique")
    if status == "unavailable" and paths:
        raise ValueError("unavailable tiers cannot claim source paths")
    if status != "unavailable" and not paths:
        raise ValueError("available or partial tiers require source paths")
    for item in paths:
        pure = PurePosixPath(item)
        if pure.is_absolute() or ".." in pure.parts:
            raise ValueError("source paths must be repository-relative")
        resolved = (root / Path(*pure.parts)).resolve()
        if not resolved.is_relative_to(root) or not resolved.is_file():
            raise ValueError(f"source path is missing or outside repository: {item}")
    return paths


def _items(raw: dict[str, Any], name: str, parser: Any) -> tuple[Any, ...]:
    values = raw.get(name)
    if not isinstance(values, list) or not values:
        raise ValueError(f"{name} must be a non-empty list")
    return tuple(parser(item) for item in values)


def _base(raw: object, identifier: str) -> tuple[dict[str, Any], str, str, str]:
    if not isinstance(raw, dict):
        raise ValueError(f"{identifier} record must be an object")
    availability, reason = _availability(raw)
    return raw, _text(identifier, raw.get(identifier)), availability, reason


def _parse_state(raw: object) -> StateBlock:
    item, key, available, reason = _base(raw, "block_id")
    return StateBlock(
        key,
        _strings("variables", item.get("variables")),
        _text("unit", item.get("unit")),
        _text("frame", item.get("frame")),
        available,
        reason,
    )


def _parse_control(raw: object) -> Control:
    item, key, available, reason = _base(raw, "control_id")
    bounds = _bounds(item.get("bounds"), "control bounds")
    actuator = item.get("actuator_id")
    return Control(
        key,
        _text("variable", item.get("variable")),
        _text("unit", item.get("unit")),
        bounds,
        None if actuator is None else _text("actuator_id", actuator),
        available,
        reason,
    )


def _parse_constraint(raw: object) -> AlgebraicConstraint:
    item, key, available, reason = _base(raw, "constraint_id")
    return AlgebraicConstraint(
        key,
        _text("kind", item.get("kind")),
        _text("expression", item.get("expression")),
        _text("residual_unit", item.get("residual_unit")),
        available,
        reason,
    )


def _parse_mode(raw: object) -> Mode:
    item, key, available, reason = _base(raw, "mode_id")
    return Mode(
        key,
        _text("invariant", item.get("invariant")),
        _text("contact_state", item.get("contact_state")),
        available,
        reason,
    )


def _parse_guard(raw: object) -> Guard:
    item, key, available, reason = _base(raw, "guard_id")
    return Guard(
        key,
        _text("source_mode", item.get("source_mode")),
        _text("target_mode", item.get("target_mode")),
        _text("surface", item.get("surface")),
        _text("direction", item.get("direction")),
        _text("uncertainty_id", item.get("uncertainty_id")),
        available,
        reason,
    )


def _parse_reset(raw: object) -> Reset:
    item, key, available, reason = _base(raw, "reset_id")
    return Reset(
        key,
        _text("guard_id", item.get("guard_id")),
        _text("state_map", item.get("state_map")),
        _text("energy_accounting", item.get("energy_accounting")),
        available,
        reason,
    )


def _parse_impact(raw: object) -> Impact:
    item, key, available, reason = _base(raw, "impact_id")
    guard = item.get("guard_id")
    return Impact(
        key,
        None if guard is None else _text("guard_id", guard),
        _text("impulse_law", item.get("impulse_law")),
        _text("energy_accounting", item.get("energy_accounting")),
        available,
        reason,
    )


def _parse_actuator(raw: object) -> ActuatorDynamics:
    item, key, available, reason = _base(raw, "actuator_id")
    state_ids = tuple(str(value).strip() for value in item.get("state_block_ids", []))
    delay = item.get("delay_s")
    delay_s = None if delay is None else float(delay)
    if delay_s is not None and (not math.isfinite(delay_s) or delay_s < 0.0):
        raise ValueError("delay_s must be finite and non-negative")
    return ActuatorDynamics(
        key,
        state_ids,
        _text("differential_law", item.get("differential_law")),
        _text("saturation", item.get("saturation")),
        delay_s,
        available,
        reason,
    )


def _parse_uncertainty(raw: object) -> EventUncertainty:
    item, key, available, reason = _base(raw, "uncertainty_id")
    return EventUncertainty(
        key,
        _text("guard_id", item.get("guard_id")),
        _text("parameterization", item.get("parameterization")),
        _bounds(item.get("bounds"), "event bounds"),
        _text("distribution_status", item.get("distribution_status")),
        available,
        reason,
    )


def _parse_observable(raw: object) -> Observable:
    item, key, available, reason = _base(raw, "observable_id")
    return Observable(
        key,
        _text("quantity", item.get("quantity")),
        _text("unit", item.get("unit")),
        _text("frame", item.get("frame")),
        available,
        reason,
    )


def _bounds(value: object, name: str) -> tuple[float, float] | None:
    if value is None:
        return None
    if not isinstance(value, list) or len(value) != 2:
        raise ValueError(f"{name} must contain two values")
    lower, upper = (float(part) for part in value)
    if not all(math.isfinite(part) for part in (lower, upper)) or lower > upper:
        raise ValueError(f"{name} must be finite and ordered")
    return lower, upper


def main(argv: list[str] | None = None) -> int:
    """Validate the registered hybrid-system contract."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("validate",))
    parser.add_argument("--path", type=Path)
    args = parser.parse_args(argv)
    root = Path(__file__).resolve().parents[3]
    path = (
        args.path
        or root
        / "docs/research/proximal_distal_energy_transfer/data/hybrid_system_contract_v1.json"
    )
    contract = HybridSystemContract.from_json(path, repository_root=root)
    statuses = {
        status: sum(tier.authority_status == status for tier in contract.tiers)
        for status in sorted(_AUTHORITY_STATUS)
    }
    print(
        json.dumps(
            {
                "schema_version": contract.schema_version,
                "tier_count": len(contract.tiers),
                "authority_statuses": statuses,
            },
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

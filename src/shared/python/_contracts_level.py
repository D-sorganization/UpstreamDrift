from __future__ import annotations

import enum
import logging
import os
from types import ModuleType
from typing import Any

logger = logging.getLogger(__name__)


class ContractLevel(enum.Enum):
    """Tri-state enforcement level for Design by Contract checks."""

    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


def _contract_level_value(level: object) -> str:
    """Return a supported value from any compatible enum definition."""
    if not isinstance(level, enum.Enum):
        raise TypeError("level must be an Enum with a supported contract level value")
    value = level.value
    if not isinstance(value, str) or value not in {"off", "warn", "enforce"}:
        raise ValueError(
            "level must be an Enum with contract level value "
            "'off', 'warn', or 'enforce'"
        )
    return value


def _contract_level_for_module(
    level: object, module: ModuleType
) -> tuple[enum.Enum, bool, Any]:
    """Prepare a module's normalized level without mutating its state."""
    target_enum = getattr(module, "ContractLevel", None)
    state = getattr(module, "_ContractState", None)
    state_level = getattr(state, "level", None)
    if not (isinstance(target_enum, type) and issubclass(target_enum, enum.Enum)):
        target_enum = type(state_level) if isinstance(state_level, enum.Enum) else None
    module_level = (
        target_enum(_contract_level_value(level))
        if target_enum is not None
        else ContractLevel(_contract_level_value(level))
    )
    off_level = getattr(target_enum, "OFF", None)
    enabled = (
        module_level != off_level
        if off_level is not None
        else _contract_level_value(module_level) != "off"
    )
    return module_level, enabled, state


def _resolve_contract_level() -> ContractLevel:
    env_val = os.environ.get("DBC_LEVEL", "").lower().strip()
    if env_val in ("off", "warn", "enforce"):
        return ContractLevel(env_val)
    return ContractLevel.ENFORCE


class _ContractState:
    level: ContractLevel = _resolve_contract_level()

    @classmethod  # type: ignore[misc]
    @property
    def enabled(cls) -> bool:
        return cls.level != ContractLevel.OFF


DBC_LEVEL: ContractLevel = _ContractState.level
CONTRACTS_ENABLED: bool = _ContractState.level != ContractLevel.OFF


def set_contract_level(level: enum.Enum) -> None:
    import sys

    level = ContractLevel(_contract_level_value(level))
    alias_updates: list[tuple[ModuleType, enum.Enum, bool, Any]] = []
    for mod_name in (
        __name__,
        "contracts",
        "shared.python.contracts",
        "src.shared.python.contracts",
    ):
        mod = sys.modules.get(mod_name)
        if mod is None:
            continue
        module_level, enabled, state = _contract_level_for_module(level, mod)
        alias_updates.append((mod, module_level, enabled, state))

    _ContractState.level = level
    for mod, module_level, enabled, state in alias_updates:
        try:
            mod.DBC_LEVEL = module_level  # type: ignore[attr-defined]
            mod.CONTRACTS_ENABLED = enabled  # type: ignore[attr-defined]
            if state is not None and state is not _ContractState:
                state.level = module_level
        except AttributeError:
            pass
    logger.info("Contract enforcement level set to %s", level.value)


def get_contract_level() -> ContractLevel:
    return _ContractState.level

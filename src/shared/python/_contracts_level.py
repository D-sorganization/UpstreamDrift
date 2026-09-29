from __future__ import annotations

import enum
import logging
import os

logger = logging.getLogger(__name__)


class ContractLevel(enum.Enum):
    """Tri-state enforcement level for Design by Contract checks."""

    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


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


def set_contract_level(level: ContractLevel) -> None:
    import sys

    _ContractState.level = level
    for mod_name in list(sys.modules):
        if mod_name in (
            __name__,
            "contracts",
            "shared.python.contracts",
            "src.shared.python.contracts",
        ):
            mod = sys.modules[mod_name]
            try:
                mod.DBC_LEVEL = level  # type: ignore[attr-defined]
                mod.CONTRACTS_ENABLED = level != ContractLevel.OFF  # type: ignore[attr-defined]
            except AttributeError:
                pass
    logger.info("Contract enforcement level set to %s", level.value)


def get_contract_level() -> ContractLevel:
    return _ContractState.level

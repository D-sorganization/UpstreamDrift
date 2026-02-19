"""Contract enforcement level configuration.

Controls how Design by Contract checks behave at runtime via the
``DBC_LEVEL`` environment variable:
  - ``enforce`` (default): Raise contract violation errors on failure.
  - ``warn``: Log violations at WARNING level but do not raise.
  - ``off``: Skip all contract checks (maximum performance).
"""

from __future__ import annotations

import enum
import os

from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class ContractLevel(enum.Enum):
    """Tri-state enforcement level for Design by Contract checks.

    Attributes:
        OFF: No checks (production hot path, maximum performance).
        WARN: Log violations at WARNING level without raising.
        ENFORCE: Raise contract violation errors on any failure.
    """

    OFF = "off"
    WARN = "warn"
    ENFORCE = "enforce"


def _resolve_contract_level() -> ContractLevel:
    """Determine the contract level from environment.

    Reads ``DBC_LEVEL`` environment variable.  Falls back to ``enforce``
    when ``__debug__`` is True (normal Python) or ``off`` when running
    with ``python -O``.
    """
    env_val = os.environ.get("DBC_LEVEL", "").lower().strip()
    if env_val in ("off", "warn", "enforce"):
        return ContractLevel(env_val)
    return ContractLevel.ENFORCE if __debug__ else ContractLevel.OFF


# Mutable state holder (avoids 'global' keyword)
_contract_state: dict[str, ContractLevel | bool] = {
    "level": _resolve_contract_level(),
    "enabled": _resolve_contract_level() != ContractLevel.OFF,
}

# Public module-level aliases (read via functions for correctness)
DBC_LEVEL: ContractLevel = _contract_state["level"]  # type: ignore[assignment]

# Legacy compatibility flag (derived from DBC_LEVEL)
CONTRACTS_ENABLED: bool = _contract_state["enabled"]  # type: ignore[assignment]


def set_contract_level(level: ContractLevel) -> None:
    """Set the global contract enforcement level at runtime.

    Args:
        level: The desired enforcement level.
    """
    _contract_state["level"] = level
    _contract_state["enabled"] = level != ContractLevel.OFF
    logger.info("Contract enforcement level set to %s", level.value)


def get_contract_level() -> ContractLevel:
    """Return the current global contract enforcement level."""
    return _contract_state["level"]  # type: ignore[return-value]


def enable_contracts() -> None:
    """Enable contract checking globally (sets level to ENFORCE)."""
    set_contract_level(ContractLevel.ENFORCE)


def disable_contracts() -> None:
    """Disable contract checking globally (sets level to OFF)."""
    set_contract_level(ContractLevel.OFF)


def contracts_enabled() -> bool:
    """Check if contracts are currently enabled."""
    return bool(_contract_state["enabled"])

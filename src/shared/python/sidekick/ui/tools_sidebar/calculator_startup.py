"""Validated startup imports for the Sidekick calculator."""

from __future__ import annotations

import importlib
import keyword
import types
from collections.abc import Mapping, MutableMapping
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class CalculatorStartupImport:
    """One optional module import exposed to calculator assistance."""

    module: str
    alias: str
    enabled: bool = True
    allow_private: bool = False

    def __post_init__(self) -> None:
        module = _clean_string(self.module)
        alias = _clean_string(self.alias)
        if not module:
            raise ValueError("startup import module must be non-empty")
        if not alias:
            raise ValueError("startup import alias must be non-empty")
        if not alias.isidentifier() or keyword.iskeyword(alias):
            raise ValueError(f"startup import alias is invalid: {alias!r}")
        if alias.startswith("_"):
            raise ValueError(f"startup import alias cannot be private: {alias!r}")
        _validate_module_path(module, allow_private=self.allow_private)
        object.__setattr__(self, "module", module)
        object.__setattr__(self, "alias", alias)

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-safe preference payload."""
        return {
            "module": self.module,
            "alias": self.alias,
            "enabled": self.enabled,
            "allow_private": self.allow_private,
        }

    @classmethod
    def from_dict(cls, payload: Mapping[str, Any]) -> CalculatorStartupImport:
        """Build a validated startup import from a persisted payload."""
        if not isinstance(payload, Mapping):
            raise TypeError("startup import payload must be a mapping")
        return cls(
            module=str(payload.get("module", "")),
            alias=str(payload.get("alias", "")),
            enabled=bool(payload.get("enabled", True)),
            allow_private=bool(payload.get("allow_private", False)),
        )


@dataclass(frozen=True)
class CalculatorStartupConfig:
    """Calculator startup dependency preferences."""

    imports: tuple[CalculatorStartupImport, ...]

    def __post_init__(self) -> None:
        imports = tuple(self.imports)
        aliases: set[str] = set()
        for startup_import in imports:
            if not isinstance(startup_import, CalculatorStartupImport):
                raise TypeError("startup imports must be CalculatorStartupImport")
            if startup_import.alias in aliases:
                raise ValueError(
                    f"duplicate startup import alias: {startup_import.alias!r}"
                )
            aliases.add(startup_import.alias)
        object.__setattr__(self, "imports", imports)

    def enabled_imports(self) -> tuple[CalculatorStartupImport, ...]:
        """Return validated imports that should run during startup."""
        return tuple(item for item in self.imports if item.enabled)

    def to_list(self) -> list[dict[str, Any]]:
        """Return a JSON-safe preference list."""
        return [item.to_dict() for item in self.imports]

    @classmethod
    def from_list(cls, payload: Any) -> CalculatorStartupConfig:
        """Build config from a JSON-safe list, falling back to defaults when absent."""
        if payload in (None, ""):
            return default_calculator_startup_config()
        if not isinstance(payload, list):
            raise TypeError("calculator startup imports must be a list")
        return cls(tuple(CalculatorStartupImport.from_dict(item) for item in payload))


@dataclass(frozen=True)
class CalculatorStartupWarning:
    """Structured diagnostic for an optional dependency that did not load."""

    module: str
    alias: str
    message: str


@dataclass(frozen=True)
class CalculatorStartupResult:
    """Outcome from applying startup imports to a namespace."""

    loaded_modules: tuple[str, ...]
    warnings: tuple[CalculatorStartupWarning, ...]


def apply_calculator_startup_imports(
    namespace: MutableMapping[str, Any],
    config: CalculatorStartupConfig,
) -> CalculatorStartupResult:
    """Import configured optional modules into ``namespace`` with diagnostics."""
    if namespace is None:
        raise ValueError("namespace must be provided")
    if not isinstance(config, CalculatorStartupConfig):
        raise TypeError("config must be CalculatorStartupConfig")

    additions: dict[str, types.ModuleType] = {}
    loaded: list[str] = []
    warnings: list[CalculatorStartupWarning] = []
    for startup_import in config.enabled_imports():
        try:
            module = importlib.import_module(startup_import.module)
        except ImportError as exc:
            warnings.append(
                CalculatorStartupWarning(
                    startup_import.module,
                    startup_import.alias,
                    f"Install optional dependency '{startup_import.module}' "
                    f"to enable alias '{startup_import.alias}': {exc}",
                )
            )
            continue
        additions[startup_import.alias] = module
        additions[startup_import.module] = module
        loaded.append(startup_import.module)

    namespace.update(additions)
    return CalculatorStartupResult(tuple(loaded), tuple(warnings))


def calculator_startup_config_from_state_payload(
    payload: Any,
) -> CalculatorStartupConfig:
    """Load startup import preferences and reject unsafe persisted entries."""
    return CalculatorStartupConfig.from_list(payload)


def _clean_string(value: str) -> str:
    return str(value).strip()


def _validate_module_path(module: str, *, allow_private: bool) -> None:
    parts = module.split(".")
    if any(not part for part in parts):
        raise ValueError(f"startup import module path is invalid: {module!r}")
    for part in parts:
        if not part.isidentifier() or keyword.iskeyword(part):
            raise ValueError(f"startup import module path is invalid: {module!r}")
        if part.startswith("_") and not allow_private:
            raise ValueError(
                f"startup import module path cannot be private: {module!r}"
            )


DEFAULT_CALCULATOR_STARTUP_IMPORTS = (
    CalculatorStartupImport("numpy", "np"),
    CalculatorStartupImport("scipy", "scipy"),
)


def default_calculator_startup_config() -> CalculatorStartupConfig:
    """Return the default optional scientific imports."""
    return CalculatorStartupConfig(DEFAULT_CALCULATOR_STARTUP_IMPORTS)

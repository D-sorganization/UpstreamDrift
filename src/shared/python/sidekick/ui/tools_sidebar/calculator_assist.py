"""Sidekick calculator help and predictive text contracts."""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import Any, Protocol

from .calculator_startup import (
    CalculatorStartupConfig,
    calculator_startup_config_from_state_payload,
)
from .registry import WorkspaceVariable
from .state import SidebarState

CALCULATOR_TAB_ID = "calculator"


@dataclass(frozen=True)
class CalculatorHelpTopic:
    """Static help metadata for the Sidekick calculator tab."""

    title: str
    summary: str
    examples: tuple[str, ...]
    tips: tuple[str, ...]

    def to_metadata(self) -> dict[str, str]:
        """Return the existing tab help registry shape."""
        return {
            "title": self.title,
            "summary": self.summary,
            "examples": "\n".join(self.examples),
            "tips": "\n".join(self.tips),
            "source": "upstream_drift_tools.ui.tools_sidebar.calculator_assist",
        }


CALCULATOR_HELP = CalculatorHelpTopic(
    title="Calculator",
    summary=(
        "Evaluate symbolic and numeric expressions, keep the last result in "
        "the calculator-local workspace, and reuse Sidekick workspace variables."
    ),
    examples=(
        "2 + 2",
        "a = 12",
        "sin(pi/2)",
        "diff(x**2, x)",
        "solve(x**2 - 4, x)",
        "latex(integrate(x**2, x))",
    ),
    tips=(
        "Use assignments such as a = 12 to keep intermediate values available.",
        "The latest result is stored as calculator_result in the local Workspace.",
        "Arrays and matrices can be built with Matrix([[1, 2], [3, 4]]).",
        "Plotting expressions should stay explicit, for example plot(sin(x)).",
        "Use solve(expression, symbol) for symbolic solving.",
        "Use latex(expression) when a rendered formula string is needed.",
    ),
)

ALLOWED_CALCULATOR_COMMANDS = (
    "diff(",
    "expand(",
    "factor(",
    "integrate(",
    "latex(",
    "limit(",
    "Matrix(",
    "plot(",
    "simplify(",
    "solve(",
    "sqrt(",
)

SCIENTIFIC_DEPENDENCY_ALIASES = {
    "numpy": ("np", "numpy"),
    "pandas": ("pd", "pandas"),
    "scipy": ("scipy",),
}


@dataclass(frozen=True)
class CalculatorSuggestion:
    """One non-executing predictive text suggestion."""

    label: str
    source: str


class CalculatorPredictionProvider(Protocol):
    """Side-effect-free calculator prediction provider."""

    def suggest(
        self,
        prefix: str,
        *,
        workspace_variables: Sequence[WorkspaceVariable] = (),
        loaded_dependencies: Iterable[str] = (),
    ) -> tuple[CalculatorSuggestion, ...]:
        """Return suggestions without evaluating calculator code."""


@dataclass(frozen=True)
class StaticCalculatorPredictionProvider:
    """Prediction provider using only allowlisted commands and visible context."""

    allowed_commands: tuple[str, ...] = ALLOWED_CALCULATOR_COMMANDS

    def suggest(
        self,
        prefix: str,
        *,
        workspace_variables: Sequence[WorkspaceVariable] = (),
        loaded_dependencies: Iterable[str] = (),
    ) -> tuple[CalculatorSuggestion, ...]:
        if not isinstance(prefix, str):
            raise TypeError("prefix must be a string")
        normalized_prefix = prefix.strip()
        if not normalized_prefix:
            return ()

        labels: list[CalculatorSuggestion] = []
        labels.extend(
            CalculatorSuggestion(command, "command")
            for command in self.allowed_commands
            if command.lower().startswith(normalized_prefix.lower())
        )
        labels.extend(
            CalculatorSuggestion(variable.name, "workspace")
            for variable in workspace_variables
            if variable.name.lower().startswith(normalized_prefix.lower())
        )
        labels.extend(
            CalculatorSuggestion(alias, "dependency")
            for alias in _dependency_aliases(loaded_dependencies)
            if alias.lower().startswith(normalized_prefix.lower())
        )
        return _dedupe_suggestions(labels)


@dataclass(frozen=True)
class CalculatorPredictiveText:
    """Preference gate for calculator prediction providers."""

    enabled: bool = False
    provider: CalculatorPredictionProvider | None = None

    def suggest(
        self,
        prefix: str,
        *,
        workspace_variables: Sequence[WorkspaceVariable] = (),
        loaded_dependencies: Iterable[str] = (),
    ) -> tuple[CalculatorSuggestion, ...]:
        if not self.enabled or self.provider is None:
            return ()
        return self.provider.suggest(
            prefix,
            workspace_variables=workspace_variables,
            loaded_dependencies=loaded_dependencies,
        )


def calculator_predictive_text_enabled(sidebar: Any) -> bool:
    """Return the persisted calculator predictive text preference from a sidebar."""
    state = getattr(sidebar, "_state", SidebarState())
    return bool(state.calculator_predictive_text_enabled)


def set_calculator_predictive_text_enabled(sidebar: Any, enabled: bool) -> None:
    """Persist the calculator predictive text preference on a sidebar host."""
    state = getattr(sidebar, "_state", None)
    if state is None:
        raise ValueError("sidebar must expose _state")
    state.calculator_predictive_text_enabled = bool(enabled)
    emit_context = getattr(sidebar, "_emit_context", None)
    if emit_context is not None:
        emit_context()


def calculator_state_fields(state: SidebarState) -> dict[str, Any]:
    """Return calculator-specific fields for SidebarState reconstruction."""
    return {
        "calculator_predictive_text_enabled": state.calculator_predictive_text_enabled,
        "calculator_startup_imports": list(state.calculator_startup_imports),
    }


def calculator_context_preferences(state: SidebarState) -> dict[str, Any]:
    """Return calculator preferences for Sidekick context payloads."""
    return {
        "calculator_predictive_text_enabled": state.calculator_predictive_text_enabled,
        "calculator_startup_imports": list(state.calculator_startup_imports),
    }


def calculator_startup_config(sidebar: Any) -> CalculatorStartupConfig:
    """Return validated calculator startup imports from a sidebar host."""
    state = getattr(sidebar, "_state", SidebarState())
    return calculator_startup_config_from_state_payload(
        state.calculator_startup_imports
    )


def set_calculator_startup_config(
    sidebar: Any,
    config: CalculatorStartupConfig,
) -> None:
    """Persist validated calculator startup imports on a sidebar host."""
    if not isinstance(config, CalculatorStartupConfig):
        raise TypeError("config must be CalculatorStartupConfig")
    state = getattr(sidebar, "_state", None)
    if state is None:
        raise ValueError("sidebar must expose _state")
    state.calculator_startup_imports = config.to_list()
    emit_context = getattr(sidebar, "_emit_context", None)
    if emit_context is not None:
        emit_context()


def _dependency_aliases(loaded_dependencies: Iterable[str]) -> tuple[str, ...]:
    loaded = {dependency for dependency in loaded_dependencies if dependency}
    aliases: list[str] = []
    for dependency in sorted(loaded):
        aliases.extend(SCIENTIFIC_DEPENDENCY_ALIASES.get(dependency, ()))
    return tuple(aliases)


def _dedupe_suggestions(
    suggestions: Iterable[CalculatorSuggestion],
) -> tuple[CalculatorSuggestion, ...]:
    seen: set[str] = set()
    result: list[CalculatorSuggestion] = []
    for suggestion in suggestions:
        if suggestion.label in seen:
            continue
        seen.add(suggestion.label)
        result.append(suggestion)
    return tuple(result)

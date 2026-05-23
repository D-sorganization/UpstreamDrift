"""Engine ↔ framework compatibility validation.

The "idiot-proof" promise: a user cannot dispatch a training job to an
engine that doesn't support the chosen framework. This module owns the
declarative compatibility map and a pure checker function that the
scheduler consults before transitioning a job to ``QUEUED``.

The default map is conservative — additional engines and frameworks
should be added here as adapters land. The map is injectable so tests
and downstream packages can override without monkey-patching.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Literal
from collections.abc import Mapping

from .config import TrainingConfig, TrainingFramework

__all__ = [
    "DEFAULT_ENGINE_FRAMEWORK_MAP",
    "CompatibilityChecker",
    "CompatibilityIssue",
    "CompatibilityReport",
]


Severity = Literal["error", "warning"]


@dataclass(frozen=True, slots=True)
class CompatibilityIssue:
    """A single complaint about a config / engine pairing.

    Attributes:
        code: Stable machine-readable identifier (e.g.
            ``"unknown_engine"``). Used by the UI to look up
            remediation copy.
        message: Human-readable explanation.
        severity: ``"error"`` blocks dispatch; ``"warning"`` is informational.
    """

    code: str
    message: str
    severity: Severity = "error"

    def __post_init__(self) -> None:
        if not isinstance(self.code, str) or not self.code:
            raise ValueError("CompatibilityIssue.code must be a non-empty string")
        if not isinstance(self.message, str) or not self.message:
            raise ValueError("CompatibilityIssue.message must be a non-empty string")
        if self.severity not in ("error", "warning"):
            raise ValueError(
                f"CompatibilityIssue.severity must be 'error' or 'warning' "
                f"(got {self.severity!r})"
            )


@dataclass(frozen=True, slots=True)
class CompatibilityReport:
    """Result of a compatibility check.

    Attributes:
        issues: Every issue found, in declaration order. May be empty.

    The :attr:`is_compatible` property is the single boolean the
    scheduler reads — it is ``True`` when no ``error``-severity issues
    are present (warnings are allowed).
    """

    issues: tuple[CompatibilityIssue, ...] = field(default_factory=tuple)

    def __post_init__(self) -> None:
        if not isinstance(self.issues, tuple):
            raise TypeError("CompatibilityReport.issues must be a tuple")
        for issue in self.issues:
            if not isinstance(issue, CompatibilityIssue):
                raise TypeError(
                    "CompatibilityReport.issues entries must be CompatibilityIssue"
                )

    @property
    def is_compatible(self) -> bool:
        """``True`` when no ``error``-severity issues are present."""

        return not any(i.severity == "error" for i in self.issues)

    @property
    def errors(self) -> tuple[CompatibilityIssue, ...]:
        """Issues whose severity is ``"error"``."""

        return tuple(i for i in self.issues if i.severity == "error")

    @property
    def warnings(self) -> tuple[CompatibilityIssue, ...]:
        """Issues whose severity is ``"warning"``."""

        return tuple(i for i in self.issues if i.severity == "warning")


DEFAULT_ENGINE_FRAMEWORK_MAP: Mapping[str, frozenset[TrainingFramework]] = (
    MappingProxyType(
        {
            "mujoco": frozenset(
                {TrainingFramework.PYTORCH, TrainingFramework.GYMNASIUM}
            ),
            "drake": frozenset({TrainingFramework.PYTORCH}),
            "pinocchio": frozenset({TrainingFramework.PYTORCH}),
            "opensim": frozenset({TrainingFramework.PYTORCH}),
            "myosim": frozenset(
                {TrainingFramework.PYTORCH, TrainingFramework.GYMNASIUM}
            ),
            "pendulum": frozenset(
                {TrainingFramework.PYTORCH, TrainingFramework.GYMNASIUM}
            ),
        }
    )
)
"""Default declarative engine → supported-framework map.

Keys match :class:`EngineType` value strings (lower-case). New engines
must be added here AND their adapter PR; missing entries surface as
``"unknown_engine"`` issues, never as silent passes.
"""


class CompatibilityChecker:
    """Validates a :class:`TrainingConfig` against a target engine.

    Construction accepts an optional override map for tests; callers
    that need the production map should pass nothing.
    """

    def __init__(
        self,
        engine_framework_map: Mapping[str, frozenset[TrainingFramework]] | None = None,
    ) -> None:
        source: Mapping[str, frozenset[TrainingFramework]] = (
            engine_framework_map
            if engine_framework_map is not None
            else DEFAULT_ENGINE_FRAMEWORK_MAP
        )
        self._map: Mapping[str, frozenset[TrainingFramework]] = MappingProxyType(
            {
                self._normalize(name): frozenset(frameworks)
                for name, frameworks in source.items()
            }
        )

    @staticmethod
    def _normalize(name: str) -> str:
        if not isinstance(name, str) or not name.strip():
            raise ValueError("engine name must be a non-empty string")
        return name.strip().lower()

    @property
    def known_engines(self) -> frozenset[str]:
        """Set of engine names registered with this checker."""

        return frozenset(self._map.keys())

    def check(self, config: TrainingConfig, engine_name: str) -> CompatibilityReport:
        """Validate the pairing.

        Args:
            config: Job configuration.
            engine_name: Target engine identifier (case-insensitive).

        Returns:
            A :class:`CompatibilityReport`. Always returns; never raises
            on a known issue type.

        Preconditions:
            - ``config`` is a :class:`TrainingConfig`.
            - ``engine_name`` is a non-empty string.
        """

        if not isinstance(config, TrainingConfig):
            raise TypeError(
                f"config must be a TrainingConfig (got {type(config).__name__})"
            )
        normalized = self._normalize(engine_name)
        issues: list[CompatibilityIssue] = []
        supported = self._map.get(normalized)
        if supported is None:
            issues.append(
                CompatibilityIssue(
                    code="unknown_engine",
                    message=(
                        f"Engine {engine_name!r} is not registered with the "
                        "training controller. Add it to the engine-framework map "
                        "or pick a different engine."
                    ),
                    severity="error",
                )
            )
        elif config.framework not in supported:
            supported_names = sorted(f.value for f in supported)
            issues.append(
                CompatibilityIssue(
                    code="framework_unsupported",
                    message=(
                        f"Engine {normalized!r} does not support framework "
                        f"{config.framework.value!r}. Supported: {supported_names}."
                    ),
                    severity="error",
                )
            )
        return CompatibilityReport(issues=tuple(issues))

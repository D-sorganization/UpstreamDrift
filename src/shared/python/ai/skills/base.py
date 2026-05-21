"""Skill abstract base class (Tools #2737)."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, ClassVar

from .contracts import SkillDescriptor, SkillInvocation, SkillResult


class Skill(ABC):
    """Abstract base for a skill.

    Subclasses MUST declare a class-level ``descriptor`` attribute. The
    runner uses the descriptor to drive preflight (precondition names) and
    postflight (postcondition names) checks.

    By default :meth:`validate_preconditions` and
    :meth:`validate_postconditions` are no-ops. Skills that declare
    preconditions / postconditions in their descriptor should override them.
    """

    descriptor: ClassVar[SkillDescriptor]

    def __init_subclass__(cls, **kwargs: Any) -> None:  # noqa: D401
        super().__init_subclass__(**kwargs)
        # Abstract intermediate subclasses (ABCs) may legitimately omit a
        # descriptor; only concrete classes are required to provide one.
        if getattr(cls, "__abstractmethods__", None):
            return
        descriptor = getattr(cls, "descriptor", None)
        if not isinstance(descriptor, SkillDescriptor):
            raise TypeError(
                f"Skill subclass {cls.__name__} must define a "
                "'descriptor: SkillDescriptor' class attribute."
            )

    def validate_preconditions(self, args: dict[str, Any]) -> None:  # noqa: B027
        """Validate input ``args``. Raise ``ValueError`` on failure.

        Intentionally a no-op on the base class so that skills with no
        preconditions need not override it. Marked ``noqa: B027`` because
        this is a deliberate default hook, not an oversight.
        """

    def validate_postconditions(self, result: dict[str, Any]) -> None:  # noqa: B027
        """Validate output ``result``. Raise ``ValueError`` on failure.

        Intentional no-op default (see :meth:`validate_preconditions`).
        """

    @abstractmethod
    async def run(self, invocation: SkillInvocation) -> SkillResult:
        """Execute the skill body. Must be async."""

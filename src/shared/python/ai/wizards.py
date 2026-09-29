"""Sidekick glue for product Wizards (Tools#5346).

Resolves the Wizard for a conversation's ``project_root``, registers its
name and persona as the Sidekick app context, and turns the latest user
message into a :class:`KnowledgeContext` for the system prompt.
"""

from __future__ import annotations

import logging
from pathlib import Path

from src.shared.python.ai.knowledge.wizard import (
    KnowledgeContext,
    WizardConfigError,
    WizardKnowledge,
    load_wizard_config,
)
from src.shared.python.ai.system_prompts import register_app_context
from src.shared.python.ai.types import ConversationContext

__all__ = ["knowledge_for_context", "reset_wizards", "wizard_for"]

logger = logging.getLogger(__name__)

_WIZARDS: dict[Path, WizardKnowledge | None] = {}


def wizard_for(project_root: Path | str) -> WizardKnowledge | None:
    """The host product's Wizard, or None when it has no ``knowledge/wizard.yml``.

    Resolved once per root; the first resolution registers the app context.
    A malformed ``wizard.yml`` is logged and treated as no Wizard, so a bad
    config never breaks chat.
    """
    root = Path(project_root).resolve()
    if root not in _WIZARDS:
        _WIZARDS[root] = _load(root)
    return _WIZARDS[root]


def knowledge_for_context(
    context: ConversationContext | None,
) -> KnowledgeContext | None:
    """Knowledge for the latest user message, if the host has a Wizard."""
    if context is None:
        return None
    root = context.metadata.get("project_root")
    if not isinstance(root, str) or not root:
        return None
    wizard = wizard_for(root)
    question = _latest_user_message(context)
    if wizard is None or not question:
        return None
    return wizard.context_for(question)


def reset_wizards() -> None:
    """Forget resolved Wizards (tests, or after a product rebuilds its config)."""
    _WIZARDS.clear()


def _load(root: Path) -> WizardKnowledge | None:
    try:
        config = load_wizard_config(root)
    except (WizardConfigError, OSError) as exc:
        logger.warning("Ignoring invalid Wizard config under %s: %s", root, exc)
        return None
    if config is None:
        return None
    register_app_context(
        config.key,
        name=config.name,
        description=config.description,
        capabilities=list(config.capabilities),
    )
    return WizardKnowledge(config)


def _latest_user_message(context: ConversationContext) -> str:
    for message in reversed(context.messages):
        if message.role == "user" and message.content:
            return str(message.content)
    return ""

"""Default tab visibility helpers for the Sidekick sidebar."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import replace
from typing import Protocol

from .state import SidebarState


class TabDefinitionLike(Protocol):
    """Minimal tab definition contract needed by visibility helpers."""

    @property
    def tab_id(self) -> str: ...

    @property
    def visible(self) -> bool: ...


def available_tab_ids(definitions: Iterable[TabDefinitionLike]) -> list[str]:
    """Return tab ids in host configuration order."""
    return [definition.tab_id for definition in definitions]


def initially_visible_tab_ids(
    definitions: Iterable[TabDefinitionLike],
    state: SidebarState,
) -> set[str]:
    """Resolve the tabs that should be visible before runtime overrides apply."""
    available = available_tab_ids(definitions)
    available_ids = set(available)
    default_visible = [
        tab_id for tab_id in state.default_visible_tabs if tab_id in available_ids
    ]
    default_hidden = {
        tab_id for tab_id in state.default_hidden_tabs if tab_id in available_ids
    }
    if default_visible:
        visible = set(default_visible)
    else:
        visible = {
            definition.tab_id for definition in definitions if definition.visible
        }
    visible -= default_hidden
    if visible or not available:
        return visible
    fallback = next(
        (
            definition.tab_id
            for definition in definitions
            if definition.tab_id not in default_hidden
        ),
        available[0],
    )
    return {fallback}


def with_default_tab_visibility(
    state: SidebarState,
    definitions: Iterable[TabDefinitionLike],
    tab_id: str,
    visible: bool,
) -> SidebarState | None:
    """Return updated default visibility state, or ``None`` if it hides all tabs."""
    available = set(available_tab_ids(definitions))
    if tab_id not in available:
        raise ValueError(f"Unknown sidebar tab id: {tab_id}")
    default_hidden = [item for item in state.default_hidden_tabs if item != tab_id]
    default_visible = [item for item in state.default_visible_tabs if item != tab_id]
    if visible:
        default_visible.append(tab_id)
    else:
        default_hidden.append(tab_id)

    candidate = replace(
        state,
        default_visible_tabs=default_visible,
        default_hidden_tabs=default_hidden,
    )
    if not initially_visible_tab_ids(definitions, candidate):
        return None
    return candidate


def without_default_tab_visibility(state: SidebarState) -> SidebarState:
    """Clear user defaults while preserving runtime tab state."""
    return replace(state, default_visible_tabs=[], default_hidden_tabs=[])


def sanitize_tab_state(
    state: SidebarState,
    available_ids: Iterable[str],
) -> SidebarState:
    """Drop persisted tab ids that are not available in the current host."""
    available = set(available_ids)
    return replace(
        state,
        tab_order=[tab_id for tab_id in state.tab_order if tab_id in available],
        hidden_tabs=[tab_id for tab_id in state.hidden_tabs if tab_id in available],
        default_visible_tabs=[
            tab_id for tab_id in state.default_visible_tabs if tab_id in available
        ],
        default_hidden_tabs=[
            tab_id for tab_id in state.default_hidden_tabs if tab_id in available
        ],
        popped_out_tabs=[
            tab_id for tab_id in state.popped_out_tabs if tab_id in available
        ],
        tab_display_names={
            tab_id: display_name
            for tab_id, display_name in state.tab_display_names.items()
            if tab_id in available
        },
    )

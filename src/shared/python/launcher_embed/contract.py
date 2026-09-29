"""Embeddable-tool contract types.

This module defines the dataclass and protocol that describe how a tool
wants to be embedded inside the launcher. It deliberately does not import
PyQt6: widget types are spelled :class:`typing.Any` so that consumers
without PyQt6 installed can still import this module.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Protocol, runtime_checkable

__all__ = ["BackgroundableTool", "EmbedCapabilities", "EmbeddableTool"]


@dataclass(frozen=True, slots=True)
class EmbedCapabilities:
    """Describes how a tool wants to be embedded inside the launcher.

    Attributes:
        supports_embedded: ``True`` if the tool can run as a child widget
            inside the launcher window. If ``False`` the launcher must
            fall back to opening it as a standalone top-level window.
        prefers_dock: ``True`` if the tool prefers to be hosted in a
            dockable side panel rather than a tab. Hosts may ignore this
            hint when dock layouts are not available.
        min_size: Minimum (width, height) in pixels the embedded widget
            requires to render correctly. Both values must be positive
            integers.
        requires_separate_qapplication: ``True`` if the tool needs its
            own ``QApplication`` instance (e.g., because it manages a
            non-default GL context). Hosts that cannot honor this should
            refuse to embed and fall back to standalone launch.

    Invariants (enforced in :meth:`__post_init__`):
        ``min_size`` is a 2-tuple of strictly positive ``int`` values.
    """

    supports_embedded: bool = True
    prefers_dock: bool = False
    min_size: tuple[int, int] = (640, 480)
    requires_separate_qapplication: bool = False
    NONE: ClassVar[EmbedCapabilities]

    def __post_init__(self) -> None:
        # DbC: validate ``min_size`` is a positive 2-tuple of ints.
        min_size = self.min_size
        if not isinstance(min_size, tuple):
            raise ValueError(
                f"min_size must be a tuple of two positive ints, "
                f"got {type(min_size).__name__}"
            )
        if len(min_size) != 2:
            raise ValueError(
                f"min_size must be a 2-tuple, got tuple of length {len(min_size)}"
            )
        width, height = min_size
        # ``bool`` is a subclass of ``int``; reject it explicitly so that
        # ``EmbedCapabilities(min_size=(True, True))`` does not silently
        # succeed.
        if (
            not isinstance(width, int)
            or not isinstance(height, int)
            or isinstance(width, bool)
            or isinstance(height, bool)
        ):
            raise ValueError(
                f"min_size must contain ints, got ({type(width).__name__}, "
                f"{type(height).__name__})"
            )
        if width <= 0 or height <= 0:
            raise ValueError(
                f"min_size must be strictly positive, got ({width}, {height})"
            )


EmbedCapabilities.NONE = EmbedCapabilities(supports_embedded=False)


@runtime_checkable
class EmbeddableTool(Protocol):
    """Protocol implemented by tools that can be embedded in the launcher.

    Implementations must provide:

    - ``tool_id``: a stable, non-empty string identifier (used as the
      registry key and surfaced in launcher UI).
    - :meth:`embed_capabilities`: returns an :class:`EmbedCapabilities`
      describing how this tool wants to be embedded.
    - :meth:`create_main_widget`: builds and returns the top-level
      ``QWidget`` for the tool. The ``parent`` argument is typed as
      :class:`typing.Any` so consumers do not need to import PyQt6 just
      to satisfy this protocol; in practice it is a ``QWidget | None``.
    - :meth:`cleanup`: tears down any resources held by the embedded
      widget. Called by the host when the tool is being closed or
      unembedded. Must be idempotent.
    - :meth:`is_dirty`: returns ``True`` if the tool has unsaved state
      that should prompt the user before close. Implementations that do
      not track dirty state should return ``False``.

    Optional lifecycle hooks (see #6013) — ``pause``, ``resume``,
    ``can_background``, and ``detach_to_window`` — are defined in the
    separate :class:`BackgroundableTool` protocol. They are deliberately
    kept *out* of this protocol so the ``runtime_checkable``
    ``isinstance`` check still accepts the ~17 existing adapters that do
    not implement them. A tool may implement any subset of those hooks;
    hosts resolve them structurally with ``getattr(tool, name, default)``.

    Notes:
        Because :class:`typing.Protocol` cannot define method bodies, the
        ``is_dirty`` "default of ``False``" is a documentation contract
        rather than an inherited implementation. Implementations should
        return ``False`` unless they have a meaningful dirty-state model.
    """

    tool_id: str

    def embed_capabilities(self) -> EmbedCapabilities:
        """Return how this tool wants to be embedded."""
        ...

    def create_main_widget(self, parent: Any) -> Any:
        """Construct and return the top-level widget for this tool.

        Args:
            parent: The intended Qt parent (``QWidget | None``). Typed as
                :class:`typing.Any` so consumers do not need to import a
                Qt binding solely to satisfy this protocol.

        Returns:
            The newly created ``QWidget``.
        """
        ...

    def cleanup(self) -> None:
        """Release any resources held by the embedded widget.

        Must be idempotent: hosts may call this multiple times during
        shutdown.
        """
        ...

    def is_dirty(self) -> bool:
        """Return ``True`` if the tool has unsaved state.

        Expected default: ``False`` for tools that do not track dirty
        state.
        """
        ...


@runtime_checkable
class BackgroundableTool(Protocol):
    """Optional backgrounding / pop-out lifecycle hooks (#6013).

    This is a **separate, additive** protocol from
    :class:`EmbeddableTool`. It is intentionally *not* folded into
    :class:`EmbeddableTool` so that the ``runtime_checkable``
    ``isinstance(tool, EmbeddableTool)`` check continues to pass for the
    ~17 existing adapters that predate this extension and do not
    implement these hooks.

    Hosts must resolve each hook structurally —
    ``getattr(tool, name, default)`` — rather than requiring conformance
    to this protocol, so that a tool may implement *any subset* of the
    six hooks (or none). The documented defaults are:

    - :meth:`pause` / :meth:`resume`: no-op.
    - :meth:`pause_widget` / :meth:`resume_widget`: fall back to adapter-level
      :meth:`pause` / :meth:`resume`.
    - :meth:`can_background`: ``True``.
    - :meth:`detach_to_window`: ``True``.
    """

    def pause(self) -> None:
        """Suspend background activity while the widget is hidden.

        Called by the host when the user backgrounds the tab
        ("keep running" close). Implementations should stop timers,
        polling, or live IPC subscriptions so a hidden tool is cheap.

        Expected default: no-op for tools with no background activity.
        """
        ...

    def pause_widget(self, widget: Any) -> None:
        """Suspend background activity for one mounted widget only.

        Hosts call this hook when a tool can be mounted more than once
        through the same adapter instance, for example as both a tab and
        a dock. Implementations should pause only ``widget`` and leave
        any still-visible sibling widgets active.
        """
        ...

    def resume(self) -> None:
        """Resume activity when a backgrounded widget is re-surfaced.

        Called by the host when a previously paused tab is reopened or
        refocused. The inverse of :meth:`pause`.

        Expected default: no-op for tools with no background activity.
        """
        ...

    def resume_widget(self, widget: Any) -> None:
        """Resume activity for one previously backgrounded widget only."""
        ...

    def can_background(self) -> bool:
        """Return ``True`` if the tool may keep running while hidden.

        When ``False`` the host skips the background prompt and applies
        legacy cleanup-on-close behaviour (the widget is destroyed via
        :meth:`EmbeddableTool.cleanup`).

        Expected default: ``True``.
        """
        ...

    def detach_to_window(self) -> bool:
        """Return ``True`` if the tool may be popped out to its own window.

        When ``False`` the tab is pin-only: the host omits the
        "Pop out" affordance and refuses programmatic pop-out.

        Expected default: ``True``.
        """
        ...

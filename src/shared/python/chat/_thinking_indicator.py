# ruff: noqa: E501
"""Reusable "AI is thinking" indicator for the shared chat dock.

This module provides :class:`ThinkingIndicator`, a small QLabel-based widget
that animates a three-dot pulser (``"Sidekick is thinking ●``, ``●●``,
``●●●"``) while the AI agent is producing a response.

Design goals (per the discoverability + DRY requirements):

* **Single widget reused everywhere** — the dock builds one instance and
  hooks it into its own chrome. Runtime tabs/panels do not render their own.
* **Law of Demeter** — the widget never reaches into a parent. The
  state-source (``_is_streaming`` flag, complete/error chunks) drives the
  widget via explicit ``start()`` / ``stop()`` calls from the dock.
* **Theme cooperation** — colors come from the injected
  :class:`ThemeProviderProtocol`, falling back to :class:`_DefaultDarkTheme`
  so the widget never hard-codes accent values.
* **Design-by-contract** — ``start()`` and ``stop()`` have explicit
  preconditions and postconditions documented in their docstrings.
* **Accessibility** — exposes ``"AI is thinking"`` as the accessible name
  so screen readers announce activity.

Tools issue: add visible thinking indicator to shared Sidekick chat dock.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtWidgets import QLabel, QWidget

from ._theme_protocol import ThemeProviderProtocol, _DefaultDarkTheme

if TYPE_CHECKING:
    pass


_DOT_FRAMES: tuple[str, str, str] = ("●", "●●", "●●●")
_TICK_MS: int = 450
_PREFIX: str = "Sidekick is thinking "


def _resolve_colors(
    theme_provider: ThemeProviderProtocol | None,
) -> dict[str, str]:
    """Return a color map, falling back to the dark defaults on failure."""
    provider: ThemeProviderProtocol = theme_provider or _DefaultDarkTheme()
    try:
        return dict(provider.get_current_colors())
    except Exception:  # noqa: BLE001 - misbehaving providers must not crash UI
        return dict(_DefaultDarkTheme().get_current_colors())


class ThinkingIndicator(QLabel):
    """Animated "AI is thinking" indicator label.

    The widget owns a :class:`QTimer` child that drives a three-frame dot
    animation at ``_TICK_MS`` cadence. The timer is parented to the widget so
    Qt's ownership tree reaps it when the widget is destroyed.

    Public API:
        * :meth:`start` — show the widget and begin animating.
        * :meth:`stop` — hide the widget and stop the timer.
        * :attr:`is_active` — read-only ``bool`` property reflecting whether
          the indicator is currently animating.

    Both ``start()`` and ``stop()`` are idempotent (preconditions document
    that calling them in the "already in target state" condition is a
    no-op, never a failure).
    """

    def __init__(
        self,
        parent: QWidget | None = None,
        theme_provider: ThemeProviderProtocol | None = None,
        accent_color: str | None = None,
    ) -> None:
        super().__init__(parent)
        colors = _resolve_colors(theme_provider)
        text_secondary = colors.get("text_secondary", "#888")
        accent = accent_color or colors.get("accent", "#58a6ff")

        self.setText("")
        self.setAccessibleName("AI is thinking")
        self.setAccessibleDescription(
            "Animated indicator showing that the AI agent is generating a response."
        )
        self.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        self.setStyleSheet(
            f"QLabel {{ color: {text_secondary}; font-size: 11px;"
            f" padding: 2px 6px; }}"
            # Slightly emphasise the dot pulse with the theme accent so the
            # eye is drawn to active animation without overpowering messages.
            f'QLabel[active="true"] {{ color: {accent}; font-weight: 600; }}'
        )

        self._frame_index: int = 0
        self._timer = QTimer(self)
        self._timer.setInterval(_TICK_MS)
        self._timer.timeout.connect(self._on_tick)

        # Start hidden — idle is the default state.
        self.hide()

    # ── public API ───────────────────────────────────────────────────

    @property
    def is_active(self) -> bool:
        """Whether the indicator is currently animating.

        Post: returns ``True`` iff :meth:`start` has been called since the
        last :meth:`stop` (or since construction).
        """
        return bool(self._timer.isActive())

    def start(self) -> None:
        """Show the indicator and begin the dot animation.

        Pre: the widget has not been destroyed by Qt.
        Pre: calling ``start()`` while already active is a no-op (idempotent),
            not a failure — this matches the queue-flush protocol where the
            dock may signal a new turn while the indicator is already shown.
        Post: ``is_active`` is ``True`` and the label text matches the first
            animation frame.
        """
        if self._timer.isActive():
            return
        self._frame_index = 0
        self._render_frame()
        self.setProperty("active", True)
        # Re-polish so the dynamic stylesheet selector picks up the change.
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.show()
        self._timer.start()

    def stop(self) -> None:
        """Hide the indicator and stop the animation timer.

        Pre: ``stop()`` is a no-op when not started (idempotent). Callers
            (e.g. the dock's ``complete``/``error`` chunk handlers and
            ``closeEvent``) may invoke it unconditionally without first
            consulting :attr:`is_active`.
        Post: ``is_active`` is ``False``, the underlying ``QTimer`` is
            stopped, and the widget is hidden.
        """
        if not self._timer.isActive() and not self.isVisible():
            return
        self._timer.stop()
        self.setProperty("active", False)
        style = self.style()
        if style is not None:
            style.unpolish(self)
            style.polish(self)
        self.hide()

    # ── internals ────────────────────────────────────────────────────

    def _on_tick(self) -> None:
        self._frame_index = (self._frame_index + 1) % len(_DOT_FRAMES)
        self._render_frame()

    def _render_frame(self) -> None:
        dots = _DOT_FRAMES[self._frame_index]
        self.setText(f"{_PREFIX}{dots}")


__all__ = ["ThinkingIndicator"]

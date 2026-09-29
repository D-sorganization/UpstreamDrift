"""One QSettings namespace and window-geometry persistence (issue #8907).

The launcher used two QSettings pairs: ``("UpstreamDrift", "Launcher")``
for check-boxes and plot theme, and ``("D-sorganization", "UpstreamDrift")``
for the UI font. The first pair dominates, so it is canonical. Keys that
only exist under a legacy pair are read from there when the canonical
store lacks them (read-old-if-new-missing) during the deprecation window;
the legacy store itself is never modified.

Windows persist their size/position with a single call::

    persist_window_geometry(self, "settings_dialog")

so no window reaches into QSettings itself. PyQt6 is imported lazily to
keep this module importable by lazily-loaded tools.
"""

from __future__ import annotations

import functools
from typing import Any

SETTINGS_ORG = "UpstreamDrift"
SETTINGS_APP = "Launcher"

#: Deprecated pairs whose keys are aliased into the canonical store.
LEGACY_SETTINGS_PAIRS: tuple[tuple[str, str], ...] = (
    ("D-sorganization", "UpstreamDrift"),
)

GEOMETRY_KEY_PREFIX = "geometry/"


def _open_settings(organization: str, application: str) -> Any:
    """Open one QSettings store (the single seam tests redirect)."""
    from PyQt6 import QtCore

    return QtCore.QSettings(organization, application)


def launcher_settings() -> Any:
    """Return the canonical launcher ``QSettings``.

    Any key present only under a legacy pair is copied into the canonical
    store first, so readers see the user's previous value. An existing
    canonical value always wins.
    """
    settings = _open_settings(SETTINGS_ORG, SETTINGS_APP)
    for organization, application in LEGACY_SETTINGS_PAIRS:
        legacy = _open_settings(organization, application)
        for key in legacy.allKeys():
            if not settings.contains(key):
                settings.setValue(key, legacy.value(key))
    return settings


def launcher_settings_scope() -> tuple[str, str]:
    """Return the canonical ``(organization, application)`` pair.

    For components (e.g. the shared ``FontManager``) that open their own
    ``QSettings``: legacy keys are aliased first so they read the old value.
    """
    launcher_settings()
    return SETTINGS_ORG, SETTINGS_APP


@functools.cache
def _geometry_persister_class() -> type:
    """Build the event-filter class lazily (keeps PyQt6 import deferred)."""
    from PyQt6.QtCore import QEvent, QObject

    class _GeometryPersister(QObject):
        """Save the watched window's geometry whenever it is hidden."""

        def __init__(self, window: Any, key: str) -> None:
            super().__init__(window)
            self._key = key

        def eventFilter(self, watched: Any, event: Any) -> bool:  # noqa: N802
            if event.type() == QEvent.Type.Hide:
                launcher_settings().setValue(self._key, watched.saveGeometry())
            return False

    return _GeometryPersister


def persist_window_geometry(window: Any, name: str) -> None:
    """Restore *window*'s saved geometry now and save it whenever it hides.

    Saving on hide (rather than only ``closeEvent``) also covers dialogs
    dismissed via ``reject()``/Esc, which hide without a close event.

    Raises:
        ValueError: if *name* is empty or *window* cannot save geometry.
    """
    if not isinstance(name, str) or not name.strip():
        raise ValueError("geometry name must be a non-empty string")
    if not callable(getattr(window, "saveGeometry", None)):
        raise ValueError(f"{type(window).__name__} cannot persist geometry")
    from PyQt6.QtCore import QByteArray

    key = GEOMETRY_KEY_PREFIX + name
    saved = launcher_settings().value(key)
    if isinstance(saved, QByteArray) and not saved.isEmpty():
        window.restoreGeometry(saved)
    window.installEventFilter(_geometry_persister_class()(window, key))


__all__ = [
    "LEGACY_SETTINGS_PAIRS",
    "SETTINGS_APP",
    "SETTINGS_ORG",
    "launcher_settings",
    "launcher_settings_scope",
    "persist_window_geometry",
]

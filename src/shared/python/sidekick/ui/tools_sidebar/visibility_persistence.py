"""Visibility persistence collaborator for :class:`UnifiedToolsSidebar` (F4).

``VisibilityPersistence`` centralises all QSettings reads and writes for the
visible-tab list.  Prior to this extraction the sidebar had three separate
write sites and two read paths; all are now funnelled through this class.

The key is project-root–scoped so two concurrently open projects never
clobber each other's visible-tab sets.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

from .qt_compat import QtCore

_logger = logging.getLogger(__name__)

__all__ = ["VisibilityPersistence"]

# Explicit org/app avoids writing to the default/empty store when the host
# hasn't called QCoreApplication.setOrganizationName().
_QS_ORG = "DSorganization"
_QS_APP = "Sidekick"
_QS_VISIBLE_TABS_KEY_GLOBAL = "sidekick_visible_tabs"
# Backward-compat alias used by the sidebar module and older tests.
_QS_VISIBLE_TABS_KEY = _QS_VISIBLE_TABS_KEY_GLOBAL


def _scoped_key(project_root: Path) -> str:
    """Return a project-root–scoped QSettings key for visible tabs.

    Using the project root as part of the key ensures two projects opened
    at the same time never clobber each other's visible-tab preferences
    (F5 two-project isolation requirement).
    """
    safe = str(project_root).replace("\\", "/").replace(":", "_").replace("/", "_")
    return f"sidekick_visible_tabs/{safe}"


class VisibilityPersistence:
    """Read and write the visible-tab list to QSettings.

    Args:
        project_root: The workspace root used to scope the QSettings key.
            Pass ``None`` to fall back to the legacy global key.

    Raises:
        TypeError: If *project_root* is provided but is not a
            :class:`~pathlib.Path` or ``str``.
    """

    def __init__(
        self,
        project_root: Path | str | None = None,
    ) -> None:
        if project_root is None:
            self._key = _QS_VISIBLE_TABS_KEY_GLOBAL
        elif isinstance(project_root, (str, Path)):
            self._key = _scoped_key(Path(project_root).expanduser().resolve())
        else:
            raise TypeError(
                f"project_root must be a Path, str, or None; got {type(project_root)}"
            )

    # ── Write ─────────────────────────────────────────────────────────────────

    def save(self, tab_ids: list[str]) -> None:
        """Persist *tab_ids* (the current visible-tab order) to QSettings.

        This is the **single** write path (F5 — no more triplicated writes).
        """
        settings = QtCore.QSettings(_QS_ORG, _QS_APP)
        settings.setValue(self._key, list(tab_ids))

    # ── Read ──────────────────────────────────────────────────────────────────

    def load(self, known_ids: set[str]) -> list[str] | None:
        """Return the persisted visible-tab list filtered to *known_ids*.

        Returns ``None`` when no value has been persisted yet, so the caller
        can fall back to the application's compile-time defaults.

        Args:
            known_ids: Set of tab ids that actually exist in this session.
                Ids not in this set are silently dropped (handles removed tabs
                from old sessions).
        """
        settings = QtCore.QSettings(_QS_ORG, _QS_APP)
        raw = settings.value(self._key, None)
        if raw is None:
            return None

        if isinstance(raw, list):
            return [str(tid) for tid in raw if str(tid) in known_ids]

        if isinstance(raw, str):
            try:
                loaded = json.loads(raw)
                if isinstance(loaded, list):
                    return [str(tid) for tid in loaded if str(tid) in known_ids]
            except Exception:
                _logger.debug(
                    "VisibilityPersistence.load: could not parse JSON value %r",
                    raw,
                )

        return None

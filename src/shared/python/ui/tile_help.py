"""Registry-driven per-tile help (issue #9413, defects #8843 / #8846).

Every tile in ``src/config/models.yaml`` declares a ``help:`` key that points
at a Markdown page under ``docs/help/``. This module is the single place that

* resolves a tile id to that page (:func:`help_path_for`),
* loads it, or explains precisely why it could not be loaded
  (:func:`load_help_markdown`), and
* attaches the *same* help affordance — an ``F1`` shortcut plus, where the
  widget owns a menu bar, a Help menu built by
  :func:`src.launchers.help_menu.build_help_menu` — to any Qt widget
  (:func:`attach_tile_help`).

Before this module the shared ``build_help_menu`` builder was dead code and
24 of 25 tool GUIs shipped with no help affordance at all (#8846), while the
help dock searched a hand-maintained rule table that had drifted away from the
tiles it was meant to serve (#8843). Both now read the registry.

The Qt imports are deliberately deferred to call time so the pure path
helpers can be imported (and unit-tested) in a headless process without a
``QApplication``.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Any

from src.config.tile_registry import REPO_ROOT, load_tile_registry
from src.shared.python.logging_pkg.logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover - typing only
    from src.config.tile_registry import TileRegistry

logger = get_logger(__name__)

#: Keyboard accelerator used for per-tile help everywhere in the suite.
HELP_SHORTCUT = "F1"

#: Attribute stamped on a widget once help has been attached, so repeated
#: calls (re-embedding a tool, detaching it to a window) stay idempotent.
_ATTACHED_ATTR = "_upstreamdrift_tile_help_id"

_GENERAL_DOCS = (
    "docs/user_guide/upstream_drift_user_manual.md",
    "docs/architecture/PROJECT_MAP.md",
    "docs/troubleshooting/FAQ.md",
)

#: Fallback tile ids for windows that cannot declare ``HELP_TILE_ID``
#: themselves.
#:
#: The normal way an engine window names its tile is the ``HELP_TILE_ID``
#: class attribute on :class:`SimulationGUIBase`. That requires editing the
#: window's module, and CI runs mypy over the *changed file set*, so adding
#: one attribute to a module that already carries type debt fails the build
#: on errors the change did not introduce (`pinocchio_golf/gui.py` and its
#: `ui/main_window.py` have pre-existing `ndarray | None` assignment and
#: mixin `self`-argument errors). Naming those windows here keeps the help
#: wiring out of files it would otherwise hold hostage.
#:
#: Keyed by class name because the two Pinocchio modules both define
#: ``PinocchioGUI`` and both mean the same tile. Prefer ``HELP_TILE_ID``
#: for anything new; this map is for modules that cannot take the edit.
WINDOW_CLASS_TILE_IDS: dict[str, str] = {
    "PinocchioGUI": "pinocchio_golf",
}


def tile_id_for_window(window: Any) -> str | None:
    """Return the tile id declared for ``window``'s class, if any.

    Args:
        window: A window instance, normally a ``SimulationGUIBase`` subclass.

    Returns:
        The tile id from :data:`WINDOW_CLASS_TILE_IDS`, or ``None`` when the
        class is not listed.
    """
    return WINDOW_CLASS_TILE_IDS.get(type(window).__name__)


def _registry(registry: TileRegistry | None = None) -> TileRegistry:
    return registry if registry is not None else load_tile_registry()


def help_relpath_for(
    tile_id: str, *, registry: TileRegistry | None = None
) -> str | None:
    """Return the ``help:`` value declared for ``tile_id``, or ``None``.

    Args:
        tile_id: Registry tile id (for example ``"ball_flight_simulator"``).
        registry: Pre-loaded registry, mainly for tests. Loaded on demand
            when omitted.

    Returns:
        The repo-relative POSIX path declared in ``models.yaml``, or ``None``
        when the tile is unknown or declares no help page.
    """
    if not tile_id:
        return None
    tile = _registry(registry).get(tile_id)
    return tile.help if tile is not None else None


def help_path_for(
    tile_id: str,
    *,
    registry: TileRegistry | None = None,
    repo_root: Path | None = None,
) -> Path | None:
    """Resolve ``tile_id`` to an existing help page on disk.

    Args:
        tile_id: Registry tile id.
        registry: Pre-loaded registry (optional).
        repo_root: Repository root to resolve against; defaults to the root
            that owns ``models.yaml``.

    Returns:
        The absolute path to the tile's help page when the tile declares one
        *and* the file exists, otherwise ``None``. Callers that need to
        distinguish "not declared" from "declared but missing" should compare
        against :func:`help_relpath_for`.
    """
    relative = help_relpath_for(tile_id, registry=registry)
    if not relative:
        return None
    candidate = (repo_root or REPO_ROOT) / relative
    return candidate if candidate.is_file() else None


def load_help_markdown(
    tile_id: str,
    *,
    registry: TileRegistry | None = None,
    repo_root: Path | None = None,
) -> str:
    """Return the Markdown help for ``tile_id``, or a diagnostic page.

    The fallback never claims "no documentation available" without saying
    what was looked for — that opacity was the substance of #8843.

    Args:
        tile_id: Registry tile id.
        registry: Pre-loaded registry (optional).
        repo_root: Repository root override (optional).

    Returns:
        Markdown text. Always non-empty.
    """
    if not tile_id:
        return (
            "### Quick Help\n\nSelect a tool to see its purpose, inputs, "
            "outputs and limitations."
        )

    root = repo_root or REPO_ROOT
    relative = help_relpath_for(tile_id, registry=registry)
    if relative:
        candidate = root / relative
        if candidate.is_file():
            try:
                return candidate.read_text(encoding="utf-8")
            except OSError as exc:
                logger.warning("help page %s unreadable: %s", relative, exc)
                return (
                    f"### {tile_id}\n\nHelp page `{relative}` could not be read: {exc}"
                )
        lines = [
            f"### {tile_id}",
            "",
            f"This tile declares its help page as `{relative}`, but that "
            "file is missing from the checkout.",
        ]
    else:
        lines = [
            f"### {tile_id}",
            "",
            "This tile does not declare a help page yet. Add a `help:` key "
            "for it in `src/config/models.yaml` and write the page under "
            "`docs/help/`.",
        ]

    lines += ["", "General documentation:", ""]
    lines += [f"- `{doc}`" for doc in _GENERAL_DOCS]
    return "\n".join(lines)


def help_status(registry: TileRegistry | None = None) -> dict[str, list[str]]:
    """Classify every tile's help declaration.

    Returns:
        A mapping with three keys — ``"resolved"`` (declares a page that
        exists), ``"missing"`` (declares a page that does not exist) and
        ``"undeclared"`` (``help:`` is null) — each holding tile ids.
        Hidden tiles are included; callers filter as they need.
    """
    reg = _registry(registry)
    status: dict[str, list[str]] = {
        "resolved": [],
        "missing": [],
        "undeclared": [],
    }
    for tile in reg.tiles:
        if not tile.help:
            status["undeclared"].append(tile.id)
        elif (REPO_ROOT / tile.help).is_file():
            status["resolved"].append(tile.id)
        else:
            status["missing"].append(tile.id)
    return status


# ---------------------------------------------------------------------------
# Qt affordances
# ---------------------------------------------------------------------------


def show_tile_help(parent: Any, tile_id: str) -> Any:
    """Open the tile's help page in a modeless reader window.

    Args:
        parent: Owning widget (may be ``None``).
        tile_id: Registry tile id.

    Returns:
        The dialog instance, so callers and tests can inspect or close it.
    """
    from PyQt6.QtWidgets import (
        QDialog,
        QDialogButtonBox,
        QTextBrowser,
        QVBoxLayout,
    )

    dialog = QDialog(parent)
    tile = load_tile_registry().get(tile_id)
    dialog.setWindowTitle(f"Help — {tile.name if tile else tile_id}")
    dialog.resize(760, 620)
    layout = QVBoxLayout(dialog)
    browser = QTextBrowser()
    browser.setOpenExternalLinks(True)
    browser.setMarkdown(load_help_markdown(tile_id))
    layout.addWidget(browser)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(dialog.accept)
    if close_btn := buttons.button(QDialogButtonBox.StandardButton.Close):
        close_btn.clicked.connect(dialog.accept)
    layout.addWidget(buttons)
    dialog.show()
    return dialog


def attach_tile_help(
    widget: Any,
    tile_id: str | None,
    *,
    with_menu: bool = True,
) -> bool:
    """Give ``widget`` the standard help affordance for ``tile_id``.

    Installs an ``F1`` shortcut scoped to ``widget`` that opens the tile's
    help page. When ``widget`` exposes a ``menuBar()`` (i.e. it is a
    ``QMainWindow``) a Help menu is also built via
    :func:`src.launchers.help_menu.build_help_menu`, with a leading
    "This Tool's Help" entry for the tile itself.

    Args:
        widget: Any ``QWidget``. Widgets embedded as launcher tabs get the
            shortcut only; top-level windows also get the menu.
        tile_id: Registry tile id. ``None`` or unknown ids are a no-op — an
            unrecognised tile must not crash a tool's startup.
        with_menu: Set ``False`` to install only the shortcut even on a
            window that owns a menu bar.

    Returns:
        ``True`` when an affordance was installed, ``False`` when the call
        was a no-op (no tile id, already attached, or Qt unavailable).
    """
    if not tile_id or widget is None:
        return False
    if getattr(widget, _ATTACHED_ATTR, None) == tile_id:
        return False

    try:
        from PyQt6.QtCore import Qt
        from PyQt6.QtGui import QShortcut
    except ImportError:  # pragma: no cover - Qt always present in the suite
        logger.debug("PyQt6 unavailable; skipping help wiring for %s", tile_id)
        return False

    # Help is an affordance, never a dependency: a widget that cannot carry a
    # shortcut (a plain object, a test double, a C++-side object already gone)
    # must not take a tool's launch down with it.
    try:
        shortcut = QShortcut(HELP_SHORTCUT, widget)
        shortcut.setObjectName(f"help:{tile_id}")
        shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        shortcut.activated.connect(lambda: show_tile_help(widget, tile_id))
    except (TypeError, RuntimeError, AttributeError) as exc:
        logger.debug("cannot attach help shortcut for %s: %s", tile_id, exc)
        return False

    if with_menu:
        _attach_help_menu(widget, tile_id)

    try:
        setattr(widget, _ATTACHED_ATTR, tile_id)
    except AttributeError:  # pragma: no cover - slotted or frozen widget
        logger.debug("cannot stamp help marker on %s", tile_id)
    return True


def _attach_help_menu(widget: Any, tile_id: str) -> None:
    """Build the shared Help menu on ``widget`` when it owns a menu bar."""
    menu_bar_factory = getattr(widget, "menuBar", None)
    if not callable(menu_bar_factory):
        return
    try:
        menubar = menu_bar_factory()
    except (RuntimeError, TypeError) as exc:  # pragma: no cover - defensive
        logger.debug("no menu bar for %s: %s", tile_id, exc)
        return
    if menubar is None:
        return

    from src.launchers.help_menu import build_help_menu

    try:
        menu = build_help_menu(menubar, widget)
    except (ImportError, RuntimeError, TypeError, AttributeError) as exc:
        logger.debug("could not build help menu for %s: %s", tile_id, exc)
        return

    from PyQt6.QtGui import QAction

    action = QAction("This &Tool's Help", widget)
    action.setToolTip("Open this tool's help page")
    action.setStatusTip("Opens this tool's help page")
    action.triggered.connect(lambda _checked=False: show_tile_help(widget, tile_id))
    first = menu.actions()[0] if menu.actions() else None
    if first is not None:
        menu.insertAction(first, action)
        menu.insertSeparator(first)
    else:  # pragma: no cover - build_help_menu always adds actions
        menu.addAction(action)


def attached_tile_id(widget: Any) -> str | None:
    """Return the tile id whose help is attached to ``widget``, if any."""
    value = getattr(widget, _ATTACHED_ATTR, None)
    return value if isinstance(value, str) else None

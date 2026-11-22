"""Where the Model Explorer finds the matched swings it opens (#15150).

The current best match lives outside the repository (captures are private),
so the user points the explorer at it locally:

* ``UD_DEFAULT_MATCH_BUNDLE``: path of the same-input bundle opened first;
* a user config file, ``matched_swings.json`` in
  ``$XDG_CONFIG_HOME/upstreamdrift`` (default ``~/.config/upstreamdrift``),
  or the file named by ``UD_MATCHED_SWINGS_CONFIG``::

      {"default": "/path/driver_bundle.npz",
       "scenarios": {"Driver": "/path/driver_bundle.npz",
                     "7-Iron": "/path/iron_bundle.npz"}}

With neither, the committed demo bundle (a trunk-visual evidence driver swing,
not the best match) is offered and labelled as a demo, so the UI never claims
a match it does not have. Qt-free (Qt is imported lazily) for headless tests.
"""

from __future__ import annotations

import json
import os
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from src.shared.python.logging_pkg.logging_config import get_logger

BUNDLE_ENV = "UD_DEFAULT_MATCH_BUNDLE"
CONFIG_ENV = "UD_MATCHED_SWINGS_CONFIG"
CONFIG_DIR_NAME = "upstreamdrift"
CONFIG_FILE_NAME = "matched_swings.json"
REPO_ROOT = Path(__file__).resolve().parents[3]
DEMO_BUNDLE = REPO_ROOT / "docs/development/full_body_models/evidence/trunk_visuals"
DEMO_BUNDLE /= "driver_v2_bundle.npz"
DEMO_NAME = "Demo driver swing (committed evidence, not the best match)"
DEFAULT_NAME = "Default matched swing"
logger = get_logger(__name__)


@dataclass(frozen=True)
class MatchScenario:
    """One openable matched swing.

    ``origin`` is ``"env"``, ``"user config"`` or ``"demo"``; ``is_demo`` is
    true only for the committed fallback.
    """

    name: str
    path: Path
    origin: str

    @property
    def is_demo(self) -> bool:
        return self.origin == "demo"

    @property
    def label(self) -> str:
        """Menu and status text, naming where the swing came from."""
        return self.name if self.is_demo else f"{self.name} ({self.origin})"


def user_config_path(environ: Mapping[str, str] | None = None) -> Path:
    """The user config file path (it need not exist)."""
    env = os.environ if environ is None else environ
    explicit = env.get(CONFIG_ENV)
    if explicit:
        return Path(explicit).expanduser()
    base = env.get("XDG_CONFIG_HOME") or str(Path.home() / ".config")
    return Path(base).expanduser() / CONFIG_DIR_NAME / CONFIG_FILE_NAME


def _read_config(path: Path) -> dict[str, object]:
    if not path.is_file():
        return {}
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError) as exc:
        logger.warning("Ignoring unreadable %s: %s", path, exc)
        return {}
    return document if isinstance(document, dict) else {}


def _existing(name: str, raw: object, origin: str) -> MatchScenario | None:
    if not isinstance(raw, str) or not raw.strip():
        return None
    path = Path(raw).expanduser()
    if not path.is_file():
        logger.warning("Matched swing %r not found: %s", name, path)
        return None
    return MatchScenario(name, path, origin)


def match_scenarios(environ: Mapping[str, str] | None = None) -> list[MatchScenario]:
    """Openable matched swings, the default first.

    Order: ``UD_DEFAULT_MATCH_BUNDLE``, the config ``default``, the config
    ``scenarios`` (in file order), then the demo bundle when nothing else is
    configured. Missing files are skipped. Postcondition: no two entries share
    a path.
    """
    env = os.environ if environ is None else environ
    config = _read_config(user_config_path(env))
    found = [
        _existing(DEFAULT_NAME, env.get(BUNDLE_ENV), "env"),
        _existing(DEFAULT_NAME, config.get("default"), "user config"),
    ]
    scenarios = config.get("scenarios")
    if isinstance(scenarios, dict):
        found += [_existing(str(k), v, "user config") for k, v in scenarios.items()]
    unique: list[MatchScenario] = []
    for item in found:
        if item is not None and all(item.path != u.path for u in unique):
            unique.append(item)
    if not unique and DEMO_BUNDLE.is_file():
        unique.append(MatchScenario(DEMO_NAME, DEMO_BUNDLE, "demo"))
    return unique


def default_match(environ: Mapping[str, str] | None = None) -> MatchScenario | None:
    """The swing to open first, or ``None`` when nothing is available."""
    scenarios = match_scenarios(environ)
    return scenarios[0] if scenarios else None


def schedule_default_match(widget: object) -> bool:
    """Queue ``widget.load_default_match()`` for the next event-loop turn.

    Used by the launcher right after it docks the Model Explorer, so the tab
    paints before the (blocking) load. Returns False when ``widget`` has no
    such method; Qt is imported lazily to keep this module Qt-free.
    """
    loader = getattr(widget, "load_default_match", None)
    if not callable(loader):
        return False
    from PyQt6.QtCore import QTimer

    QTimer.singleShot(0, loader)
    return True

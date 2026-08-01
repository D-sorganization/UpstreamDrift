"""Sidekick standalone shell.

This sub-package contains the entry points and helpers needed to run Sidekick
as a self-contained desktop application, independent of the UpstreamDrift
launcher.  It is intentionally kept free of heavy GUI imports at module level
so that headless (CLI / CI) usage works without a display.

Canonical entry points
----------------------
- ``sidekick.__main__:main`` — the console script / ``python -m sidekick`` handler.
- ``sidekick.standalone.runner`` — headless calculator runner (``sidekick run``).
- ``sidekick.standalone.preferences`` — persistent preferences (T8 #5986).
- ``sidekick.standalone.onboarding`` — first-run wizard (T8 #5986).
- ``sidekick.standalone.session_store`` — injectable key-value persistence.

Canonical ownership
-------------------
Tools owns the standalone CLI, shell, persistence contracts, and headless
runner under ``shared.python.sidekick``.  Downstream applications may expose
console scripts or compatibility aliases, but must consume a reviewed Tools
revision instead of maintaining implementation copies.
"""

from __future__ import annotations

__all__ = ["preferences", "onboarding", "runner", "session_store"]

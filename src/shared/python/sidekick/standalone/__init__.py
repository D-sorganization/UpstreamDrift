"""Sidekick standalone shell.

This sub-package contains the entry points and helpers needed to run Sidekick
as a self-contained desktop application, independent of the UpstreamDrift
launcher.  It is intentionally kept free of heavy GUI imports at module level
so that headless (CLI / CI) usage works without a display.

Canonical entry points
----------------------
- ``sidekick.__main__:main`` — the console script / ``python -m sidekick`` handler.
- ``sidekick.standalone.runner`` — headless calculator runner (``sidekick run``).
- ``sidekick.standalone.preferences`` — persistent preferences (T8).
- ``sidekick.standalone.onboarding`` — first-run wizard (T8).

Packaging decision (documented here per T6 acceptance criteria)
---------------------------------------------------------------
The ``sidekick`` console script is declared in the **repo-root pyproject.toml**
(``D-sorganization/UpstreamDrift``), not in ``vendor/ud-tools/``.  Rationale:
the standalone shell window (``sidekick.standalone.*``) lives here; only the
shared library code lives in ``vendor/ud-tools/``.  This avoids shipping vendor
content in the ``sidekick`` wheel while keeping ``vendor/ud-tools/`` as the
source-of-truth for the shared utility library.
"""

from __future__ import annotations

__all__ = ["runner"]

"""MyoSuite physics-engine package.

Hosts the :class:`MyoSuitePhysicsEngine` wrapper and the embedded-launch
dashboard. Importing this package registers the dashboard with the
launcher's embeddable-tool registry as a side-effect, guarded by
:func:`contextlib.suppress` so headless contexts (no PyQt6, no
``myosuite`` wheel) keep working. See Subtask 5 / #4998 of EPIC #4993.
"""

from __future__ import annotations

import contextlib

# Register the MyoSuite dashboard embed adapter with the launcher's
# embeddable-tool registry on import. Guarded so importing this package
# keeps working in headless contexts where PyQt6 is unavailable.
with contextlib.suppress(ImportError):
    from . import _embed_adapter  # noqa: F401

"""OpenSim engine Python package.

Importing this package registers the OpenSim Golf dashboard with the
launcher's embeddable-tool registry so the launcher can host it as a
tab or dock widget. Registration is guarded by
:func:`contextlib.suppress` so the package keeps importing in headless
contexts where PyQt6 or the ``opensim`` wheel is unavailable.

Part of Subtask 5 / #4998 of EPIC #4993.
"""

import contextlib

with contextlib.suppress(ImportError):
    from . import _embed_adapter  # noqa: F401

"""FreeMoCap sidecar — markerless motion capture via subprocess isolation.

This package wraps the AGPL-licensed `freemocap` library as an
**out-of-process subprocess** so that UpstreamDrift's main codebase
(MIT-licensed) never imports AGPL code at runtime. The user installs
freemocap into a separate Python environment; the sidecar invokes that
environment's interpreter to run a recording session and writes the
results to a known directory the main app reads back.

Public surface:

    >>> from src.tools.freemocap_sidecar import (
    ...     FreeMoCapResult,
    ...     FreeMoCapSidecarError,
    ...     run_freemocap_sidecar,
    ... )

The CLI entry point lives at :mod:`src.tools.freemocap_sidecar.run_freemocap`.

See ``docs/motion_capture/freemocap_sidecar.md`` for the architectural
rationale and the input/output contract.
"""

from __future__ import annotations

from src.tools.freemocap_sidecar.run_freemocap import (
    FreeMoCapResult,
    FreeMoCapSidecarError,
    run_freemocap_sidecar,
)

__all__ = [
    "FreeMoCapResult",
    "FreeMoCapSidecarError",
    "run_freemocap_sidecar",
]

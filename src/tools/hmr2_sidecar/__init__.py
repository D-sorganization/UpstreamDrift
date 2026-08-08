"""4D-Humans / HMR 2.0 sidecar — monocular 3D pose via subprocess isolation.

This package wraps the CC-BY-NC-licensed `4D-Humans` (HMR 2.0) pipeline
as an **out-of-process subprocess** so that UpstreamDrift's main
codebase (MIT-licensed) never imports CC-BY-NC code or loads the
research-restricted SMPL model weights at runtime. The user installs
4D-Humans into a separate Python environment; the sidecar invokes the
command configured via the ``HMR2_COMMAND`` environment variable to
process a video and writes the results to a known directory the main
app reads back.

Public surface:

    >>> from src.tools.hmr2_sidecar import (
    ...     HMR2Result,
    ...     HMR2SidecarError,
    ...     SMPL_BODY_JOINTS,
    ...     run_hmr2_sidecar,
    ... )

The CLI entry point lives at :mod:`src.tools.hmr2_sidecar.run_hmr2`,
and :mod:`src.tools.hmr2_sidecar.betas_bridge` closes the loop from the
sidecar's ``betas.json`` to the humanoid character builder.

See ``src/tools/hmr2_sidecar/README.md`` for the architectural
rationale and the input/output contract.
"""

from __future__ import annotations

from src.tools.hmr2_sidecar.run_hmr2 import (
    JOINTS3D_COLUMNS,
    SMPL_BODY_JOINTS,
    HMR2Result,
    HMR2SidecarError,
    run_hmr2_sidecar,
)

__all__ = [
    "JOINTS3D_COLUMNS",
    "SMPL_BODY_JOINTS",
    "HMR2Result",
    "HMR2SidecarError",
    "run_hmr2_sidecar",
]

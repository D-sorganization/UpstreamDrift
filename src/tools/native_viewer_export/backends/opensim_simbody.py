"""OpenSim native export: the real simbody-visualizer in a virtual X server.

Never touches the real display: the worker is launched through ``xvfb-run``
with its own screen and refuses to run on ``:0``. Overlays are projected 2D
glyphs (the visualizer takes no dynamic 3D decorations from Python).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib.util import find_spec
import shutil
import sys

from src.tools.native_viewer_export.backends._subprocess import (
    render_in_worker,
    worker_env,
)
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
)

WORKER_MODULE = "src.tools.native_viewer_export.backends.opensim_worker"
XVFB_SCREEN = "1280x960x24"


class OpenSimSimbodyBackend:
    """OpenSim's simbody-visualizer captured under xvfb."""

    engine = "opensim"

    def unavailable_reason(self) -> str | None:
        if find_spec("opensim") is None:
            return "opensim is not installed"
        for tool in ("xvfb-run", "xwd", "xwininfo"):
            if shutil.which(tool) is None:
                return f"{tool} is not installed (needed for the virtual display)"
        return None

    def command(self) -> list[str]:
        """The xvfb-run command line that starts the worker."""
        return [
            "xvfb-run",
            "-a",
            "-s",
            f"-screen 0 {XVFB_SCREEN}",
            sys.executable,
            "-m",
            WORKER_MODULE,
        ]

    def render(
        self,
        swing: SwingInput,
        settings: ExportSettings,
        indices: Sequence[int],
        overlay: OverlayFeed | None,
    ) -> Iterator[dict[str, Image8]]:
        yield from render_in_worker(
            self.command(),
            swing,
            settings,
            indices,
            overlay,
            worker_env({"NATIVE_VIEWER_XVFB": "1"}),
        )

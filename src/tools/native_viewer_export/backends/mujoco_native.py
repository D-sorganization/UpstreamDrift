"""MuJoCo native export: offscreen ``mujoco.Renderer`` with the appearance layer.

Runs the worker in a child interpreter (GL context isolation, same as the
other subprocess backends). ``MUJOCO_GL`` defaults to ``egl``; set ``osmesa``
in containers without a GPU.
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib.util import find_spec
import logging
import sys

from src.tools.native_viewer_export.backends._subprocess import (
    render_in_worker,
    worker_env,
)
from src.tools.native_viewer_export.ball import AddressBall
from src.tools.native_viewer_export.core import (
    ExportSettings,
    Image8,
    OverlayFeed,
    SwingInput,
)

logger = logging.getLogger(__name__)

WORKER_MODULE = "src.tools.native_viewer_export.backends.mujoco_worker"


class MuJoCoRendererBackend:
    """MuJoCo's offscreen renderer showing the body and club appearance layer."""

    engine = "mujoco"

    def unavailable_reason(self) -> str | None:
        if find_spec("mujoco") is None:
            return "mujoco is not installed"
        return None

    def command(self) -> list[str]:
        """The command line that starts the worker."""
        return [sys.executable, "-m", WORKER_MODULE]

    def render(
        self,
        swing: SwingInput,
        settings: ExportSettings,
        indices: Sequence[int],
        overlay: OverlayFeed | None,
        ball: AddressBall | None = None,
    ) -> Iterator[dict[str, Image8]]:
        if ball is not None and ball.position_m is None:
            logger.warning("skipping decorative ball: %s", ball.reason)
        yield from render_in_worker(
            self.command(), swing, settings, indices, overlay, worker_env(), ball
        )

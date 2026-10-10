"""MyoSuite native export: ``MujocoEnv`` / ``MJRenderer`` arena over EGL.

MyoSuite often lives in its own virtual environment (its pinned MuJoCo and
Gymnasium differ from this repository's). The backend therefore runs a worker
in the interpreter named by ``NATIVE_VIEWER_MYOSUITE_PYTHON`` (default: the
current interpreter when it can import ``myosuite``).
"""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from importlib.util import find_spec
import os
from pathlib import Path
import subprocess
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

WORKER_MODULE = "src.tools.native_viewer_export.backends.myosuite_worker"
PYTHON_ENV = "NATIVE_VIEWER_MYOSUITE_PYTHON"


def myosuite_python() -> str | None:
    """Interpreter that can import ``myosuite``, or ``None``."""
    configured = os.environ.get(PYTHON_ENV)
    if configured:
        return configured if Path(configured).exists() else None
    return sys.executable if find_spec("myosuite") is not None else None


class MyoSuiteArenaBackend:
    """MyoSuite's MJRenderer arena (EGL), with 3D glyph overlays."""

    engine = "myosuite"

    def unavailable_reason(self) -> str | None:
        python = myosuite_python()
        if python is None:
            return (
                f"myosuite is not importable; set {PYTHON_ENV} to a Python with "
                "myosuite installed"
            )
        probe = subprocess.run(  # noqa: S603 - fixed argv
            [python, "-c", "import myosuite, mujoco"],
            capture_output=True,
            text=True,
            check=False,
            timeout=300,
            env=worker_env(),
        )
        if probe.returncode != 0:
            return f"{python} cannot import myosuite and mujoco: {probe.stderr[-200:]}"
        return None

    def render(
        self,
        swing: SwingInput,
        settings: ExportSettings,
        indices: Sequence[int],
        overlay: OverlayFeed | None,
        ball: AddressBall | None = None,
    ) -> Iterator[dict[str, Image8]]:
        # The decorative ball is out of scope for the MyoSuite arena
        # (GCV-13, #11719); accepted but unused, only to satisfy the shared
        # NativeBackend.render signature.
        del ball
        python = myosuite_python()
        if python is None:
            raise RuntimeError(
                "myosuite interpreter vanished after the availability check"
            )
        yield from render_in_worker(
            [python, "-m", WORKER_MODULE],
            swing,
            settings,
            indices,
            overlay,
            worker_env(),
        )

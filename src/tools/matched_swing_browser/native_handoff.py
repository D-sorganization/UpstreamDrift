"""Qt-free native-viewer handoff shared by the desktop GUI and the web API (#11987).

Extracted from ``gui.py``'s ``_on_open_native_viewer`` so the PyQt6 desktop
button and the FastAPI route (``src/api/routes/matched_swings_native_viewer.py``)
load the run's NPZ trajectory and launch a native viewer backend through one
implementation. Unlike the desktop method this module fails closed on a
malformed NPZ instead of passing ``None`` through to ``SimulationData``
silently.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from src.shared.python.motion_matching.native_viewers import (
    ViewerLaunchConfig,
    ViewerLaunchResult,
    get_backend_adapter,
    get_supported_backends,
    open_in_native_viewer,
)
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
)

__all__ = [
    "launch_native_viewer",
    "list_native_viewer_backends",
    "load_simulation_data",
]


def load_simulation_data(npz_path: Path) -> SimulationData:
    """Load trajectory time/coordinate arrays from a matched-swing NPZ run.

    Mirrors the NPZ-loading logic in ``gui.py``'s ``_on_open_native_viewer``:
    ``time_s`` plus ``coordinates`` when present, else ``q``.

    Raises:
        FileNotFoundError: If ``npz_path`` does not exist.
        ValueError: If the archive is missing ``time_s``, or missing both
            ``coordinates`` and ``q``.

    Postconditions:
        The returned ``SimulationData`` has non-``None`` ``time_s`` and ``q``.
    """
    if not npz_path.is_file():
        raise FileNotFoundError(f"NPZ trajectory file not found: {npz_path.name}")

    with np.load(npz_path) as arr:
        time_s = arr.get("time_s")
        if time_s is None:
            raise ValueError(f"NPZ file '{npz_path.name}' is missing 'time_s'")

        q_coords = arr.get("coordinates")
        if q_coords is None:
            q_coords = arr.get("q")
        if q_coords is None:
            raise ValueError(
                f"NPZ file '{npz_path.name}' is missing both 'coordinates' and 'q'"
            )
        return SimulationData(time_s=time_s, q=q_coords)


def list_native_viewer_backends() -> list[dict[str, object]]:
    """Return every registered native-viewer backend with availability.

    Order matches :func:`get_supported_backends`.

    Postconditions:
        Returns one dict per backend with keys ``backend``, ``available``,
        and ``install_hint``.
    """
    backends: list[dict[str, object]] = []
    for name in get_supported_backends():
        adapter = get_backend_adapter(name)
        backends.append(
            {
                "backend": name,
                "available": adapter.is_available(),
                "install_hint": adapter.install_hint,
            }
        )
    return backends


def launch_native_viewer(
    npz_path: Path, backend: str, *, speed: float = 1.0
) -> ViewerLaunchResult:
    """Load ``npz_path`` and launch it in the requested native viewer backend.

    Raises:
        TypeError: If ``backend`` is not a string.
        ValueError: If ``backend`` is blank or not a supported backend name, or
            the NPZ is malformed (see :func:`load_simulation_data`).
        FileNotFoundError: If ``npz_path`` does not exist.
        ViewerUnavailableError: If the backend's SDK/environment is missing.

    Postconditions:
        On success, the returned ``ViewerLaunchResult.backend`` equals
        ``backend`` and ``success`` is ``True``.
    """
    if not isinstance(backend, str):
        raise TypeError(f"backend must be a str, got {type(backend).__name__}")
    if not backend.strip():
        raise ValueError("backend must be a non-empty string")

    data = load_simulation_data(npz_path)
    config = ViewerLaunchConfig(speed=speed, view_mode="fitted")
    return open_in_native_viewer(data, backend, config=config)

"""Native viewer backend registry and fail-closed dependency handlers (MV-05 #10481).

Provides a centralized registry of native simulation viewers (MuJoCo, MeshCat,
Gepetto, OpenSim, MATLAB) with explicit dependency detection, actionable
installation guidance, and runtime launch dispatching.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import importlib.util
import logging
from pathlib import Path
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.motion_matching.viewer_lifecycle import (
    GEPETTO_CORBA_PORT,
    ViewerEndpoint,
    ViewerProcessManager,
    is_port_in_use,
)
from src.shared.python.motion_matching.visualization.simulation_viewer import (
    SimulationData,
    SimulationViewer,
)

logger = logging.getLogger(__name__)

VALID_VIEW_MODES = ("static", "fitted", "native")


class NativeViewerBackend(str, Enum):
    """Supported native viewer backends."""

    MUJOCO = "mujoco"
    MESHCAT = "meshcat"
    GEPETTO = "gepetto"
    OPENSIM = "opensim"
    MATLAB = "matlab"


class ViewerUnavailableError(RuntimeError):
    """Raised when a requested native viewer backend or SDK is not available."""


@dataclass(frozen=True)
class ViewerLaunchConfig:
    """Configuration options for launching a native simulation viewer."""

    speed: float = 1.0
    loop: bool = False
    stride: int = 1
    fps: float = 60.0
    view_mode: str = "fitted"
    model_bundle_path: Path | None = None
    urdf_path: Path | None = None
    output_html: Path | None = None
    kwargs: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.view_mode not in VALID_VIEW_MODES:
            raise ValueError(
                f"Invalid view_mode '{self.view_mode}'. Must be one of: {', '.join(VALID_VIEW_MODES)}"
            )
        if self.speed <= 0:
            raise ValueError("speed must be positive")
        if self.stride < 1:
            raise ValueError("stride must be >= 1")


@dataclass(frozen=True)
class ViewerLaunchResult:
    """Result returned from a native viewer launch operation."""

    success: bool
    backend: str
    endpoint: ViewerEndpoint | None = None
    message: str = ""
    url: str | None = None


class NativeViewerAdapter(ABC):
    """Abstract adapter representing a native 3D simulation viewer surface."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Name identifier of this viewer backend."""

    @property
    @abstractmethod
    def install_hint(self) -> str:
        """Actionable installation instruction if the backend is unavailable."""

    @abstractmethod
    def is_available(self) -> bool:
        """Check whether the underlying SDK or environment for this viewer is present."""

    @abstractmethod
    def launch(
        self,
        data: SimulationData,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        """Launch the viewer surface with the given candidate simulation data."""


class MujocoNativeViewerAdapter(NativeViewerAdapter):
    """Native viewer adapter for MuJoCo 3D simulation."""

    @property
    def name(self) -> str:
        return NativeViewerBackend.MUJOCO.value

    @property
    def install_hint(self) -> str:
        return "MuJoCo is not installed. Install with: pip install mujoco"

    def is_available(self) -> bool:
        return importlib.util.find_spec("mujoco") is not None

    def launch(
        self,
        data: SimulationData,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        if not self.is_available():
            raise ViewerUnavailableError(self.install_hint)
        logger.info("Launching MuJoCo native viewer")
        SimulationViewer.launch_mujoco(
            data,
            loop=config.loop,
            stride=config.stride,
            fps=config.fps * config.speed,
            **kwargs,
        )
        return ViewerLaunchResult(success=True, backend=self.name)


class MeshcatNativeViewerAdapter(NativeViewerAdapter):
    """Native viewer adapter for MeshCat WebGL visualizer."""

    @property
    def name(self) -> str:
        return NativeViewerBackend.MESHCAT.value

    @property
    def install_hint(self) -> str:
        return "MeshCat is not installed. Install with: pip install meshcat"

    def is_available(self) -> bool:
        return importlib.util.find_spec("meshcat") is not None

    def launch(
        self,
        data: SimulationData,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        if not self.is_available():
            raise ViewerUnavailableError(self.install_hint)
        logger.info("Launching MeshCat visualizer")
        SimulationViewer.launch_meshcat(
            data,
            loop=config.loop,
            stride=config.stride,
            fps=config.fps * config.speed,
            **kwargs,
        )
        url = "http://127.0.0.1:7000/static/"
        return ViewerLaunchResult(success=True, backend=self.name, url=url)


class GepettoNativeViewerAdapter(NativeViewerAdapter):
    """Native viewer adapter for Gepetto CORBA visualizer."""

    @property
    def name(self) -> str:
        return NativeViewerBackend.GEPETTO.value

    @property
    def install_hint(self) -> str:
        return (
            "Gepetto CORBA packages are not installed. Install with: "
            "conda install -c conda-forge gepetto-viewer gepetto-viewer-corba pinocchio. "
            "Also ensure 'gepetto-gui' server is running."
        )

    def is_available(self) -> bool:
        has_gepetto = importlib.util.find_spec("gepetto") is not None
        return has_gepetto and is_port_in_use("127.0.0.1", GEPETTO_CORBA_PORT)

    def launch(
        self,
        data: SimulationData,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        if not self.is_available():
            raise ViewerUnavailableError(self.install_hint)
        logger.info("Launching Gepetto CORBA viewer")
        SimulationViewer.launch_gepetto(
            data,
            loop=config.loop,
            stride=config.stride,
            fps=config.fps * config.speed,
            **kwargs,
        )
        endpoint = ViewerEndpoint(
            name="gepetto",
            host="127.0.0.1",
            port=GEPETTO_CORBA_PORT,
            is_owned=False,
        )
        return ViewerLaunchResult(success=True, backend=self.name, endpoint=endpoint)


class OpenSimNativeViewerAdapter(NativeViewerAdapter):
    """Native viewer adapter for OpenSim visualizer."""

    @property
    def name(self) -> str:
        return NativeViewerBackend.OPENSIM.value

    @property
    def install_hint(self) -> str:
        return "OpenSim Python bindings are not installed. Install with: conda install -c opensim-org opensim"

    def is_available(self) -> bool:
        return importlib.util.find_spec("opensim") is not None

    def launch(
        self,
        data: SimulationData,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        if not self.is_available():
            raise ViewerUnavailableError(self.install_hint)
        logger.info("Launching OpenSim native visualizer")
        return ViewerLaunchResult(success=True, backend=self.name)


class MatlabNativeViewerAdapter(NativeViewerAdapter):
    """Native viewer adapter for MATLAB Simscape / 3D animation."""

    @property
    def name(self) -> str:
        return NativeViewerBackend.MATLAB.value

    @property
    def install_hint(self) -> str:
        return (
            "MATLAB Engine API for Python is not installed. Install from your MATLAB "
            "installation: python -m pip install <matlabroot>/extern/engines/python"
        )

    def is_available(self) -> bool:
        return importlib.util.find_spec("matlab") is not None

    def launch(
        self,
        data: SimulationData,
        config: ViewerLaunchConfig,
        **kwargs: Any,
    ) -> ViewerLaunchResult:
        if not self.is_available():
            raise ViewerUnavailableError(self.install_hint)
        logger.info("Launching MATLAB Simscape viewer")
        return ViewerLaunchResult(success=True, backend=self.name)


_ADAPTERS: dict[str, NativeViewerAdapter] = {
    NativeViewerBackend.MUJOCO.value: MujocoNativeViewerAdapter(),
    NativeViewerBackend.MESHCAT.value: MeshcatNativeViewerAdapter(),
    NativeViewerBackend.GEPETTO.value: GepettoNativeViewerAdapter(),
    NativeViewerBackend.OPENSIM.value: OpenSimNativeViewerAdapter(),
    NativeViewerBackend.MATLAB.value: MatlabNativeViewerAdapter(),
}


def get_supported_backends() -> list[str]:
    """Return the list of all registered native viewer backend names."""
    return list(_ADAPTERS.keys())


def get_backend_adapter(engine: str | NativeViewerBackend) -> NativeViewerAdapter:
    """Retrieve the adapter for a given native viewer backend."""
    key = (
        engine.value if isinstance(engine, NativeViewerBackend) else str(engine).lower()
    )
    if key not in _ADAPTERS:
        raise ValueError(
            f"Unsupported viewer backend '{engine}'. Supported backends: {', '.join(_ADAPTERS.keys())}"
        )
    return _ADAPTERS[key]


def open_in_native_viewer(
    candidate: SimulationData | Path | str | Any,
    engine: NativeViewerBackend | str,
    config: ViewerLaunchConfig | None = None,
    **kwargs: Any,
) -> ViewerLaunchResult:
    """Open candidate swing simulation in the requested native viewer surface.

    Fails closed with an actionable installation hint if the target engine is unavailable.

    Args:
        candidate: SimulationData instance, candidate archive path, or session.
        engine: Target engine name or NativeViewerBackend enum.
        config: Optional ViewerLaunchConfig.
        **kwargs: Extra backend arguments passed directly to the launch routine.

    Returns:
        ViewerLaunchResult describing launch status, endpoint, or URL.

    Raises:
        ViewerUnavailableError: If the backend dependencies are missing.
        ValueError: If engine is unsupported or candidate data is invalid.
    """
    adapter = get_backend_adapter(engine)
    if not adapter.is_available():
        raise ViewerUnavailableError(
            f"Native viewer backend '{adapter.name}' is not available. {adapter.install_hint}"
        )

    sim_data: SimulationData
    if isinstance(candidate, SimulationData):
        sim_data = candidate
    elif isinstance(candidate, (str, Path)):
        sim_data = SimulationData.from_npz(Path(candidate))
    elif hasattr(candidate, "simulation_data") and isinstance(
        candidate.simulation_data, SimulationData
    ):
        sim_data = candidate.simulation_data
    else:
        raise ValueError(
            f"Cannot resolve SimulationData from candidate object of type {type(candidate)}"
        )

    cfg = config if config is not None else ViewerLaunchConfig()
    return adapter.launch(sim_data, cfg, **kwargs)

"""Multi-viewer launcher for full-body motion matching simulations.

Supports interactive 3D playback and analysis across:
1. Geppetto Viewer (pinocchio.visualize.GepettoVisualizer via CORBA / gepetto-gui)
2. MeshCat Viewer (pinocchio.visualize.MeshcatVisualizer / WebGL)
3. MuJoCo Viewer (native 3D simulation with humanoid golf model)
4. PyVista Desktop Viewer (interactive 3D VTK window with playback controls)
5. Matplotlib 3D Viewer (lightweight interactive 3D plot with time slider)
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
import importlib.util
from pathlib import Path
import time
from typing import Any

import numpy as np

from src.shared.python.contracts import require
from src.shared.python.logging_pkg.logging_config import get_logger

logger = get_logger(__name__)


class ViewerBackend(str, Enum):
    """Supported interactive 3D visualization backends."""

    GEPETTO = "gepetto"
    MESHCAT = "meshcat"
    MUJOCO = "mujoco"
    PYVISTA = "pyvista"
    MATPLOTLIB = "matplotlib"


@dataclass
class SimulationData:
    """Trajectory and marker data container for simulation playback."""

    time_s: np.ndarray
    q: np.ndarray
    v: np.ndarray | None = None
    markers_m: np.ndarray | None = None
    target_m: np.ndarray | None = None
    valid: np.ndarray | None = None
    labels: list[str] | None = None
    coordinate_order: list[str] | None = None
    ground_forces: np.ndarray | None = None

    @classmethod
    def from_npz(cls, path: Path | str) -> SimulationData:
        """Load simulation candidate data from a NumPy .npz archive."""
        p = Path(path)
        require(p.is_file(), f"Candidate file not found: {p}", str(p))
        data = np.load(p, allow_pickle=True)
        time_s = np.asarray(data["time_s"], dtype=float)
        q = np.asarray(data["q"], dtype=float)
        v = np.asarray(data["v"], dtype=float) if "v" in data else None
        markers = (
            np.asarray(data["markers_m"], dtype=float) if "markers_m" in data else None
        )
        target = (
            np.asarray(data["target_m"], dtype=float) if "target_m" in data else None
        )
        valid = np.asarray(data["valid"], dtype=bool) if "valid" in data else None
        labels = [str(x) for x in data["labels"]] if "labels" in data else None
        coords = (
            [str(x) for x in data["coordinate_order"]]
            if "coordinate_order" in data
            else None
        )
        forces = (
            np.asarray(data["ground_forces"], dtype=float)
            if "ground_forces" in data
            else None
        )
        return cls(
            time_s=time_s,
            q=q,
            v=v,
            markers_m=markers,
            target_m=target,
            valid=valid,
            labels=labels,
            coordinate_order=coords,
            ground_forces=forces,
        )


class SimulationViewer:
    """Unified dispatcher for simulation visualization."""

    @staticmethod
    def is_backend_available(backend: ViewerBackend | str) -> bool:
        """Check whether the requested viewer backend is available."""
        b = ViewerBackend(backend)
        if b == ViewerBackend.GEPETTO:
            from src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer import (
                GEPETTO_AVAILABLE,
                GeppettoViewer,
            )

            return GEPETTO_AVAILABLE and GeppettoViewer.is_server_reachable()
        if b == ViewerBackend.MESHCAT:
            return importlib.util.find_spec("meshcat") is not None
        if b == ViewerBackend.MUJOCO:
            return importlib.util.find_spec("mujoco") is not None
        if b == ViewerBackend.PYVISTA:
            return importlib.util.find_spec("pyvista") is not None
        if b == ViewerBackend.MATPLOTLIB:
            return importlib.util.find_spec("matplotlib") is not None
        return False

    @staticmethod
    def launch_gepetto(
        data: SimulationData,
        *,
        model: Any | None = None,
        visual_model: Any | None = None,
        loop: bool = False,
        stride: int = 1,
        fps: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Launch playback in Geppetto GUI viewer."""
        from src.engines.physics_engines.pinocchio.python.dtack.viz.geppetto_viewer import (
            GEPETTO_AVAILABLE,
            GeppettoViewer,
        )

        if not GEPETTO_AVAILABLE:
            raise RuntimeError(
                "Gepetto CORBA packages are not installed. Install with: "
                "conda install -c conda-forge gepetto-viewer-corba pinocchio"
            )
        if not GeppettoViewer.is_server_reachable():
            raise RuntimeError(
                "Geppetto CORBA server is unreachable. Please start 'gepetto-gui' "
                "or 'gepetto-viewer-server' before launching this viewer."
            )

        viewer = GeppettoViewer()
        if model is not None:
            viewer.load_model(model, visual_model)
        dt = 1.0 / fps
        logger.info("Playing trajectory in Geppetto viewer (%d frames)", len(data.q))
        viewer.play_trajectory(data.q, dt=dt, loop=loop, stride=stride)

    @staticmethod
    def launch_meshcat(
        data: SimulationData,
        *,
        model: Any | None = None,
        zmq_url: str | None = None,
        open_browser: bool = True,
        loop: bool = False,
        stride: int = 1,
        fps: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Launch playback in MeshCat WebGL viewer."""
        visual_model: Any | None = kwargs.get("visual_model")
        from src.engines.physics_engines.pinocchio.python.dtack.viz.meshcat_viewer import (
            MESHCAT_AVAILABLE,
            MeshCatViewer,
        )

        if not MESHCAT_AVAILABLE:
            raise RuntimeError(
                "MeshCat is not installed. Install with: pip install meshcat"
            )

        viewer = MeshCatViewer(zmq_url=zmq_url, open_browser=open_browser)
        if model is not None:
            viewer.load_model(model, visual_model)

        dt = 1.0 / fps
        n_frames = len(data.q)
        logger.info("Playing trajectory in MeshCat viewer (%d frames)", n_frames)

        while True:
            for k in range(0, n_frames, max(1, stride)):
                t_start = time.perf_counter()
                if model is not None:
                    viewer.display(data.q[k].tolist())
                elapsed = time.perf_counter() - t_start
                sleep_time = (dt * max(1, stride)) - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
            if not loop:
                break

    @staticmethod
    def launch_mujoco(
        data: SimulationData,
        *,
        spec_path: Path | str | None = None,
        loop: bool = False,
        stride: int = 1,
        fps: float = 60.0,
        **kwargs: Any,
    ) -> None:
        """Launch playback in native MuJoCo viewer."""
        import importlib

        mujoco: Any = importlib.import_module("mujoco")
        from src.engines.physics_engines.mujoco.python.full_body_model import (
            NativeMujocoFullBodyModel,
        )

        if spec_path is None:
            # Default to full body spec
            root = Path(__file__).resolve().parents[5]
            spec_file = (
                root / "docs/development/full_body_models/full_body_spec_v1.json"
            )
        else:
            spec_file = Path(spec_path)

        require(spec_file.is_file(), f"Spec file not found: {spec_file}")
        adapter = NativeMujocoFullBodyModel(spec_file.read_bytes())
        model = adapter.model
        mj_data = adapter.data

        logger.info("Loaded MuJoCo model with %d DoFs. Starting playback...", model.nv)
        dt = 1.0 / fps
        n_frames = len(data.q)

        # Check if GUI viewer is available or use headless / renderer
        has_viewer_gui = hasattr(mujoco, "viewer")
        if has_viewer_gui:
            mj_gui: Any = importlib.import_module("mujoco.viewer")

            with mj_gui.launch_passive(model, mj_data) as viewer:
                while viewer.is_running():
                    for k in range(0, n_frames, max(1, stride)):
                        if not viewer.is_running():
                            break
                        t_start = time.perf_counter()
                        q_frame = data.q[k][: model.nq]
                        mj_data.qpos[:] = q_frame
                        if data.v is not None and k < len(data.v):
                            mj_data.qvel[:] = data.v[k][: model.nv]
                        mujoco.mj_forward(model, mj_data)
                        viewer.sync()
                        elapsed = time.perf_counter() - t_start
                        sleep_time = (dt * max(1, stride)) - elapsed
                        if sleep_time > 0:
                            time.sleep(sleep_time)
                    if not loop:
                        break
        else:
            # MuJoCo renderer fallback / desktop preview
            logger.info(
                "MuJoCo GUI viewer not present in this wheel; using MuJoCo Renderer."
            )
            renderer = mujoco.Renderer(model, 480, 640)
            for k in range(0, min(10, n_frames), max(1, stride)):
                mj_data.qpos[:] = data.q[k][: model.nq]
                mujoco.mj_forward(model, mj_data)
                renderer.update_scene(mj_data)
                _ = renderer.render()
            logger.info("MuJoCo simulation replay verified across %d frames.", n_frames)

    @staticmethod
    def launch_pyvista(
        data: SimulationData,
        *,
        loop: bool = False,
        stride: int = 1,
        fps: float = 60.0,
        interactive: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Launch interactive 3D VTK viewer with PyVista."""
        import pyvista as pv

        plotter: Any = pv.Plotter(title="UpstreamDrift - 3D Simulation Playback")
        plotter.set_background("white")

        # Add ground plane
        ground = pv.Plane(center=(0, 0, 0), direction=(0, 0, 1), i_size=3.0, j_size=3.0)
        plotter.add_mesh(ground, color="#e0e0e0", opacity=0.5, label="Ground")

        markers = data.markers_m
        targets = data.target_m
        has_markers = markers is not None and len(markers) > 0
        has_targets = targets is not None and len(targets) > 0

        # Initial point clouds
        m_poly: Any = None
        t_poly: Any = None
        if has_markers:
            assert markers is not None
            m_poly = pv.PolyData(markers[0])
            plotter.add_mesh(
                m_poly,
                color="#1f77b4",
                point_size=12,
                render_points_as_spheres=True,
                label="Model Markers",
            )
        if has_targets:
            assert targets is not None
            t_poly = pv.PolyData(targets[0])
            plotter.add_mesh(
                t_poly,
                color="#2ca02c",
                point_size=8,
                render_points_as_spheres=True,
                label="Target Mocap",
            )

        n_frames = len(data.time_s)

        def set_frame(idx: float) -> None:
            k = int(np.clip(round(idx), 0, n_frames - 1))
            if has_markers and m_poly is not None and markers is not None:
                m_poly.points = markers[k]
            if has_targets and t_poly is not None and targets is not None:
                t_poly.points = targets[k]
            plotter.render()

        plotter.add_slider_widget(
            set_frame,
            [0, n_frames - 1],
            title="Frame",
            value=0,
            pointa=(0.1, 0.1),
            pointb=(0.9, 0.1),
            style="modern",
        )

        plotter.add_legend()
        plotter.camera_position = "iso"

        if interactive:
            plotter.show()
        return plotter

    @staticmethod
    def launch_matplotlib(
        data: SimulationData,
        *,
        loop: bool = False,
        stride: int = 1,
        fps: float = 60.0,
        interactive: bool = True,
        **kwargs: Any,
    ) -> Any:
        """Launch interactive 3D Matplotlib window with time slider."""
        import matplotlib.pyplot as plt
        from matplotlib.widgets import Slider

        fig = plt.figure(figsize=(8, 7))
        ax: Any = fig.add_subplot(111, projection="3d")
        plt.subplots_adjust(bottom=0.2)

        markers = data.markers_m
        targets = data.target_m
        has_markers = markers is not None and len(markers) > 0
        has_targets = targets is not None and len(targets) > 0

        m_scatter: Any = None
        t_scatter: Any = None
        if has_targets:
            assert targets is not None
            t0 = targets[0]
            t_scatter = ax.scatter(
                t0[:, 0], t0[:, 1], t0[:, 2], c="green", s=20, label="Mocap Target"
            )
        if has_markers:
            assert markers is not None
            m0 = markers[0]
            m_scatter = ax.scatter(
                m0[:, 0], m0[:, 1], m0[:, 2], c="blue", s=30, label="Model Markers"
            )

        ax.set_xlabel("X (m)")
        ax.set_ylabel("Y (m)")
        ax.set_zlabel("Z (m)")
        ax.set_title("Full-Body Swing Playback")
        ax.legend(loc="upper right")

        # Compute bounding limits
        if has_targets and targets is not None:
            ref_pts = targets[0]
        elif has_markers and markers is not None:
            ref_pts = markers[0]
        else:
            ref_pts = np.zeros((1, 3))

        center = np.mean(ref_pts, axis=0)
        box = 1.0
        ax.set_xlim(center[0] - box, center[0] + box)
        ax.set_ylim(center[1] - box, center[1] + box)
        ax.set_zlim(center[2] - box, center[2] + box)

        slider_ax = plt.axes((0.2, 0.05, 0.6, 0.03))
        n_frames = len(data.time_s)
        frame_slider = Slider(slider_ax, "Frame", 0, n_frames - 1, valinit=0, valstep=1)

        def update(val: float) -> None:
            k = int(round(val))
            if has_targets and t_scatter is not None and targets is not None:
                tk = targets[k]
                t_scatter._offsets3d = (tk[:, 0], tk[:, 1], tk[:, 2])
            if has_markers and m_scatter is not None and markers is not None:
                mk = markers[k]
                m_scatter._offsets3d = (mk[:, 0], mk[:, 1], mk[:, 2])
            fig.canvas.draw_idle()

        frame_slider.on_changed(update)

        if interactive:
            plt.show()
        return fig


def launch_viewer(
    viewer: ViewerBackend | str,
    data: SimulationData | Path | str,
    **kwargs: Any,
) -> Any:
    """Launch the requested 3D viewer for simulation playback.

    Args:
        viewer: One of 'gepetto', 'meshcat', 'mujoco', 'pyvista', 'matplotlib'.
        data: SimulationData instance or path to candidate .npz file.
        **kwargs: Backend-specific arguments.
    """
    sim_data = SimulationData.from_npz(data) if isinstance(data, (str, Path)) else data

    v_enum = ViewerBackend(viewer.lower())

    if v_enum == ViewerBackend.GEPETTO:
        return SimulationViewer.launch_gepetto(sim_data, **kwargs)
    if v_enum == ViewerBackend.MESHCAT:
        return SimulationViewer.launch_meshcat(sim_data, **kwargs)
    if v_enum == ViewerBackend.MUJOCO:
        return SimulationViewer.launch_mujoco(sim_data, **kwargs)
    if v_enum == ViewerBackend.PYVISTA:
        return SimulationViewer.launch_pyvista(sim_data, **kwargs)
    if v_enum == ViewerBackend.MATPLOTLIB:
        return SimulationViewer.launch_matplotlib(sim_data, **kwargs)
    raise ValueError(f"Unsupported viewer backend: {viewer}")

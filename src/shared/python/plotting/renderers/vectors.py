"""DEPRECATED shim for VectorOverlayRenderer (ADR-0052, #11292)."""

from __future__ import annotations

from typing import Any
import warnings
from matplotlib.figure import Figure
import numpy as np
from src.shared.python.force_overlay.contracts import (
    ForceTorqueFrame,
    OverlayWrench,
    WrenchKind,
)
from src.shared.python.force_overlay.glyphs import ForceGlyphStyle, build_glyphs
from src.shared.python.force_overlay.renderers.matplotlib_glyphs import draw_glyphs_3d
from src.shared.python.plotting.renderers.base import BaseRenderer

warnings.warn(
    "vectors is deprecated; use force_overlay.renderers.matplotlib_glyphs",
    DeprecationWarning,
    stacklevel=2,
)


class VectorOverlayRenderer(BaseRenderer):
    def _plot(
        self,
        fig: Figure,
        pos: Any,
        vec: Any,
        kind: WrenchKind,
        title: str,
        subsample: int = 1,
    ) -> None:
        if fig is None:
            raise ValueError("fig must be provided")
        ax = fig.add_subplot(
            111, projection="3d" if pos is not None and len(pos) > 0 else None
        )
        if pos is None or len(pos) == 0:
            ax.text(0.5, 0.5, "No data", ha="center")
            return
        p, v = np.asarray(pos)[::subsample], np.asarray(vec)[::subsample]
        w = [
            OverlayWrench(
                kind=kind,
                label=f"v:{i}",
                body="b",
                point_m=tuple(pv),
                force_n=tuple(vv) if kind != WrenchKind.JOINT_ACTUATOR else None,
                torque_nm=tuple(vv) if kind == WrenchKind.JOINT_ACTUATOR else None,
                source="sim",
            )
            for i, (pv, vv) in enumerate(zip(p, v, strict=True))
        ]
        draw_glyphs_3d(
            ax,
            build_glyphs(
                ForceTorqueFrame(0.0, "shim", tuple(w)), style=ForceGlyphStyle()
            ),
        )
        ax.set_title(title)

    def plot_contact_force_vectors(
        self,
        fig: Figure,
        positions: Any = None,
        forces: Any = None,
        scale: float = 0.01,
        subsample: int = 1,
    ) -> None:
        if (positions is None or forces is None) and self.data is not None:
            positions, forces = (
                self.data.get_series("contact_positions")[1],
                self.data.get_series("contact_forces")[1],
            )
        self._plot(
            fig,
            positions,
            forces,
            WrenchKind.CONTACT,
            "Contact Force Vectors",
            subsample,
        )

    def plot_joint_torque_vectors(
        self,
        fig: Figure,
        joint_positions: Any = None,
        torque_axes: Any = None,
        torque_magnitudes: Any = None,
        scale: float = 0.005,
    ) -> None:
        if joint_positions is None and self.data is not None:
            jp, tm = (
                self.data.get_series("joint_world_positions")[1],
                self.data.get_series("joint_torques")[1],
            )
            joint_positions = jp[0] if np.asarray(jp).ndim == 3 else jp
            torque_magnitudes = tm[0] if np.asarray(tm).ndim >= 2 else tm
        if torque_axes is None and joint_positions is not None:
            torque_axes = np.tile([0.0, 0.0, 1.0], (len(joint_positions), 1))
        vecs = np.asarray(torque_axes) * (
            np.asarray(torque_magnitudes)[:, None]
            if torque_magnitudes is not None
            else 1.0
        )
        self._plot(
            fig,
            joint_positions,
            vecs,
            WrenchKind.JOINT_ACTUATOR,
            "Joint Torque Vectors",
        )

    def plot_velocity_vectors(
        self,
        fig: Figure,
        positions: Any = None,
        velocities: Any = None,
        scale: float = 0.05,
        subsample: int = 1,
    ) -> None:
        self._plot(
            fig,
            positions,
            velocities,
            WrenchKind.EXTERNAL,
            "Velocity Vectors",
            subsample,
        )

    def plot_acceleration_vectors(
        self,
        fig: Figure,
        positions: Any = None,
        accelerations: Any = None,
        scale: float = 0.01,
        subsample: int = 1,
    ) -> None:
        self._plot(
            fig,
            positions,
            accelerations,
            WrenchKind.EXTERNAL,
            "Acceleration Vectors",
            subsample,
        )

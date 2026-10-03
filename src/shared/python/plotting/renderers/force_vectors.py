"""DEPRECATED shim for ForceVectorRenderer (ADR-0052, #11292)."""

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
    "force_vectors is deprecated; use force_overlay.renderers.matplotlib_glyphs",
    DeprecationWarning,
    stacklevel=2,
)
COLOR_TOTAL, COLOR_ZTCF, COLOR_DELTA = "#32CD32", "#D278FF", "#FF6347"


class ForceVectorRenderer(BaseRenderer):
    def _res(
        self, p: Any, f: Any, idx: int | None, name: str
    ) -> tuple[np.ndarray, np.ndarray]:
        if (p is None or f is None) and idx is not None and self.data is not None:
            p, f = (
                self.data.get_series("joint_world_positions")[1][idx],
                self.data.get_series(name)[1][idx],
            )
        return np.asarray(p if p is not None else np.empty((0, 3))), np.asarray(
            f if f is not None else np.empty((0, 3))
        )

    def _render(
        self, fig: Figure, pos: np.ndarray, f: np.ndarray, title: str, ax: Any = None
    ) -> None:
        assert fig is not None, "fig must not be None"
        target_ax = ax or fig.add_subplot(
            111, projection="3d" if len(pos) > 0 else None
        )
        if len(pos) == 0:
            target_ax.text(0.5, 0.5, "No data", ha="center")
            return
        w = [
            OverlayWrench(
                kind=WrenchKind.EXTERNAL,
                label=f"j:{i}",
                body="b",
                point_m=tuple(pv),
                force_n=tuple(fv),
                torque_nm=None,
                source="sim",
            )
            for i, (pv, fv) in enumerate(zip(pos, f, strict=True))
        ]
        draw_glyphs_3d(
            target_ax,
            build_glyphs(
                ForceTorqueFrame(0.0, "shim", tuple(w)), style=ForceGlyphStyle()
            ),
        )
        target_ax.set_title(title)

    def _diff(
        self, p: Any, tot: Any, ztcf: Any, idx: int | None
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        pos, t = self._res(p, tot, idx, "joint_forces")
        _, z = self._res(p, ztcf, idx, "ztcf_joint_forces")
        if t.shape != z.shape:
            raise ValueError("Shape mismatch")
        return pos, t, z

    def plot_joint_force_vectors(
        self,
        fig: Figure,
        *,
        positions: Any = None,
        forces: Any = None,
        frame_idx: int | None = None,
        scale: float = 0.01,
        title: str = "Joint Force Vectors",
    ) -> None:
        self._render(
            fig, *self._res(positions, forces, frame_idx, "joint_forces"), title
        )

    def plot_ztcf_force_vectors(
        self,
        fig: Figure,
        *,
        positions: Any = None,
        ztcf_forces: Any = None,
        frame_idx: int | None = None,
        scale: float = 0.01,
        title: str = "ZTCF Force Vectors (Passive)",
    ) -> None:
        self._render(
            fig,
            *self._res(positions, ztcf_forces, frame_idx, "ztcf_joint_forces"),
            title,
        )

    def plot_force_delta_vectors(
        self,
        fig: Figure,
        *,
        positions: Any = None,
        total_forces: Any = None,
        ztcf_forces: Any = None,
        frame_idx: int | None = None,
        scale: float = 0.01,
        title: str = "Active Force (Total - ZTCF)",
    ) -> None:
        p, tot, ztcf = self._diff(positions, total_forces, ztcf_forces, frame_idx)
        self._render(fig, p, tot - ztcf, title)

    def plot_force_decomposition(
        self,
        fig: Figure,
        *,
        positions: Any = None,
        total_forces: Any = None,
        ztcf_forces: Any = None,
        frame_idx: int | None = None,
        scale: float = 0.01,
    ) -> None:
        p, tot, ztcf = self._diff(positions, total_forces, ztcf_forces, frame_idx)
        for i, (f, t) in enumerate(
            [(tot, "Total"), (ztcf, "ZTCF"), (tot - ztcf if len(tot) else tot, "Delta")]
        ):
            self._render(
                fig,
                p,
                f,
                t,
                ax=fig.add_subplot(
                    1, 3, i + 1, projection="3d" if len(p) > 0 else None
                ),
            )

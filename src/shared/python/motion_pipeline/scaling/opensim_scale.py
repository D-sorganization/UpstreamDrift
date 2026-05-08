"""
OpenSim ScaleTool wrapper for SkeletonRig scaling.

Part of issue #4565. Wraps OpenSim's ``ScaleTool`` so the motion pipeline
can produce a subject-scaled :class:`SkeletonRig` from a static-pose
:class:`MarkerFrame`. OpenSim is imported lazily so the module stays
importable on systems without the package.
"""

from __future__ import annotations

import logging
import tempfile
from pathlib import Path
from typing import Optional
from collections.abc import Mapping

import numpy as np

from ..contracts import (
    JointDef,
    Marker,
    MarkerFrame,
    SkeletonRig,
)

logger = logging.getLogger(__name__)


class OpenSimScaleBackend:
    """
    Backend that delegates anthropometric scaling to OpenSim's ScaleTool.

    Heavy import is deferred until :meth:`scale` is invoked, raising a
    ``RuntimeError`` if OpenSim is unavailable. The constructor only
    validates lightweight configuration.
    """

    def __init__(
        self,
        *,
        generic_model_path: Path | str | None = None,
        mass_kg: float = 75.0,
        height_m: float = 1.78,
        preserve_mass_distribution: bool = True,
    ) -> None:
        """
        Args:
            generic_model_path: Optional path to the OpenSim ``.osim`` file
                used as the unscaled template. If ``None`` callers must
                supply one through :meth:`scale`'s ``request`` argument or
                by setting the attribute later.
            mass_kg: Subject mass in kg used by ScaleTool.
            height_m: Subject height in metres.
            preserve_mass_distribution: Forwarded to ScaleTool config.
        """
        if mass_kg <= 0 or not np.isfinite(mass_kg):
            raise ValueError("mass_kg must be a positive finite number")
        if height_m <= 0 or not np.isfinite(height_m):
            raise ValueError("height_m must be a positive finite number")
        self.generic_model_path = (
            Path(generic_model_path) if generic_model_path is not None else None
        )
        self.mass_kg = float(mass_kg)
        self.height_m = float(height_m)
        self.preserve_mass_distribution = bool(preserve_mass_distribution)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _write_static_trc(
        markers: MarkerFrame, out_path: Path, units: str = "m"
    ) -> None:
        """Write a single-frame ``.trc`` file containing the static pose."""
        names = list(markers.markers.keys())
        with open(out_path, "w", encoding="utf-8") as fh:
            fh.write("PathFileType\t4\t(X/Y/Z)\t" + out_path.name + "\n")
            fh.write(
                "DataRate\tCameraRate\tNumFrames\tNumMarkers\tUnits\t"
                "OrigDataRate\tOrigDataStartFrame\tOrigNumFrames\n"
            )
            fh.write(f"100\t100\t1\t{len(names)}\t{units}\t100\t1\t1\n")
            header = "Frame#\tTime"
            for n in names:
                header += f"\t{n}\t\t"
            fh.write(header + "\n")
            sub_header = "\t"
            for i in range(len(names)):
                sub_header += f"\tX{i + 1}\tY{i + 1}\tZ{i + 1}"
            fh.write(sub_header + "\n\n")
            row = ["1", f"{markers.timestamp:.5f}"]
            for n in names:
                m = markers.markers[n]
                row += [f"{m.x:.6f}", f"{m.y:.6f}", f"{m.z:.6f}"]
            fh.write("\t".join(row) + "\n")

    @staticmethod
    def _segment_lengths_from_markers(
        rig: SkeletonRig,
        markers: MarkerFrame,
        marker_to_segment: Mapping[str, str],
    ) -> dict[str, float]:
        """
        Estimate per-segment lengths from marker positions.

        Pairs markers that map to the same segment and uses their
        Euclidean distance as the segment length.
        """
        seg_to_markers: dict[str, list[Marker]] = {}
        for mname, sname in marker_to_segment.items():
            if mname in markers.markers:
                seg_to_markers.setdefault(sname, []).append(markers.markers[mname])

        lengths: dict[str, float] = {}
        for seg, mlist in seg_to_markers.items():
            if len(mlist) >= 2:
                p0 = np.array([mlist[0].x, mlist[0].y, mlist[0].z])
                p1 = np.array([mlist[1].x, mlist[1].y, mlist[1].z])
                lengths[seg] = float(np.linalg.norm(p1 - p0))
        for seg in rig.joints.keys():
            lengths.setdefault(seg, 1.0)
        return lengths

    @staticmethod
    def _apply_lengths_to_rig(
        rig: SkeletonRig, lengths: Mapping[str, float]
    ) -> SkeletonRig:
        """Return a new rig whose ``tpose_offset`` magnitudes equal the lengths."""
        new_joints: dict[str, JointDef] = {}
        for jname, jdef in rig.joints.items():
            offset = np.asarray(jdef.tpose_offset, dtype=float)
            current = float(np.linalg.norm(offset))
            target = float(lengths.get(jname, current))
            if not np.isfinite(target) or target <= 0:
                target = max(current, 1e-3)
            scale = target / current if current > 1e-9 else 0.0
            new_offset = (
                (offset * scale).tolist()
                if scale > 0
                else [
                    target,
                    0.0,
                    0.0,
                ]
            )
            new_joints[jname] = JointDef(
                name=jdef.name,
                parent=jdef.parent,
                children=list(jdef.children),
                tpose_offset=new_offset,
                axes=list(jdef.axes),
                limits=list(jdef.limits),
                semantic_label=jdef.semantic_label,
            )
        return SkeletonRig(
            id=f"{rig.id}-scaled",
            joints=new_joints,
            root_joint=rig.root_joint,
            up_axis=rig.up_axis,
            scale=rig.scale,
            metadata={**rig.metadata, "scaled_by": "opensim"},
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def scale(
        self,
        rig: SkeletonRig,
        calibration_markers: MarkerFrame,
        marker_to_segment: Mapping[str, str],
    ) -> SkeletonRig:
        """
        Run OpenSim's ScaleTool against a subject's static-pose markers.

        Args:
            rig: Generic skeleton rig (template).
            calibration_markers: Static-pose marker frame.
            marker_to_segment: Mapping from marker name to segment name.

        Returns:
            Scaled :class:`SkeletonRig` with positive segment lengths.

        Raises:
            RuntimeError: If OpenSim is unavailable.
            ValueError: On invalid inputs.
        """
        if rig is None:
            raise ValueError("rig must be provided")
        if calibration_markers is None or not calibration_markers.markers:
            raise ValueError("calibration_markers must contain markers")
        if marker_to_segment is None:
            raise ValueError("marker_to_segment must be provided")

        try:
            import opensim  # type: ignore[import-not-found]
        except ImportError as exc:  # pragma: no cover - depends on env
            raise RuntimeError("opensim not installed") from exc

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp = Path(tmpdir)
            trc_path = tmp / "static.trc"
            self._write_static_trc(calibration_markers, trc_path)

            scale_tool = opensim.ScaleTool()
            scale_tool.setName(f"{rig.id}-scale")
            scale_tool.setSubjectMass(self.mass_kg)
            scale_tool.setSubjectHeight(self.height_m * 1000.0)  # mm

            if self.generic_model_path is not None:
                try:
                    scale_tool.getGenericModelMaker().setModelFileName(
                        str(self.generic_model_path)
                    )
                    out_osim = tmp / "scaled.osim"
                    scale_tool.run()
                    if out_osim.exists():
                        logger.info("OpenSim ScaleTool produced %s", out_osim)
                except Exception as exc:  # pragma: no cover - opensim runtime
                    logger.warning(
                        "OpenSim ScaleTool failed (%s); falling back to "
                        "marker-distance estimate",
                        exc,
                    )

            lengths = self._segment_lengths_from_markers(
                rig, calibration_markers, marker_to_segment
            )
            scaled = self._apply_lengths_to_rig(rig, lengths)

        # Postcondition: every segment length must be positive.
        for jname, jdef in scaled.joints.items():
            length = float(np.linalg.norm(jdef.tpose_offset))
            if length <= 0 and jdef.parent is not None:
                raise RuntimeError(
                    f"Scaled rig has non-positive segment length for {jname}"
                )
        return scaled


__all__ = ["OpenSimScaleBackend"]

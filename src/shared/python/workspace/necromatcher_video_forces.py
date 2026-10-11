"""Opt-in force/torque and segment-shading layer for Necromatcher video exports.

The layer is drawn through the shared FTO-8 renderer with the fit's own camera
hypothesis. Torques from a replayed or fitted motion are not measured historical
torques, and every legend carries that qualification. This module is SDK-free:
the MuJoCo provider is injected by the clean worker (``necromatcher_video_worker``).
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace
from fractions import Fraction
import math
from typing import Any, Protocol

import numpy as np

from src.shared.python.estimation import CubicHermiteSplineTrajectory
from src.shared.python.force_overlay import ForceGlyphStyle, build_glyphs
from src.shared.python.force_overlay.contracts import ForceTorqueFrame, WrenchKind
from src.shared.python.force_overlay.conversions import SegmentAxis
from src.shared.python.force_overlay.renderers.opencv_glyphs import (
    HypothesisProjector,
    draw_glyphs_on_frame,
)
from src.shared.python.force_overlay.renderers.opencv_segments import (
    draw_segment_meshes_on_frame,
    segment_poses_from_axes,
)

from .necromatcher_native import NativeFitBinding

QUALIFICATION = "research fit - unqualified camera and dynamics; not measured forces"
_SEGMENT_RADIUS_M = 0.04
_MIN_SEGMENT_M = 1e-6
_DEFAULT_KINDS = (WrenchKind.JOINT_REACTION.value,)


@dataclass(frozen=True)
class ForceLayer:
    """Validated, deterministic layer settings; disabled by default."""

    enabled: bool = False
    kinds: tuple[str, ...] = _DEFAULT_KINDS
    scale: float = 1.0
    segment_shading: bool = False

    def __post_init__(self) -> None:
        known = {kind.value for kind in WrenchKind}
        kinds = tuple(sorted(set(self.kinds)))
        if not kinds or any(kind not in known for kind in kinds):
            raise ValueError(f"Force layer kinds must be a non-empty subset of {known}")
        if (
            isinstance(self.scale, bool)
            or not isinstance(self.scale, (int, float))
            or not math.isfinite(self.scale)
            or self.scale <= 0
        ):
            raise ValueError("Force layer scale must be finite and positive")
        object.__setattr__(self, "kinds", kinds)
        object.__setattr__(self, "scale", float(self.scale))

    def to_dict(self) -> dict[str, Any]:
        return {
            "enabled": self.enabled,
            "kinds": list(self.kinds),
            "scale": self.scale,
            "segment_shading": self.segment_shading,
        }


class ForceSampler(Protocol):
    """Provider boundary: world-frame wrenches for one fitted state."""

    def sample(
        self, q: np.ndarray, v: np.ndarray, a: np.ndarray, time_s: float
    ) -> ForceTorqueFrame: ...


ForceSamplerFactory = Callable[[NativeFitBinding], ForceSampler]


def frame_time_s(frame: dict[str, Any]) -> float:
    """Source presentation time in seconds from exact PTS integers."""
    return float(
        Fraction(
            frame["pts_ticks"] * frame["timebase_numerator"],
            frame["timebase_denominator"],
        )
    )


def fit_derivatives(
    binding: NativeFitBinding,
) -> tuple[np.ndarray | None, np.ndarray | None, str | None]:
    """Analytic ``(v, a, reason)`` per exported frame from the preserved spline.

    Locked coordinates have zero rate and acceleration. Nothing is estimated by
    finite differences: a fit without a preserved spline is kinematic-only and
    returns ``(None, None, reason)``.
    """
    original = binding.fit["evidence"].get("original_fit", {})
    needed = ("free_coordinates", "knot_times", "spline_coefficients")
    if any(name not in original for name in needed):
        return None, None, "fit stores no spline derivatives (kinematic-only export)"
    order = list(binding.plant.coordinate_order)
    free = [order.index(name) for name in original["free_coordinates"]]
    knots = np.asarray(original["knot_times"], dtype=float)
    times = np.array([frame_time_s(f) for f in binding.fit["frames"]])
    if times[0] < knots[0] or times[-1] > knots[-1]:
        return None, None, "frame times lie outside the preserved spline interval"
    evaluation = CubicHermiteSplineTrajectory(knots, len(free)).evaluate(
        np.asarray(original["spline_coefficients"], dtype=float), times
    )
    velocity = np.zeros((len(times), len(order)))
    acceleration = np.zeros_like(velocity)
    velocity[:, free], acceleration[:, free] = evaluation.v, evaluation.a
    return velocity, acceleration, None


class _SegmentProjector(HypothesisProjector):
    """Expose camera-frame points so segment depth sorting uses the same camera."""

    @property
    def camera(self) -> _SegmentProjector:
        return self

    def camera_from_world(self, points: np.ndarray) -> np.ndarray:
        projection = self.projection
        return (
            np.asarray(points, float) @ projection.rotation.T + projection.translation
        )


class ForceLayerRenderer:
    """Per-export state: one projector, one provider, derivatives computed once."""

    def __init__(
        self,
        binding: NativeFitBinding,
        layer: ForceLayer,
        factory: ForceSamplerFactory,
        edges: Sequence[dict[str, str]],
    ) -> None:
        camera, _ = binding.review_inputs()
        self._projector = _SegmentProjector(camera)
        self._layer = layer
        self._edges = tuple(edges)
        self._velocity, self._acceleration, self.reason = fit_derivatives(binding)
        self._sampler = factory(binding) if self.reason is None else None
        base = ForceGlyphStyle()
        self._style = replace(
            base,
            kinds=frozenset(WrenchKind(kind) for kind in layer.kinds),
            force_scale_m_per_n=base.force_scale_m_per_n * layer.scale,
            torque_scale_m_per_nm=base.torque_scale_m_per_nm * layer.scale,
        )
        self._times = [frame_time_s(f) for f in binding.fit["frames"]]

    @property
    def available(self) -> bool:
        return self.reason is None

    def manifest(self) -> dict[str, Any]:
        return {
            "schema": "necromatcher/video-force-layer/1",
            "settings": self._layer.to_dict(),
            "available": self.available,
            "reason": self.reason,
            "qualification": QUALIFICATION,
            "derivative_source": "preserved cubic-Hermite spline, analytic v and a",
            "torque_semantics": "model-replay joint reactions; not measured forces",
        }

    def _shade(
        self, image: np.ndarray, origins: dict[str, np.ndarray]
    ) -> dict[str, int]:
        axes = [
            SegmentAxis(
                edge["b"],
                edge["b"],
                tuple(origins[edge["a"]]),
                tuple(origins[edge["b"]]),
            )
            for edge in self._edges
            if np.linalg.norm(origins[edge["a"]] - origins[edge["b"]]) > _MIN_SEGMENT_M
        ]
        shaded, receipt = draw_segment_meshes_on_frame(
            image, segment_poses_from_axes(axes, _SEGMENT_RADIUS_M), self._projector
        )
        image[:] = shaded
        return {
            "triangles_drawn": receipt.triangles_drawn,
            "triangles_culled": receipt.triangles_culled,
            "segments_rendered": receipt.segments_rendered,
            "segments_without_loads": receipt.segments_without_loads,
        }

    def draw(
        self,
        image: np.ndarray,
        position: int,
        pose: np.ndarray,
        origins: dict[str, np.ndarray],
    ) -> dict[str, Any]:
        """Draw onto ``image`` in place; return per-frame receipts for the manifest."""
        record: dict[str, Any] = {}
        if self._layer.segment_shading:
            record["segment_shading"] = self._shade(image, origins)
        if self._sampler is not None:
            if self._velocity is None or self._acceleration is None:
                raise ValueError("Available force layer requires fit derivatives")
            frame = self._sampler.sample(
                pose,
                self._velocity[position],
                self._acceleration[position],
                self._times[position],
            )
            receipt = draw_glyphs_on_frame(
                image,
                build_glyphs(frame, self._style),
                self._projector,
                inplace=True,
                qualification=QUALIFICATION,
            )
            record["force_glyphs"] = receipt.to_dict()
        return record

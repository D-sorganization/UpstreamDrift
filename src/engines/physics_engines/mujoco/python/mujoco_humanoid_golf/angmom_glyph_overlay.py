"""Headless bridge from the MuJoCo engine to angular-momentum glyphs (#15398)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from src.shared.python.body_part_viz.angular_momentum_glyphs import (
    AngularMomentumArrowStyle,
    build_angular_momentum_glyphs,
)
from src.shared.python.body_part_viz.mujoco_angular_momentum_anchors import (
    angular_momentum_anchors,
)
from src.shared.python.body_part_viz.mujoco_segment_angular_momentum import (
    MujocoSegmentAngularMomentumProvider,
)
from src.shared.python.force_overlay.glyphs import GlyphSet

if TYPE_CHECKING:
    import mujoco

    from src.shared.python.body_part_viz.segment_angular_momentum import (
        SegmentGroupCatalog,
    )


def build_angmom_overlay_glyphs(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    catalog: SegmentGroupCatalog,
    enabled: bool,
    style: AngularMomentumArrowStyle,
) -> GlyphSet | None:
    """Sample the provider once and return angular-momentum ``GlyphSet``."""
    if not enabled:
        return None
    time_s = float(data.time)
    provider = MujocoSegmentAngularMomentumProvider(
        model, data, catalog=catalog, time_s=time_s
    )
    frame = provider.get_segment_angular_momentum()
    if frame is None:
        return None
    anchors = angular_momentum_anchors(model, data, catalog)
    return build_angular_momentum_glyphs(
        frame,
        style=style,
        anchors_m=anchors,
        enabled=True,
        catalog=catalog,
    )

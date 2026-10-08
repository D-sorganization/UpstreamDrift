"""Club impact parameters relative to an explicit target line (GCV-15, #11721).

Engine-agnostic extractor producing launch-monitor-style parameters from a
world-frame clubhead series.  Every value is computed or explicitly
unavailable with a reason; nothing is invented.

Definitions (binding; right-handed golfer, left-handed mirrors the lateral
sign only)
---------------------------------------------------------------------------
Target axes: ``x_t`` = target direction (horizontal unit), ``z_t`` = up,
``y_t = z_t x x_t`` (left of target for a right-handed golfer).  ``v`` is the
face-centre velocity at the last pre-contact sample; ``n`` the face normal.
Subscript ``h`` is the projection onto the plane orthogonal to ``z_t``.

* Clubhead speed ``|v|`` (m/s and mph).
* Attack angle ``AoA = atan2(v.z_t, |v_h|)``; negative = descending.
* Club path ``= atan2(-s v.y_t, v.x_t)``, ``s = +1`` RH, ``-1`` LH;
  positive = in-to-out.
* Face angle: the same formula applied to ``n``; positive = open.
* Face-to-path = face angle - club path (wrapped to +/-180 deg).
* Dynamic loft = ``atan2(n.z_t, |n_h|)``.
* Spin loft = 3-D angle ``acos(v_hat . n_hat)``; the planar spin loft and the
  spin-axis tilt come from the Tools D-plane (model estimates).
* Swing direction and vertical swing-plane angle: least-squares plane through
  face-centre samples within +/-20 ms of impact; direction is the horizontal
  heading of the in-plane travel, the plane angle is its inclination from
  horizontal.
* Low point: minimum face-centre height from the top of the swing to the end
  of the series; reported as ``(p - ball) . x_t`` (positive = target side) and
  height above ``ground_height_m``.
* Impact state: last pre-contact sample (``contact_index - 1``), otherwise the
  explicit index or the peak-speed index of ``detect_impact_index``.
* Unobservable club axial rotation (MS-108): face-derived fields are
  unavailable with the recorded reason.

Default frame (ADR-0041): Z-up, golfer faces -X, target line -Y.  The frame is
recorded in every result.
"""

from __future__ import annotations

from .clubhead_series import ClubheadSeries
from .extract import BallObservation, ImpactParameters, extract_impact_parameters
from .target_frame import TargetFrame
from .tools_gateway import (
    ToolsDeliveryEstimate,
    ToolsDeliveryGateway,
    ToolsDeliveryUnavailableError,
    load_tools_delivery_gateway,
)

__all__ = [
    "ClubheadSeries",
    "ImpactParameters",
    "TargetFrame",
    "ToolsDeliveryEstimate",
    "ToolsDeliveryGateway",
    "ToolsDeliveryUnavailableError",
    "BallObservation",
    "extract_impact_parameters",
    "load_tools_delivery_gateway",
]

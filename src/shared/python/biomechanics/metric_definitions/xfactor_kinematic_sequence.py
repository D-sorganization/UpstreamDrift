"""Cheetham X-factor stretch and kinematic sequence definitions (#14426, GM-3).

References: Cheetham et al. (2001, 2008).
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.biomechanics.metric_definitions.schema import (
    DimensionKind,
    FilterDifferentiationDescriptor,
    GolfMetricDefinitionEntry,
    NormalisationKind,
    SourceKind,
)
from src.shared.python.core.contracts import ensure, require


class VelocityComponentKind(str, Enum):
    """Component of segment angular velocity evaluated for kinematic sequence."""

    swing_plane_normal = "swing_plane_normal"
    axial = "axial"


@dataclass(frozen=True)
class CheethamXFactorStretchResult:
    x_factor_at_top_deg: float
    max_downswing_x_factor_deg: float
    stretch_deg: float
    stretch_pct: float
    top_time_s: float | None = None
    max_downswing_time_s: float | None = None


@dataclass(frozen=True)
class KinematicSegmentPeak:
    segment: str
    peak_speed_deg_s: float
    peak_time_s: float
    peak_time_rel_impact_s: float


@dataclass(frozen=True)
class KinematicSequenceGain:
    proximal_segment: str
    distal_segment: str
    speed_gain_diff_deg_s: float
    speed_gain_ratio: float


@dataclass(frozen=True)
class CheethamKinematicSequenceResult:
    velocity_component: VelocityComponentKind
    filter_descriptor: FilterDifferentiationDescriptor
    peaks: dict[str, KinematicSegmentPeak]
    gains: list[KinematicSequenceGain]
    sequence_order: list[str]
    is_valid_sequence: bool
    impact_time_s: float


_BUTTERWORTH_12HZ = FilterDifferentiationDescriptor(
    lowpass_cutoff_hz=12.0,
    method="butterworth_4th_order_zero_lag",
    min_sampling_hz=60.0,
)

XFACTOR_STRETCH_CHEETHAM_ENTRY = GolfMetricDefinitionEntry(
    metric_id="x_factor.stretch.cheetham_2001",
    segments=("pelvis", "thorax"),
    landmark_proxy="joint_centres",
    reference_frame="pelvis_local",
    sign_convention="right_handed_positive_axial",
    units="deg",
    event_anchors=("top", "early_downswing", "impact"),
    cites=("Cheetham et al. (2001)", "Optimising Performance in Golf"),
    dimension=DimensionKind.three_d,
    cardan_order="ZXY",
    normalisation=NormalisationKind.none,
    source=SourceKind.measured,
    definition_version="1.0.0",
    filter_descriptor=_BUTTERWORTH_12HZ,
)

KINEMATIC_SEQUENCE_CHEETHAM_ENTRY = GolfMetricDefinitionEntry(
    metric_id="kinematic_sequence.cheetham_2008",
    segments=("pelvis", "thorax", "lead_arm", "club"),
    landmark_proxy="segment_principal_axes",
    reference_frame="swing_plane_normal_or_axial",
    sign_convention="proximal_to_distal_acceleration",
    units="deg/s",
    event_anchors=("downswing", "impact"),
    cites=("Cheetham et al. (2008)", "World Scientific Congress of Golf V"),
    dimension=DimensionKind.three_d,
    cardan_order="ZXY",
    normalisation=NormalisationKind.none,
    source=SourceKind.measured,
    definition_version="1.0.0",
    filter_descriptor=_BUTTERWORTH_12HZ,
)


def _matched_degree_series(
    values: NDArray[np.floating] | list[float],
    times_s: NDArray[np.floating] | list[float],
) -> tuple[NDArray[np.floating], NDArray[np.floating]]:
    xf = np.asarray(values, dtype=float).reshape(-1)
    ts = np.asarray(times_s, dtype=float).reshape(-1)
    require(
        xf.size == ts.size and xf.size >= 2,
        "arrays must match length >= 2",
        (xf.size, ts.size),
    )
    return xf, ts


def compute_cheetham_xfactor_stretch(
    x_factor_deg: NDArray[np.floating] | list[float],
    times_s: NDArray[np.floating] | list[float],
    top_idx: int,
    impact_idx: int | None = None,
) -> CheethamXFactorStretchResult:
    """Compute X-factor stretch per Cheetham et al. (2001)."""
    xf, ts = _matched_degree_series(x_factor_deg, times_s)
    require(0 <= top_idx < xf.size, "top_idx out of bounds", top_idx)
    if impact_idx is not None:
        require(
            top_idx <= impact_idx < xf.size,
            "impact_idx must be >= top_idx",
            (top_idx, impact_idx),
        )
        ds_xf, ds_ts = xf[top_idx : impact_idx + 1], ts[top_idx : impact_idx + 1]
    else:
        ds_xf, ds_ts = xf[top_idx:], ts[top_idx:]

    top_val, max_idx_rel = float(xf[top_idx]), int(np.argmax(ds_xf))
    max_val, max_time = float(ds_xf[max_idx_rel]), float(ds_ts[max_idx_rel])
    stretch_deg = max(0.0, max_val - top_val)
    stretch_pct = (stretch_deg / abs(top_val) * 100.0) if abs(top_val) > 1e-6 else 0.0

    ensure(stretch_deg >= 0.0, "stretch_deg non-negative", stretch_deg)
    return CheethamXFactorStretchResult(
        top_val, max_val, stretch_deg, stretch_pct, float(ts[top_idx]), max_time
    )


def compute_xfactor_velocity(
    x_factor_deg: NDArray[np.floating] | list[float],
    times_s: NDArray[np.floating] | list[float],
) -> tuple[NDArray[np.floating], float]:
    """Compute X-factor velocity (d/dt) and peak absolute rate."""
    xf, ts = _matched_degree_series(x_factor_deg, times_s)
    vel = np.gradient(xf, ts)
    return vel, float(np.max(np.abs(vel)))


def compute_cheetham_kinematic_sequence(
    segment_velocities: dict[str, Any],
    times_s: NDArray[np.floating] | list[float] | None,
    impact_time_s: float,
    *,
    velocity_component: VelocityComponentKind
    | str = VelocityComponentKind.swing_plane_normal,
    filter_descriptor: FilterDifferentiationDescriptor | None = None,
    expected_order: tuple[str, ...] = ("pelvis", "thorax", "lead_arm", "club"),
) -> CheethamKinematicSequenceResult:
    """Analyze kinematic sequence per Cheetham et al. (2008). Missing times raises."""
    if times_s is None:
        raise ValueError(
            "Missing timestamps: kinematic sequence requires explicit time array"
        )
    ts = np.asarray(times_s, dtype=float).reshape(-1)
    require(
        ts.size >= 2 and len(segment_velocities) > 0,
        "valid inputs required",
        (ts.size, len(segment_velocities)),
    )

    v_comp = VelocityComponentKind(velocity_component)
    f_desc = filter_descriptor or KINEMATIC_SEQUENCE_CHEETHAM_ENTRY.filter_descriptor

    peaks: dict[str, KinematicSegmentPeak] = {}
    for seg, data in segment_velocities.items():
        v = np.asarray(data, dtype=float).reshape(-1)
        require(v.size == ts.size, f"segment '{seg}' size mismatch", (v.size, ts.size))
        idx = int(np.argmax(np.abs(v)))
        peak_t = float(ts[idx])
        peaks[seg] = KinematicSegmentPeak(
            seg, float(np.abs(v[idx])), peak_t, float(peak_t - impact_time_s)
        )

    gains: list[KinematicSequenceGain] = []
    order_present = [s for s in expected_order if s in peaks]
    for i in range(len(order_present) - 1):
        p_name, d_name = order_present[i], order_present[i + 1]
        p_peak, d_peak = peaks[p_name], peaks[d_name]
        diff = float(d_peak.peak_speed_deg_s - p_peak.peak_speed_deg_s)
        ratio = (
            float(d_peak.peak_speed_deg_s / p_peak.peak_speed_deg_s)
            if p_peak.peak_speed_deg_s > 1e-6
            else 0.0
        )
        gains.append(KinematicSequenceGain(p_name, d_name, diff, ratio))

    actual_order = sorted(peaks.keys(), key=lambda s: peaks[s].peak_time_s)
    is_valid = (
        actual_order == order_present
        if len(actual_order) == len(order_present)
        else False
    )
    return CheethamKinematicSequenceResult(
        v_comp, f_desc, peaks, gains, actual_order, is_valid, float(impact_time_s)
    )

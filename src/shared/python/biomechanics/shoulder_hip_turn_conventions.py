"""Shoulder and hip (pelvis) turn conventions (GM-1c / #14424).

Splits address-relative from target-line/lab-frame shoulder and hip turn
conventions with declared segment proxies and Chu et al. (2010) event anchors.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Sequence

import numpy as np
from scipy.spatial.transform import Rotation

from src.shared.python.biomechanics.metric_definitions.schema import (
    DimensionKind,
    FilterDifferentiationDescriptor,
    GolfMetricDefinitionEntry,
    NormalisationKind,
    RegistryDocument,
    SourceKind,
)
from src.shared.python.contracts import ensure, require
from src.shared.python.swing_comparison.events import SwingEvents


class SegmentProxy(str, Enum):
    """Declared anatomical proxy used to materialize the segment line."""

    joint_centre_line = "joint_centre_line"
    thorax_pelvis_landmarks = "thorax_pelvis_landmarks"


class TurnReferenceKind(str, Enum):
    """Reference orientation against which turn is reported."""

    address_relative = "address_relative"
    target_line_lab_frame = "target_line_lab_frame"


TURN_CONVENTIONS_REGISTRY_VERSION = "1.0.0"
CHU_2010_EVENT_ANCHORS: tuple[str, ...] = ("address", "top", "impact", "finish")
_CITE_CHU2010 = "doi:10.1080/02640414.2010.507249"


def _build_registry() -> RegistryDocument:
    base = {
        "dimension": DimensionKind.three_d,
        "cardan_order": "ZXY",
        "sign_convention": "right_handed_positive_backswing",
        "units": "deg",
        "event_anchors": CHU_2010_EVENT_ANCHORS,
        "source": SourceKind.measured,
        "definition_version": TURN_CONVENTIONS_REGISTRY_VERSION,
        "cites": (_CITE_CHU2010,),
        "filter_descriptor": FilterDifferentiationDescriptor(
            method="none", min_sampling_hz=30.0
        ),
    }
    pairs = [
        (s, seg, r, nrm, rf, pk, prx)
        for s, seg in (("shoulder", "thorax"), ("hip", "pelvis"))
        for r, nrm, rf in (
            ("address_relative", NormalisationKind.address_relative, "address_frame"),
            ("target_line", NormalisationKind.none, "target_line_lab_frame"),
        )
        for pk, prx in (
            ("joint_centres", SegmentProxy.joint_centre_line.value),
            ("landmarks", SegmentProxy.thorax_pelvis_landmarks.value),
        )
    ]
    return {
        f"{s}_turn_{r}_{pk}": GolfMetricDefinitionEntry(
            metric_id=f"{s}_turn_{r}_{pk}",
            segments=(seg,),
            landmark_proxy=prx,
            reference_frame=rf,
            normalisation=nrm,
            **base,  # type: ignore[arg-type]
        )
        for s, seg, r, nrm, rf, pk, prx in pairs
    }


TURN_METRIC_DEFINITIONS: RegistryDocument = _build_registry()


def turn_metric_definition(metric_id: str) -> GolfMetricDefinitionEntry:
    """Retrieve entry for ``metric_id`` with DbC enforcement."""
    if metric_id not in TURN_METRIC_DEFINITIONS:
        raise ValueError(f"unknown turn metric_id: {metric_id!r}")
    return TURN_METRIC_DEFINITIONS[metric_id]


def wrap_angle_deg(angle_deg: float | np.ndarray) -> Any:
    """Wrap angles to [-180°, 180°)."""
    wrapped = (np.asarray(angle_deg, dtype=float) + 180.0) % 360.0 - 180.0
    return float(wrapped) if isinstance(angle_deg, (int, float, np.number)) else wrapped


def unwrap_turn_deg(angles_deg: np.ndarray | Sequence[float]) -> np.ndarray:
    """Unwrap turn series across ±180° preserving NaN frames."""
    arr = np.asarray(angles_deg, dtype=np.float64)
    require(arr.ndim == 1, "angles_deg must be 1D", arr.shape)
    if arr.size == 0:
        return np.empty(0, dtype=np.float64)
    out = np.full_like(arr, np.nan, dtype=np.float64)
    valid = np.isfinite(arr)
    if valid.any():
        out[valid] = np.unwrap(arr[valid], period=360.0)
    ensure(out.shape == arr.shape, "unwrap_turn_deg preserves input shape")
    return out


@dataclass(frozen=True)
class TurnSeriesResult:
    """Evaluated turn series with metadata and event values."""

    metric_id: str
    turn_deg: np.ndarray
    event_values_deg: dict[str, float] | None
    reference_kind: TurnReferenceKind
    proxy: SegmentProxy
    definition: GolfMetricDefinitionEntry


def _resolve_basis(
    up: Any, target: Any, target_line: Any
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    u_arr = np.asarray(up, dtype=float)
    t_raw = (
        target
        if target is not None
        else (target_line if target_line is not None else (0.0, 1.0, 0.0))
    )
    t_arr = np.asarray(t_raw, dtype=float)
    if u_arr.shape != (3,) or t_arr.shape != (3,):
        raise ValueError("up and target must have shape (3,)")
    u_hat = u_arr / float(np.linalg.norm(u_arr))
    t_proj = t_arr - np.dot(t_arr, u_hat) * u_hat
    t_norm = float(np.linalg.norm(t_proj))
    if t_norm <= 1e-6:
        raise ValueError("target and up vectors must not be parallel")
    t_hat = t_proj / t_norm
    return u_hat, t_hat, np.cross(u_hat, t_hat)


def _extract_events(
    turn_deg: np.ndarray, ev: SwingEvents | None
) -> dict[str, float] | None:
    if ev is None:
        return None
    res = {}
    for name, idx in (
        ("address", ev.address_idx),
        ("top", ev.top_idx),
        ("impact", ev.impact_idx),
        ("finish", ev.finish_idx),
    ):
        if not (0 <= idx < len(turn_deg)):
            raise ValueError(
                f"event index {idx} for {name} out of range [0, {len(turn_deg)})"
            )
        res[name] = float(turn_deg[idx])
    return res


def _make_result(
    metric_id: str, raw_yaw: np.ndarray, ev: SwingEvents | None
) -> TurnSeriesResult:
    defn = turn_metric_definition(metric_id)
    unwrapped = unwrap_turn_deg(raw_yaw)
    if defn.normalisation == NormalisationKind.address_relative:
        addr = ev.address_idx if ev is not None else 0
        if not (0 <= addr < len(unwrapped)):
            raise ValueError("address_idx out of range")
        turn_deg, ref = unwrapped - unwrapped[addr], TurnReferenceKind.address_relative
    else:
        turn_deg, ref = unwrapped, TurnReferenceKind.target_line_lab_frame
    proxy = (
        SegmentProxy.joint_centre_line
        if "joint_centres" in metric_id
        else SegmentProxy.thorax_pelvis_landmarks
    )
    return TurnSeriesResult(
        metric_id, turn_deg, _extract_events(turn_deg, ev), ref, proxy, defn
    )


def evaluate_turn_from_rotations(
    metric_id: str,
    rotations: Rotation | np.ndarray,
    *,
    events: SwingEvents | None = None,
    up: Any = (0.0, 0.0, 1.0),
    target: Any = None,
    target_line: Any = None,
) -> TurnSeriesResult:
    """Evaluate turn series from 3D rotations."""
    turn_metric_definition(metric_id)
    _, t_hat, n_hat = _resolve_basis(up, target, target_line)
    if isinstance(rotations, Rotation):
        r = rotations
    elif (
        isinstance(rotations, np.ndarray)
        and rotations.ndim == 3
        and rotations.shape[1:] == (3, 3)
    ):
        r = Rotation.from_matrix(rotations)
    else:
        raise ValueError(f"unsupported rotations: {type(rotations)}")
    dirs = r.apply(t_hat)
    yaw = np.degrees(np.arctan2(np.dot(dirs, n_hat), np.dot(dirs, t_hat)))
    return _make_result(metric_id, yaw, events)


def evaluate_turn_from_points(
    metric_id: str,
    left: np.ndarray,
    right: np.ndarray,
    *,
    events: SwingEvents | None = None,
    up: Any = (0.0, 0.0, 1.0),
    target: Any = None,
    target_line: Any = None,
) -> TurnSeriesResult:
    """Evaluate turn series from 3D points."""
    turn_metric_definition(metric_id)
    l_arr, r_arr = np.asarray(left, dtype=float), np.asarray(right, dtype=float)
    if l_arr.shape != r_arr.shape or l_arr.ndim != 2 or l_arr.shape[1] != 3:
        raise ValueError(f"mismatched or invalid shapes: {l_arr.shape}, {r_arr.shape}")
    _, t_hat, n_hat = _resolve_basis(up, target, target_line)
    vec = l_arr - r_arr
    valid = np.isfinite(vec).all(axis=1)
    raw_yaw: np.ndarray = np.full(len(vec), np.nan, dtype=np.float64)
    if valid.any():
        raw_yaw[valid] = np.degrees(
            np.arctan2(np.dot(vec[valid], n_hat), np.dot(vec[valid], t_hat))
        )
    return _make_result(metric_id, raw_yaw, events)


def evaluate_turn_series(metric_id: str, series: Any, **kw: Any) -> TurnSeriesResult:
    """Dispatch evaluation of turn series from rotations, points, or angle array."""
    if isinstance(series, Rotation) or (
        isinstance(series, np.ndarray) and series.ndim == 3
    ):
        return evaluate_turn_from_rotations(metric_id, series, **kw)
    if isinstance(series, (tuple, list)) and len(series) == 2:
        return evaluate_turn_from_points(metric_id, series[0], series[1], **kw)
    return _make_result(metric_id, np.asarray(series, dtype=float), kw.get("events"))


def turn_publish_payload(
    result_or_metric: TurnSeriesResult | str,
    values_deg: Any = None,
    *,
    events: SwingEvents | None = None,
    event_values_deg: dict[str, float] | None = None,
    registry_version: str | None = None,
) -> dict[str, Any]:
    """Build a JSON-serializable dictionary with mandatory registry labels."""
    if isinstance(result_or_metric, TurnSeriesResult):
        r = result_or_metric
        mid, defn, vals, ev = r.metric_id, r.definition, r.turn_deg, r.event_values_deg
    else:
        mid, defn = result_or_metric, turn_metric_definition(result_or_metric)
        require(values_deg is not None, "values_deg required")
        vals = np.asarray(values_deg, dtype=float)
        ev = (
            event_values_deg
            if event_values_deg is not None
            else _extract_events(vals, events)
        )
    return {
        "metric_id": mid,
        "registry_version": registry_version or TURN_CONVENTIONS_REGISTRY_VERSION,
        "reference_frame": defn.reference_frame,
        "normalisation": defn.normalisation.value,
        "landmark_proxy": defn.landmark_proxy,
        "segments": list(defn.segments),
        "cites": list(defn.cites),
        "units": defn.units,
        "values_deg": [float(v) if np.isfinite(v) else float("nan") for v in vals],
        "events_deg": ev,
    }


__all__ = [
    "CHU_2010_EVENT_ANCHORS",
    "SegmentProxy",
    "TURN_CONVENTIONS_REGISTRY_VERSION",
    "TURN_METRIC_DEFINITIONS",
    "TurnReferenceKind",
    "TurnSeriesResult",
    "evaluate_turn_from_points",
    "evaluate_turn_from_rotations",
    "evaluate_turn_series",
    "turn_metric_definition",
    "turn_publish_payload",
    "unwrap_turn_deg",
    "wrap_angle_deg",
]

# fmt: off
"""Deterministic frame-level replay harness for CoP and GRF chatter (#15776 WP-1).

Captures exact contact geometry, origins, horizontal couples, normal loads,
gate transitions, pressure centroid, and wrench CoP excursions outside foot hulls.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence
import numpy as np

__all__ = [
    "ContactPointRecord",
    "FrameReplayRecord",
    "ReplayHarness",
    "compute_pressure_centroid",
    "compute_wrench_cop",
    "detect_chatter",
    "point_in_polygon_2d",
    "signed_distance_to_polygon_2d",
]


@dataclass(frozen=True)
class ContactPointRecord:
    contact_id: str
    body_id: str
    foot: str
    point: tuple[float, float, float]
    force_n: tuple[float, float, float]
    torque_nm: tuple[float, float, float] = (0.0, 0.0, 0.0)


@dataclass
class FrameReplayRecord:
    frame_id: str
    time_s: float
    source_ref: str
    contacts: list[ContactPointRecord]
    raw_wrench_o: tuple[tuple[float, float, float], tuple[float, float, float]]
    wrench_cop: tuple[float, float, float] | None
    pressure_centroid: tuple[float, float, float] | None
    contact_state: str
    gate_alpha: float
    bos_valid: bool
    bos_margin_m: float
    is_excursion: bool


def point_in_polygon_2d(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> bool:
    """Ray-casting algorithm for 2D polygon containment."""
    x, y = point
    inside = False
    n = len(polygon)
    if n < 3:
        return False
    p1x, p1y = polygon[0]
    for i in range(1, n + 1):
        p2x, p2y = polygon[i % n]
        if y > min(p1y, p2y):
            if y <= max(p1y, p2y):
                if x <= max(p1x, p2x):
                    if p1y != p2y:
                        xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                    if p1x == p2x or x <= xinters:
                        inside = not inside
        p1x, p2x = p2x, p2x
        p1y, p2y = p2y, p2y
    return inside


def signed_distance_to_polygon_2d(point: tuple[float, float], polygon: Sequence[tuple[float, float]]) -> float:
    """Signed distance from point to polygon boundary (+ inside, - outside)."""
    if len(polygon) < 3:
        return 0.0
    px, py = point
    min_dist_sq = float("inf")
    n = len(polygon)
    for i in range(n):
        x1, y1 = polygon[i]
        x2, y2 = polygon[(i + 1) % n]
        dx = x2 - x1
        dy = y2 - y1
        line_len_sq = dx * dx + dy * dy
        if line_len_sq == 0.0:
            dist_sq = (px - x1) ** 2 + (py - y1) ** 2
        else:
            t = max(0.0, min(1.0, ((px - x1) * dx + (py - y1) * dy) / line_len_sq))
            proj_x = x1 + t * dx
            proj_y = y1 + t * dy
            dist_sq = (px - proj_x) ** 2 + (py - proj_y) ** 2
        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq

    dist = math.sqrt(min_dist_sq)
    inside = point_in_polygon_2d(point, polygon)
    return dist if inside else -dist


def compute_pressure_centroid(
    contacts: Sequence[ContactPointRecord], ground_z: float = 0.0
) -> tuple[float, float, float] | None:
    """Centroid of non-negative coplanar normal loads; guaranteed inside contact convex hull."""
    total_fz = 0.0
    weighted_x = 0.0
    weighted_y = 0.0
    for c in contacts:
        fz = max(0.0, float(c.force_n[2]))
        if fz > 0.0:
            total_fz += fz
            weighted_x += float(c.point[0]) * fz
            weighted_y += float(c.point[1]) * fz
    if total_fz <= 0.0:
        return None
    return (weighted_x / total_fz, weighted_y / total_fz, float(ground_z))


def compute_wrench_cop(
    contacts: Sequence[ContactPointRecord],
    origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
    ground_z: float = 0.0,
    min_fz: float = 5.0,
) -> tuple[tuple[float, float, float], tuple[float, float, float], tuple[float, float, float] | None]:
    """Wrench CoP with the moment about origin O transported to the ground plane first."""
    ox, oy, oz = origin
    total_f = np.zeros(3)
    total_m_o = np.zeros(3)
    for c in contacts:
        f = np.asarray(c.force_n, dtype=float)
        p = np.asarray(c.point, dtype=float)
        tau = np.asarray(c.torque_nm, dtype=float)
        total_f += f
        r = p - np.array([ox, oy, oz])
        total_m_o += np.cross(r, f) + tau

    fz = total_f[2]
    if fz < min_fz:
        cop = None
    else:
        # Moment transport O -> G, G = (ox, oy, ground_z). M_B = M_A + (A - B) x F, so with
        # O - G = (0, 0, k) and k = oz - ground_z: (0, 0, k) x (Fx, Fy, Fz) = (-k*Fy, k*Fx, 0),
        # giving M_G = (M_x - k*F_y, M_y + k*F_x, M_z). k = 0 (origin on the ground plane) is
        # the identity, and a vertical-only resultant (Fx = Fy = 0) needs no transport.
        k = oz - float(ground_z)
        mg_x = total_m_o[0] - k * total_f[1]
        mg_y = total_m_o[1] + k * total_f[0]
        # c = G + (n x M_G) / Fz where n = (0, 0, 1)
        # n x M_G = (-M_G_y, M_G_x, 0)
        cop = (
            float(ox - mg_y / fz),
            float(oy + mg_x / fz),
            float(ground_z),
        )
    return (
        (float(total_f[0]), float(total_f[1]), float(total_f[2])),
        (float(total_m_o[0]), float(total_m_o[1]), float(total_m_o[2])),
        cop,
    )


def _convex_hull_2d(points: list[tuple[float, float]]) -> list[tuple[float, float]]:
    """Andrew's monotone chain: returns CCW convex hull vertices (no duplicates)."""
    pts = sorted(set(points))
    if len(pts) < 3:
        return list(pts)

    def _cross(o: tuple[float, float], a: tuple[float, float], b: tuple[float, float]) -> float:
        """Return the 2D cross product of vectors (a - o) and (b - o)."""
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[tuple[float, float]] = []
    for p in pts:
        while len(lower) >= 2 and _cross(lower[-2], lower[-1], p) <= 0:
            lower.pop()
        lower.append(p)
    upper: list[tuple[float, float]] = []
    for p in reversed(pts):
        while len(upper) >= 2 and _cross(upper[-2], upper[-1], p) <= 0:
            upper.pop()
        upper.append(p)
    return lower[:-1] + upper[:-1]


def detect_chatter(
    transitions: Sequence[tuple[str, float, str]], min_oscillations: int = 2
) -> bool:
    """Detect rapid state oscillation across contact thresholds."""
    return len(transitions) >= min_oscillations


class ReplayHarness:
    """Deterministic replay harness capturing CoP excursions and GRF chatter windows."""

    def __init__(
        self,
        foot_hulls: Mapping[str, Sequence[tuple[float, float]]],
        engage_threshold_n: float = 20.0,
        release_threshold_n: float = 10.0,
        dwell_time_s: float = 0.015,
        ground_z: float = 0.0,
        initial_state: str = "stance",
    ) -> None:
        """Configure foot hulls, hysteresis thresholds, dwell timing, and ground plane."""
        self.foot_hulls = dict(foot_hulls)
        self.engage_threshold_n = engage_threshold_n
        self.release_threshold_n = release_threshold_n
        self.dwell_time_s = dwell_time_s
        self.ground_z = ground_z
        self.records: list[FrameReplayRecord] = []
        self._state = initial_state
        self._last_state_change_time = 0.0
        self._transitions: list[tuple[str, float, str]] = []
        self._pending_state: str | None = None
        self._pending_since: float = 0.0

    def record_frame(
        self,
        frame_id: str,
        time_s: float,
        source_ref: str,
        contacts: Sequence[ContactPointRecord],
    ) -> FrameReplayRecord:
        """Record one replay frame, advancing state, gate alpha, and BoS excursion flags."""
        net_f, net_m, cop_w = compute_wrench_cop(
            contacts, origin=(0.0, 0.0, self.ground_z), ground_z=self.ground_z, min_fz=self.release_threshold_n
        )
        p_centroid = compute_pressure_centroid(contacts, ground_z=self.ground_z)

        raw_fz = net_f[2]
        old_state = self._state
        # Determine candidate: None means hysteresis band (no zone change)
        if raw_fz >= self.engage_threshold_n:
            candidate: str | None = "stance"
        elif raw_fz < self.release_threshold_n:
            candidate = "flight"
        else:
            candidate = None

        # Option A dwell: hysteresis band leaves pending untouched; return to current zone clears it
        if candidate is not None and candidate != old_state:
            if self._pending_state != candidate:
                self._pending_state = candidate
                self._pending_since = time_s
            elif time_s - self._pending_since >= self.dwell_time_s:
                self._transitions.append((frame_id, time_s, f"{old_state}->{candidate}"))
                self._state = candidate
                self._last_state_change_time = time_s
                self._pending_state = None
        elif candidate == old_state:
            self._pending_state = None

        # Gate alpha calculation
        if raw_fz <= self.release_threshold_n:
            alpha = 0.0
        elif raw_fz >= self.engage_threshold_n:
            alpha = 1.0
        else:
            alpha = (raw_fz - self.release_threshold_n) / (self.engage_threshold_n - self.release_threshold_n)

        # Base of support check against combined convex hull of active foot polygons
        bos_valid = True
        bos_margin = 0.0
        is_excursion = False
        if cop_w is not None and self.foot_hulls:
            # Gate on aggregated per-foot normal load: a foot whose load is split across
            # contacts is loaded even when no single contact clears the threshold.
            fz_by_foot: dict[str, float] = {}
            for c in contacts:
                if c.foot in self.foot_hulls:
                    fz_by_foot[c.foot] = fz_by_foot.get(c.foot, 0.0) + float(c.force_n[2])
            active_feet = {
                foot for foot, foot_fz in fz_by_foot.items()
                if foot_fz >= self.release_threshold_n
            }
            if not active_feet:
                active_feet = set(self.foot_hulls.keys())
            hull_verts: list[tuple[float, float]] = []
            for foot in active_feet:
                hull_verts.extend(self.foot_hulls[foot])
            combined_hull = _convex_hull_2d(hull_verts)
            bos_margin = signed_distance_to_polygon_2d(cop_w[:2], combined_hull)
            if bos_margin < 0.0:
                bos_valid = False
                is_excursion = True

        rec = FrameReplayRecord(
            frame_id=frame_id,
            time_s=time_s,
            source_ref=source_ref,
            contacts=list(contacts),
            raw_wrench_o=(net_f, net_m),
            wrench_cop=cop_w,
            pressure_centroid=p_centroid,
            contact_state=self._state,
            gate_alpha=alpha,
            bos_valid=bos_valid,
            bos_margin_m=bos_margin,
            is_excursion=is_excursion,
        )
        self.records.append(rec)
        return rec

    def get_first_excursion(self) -> dict[str, Any] | None:
        """Return the first recorded CoP excursion outside the base of support, or None."""
        for r in self.records:
            if r.is_excursion:
                return {
                    "frame_id": r.frame_id,
                    "time_s": r.time_s,
                    "source_ref": r.source_ref,
                    "wrench_cop": r.wrench_cop,
                    "pressure_centroid": r.pressure_centroid,
                    "bos_valid": r.bos_valid,
                    "bos_margin_m": r.bos_margin_m,
                    "raw_wrench_o": r.raw_wrench_o,
                    "contacts": [
                        {
                            "id": c.contact_id,
                            "body": c.body_id,
                            "foot": c.foot,
                            "point": c.point,
                            "force": c.force_n,
                            "torque": c.torque_nm,
                        }
                        for c in r.contacts
                    ],
                }
        return None

    def get_first_chatter_window(self) -> dict[str, Any] | None:
        """Return the first chatter window spanning recorded state transitions, or None."""
        if len(self._transitions) >= 2:
            return {
                "start_frame": self._transitions[0][0],
                "end_frame": self._transitions[-1][0],
                "transitions": self._transitions,
            }
        return None
# fmt: on

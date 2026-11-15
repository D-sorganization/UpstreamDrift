"""Penetration and anthropometric metrics for hand visual comparison."""

from __future__ import annotations

import math

import numpy as np

from src.shared.python.model_appearance.geometry import Mesh


def grip_shaft_penetration_fraction(
    mesh: Mesh,
    *,
    grip_radius_m: float,
    tolerance_m: float,
) -> float:
    """Fraction of vertices inside the grip cylinder minus tolerance."""
    rho = np.hypot(mesh.vertices[:, 0], mesh.vertices[:, 1])
    limit = float(grip_radius_m) - float(tolerance_m)
    if limit <= 0.0:
        raise ValueError("grip_radius_m must exceed shaft_penetration_tolerance_m")
    inside = rho < limit
    return float(inside.sum()) / float(len(mesh.vertices))


def _pairwise_min_distances(
    samples_a: np.ndarray,
    samples_b: np.ndarray,
    chunk: int = 128,
) -> float:
    best = float("inf")
    for i in range(0, len(samples_a), chunk):
        block = samples_a[i : i + chunk]
        diff = block[:, None, :] - samples_b[None, :, :]
        dist = np.linalg.norm(diff, axis=2)
        best = min(best, float(dist.min()))
    return best


def inter_finger_penetration_fraction(
    components: list[tuple[str, np.ndarray]],
    *,
    tolerance_m: float,
    samples_per_finger: int = 24,
) -> float:
    """Share of cross-finger sample pairs closer than ``tolerance_m``."""
    if samples_per_finger < 4:
        raise ValueError("samples_per_finger must be >= 4")
    if len(components) < 2:
        return 0.0
    clouds: list[tuple[str, np.ndarray]] = []
    for name, verts in components:
        n = len(verts)
        if n <= samples_per_finger:
            pick = verts
        else:
            pick = verts[np.linspace(0, n - 1, samples_per_finger, dtype=int)]
        clouds.append((name, pick))
    violations = 0
    pairs = 0
    for i in range(len(clouds)):
        for j in range(i + 1, len(clouds)):
            dist = _pairwise_min_distances(clouds[i][1], clouds[j][1])
            pairs += 1
            if dist < float(tolerance_m):
                violations += 1
    return violations / float(pairs) if pairs else 0.0


def palm_arch_angle_deg(mesh: Mesh) -> float:
    """Angle between the palm plane and the shaft-normal plane (visual proxy)."""
    verts = mesh.vertices
    z_mid = float(np.median(verts[:, 2]))
    palm_band = verts[np.abs(verts[:, 2] - z_mid) < 0.015]
    if len(palm_band) < 12:
        palm_band = verts
    centroid = palm_band.mean(axis=0)
    centered = palm_band - centroid
    _, _, vh = np.linalg.svd(centered, full_matrices=False)
    normal = vh[2]
    if normal[2] < 0.0:
        normal = -normal
    shaft = np.array([0.0, 0.0, 1.0], dtype=float)
    cos_angle = float(np.clip(np.dot(normal, shaft), -1.0, 1.0))
    return math.degrees(math.acos(cos_angle))


def anthropometric_vs_reference(
    mesh: Mesh,
    *,
    reference: object,
    middle_finger_length_m: float | None = None,
) -> dict[str, float | bool]:
    """Compare measured mesh extents to :class:`HandAnthropometryReference`."""
    from src.shared.python.model_appearance.hand_backends.settings import (
        HandAnthropometryReference,
    )

    if not isinstance(reference, HandAnthropometryReference):
        raise TypeError("reference must be HandAnthropometryReference")
    measured = anthropometric_ratios(mesh)
    tol = reference.ratio_tolerance_rel
    checks: dict[str, float | bool] = {
        "reference_hand_length_m": reference.reference_hand_length_m,
        "reference_palm_width_m": reference.reference_palm_width_m,
        "reference_middle_finger_length_m": reference.middle_finger_length_m,
        "hand_length_within_tolerance": _within(
            measured["hand_length_m"], reference.reference_hand_length_m, tol
        ),
        "palm_width_within_tolerance": _within(
            measured["hand_breadth_m"], reference.reference_palm_width_m, tol
        ),
    }
    if middle_finger_length_m is not None:
        checks["middle_finger_length_m"] = middle_finger_length_m
        checks["middle_finger_within_tolerance"] = _within(
            middle_finger_length_m,
            reference.middle_finger_length_m,
            tol,
        )
    checks["anthropometrics_accepted"] = all(
        checks[k] for k in checks if k.endswith("_within_tolerance")
    )
    return checks


def _within(measured: float, target: float, rel_tol: float) -> bool:
    if target <= 0.0:
        return False
    return abs(measured - target) / target <= rel_tol


def anthropometric_ratios(mesh: Mesh) -> dict[str, float]:
    """Hand length/breadth and finger-length ratios from axis-aligned bounds."""
    verts = mesh.vertices
    length = float(verts[:, 2].max() - verts[:, 2].min())
    breadth = float(verts[:, 0].max() - verts[:, 0].min())
    depth = float(verts[:, 1].max() - verts[:, 1].min())
    if breadth < 1e-9 or length < 1e-9:
        raise ValueError("degenerate hand mesh for anthropometrics")
    z_bins = np.linspace(verts[:, 2].min(), verts[:, 2].max(), 5)
    segment_lengths = []
    for lo, hi in zip(z_bins[:-1], z_bins[1:], strict=True):
        mask = (verts[:, 2] >= lo) & (verts[:, 2] < hi)
        if mask.sum() < 8:
            continue
        segment_lengths.append(float(hi - lo))
    ratios = {
        "hand_length_m": length,
        "hand_breadth_m": breadth,
        "hand_depth_m": depth,
        "length_over_breadth": length / breadth,
    }
    if len(segment_lengths) >= 3:
        base = segment_lengths[0]
        for i, seg in enumerate(segment_lengths[1:4], start=1):
            if base > 1e-9:
                ratios[f"finger_segment_ratio_{i}"] = seg / base
    return ratios


__all__ = [
    "anthropometric_ratios",
    "anthropometric_vs_reference",
    "grip_shaft_penetration_fraction",
    "inter_finger_penetration_fraction",
    "palm_arch_angle_deg",
]

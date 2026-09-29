"""Marker-cluster C3D convention helpers (private).

Implements the validated schema for the cluster-marker mocap files in
``src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/Data/Mocap C3D
Files/`` (issue #013 follow-up).  The schema is:

* 28 anatomical Vicon Plug-in-Gait markers (skin-mounted body markers).
* Two 3-marker rigid clusters on the club:

  - ``Marker_2:2:{1,2,3}`` -- clubhead cluster.
  - ``Marker_3:3:{1,2,3}`` -- grip / butt cluster.

* ``Marker_0:0:0`` is a stuck-value sentinel and is excluded.
* ``RShoulderTop`` is occluded across the driver trace and is excluded.
* Source units are metres; vertical axis is ``+Y`` (Vicon convention).
* Cluster markers contain short NaN gaps (<= 5 frames typical) which are
  spline-filled before differentiation.

The clubhead and grip positions are computed as cluster centroids; the
rigid-body pose is solved via a Procrustes / Kabsch fit against the
frame-0 (address) cluster geometry.  Output frames are converted Y-up ->
Z-up by the right-handed swap ``(x, y, z) -> (x, -z, y)`` so the results
land in the Simscape Z-up world frame.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Marker schema constants
# ---------------------------------------------------------------------------

CLUBHEAD_CLUSTER: tuple[str, ...] = (
    "Marker_2:2:1",
    "Marker_2:2:2",
    "Marker_2:2:3",
)
"""Three-marker rigid cluster on the clubhead."""

GRIP_CLUSTER: tuple[str, ...] = (
    "Marker_3:3:1",
    "Marker_3:3:2",
    "Marker_3:3:3",
)
"""Three-marker rigid cluster on the grip / butt end."""

SENTINEL_MARKERS: tuple[str, ...] = ("Marker_0:0:0",)
"""Markers known to carry stuck sentinel values; ignore."""

OCCLUDED_MARKERS: tuple[str, ...] = ("RShoulderTop",)
"""Markers known-occluded across the validated traces."""

EXCLUDED_MARKERS: frozenset[str] = frozenset(SENTINEL_MARKERS + OCCLUDED_MARKERS)
"""Union of markers that must never be used by downstream consumers."""

MAX_GAP_FRAMES: int = 5
"""Maximum NaN gap length (frames) eligible for spline interpolation."""


@dataclass(frozen=True)
class ClusterClubPose:
    """Per-frame club kinematics derived from the rigid marker clusters.

    Attributes:
        clubhead: ``(N, 3)`` centroid of the clubhead cluster, Z-up metres.
        butt:     ``(N, 3)`` centroid of the grip cluster, Z-up metres.
        rotation: ``(N, 3, 3)`` rigid-body rotation w.r.t. the frame-0
                  (address) cluster reference; ``det(R) == +1``.
    """

    clubhead: np.ndarray
    butt: np.ndarray
    rotation: np.ndarray


# ---------------------------------------------------------------------------
# Schema detection
# ---------------------------------------------------------------------------


def has_marker_clusters(filename: str | None, marker_labels: list[str]) -> bool:
    """True if the C3D file follows the cluster-marker convention.

    A file is considered cluster-formatted if **either**:

    * the bare filename starts with ``"C3DExport"`` (case-insensitive), or
    * the marker list contains the canonical clubhead-cluster name
      ``Marker_2:2:1``.
    """
    if filename and str(filename).lower().startswith("c3dexport"):
        return True
    return CLUBHEAD_CLUSTER[0] in set(marker_labels)


# ---------------------------------------------------------------------------
# Coordinate-frame conversion
# ---------------------------------------------------------------------------


def y_up_to_z_up(points: np.ndarray) -> np.ndarray:
    """Convert a Y-up trajectory to a right-handed Z-up trajectory.

    Mapping ``(x, y, z) -> (x, -z, y)``.  The corresponding rotation
    matrix is

    .. code-block:: text

        R = [[1, 0, 0],
             [0, 0, -1],
             [0, 1, 0]]

    which has ``det(R) == +1``, so the handedness is preserved.

    Args:
        points: ``(N, 3)`` or ``(N, M, 3)`` array of positions.

    Returns:
        Array with the same shape, in Z-up coordinates.
    """
    if points.ndim < 2 or points.shape[-1] != 3:
        raise ValueError(f"Expected last axis of length 3, got {points.shape}")
    out = np.empty_like(points, dtype=np.float64)
    out[..., 0] = points[..., 0]
    out[..., 1] = -points[..., 2]
    out[..., 2] = points[..., 1]
    return out


def y_up_to_z_up_rotation() -> np.ndarray:
    """Return the 3x3 rotation matrix used by :func:`y_up_to_z_up`."""
    return np.array(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 1.0, 0.0],
        ],
        dtype=np.float64,
    )


# ---------------------------------------------------------------------------
# Gap interpolation
# ---------------------------------------------------------------------------


def fill_short_gaps(arr: np.ndarray, max_gap: int = MAX_GAP_FRAMES) -> np.ndarray:
    """Spline-fill NaN gaps of length ``<= max_gap`` along axis 0.

    Longer gaps and leading / trailing NaNs are left untouched (the caller
    decides how to crop).  Each column is interpolated independently with
    a natural cubic spline.
    """
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError(f"fill_short_gaps expects (N, 3), got {arr.shape}")
    if max_gap < 1:
        raise ValueError("max_gap must be >= 1")
    out = arr.astype(np.float64, copy=True)
    for col in range(3):
        _fill_column_gaps(out[:, col], max_gap)
    return out


def _iter_internal_gaps(c: np.ndarray):
    """Yield ``(start, end)`` half-open ranges of internal NaN runs."""
    n = c.shape[0]
    i = 0
    while i < n:
        if not np.isnan(c[i]):
            i += 1
            continue
        start = i
        while i < n and np.isnan(c[i]):
            i += 1
        if start > 0 and i < n:
            yield start, i


def _fill_column_gaps(c: np.ndarray, max_gap: int) -> None:
    """In-place spline-fill of internal NaN gaps in a 1-D column."""
    valid = ~np.isnan(c)
    if valid.sum() < 4:
        return
    for start, end in _iter_internal_gaps(c):
        if (end - start) > max_gap:
            continue
        left_idx = np.where(valid[:start])[0]
        right_idx = np.where(valid[end:])[0] + end
        anchors = np.concatenate(
            [left_idx[-3:] if len(left_idx) >= 3 else left_idx, right_idx[:3]]
        )
        if anchors.size < 2:
            continue
        order = 3 if anchors.size >= 4 else anchors.size - 1
        poly = np.polyfit(anchors.astype(np.float64), c[anchors], deg=order)
        c[start:end] = np.polyval(poly, np.arange(start, end, dtype=np.float64))


# ---------------------------------------------------------------------------
# Cluster pose: Procrustes / Kabsch
# ---------------------------------------------------------------------------


def pose_from_cluster(
    cluster_t: np.ndarray, reference: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Solve rigid-body pose at each frame against a reference cluster.

    For every frame ``t``, find ``R_t`` (proper rotation, ``det == +1``)
    and ``c_t`` (centroid) such that

    .. math::

        cluster\\_t[t] \\approx R_t \\, (reference - mean(reference)) + c_t

    Uses the SVD-based Kabsch algorithm with a reflection guard.

    Args:
        cluster_t: ``(N, 3, 3)`` cluster positions per frame
                   ``[frame, marker, xyz]``.  NaN-rows propagate.
        reference: ``(3, 3)`` cluster geometry at the reference frame.

    Returns:
        Tuple ``(rotations, centroids)``:

        * rotations: ``(N, 3, 3)`` proper rotations.
        * centroids: ``(N, 3)`` cluster centroids.

        Frames in which any reference / current marker is non-finite
        produce ``NaN``-filled rows.
    """
    if cluster_t.ndim != 3 or cluster_t.shape[1:] != (3, 3):
        raise ValueError(f"cluster_t must be (N, 3, 3), got {cluster_t.shape}")
    if reference.shape != (3, 3):
        raise ValueError(f"reference must be (3, 3), got {reference.shape}")

    n = cluster_t.shape[0]
    rotations = np.full((n, 3, 3), np.nan, dtype=np.float64)
    centroids = np.full((n, 3), np.nan, dtype=np.float64)

    ref_centroid = reference.mean(axis=0)
    ref_centered = reference - ref_centroid
    if not np.all(np.isfinite(ref_centered)):
        raise ValueError("reference cluster must be all-finite")

    for i in range(n):
        frame = cluster_t[i]
        if not np.all(np.isfinite(frame)):
            continue
        c = frame.mean(axis=0)
        cur = frame - c
        h = ref_centered.T @ cur
        u, _s, vt = np.linalg.svd(h)
        d = np.sign(np.linalg.det(vt.T @ u.T))
        if d == 0:
            d = 1.0
        s_diag = np.diag([1.0, 1.0, d])
        r = vt.T @ s_diag @ u.T
        rotations[i] = r
        centroids[i] = c
    return rotations, centroids


# ---------------------------------------------------------------------------
# Top-level cluster-pose extraction
# ---------------------------------------------------------------------------


def extract_cluster_club_pose(
    points: dict[str, np.ndarray],
    *,
    address_frame: int = 0,
    convert_to_z_up: bool = True,
) -> ClusterClubPose:
    """Extract clubhead/butt centroids + rigid-body pose from cluster markers.

    Args:
        points: mapping ``label -> (N, 3) array`` of marker trajectories
                in source coordinates (metres, Y-up).  Sentinel and
                occluded markers are silently ignored if present.
        address_frame: frame index used as the rigid-body reference.
        convert_to_z_up: if True (default), apply the Y-up -> Z-up swap.

    Returns:
        :class:`ClusterClubPose` with ``clubhead`` and ``butt`` centroids
        plus the per-frame rigid rotation of the clubhead cluster.

    Raises:
        ValueError: if any required cluster marker is absent or the
                    address frame is non-finite.
    """
    missing = [
        m
        for m in (*CLUBHEAD_CLUSTER, *GRIP_CLUSTER)
        if m not in points or m in EXCLUDED_MARKERS
    ]
    if missing:
        raise ValueError(f"Missing required cluster markers: {missing}")

    head_arr = np.stack([fill_short_gaps(points[m]) for m in CLUBHEAD_CLUSTER], axis=1)
    butt_arr = np.stack([fill_short_gaps(points[m]) for m in GRIP_CLUSTER], axis=1)

    if address_frame < 0 or address_frame >= head_arr.shape[0]:
        raise ValueError(f"address_frame {address_frame} out of range")
    head_ref = head_arr[address_frame]
    if not np.all(np.isfinite(head_ref)):
        raise ValueError(f"clubhead cluster at address_frame={address_frame} has NaNs")

    rotations, head_centroids = pose_from_cluster(head_arr, head_ref)
    _butt_rot, butt_centroids = pose_from_cluster(butt_arr, butt_arr[address_frame])

    if convert_to_z_up:
        r_swap = y_up_to_z_up_rotation()
        head_centroids = y_up_to_z_up(head_centroids)
        butt_centroids = y_up_to_z_up(butt_centroids)
        # Apply the same world-frame change to the rotations: R' = R_swap @ R.
        for i in range(rotations.shape[0]):
            if np.all(np.isfinite(rotations[i])):
                rotations[i] = r_swap @ rotations[i]

    return ClusterClubPose(
        clubhead=head_centroids,
        butt=butt_centroids,
        rotation=rotations,
    )

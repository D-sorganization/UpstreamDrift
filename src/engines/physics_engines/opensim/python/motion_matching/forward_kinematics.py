"""Forward-kinematics extraction for the OpenSim golf humanoid.

This module provides the boundary between OpenSim's internal ``SimTK::State``
and the canonical cross-engine ``SimOut`` schema. It pulls grip, clubhead,
and other anatomical landmark poses out of a realised state in the
canonical world frame.

Canonical conventions (per
``src/engines/CROSS_ENGINE_PARITY_SPEC.md`` and
``OPENSIM_PARITY_SPEC.md`` §3.3):

* Position: 3-vector ``[x, y, z]`` in metres, world frame.
* Quaternion: 4-vector ``[w, x, y, z]``, unit norm.

OpenSim itself returns rotation matrices and Eigen-style quaternions
``[x, y, z, w]``. All conversion to canonical form happens here so callers
never see the OpenSim convention.

The MVP model ships two landmark frames as ``PhysicalOffsetFrame`` objects:

* ``/bodyset/Club/club_grip_offset`` -> grip
* ``/bodyset/Club/club_head_offset`` -> clubhead

If a future model adds extra frames (e.g. butt, mid-hands, lead-shoulder)
they can be added to :data:`CANONICAL_LANDMARKS` and will surface in
:func:`extract_full_pose` automatically.

This module is the OpenSim-side counterpart of the Simscape
``compute_skeleton_fk.m`` helper. The output schema matches the canonical
``SimOut`` fields read by ``src/shared/python/motion_matching/cost.py``.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

from src.engines.physics_engines.opensim.python.opensim_golf.core import (
    OpenSimNotInstalledError,
)

if TYPE_CHECKING:  # pragma: no cover - typing only
    pass


# --- Canonical landmark catalogue ----------------------------------------
#
# Maps a canonical landmark name to the OpenSim frame component path. The
# frames live on the ``Club`` body (rigidly welded to the right hand in the
# MVP model). If a downstream model needs more landmarks, extend this dict;
# every consumer of :func:`extract_full_pose` will pick them up.

CANONICAL_LANDMARKS: dict[str, str] = {
    "grip": "/bodyset/Club/club_grip_offset",
    "clubhead": "/bodyset/Club/club_head_offset",
}


# --- Quaternion conversion -----------------------------------------------


def _rotation_matrix_to_quat_wxyz(
    rot: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a unit quaternion ``[w, x, y, z]``.

    Uses Shepperd's numerically-stable branch selection: pick the largest
    diagonal-related quantity to avoid divide-by-near-zero. Pure-Python /
    NumPy so it can be unit tested without OpenSim installed.

    Args:
        rot: ``(3, 3)`` rotation matrix. Must be orthonormal.

    Returns:
        ``(4,)`` unit quaternion in canonical ``[w, x, y, z]`` order with
        ``w >= 0`` (sign-canonicalised so the scalar component is
        non-negative — there are two equivalent quaternions for every
        rotation).

    Raises:
        ValueError: If ``rot`` is not a ``(3, 3)`` array.
    """
    rot = np.asarray(rot, dtype=np.float64)
    if rot.shape != (3, 3):
        raise ValueError(f"rot must be (3, 3); got {rot.shape}")

    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0.0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (rot[2, 1] - rot[1, 2]) * s
        y = (rot[0, 2] - rot[2, 0]) * s
        z = (rot[1, 0] - rot[0, 1]) * s
    elif rot[0, 0] > rot[1, 1] and rot[0, 0] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2])
        w = (rot[2, 1] - rot[1, 2]) / s
        x = 0.25 * s
        y = (rot[0, 1] + rot[1, 0]) / s
        z = (rot[0, 2] + rot[2, 0]) / s
    elif rot[1, 1] > rot[2, 2]:
        s = 2.0 * np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2])
        w = (rot[0, 2] - rot[2, 0]) / s
        x = (rot[0, 1] + rot[1, 0]) / s
        y = 0.25 * s
        z = (rot[1, 2] + rot[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1])
        w = (rot[1, 0] - rot[0, 1]) / s
        x = (rot[0, 2] + rot[2, 0]) / s
        y = (rot[1, 2] + rot[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    # Sign-canonicalise so w >= 0 (q and -q represent the same rotation).
    if quat[0] < 0.0:
        quat = -quat
    # Renormalise to absorb numerical drift.
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("Degenerate rotation produced zero-norm quaternion")
    return quat / norm


# --- OpenSim helpers (LOD <= 2) ------------------------------------------


def _ensure_position_realised(model: Any, state: Any) -> None:
    """Realise the model to Position stage so frame transforms are valid.

    OpenSim raises if you query a frame's location-in-ground when the
    state has not been realised at least to ``Position``. This helper is
    idempotent (re-realising a realised state is cheap) and isolates the
    SWIG call from callers.
    """
    try:
        model.realizePosition(state)
    except Exception as exc:  # noqa: BLE001 — surface OpenSim errors uniformly
        raise RuntimeError(
            f"OpenSim realizePosition failed: {exc}. "
            "Verify the state was produced by this model's initSystem()."
        ) from exc


def _frame_pose_in_ground(
    model: Any, state: Any, frame_path: str
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(pos_xyz, quat_wxyz)`` for a named frame in world frame.

    Resolves ``frame_path`` via ``model.getComponent``, queries
    ``getTransformInGround(state)``, and converts to canonical NumPy types.

    Args:
        model: An ``opensim.Model`` whose system has been initialised.
        state: A realised ``simbody::State`` (Position stage or higher).
        frame_path: Component path, e.g. ``"/bodyset/Club/club_grip_offset"``.

    Returns:
        ``(pos, quat)`` where ``pos`` is ``(3,) float64`` in metres and
        ``quat`` is ``(4,) float64`` ``[w, x, y, z]`` unit quaternion.
    """
    try:
        frame = model.getComponent(frame_path)
    except Exception as exc:  # noqa: BLE001 — OpenSim raises generic errors
        raise KeyError(
            f"Frame {frame_path!r} not found in model. "
            f"Check the model exposes this PhysicalOffsetFrame."
        ) from exc

    # getTransformInGround returns a SimTK::Transform with .R() (Rotation)
    # and .p() (Vec3 translation).
    transform = frame.getTransformInGround(state)
    p = transform.p()
    pos = np.array([p.get(0), p.get(1), p.get(2)], dtype=np.float64)

    rotation = transform.R()
    rot_mat = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rot_mat[i, j] = rotation.get(i, j)
    quat = _rotation_matrix_to_quat_wxyz(rot_mat)
    return pos, quat


def _require_opensim() -> None:
    """Raise :class:`OpenSimNotInstalledError` if the SWIG bindings are missing.

    All public extraction functions are usable only inside the OpenSim
    Python environment because they consume a live ``simbody::State``. This
    helper runs the import probe so the error message is consistent.
    """
    try:
        import opensim  # type: ignore[import-untyped]  # noqa: F401
    except ImportError as exc:  # pragma: no cover - exercised in env without opensim
        raise OpenSimNotInstalledError(
            "OpenSim Python bindings are required for forward-kinematics "
            "extraction. Install via 'conda install -c opensim-org opensim' "
            "or 'pip install opensim'."
        ) from exc


# --- Public API -----------------------------------------------------------


def extract_grip_pose(
    state: Any, model: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(grip_position_xyz, grip_quaternion_wxyz)`` in world frame.

    Args:
        state: A ``simbody::State`` produced by ``model.initSystem()`` (or
            equivalent) that has been realised to at least Position stage.
            This helper will realise it if needed.
        model: The ``opensim.Model`` whose ``Club`` body carries the
            ``club_grip_offset`` frame.

    Returns:
        ``(pos, quat)`` with ``pos`` shape ``(3,)`` in metres and ``quat``
        shape ``(4,)`` in canonical ``[w, x, y, z]`` order.

    Raises:
        OpenSimNotInstalledError: If the OpenSim Python bindings are missing.
        KeyError: If the model does not expose ``club_grip_offset``.
        ValueError: If preconditions fail.
    """
    if state is None or model is None:
        raise ValueError("state and model are both required")
    _require_opensim()
    _ensure_position_realised(model, state)
    return _frame_pose_in_ground(model, state, CANONICAL_LANDMARKS["grip"])


def extract_clubhead_pose(
    state: Any, model: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(clubhead_position_xyz, clubhead_quaternion_wxyz)``.

    Args:
        state: Realised ``simbody::State``.
        model: ``opensim.Model`` carrying the ``club_head_offset`` frame.

    Returns:
        ``(pos, quat)`` — see :func:`extract_grip_pose`.

    Raises:
        OpenSimNotInstalledError: If the OpenSim Python bindings are missing.
        KeyError: If the model does not expose ``club_head_offset``.
        ValueError: If preconditions fail.
    """
    if state is None or model is None:
        raise ValueError("state and model are both required")
    _require_opensim()
    _ensure_position_realised(model, state)
    return _frame_pose_in_ground(model, state, CANONICAL_LANDMARKS["clubhead"])


def extract_full_pose(state: Any, model: Any) -> dict[str, NDArray[np.float64]]:
    """Extract every canonical landmark in one realisation pass.

    Returns a dict shaped to match the cross-engine ``SimOut`` (per-step)
    schema, with two flat keys per landmark:

    * ``"<landmark>_pos"`` -> ``(3,) float64`` position in world frame
    * ``"<landmark>_quat"`` -> ``(4,) float64`` quaternion ``[w, x, y, z]``

    For the MVP model that yields:

    * ``grip_pos``, ``grip_quat``
    * ``clubhead_pos``, ``clubhead_quat``

    The single-pass form avoids redundant ``realizePosition`` calls when
    populating a per-step ``SimOut`` row inside an integration loop.

    Args:
        state: Realised ``simbody::State``.
        model: An ``opensim.Model`` exposing the canonical landmark frames.

    Returns:
        Dict from canonical key to NumPy array.

    Raises:
        OpenSimNotInstalledError: If the OpenSim Python bindings are missing.
        KeyError: If a canonical landmark frame is missing from the model.
        ValueError: If preconditions fail.
    """
    if state is None or model is None:
        raise ValueError("state and model are both required")
    _require_opensim()
    _ensure_position_realised(model, state)

    out: dict[str, NDArray[np.float64]] = {}
    for landmark, path in CANONICAL_LANDMARKS.items():
        pos, quat = _frame_pose_in_ground(model, state, path)
        out[f"{landmark}_pos"] = pos
        out[f"{landmark}_quat"] = quat
    return out

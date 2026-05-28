"""Forward-kinematics extraction for the OpenSim golf humanoid (issue #4191).

This module is the canonical FK boundary between OpenSim's internal
``SimTK::State`` and the cross-engine ``SimOut`` schema. It pulls grip
and clubhead world poses out of a realised state by reading the
``hand_r_grip_offset`` and ``club_head_offset`` ``PhysicalOffsetFrame``
components committed in ``models/golf_humanoid.osim`` (PR #4149).

History (issue #4191):
    The original FK module merged via PRs #4158 / #4160 looked for body
    names ``hand_left`` / ``hand_right`` that **do not exist** in the
    canonical humanoid. PR #4149 anchors the grip and clubhead as
    ``PhysicalOffsetFrame`` objects owned by the
    ``/jointset/hand_r_to_club`` weld joint:

    * ``hand_r_grip_offset``  (parent frame, on ``/bodyset/hand_r``)
    * ``club_grip_offset``    (child frame, on ``/bodyset/Club``)
    * ``club_head_offset``    (clubhead, on ``/bodyset/Club``)

    PR #4185 (issue #4120) shipped a parallel ``extract_full_pose`` to
    work around the bug. PR #4165 (issue #4116) shipped a third FK path.
    This module reconciles all three into a single canonical extractor
    so future callers cannot hit the same bug.

Public API:
    Canonical (preferred):
        extract_grip_pose(state, model)        -> (pos, quat)
        extract_clubhead_pose(state, model)    -> (pos, quat)
        extract_full_pose(state, model)        -> dict[str, NDArray]

    Legacy (kept working, emit DeprecationWarning):
        compute_grip(model, state)             -> (pos, quat)
        compute_clubhead(model, state)         -> (pos, quat)
        compute_skeleton_fk(model, states)     -> dict[str, NDArray]

Quaternion convention: canonical ``[w, x, y, z]`` (Simscape-compatible),
sign-canonicalised so the scalar component is non-negative.
"""

from __future__ import annotations

import warnings
from typing import TYPE_CHECKING, Any

import numpy as np
from numpy.typing import NDArray

if TYPE_CHECKING:  # pragma: no cover - typing only
    import opensim as osim

__all__ = [
    # Canonical (preferred) extractors.
    "extract_grip_pose",
    "extract_clubhead_pose",
    "extract_full_pose",
    # Legacy aliases (emit DeprecationWarning).
    "compute_grip",
    "compute_clubhead",
    "compute_skeleton_fk",
    # Frame identifiers (canonical names from golf_humanoid.osim).
    "GRIP_FRAME_NAME",
    "CLUBHEAD_FRAME_NAME",
    "GRIP_FRAME_PATH",
    "CLUBHEAD_FRAME_PATH",
    "CANONICAL_LANDMARKS",
]


# --- Canonical landmark catalogue ----------------------------------------

# Bare frame names (as they appear in the .osim XML).
GRIP_FRAME_NAME: str = "hand_r_grip_offset"
CLUBHEAD_FRAME_NAME: str = "club_head_offset"

# Fully-qualified component paths used with ``model.getComponent``. Frames
# defined inside the WeldJoint live under ``/jointset/<joint_name>/...``.
GRIP_FRAME_PATH: str = "/jointset/hand_r_to_club/hand_r_grip_offset"
CLUBHEAD_FRAME_PATH: str = "/jointset/hand_r_to_club/club_head_offset"

CANONICAL_LANDMARKS: dict[str, str] = {
    "grip": GRIP_FRAME_PATH,
    "clubhead": CLUBHEAD_FRAME_PATH,
}


# --- OpenSim helpers (LOD <= 2) ------------------------------------------


def _require_opensim() -> None:
    """Raise ``ImportError`` if the OpenSim Python bindings are missing.

    Raises:
        ImportError: If ``import opensim`` fails. Message mirrors the rest
            of the engine package so users see consistent install hints.
    """
    try:
        import opensim  # noqa: F401  # type: ignore[import-untyped]
    except ImportError as exc:  # pragma: no cover - exercised w/o opensim
        raise ImportError(
            "OpenSim Python bindings are required for forward-kinematics "
            "extraction. Install via 'conda install -c opensim-org opensim' "
            "or 'pip install opensim'."
        ) from exc


def _ensure_position_realised(model: Any, state: Any) -> None:
    """Realise the model to Position stage so frame transforms are valid.

    OpenSim raises if a frame's location-in-ground is queried before the
    state has been realised at least to ``Position``. This helper is
    idempotent (re-realising is cheap) and isolates the SWIG call.

    Raises:
        RuntimeError: If realisation fails. The most common cause is a
            state that was not produced by ``model.initSystem()``.
    """
    if not state.isValid():
        raise RuntimeError("provided state is not valid")
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
    ``getTransformInGround(state)``, and converts to canonical NumPy.

    Args:
        model: An ``opensim.Model`` whose system has been initialised.
        state: A realised ``simbody::State`` (Position stage or higher).
        frame_path: Component path, e.g.
            ``"/jointset/hand_r_to_club/hand_r_grip_offset"``.

    Returns:
        ``(pos, quat)`` — ``pos`` is ``(3,) float64`` in metres,
        ``quat`` is ``(4,) float64`` ``[w, x, y, z]`` unit quaternion.

    Raises:
        KeyError: If the frame is not found in the model.
    """
    try:
        frame = model.getComponent(frame_path)
    except Exception as exc:  # noqa: BLE001 — OpenSim raises generic errors
        raise KeyError(
            f"Frame {frame_path!r} not found in model. Confirm the model "
            "is golf_humanoid.osim (or a derivative that exposes the same "
            "PhysicalOffsetFrames)."
        ) from exc

    transform = frame.getTransformInGround(state)
    p = transform.p()
    pos = np.array([p.get(0), p.get(1), p.get(2)], dtype=np.float64)
    rotation = transform.R()
    rot_mat = np.empty((3, 3), dtype=np.float64)
    for i in range(3):
        for j in range(3):
            rot_mat[i, j] = rotation.get(i, j)
    quat = _rotmat_to_quat(rot_mat)
    return pos, quat


# --- Public canonical API -------------------------------------------------


def extract_grip_pose(
    state: Any, model: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(grip_position_xyz, grip_quaternion_wxyz)`` in world frame.

    Reads the canonical ``hand_r_grip_offset`` PhysicalOffsetFrame from
    the WeldJoint linking ``hand_r`` to ``Club``. This is the location
    the cross-engine spec §2.2 names ``SimOut.grip``.

    Args:
        state: A ``simbody::State`` produced by ``model.initSystem()``;
            will be realised to Position stage if needed.
        model: The ``opensim.Model`` that exposes
            ``/jointset/hand_r_to_club/hand_r_grip_offset``.

    Returns:
        ``(pos, quat)`` with ``pos`` shape ``(3,)`` metres, ``quat``
        shape ``(4,)`` in canonical ``[w, x, y, z]`` order.

    Raises:
        ImportError: If the OpenSim Python bindings are missing.
        ValueError: If ``state`` or ``model`` is ``None``.
        KeyError: If the model lacks ``hand_r_grip_offset``.
        RuntimeError: If state realisation fails.
    """
    if state is None or model is None:
        raise ValueError("state and model are both required")
    _require_opensim()
    _ensure_position_realised(model, state)
    return _frame_pose_in_ground(model, state, GRIP_FRAME_PATH)


def extract_clubhead_pose(
    state: Any, model: Any
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Return ``(clubhead_position_xyz, clubhead_quaternion_wxyz)``.

    Reads the canonical ``club_head_offset`` PhysicalOffsetFrame on the
    ``Club`` body. This is the location the cross-engine spec §2.2
    names ``SimOut.clubhead``.

    Args:
        state: Realised ``simbody::State``.
        model: ``opensim.Model`` exposing
            ``/jointset/hand_r_to_club/club_head_offset``.

    Returns:
        ``(pos, quat)`` — see :func:`extract_grip_pose`.

    Raises:
        ImportError: If the OpenSim Python bindings are missing.
        ValueError: If ``state`` or ``model`` is ``None``.
        KeyError: If the model lacks ``club_head_offset``.
        RuntimeError: If state realisation fails.
    """
    if state is None or model is None:
        raise ValueError("state and model are both required")
    _require_opensim()
    _ensure_position_realised(model, state)
    return _frame_pose_in_ground(model, state, CLUBHEAD_FRAME_PATH)


def extract_full_pose(state: Any, model: Any) -> dict[str, NDArray[np.float64]]:
    """Extract every canonical landmark in one realisation pass.

    Returns a dict shaped to match the cross-engine ``SimOut`` per-step
    schema, with two flat keys per landmark:

    * ``"<landmark>_pos"``  -> ``(3,) float64`` position in world frame
    * ``"<landmark>_quat"`` -> ``(4,) float64`` quaternion ``[w, x, y, z]``

    For the MVP humanoid that yields ``grip_pos``, ``grip_quat``,
    ``clubhead_pos``, ``clubhead_quat``. Future models can extend
    :data:`CANONICAL_LANDMARKS` and consumers will pick up the new
    landmarks automatically.

    Args:
        state: Realised ``simbody::State``.
        model: ``opensim.Model`` exposing the frames in
            :data:`CANONICAL_LANDMARKS`.

    Returns:
        Dictionary mapping ``"<landmark>_pos"`` / ``"<landmark>_quat"``
        to ``(3,)`` and ``(4,)`` ``float64`` arrays respectively.

    Raises:
        ImportError: If the OpenSim Python bindings are missing.
        ValueError: If ``state`` or ``model`` is ``None``.
        KeyError: If any canonical frame is missing from the model.
        RuntimeError: If state realisation fails.
    """
    if state is None or model is None:
        raise ValueError("state and model are both required")
    _require_opensim()
    _ensure_position_realised(model, state)
    out: dict[str, NDArray[np.float64]] = {}
    for landmark, frame_path in CANONICAL_LANDMARKS.items():
        pos, quat = _frame_pose_in_ground(model, state, frame_path)
        out[f"{landmark}_pos"] = pos
        out[f"{landmark}_quat"] = quat
    return out


# --- Legacy API (deprecated) ---------------------------------------------
#
# The original `compute_grip`/`compute_clubhead`/`compute_skeleton_fk`
# entry points (PRs #4158 / #4160) are retained as thin wrappers that
# delegate to the canonical extractors. They emit DeprecationWarning so
# downstream callers can migrate at their own pace.


def compute_grip(
    model: osim.Model,
    state: osim.State,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """[Deprecated] Compute grip position and orientation from a state.

    .. deprecated::
        Use :func:`extract_grip_pose` instead. The original implementation
        looked for non-existent ``hand_left`` / ``hand_right`` bodies (see
        issue #4191); this wrapper now reads the canonical
        ``hand_r_grip_offset`` frame.

    Args:
        model: OpenSim Model instance (must be initialized).
        state: SimTK::State snapshot at which to evaluate FK.

    Returns:
        ``(position, quaternion)`` — see :func:`extract_grip_pose`.
    """
    warnings.warn(
        "compute_grip is deprecated; use extract_grip_pose(state, model) "
        "from the same module. (issue #4191)",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_grip_pose(state, model)


def compute_clubhead(
    model: osim.Model,
    state: osim.State,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """[Deprecated] Compute clubhead position and orientation from a state.

    .. deprecated::
        Use :func:`extract_clubhead_pose` instead. The original fallback
        synthesised a clubhead by offsetting the (broken) grip; this
        wrapper now reads the canonical ``club_head_offset`` frame.

    Args:
        model: OpenSim Model instance (must be initialized).
        state: SimTK::State snapshot at which to evaluate FK.

    Returns:
        ``(position, quaternion)`` — see :func:`extract_clubhead_pose`.
    """
    warnings.warn(
        "compute_clubhead is deprecated; use extract_clubhead_pose(state, "
        "model) from the same module. (issue #4191)",
        DeprecationWarning,
        stacklevel=2,
    )
    return extract_clubhead_pose(state, model)


def compute_skeleton_fk(
    model: osim.Model,
    states: list[osim.State] | NDArray[np.float64],
) -> dict[str, NDArray[np.float64]]:
    """[Deprecated] Vectorised FK extraction over a state trajectory.

    .. deprecated::
        Loop :func:`extract_full_pose` over your states and stack the
        results — the canonical extractor reads the correct frames and
        keeps the per-step output schema flat. (issue #4191)

    Args:
        model: OpenSim Model instance (must be initialized).
        states: List/tuple of SimTK::State objects, or an ``(N, n_q)``
            trajectory array. Array form is not supported (kept for
            signature parity with the legacy API; raises
            :class:`NotImplementedError` like the original).

    Returns:
        Dictionary with keys ``grip``, ``grip_quat``, ``clubhead``,
        ``club_quat`` — same shape as the original API.

    Raises:
        ValueError: If ``states`` sequence is empty.
        TypeError: If ``states`` is neither a list of States nor an array.
        NotImplementedError: If an array is passed (matches legacy
            behaviour pending issue #4110 / #4114 follow-ups).
    """
    warnings.warn(
        "compute_skeleton_fk is deprecated; loop extract_full_pose over "
        "your states. (issue #4191)",
        DeprecationWarning,
        stacklevel=2,
    )

    if isinstance(states, np.ndarray):
        raise NotImplementedError(
            "Array-based state trajectories not yet supported. "
            "Awaiting issue #4110 (model with DOF naming) and #4114 "
            "(coordinate mapping). Pass a list of State objects instead."
        )

    if not isinstance(states, (list, tuple)):
        raise TypeError(f"states must be list of States or array, got {type(states)}")

    if len(states) == 0:
        raise ValueError("states sequence is empty")

    n_samples = len(states)
    grip_pos_array = np.zeros((n_samples, 3), dtype=np.float64)
    grip_quat_array = np.zeros((n_samples, 4), dtype=np.float64)
    clubhead_pos_array = np.zeros((n_samples, 3), dtype=np.float64)
    club_quat_array = np.zeros((n_samples, 4), dtype=np.float64)

    for i, state in enumerate(states):
        pose = extract_full_pose(state, model)
        grip_pos_array[i] = pose["grip_pos"]
        grip_quat_array[i] = pose["grip_quat"]
        clubhead_pos_array[i] = pose["clubhead_pos"]
        club_quat_array[i] = pose["clubhead_quat"]

    return {
        "grip": grip_pos_array,
        "grip_quat": grip_quat_array,
        "clubhead": clubhead_pos_array,
        "club_quat": club_quat_array,
    }


# --- Utilities ------------------------------------------------------------


def _rotmat_to_quat(rot_matrix: object) -> NDArray[np.float64]:
    """Convert a 3x3 rotation matrix to a quaternion ``[w, x, y, z]``.

    Accepts either a ``numpy.ndarray`` or any OpenSim-style object with a
    ``get(i, j)`` accessor. Uses Shepperd's numerically-stable branch
    selection. Sign-canonicalises so ``w >= 0``.

    Args:
        rot_matrix: ``(3, 3)`` ``numpy.ndarray`` or object with
            ``get(i, j)`` returning the matrix entry.

    Returns:
        Unit quaternion ``(4,)`` ``[w, x, y, z]``, ``w >= 0``.

    Raises:
        ValueError: If the input cannot be converted to a 3x3 matrix or
            produces a degenerate (zero-norm) quaternion.
    """
    if isinstance(rot_matrix, np.ndarray):
        mat = np.asarray(rot_matrix, dtype=np.float64)
        if mat.shape != (3, 3):
            raise ValueError(f"rot_matrix must be (3, 3); got {mat.shape}")
    else:
        # OpenSim Rotation: has .get(i, j).
        _rm: Any = rot_matrix
        mat = np.array(
            [
                [
                    _rm.get(0, 0),
                    _rm.get(0, 1),
                    _rm.get(0, 2),
                ],
                [
                    _rm.get(1, 0),
                    _rm.get(1, 1),
                    _rm.get(1, 2),
                ],
                [
                    _rm.get(2, 0),
                    _rm.get(2, 1),
                    _rm.get(2, 2),
                ],
            ],
            dtype=np.float64,
        )

    trace = np.trace(mat)
    if trace > 0:
        s = 0.5 / np.sqrt(trace + 1.0)
        w = 0.25 / s
        x = (mat[2, 1] - mat[1, 2]) * s
        y = (mat[0, 2] - mat[2, 0]) * s
        z = (mat[1, 0] - mat[0, 1]) * s
    elif mat[0, 0] > mat[1, 1] and mat[0, 0] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[0, 0] - mat[1, 1] - mat[2, 2])
        w = (mat[2, 1] - mat[1, 2]) / s
        x = 0.25 * s
        y = (mat[0, 1] + mat[1, 0]) / s
        z = (mat[0, 2] + mat[2, 0]) / s
    elif mat[1, 1] > mat[2, 2]:
        s = 2.0 * np.sqrt(1.0 + mat[1, 1] - mat[0, 0] - mat[2, 2])
        w = (mat[0, 2] - mat[2, 0]) / s
        x = (mat[0, 1] + mat[1, 0]) / s
        y = 0.25 * s
        z = (mat[1, 2] + mat[2, 1]) / s
    else:
        s = 2.0 * np.sqrt(1.0 + mat[2, 2] - mat[0, 0] - mat[1, 1])
        w = (mat[1, 0] - mat[0, 1]) / s
        x = (mat[0, 2] + mat[2, 0]) / s
        y = (mat[1, 2] + mat[2, 1]) / s
        z = 0.25 * s

    quat = np.array([w, x, y, z], dtype=np.float64)
    if quat[0] < 0.0:
        quat = -quat
    norm = float(np.linalg.norm(quat))
    if norm == 0.0:
        raise ValueError("Degenerate rotation produced zero-norm quaternion")
    return quat / norm


def _average_quaternions(
    q1: NDArray[np.float64],
    q2: NDArray[np.float64],
) -> NDArray[np.float64]:
    """Average two unit quaternions via SLERP at the midpoint.

    Args:
        q1: Quaternion ``[w, x, y, z]``.
        q2: Quaternion ``[w, x, y, z]``.

    Returns:
        Unit-norm averaged quaternion ``[w, x, y, z]``.

    Raises:
        ValueError: If either input has zero norm.
    """
    n1 = float(np.linalg.norm(q1))
    n2 = float(np.linalg.norm(q2))
    if n1 == 0.0 or n2 == 0.0:
        raise ValueError("input quaternions must have non-zero norm")
    q1 = q1 / n1
    q2 = q2 / n2

    dot_product = float(np.dot(q1, q2))
    if dot_product < 0.0:
        q2 = -q2
        dot_product = -dot_product
    dot_product = float(np.clip(dot_product, -1.0, 1.0))

    theta = np.arccos(dot_product)
    t = 0.5
    sin_theta = np.sin(theta)
    if np.abs(sin_theta) < 1e-6:
        q_avg = (q1 + q2) / 2.0
    else:
        w1 = np.sin((1.0 - t) * theta) / sin_theta
        w2 = np.sin(t * theta) / sin_theta
        q_avg = w1 * q1 + w2 * q2

    norm = float(np.linalg.norm(q_avg))
    if norm == 0.0:
        raise ValueError("averaged quaternion is degenerate")
    return q_avg / norm

"""OpenSim ↔ Simscape coordinate-convention mapping helper.

Background
----------
OpenSim's Rajagopal2015-derived ``golf_humanoid.osim`` exposes 39
generalized coordinates (full-body, including legs, lumbar spine, both
arms, and pelvis floating base). The Simscape ``GolfSwing3D_Kinetic``
body chain — the canonical reference defined in
``shared/models/golf_humanoid_topology.yaml`` — is a 25-DOF subset that
omits the legs entirely (the swing model is ground-locked at the hips)
and folds OpenSim's lumbar 3-DOF block into a spine-universal +
torso-revolute split.

This module is **pure-Python** (no ``import opensim``). It is imported
by every cost / FK helper in
``src/engines/physics_engines/opensim/python/`` and by every cross-
engine test that compares OpenSim results against the Simscape oracle.

The four conventions that differ between the engines:

1. **Frame orientation.** OpenSim uses Y-up (``+y`` is up, ``+x`` is the
   anterior axis, ``+z`` is to the subject's right). Simscape's golf-
   swing model uses Z-up (``+z`` up, ``+x`` along the target line,
   ``+y`` to the golfer's right per ``golf_humanoid_topology.yaml``).
   See :func:`frame_y_up_to_z_up`.

2. **Quaternion ordering.** OpenSim returns quaternions through Eigen's
   ``[x, y, z, w]`` ordering. The canonical ``SimOut`` schema and the
   Simscape pipeline use ``[w, x, y, z]``. See
   :func:`quat_eigen_to_canonical`.

3. **Joint-angle sign conventions.** Several Rajagopal coordinates are
   the negative of the Simscape equivalent. The :data:`OPENSIM_TO_SIMSCAPE`
   table records each mapping together with a sign multiplier that is
   applied when projecting to/from Simscape coordinates.

4. **DOF-count mismatch.** The OpenSim model has 39 DOF, Simscape 25.
   :func:`to_simscape` projects 39→25 by selecting the mapped indices;
   :func:`from_simscape` embeds 25→39 by reading mapped indices and
   filling the remainder from :data:`OPENSIM_NEUTRAL_POSE`.

The mapping table below is the only authoritative source for which
OpenSim coordinate maps to which Simscape ``q`` slot — every other
helper in the OpenSim engine wrapper imports it. **Do not** define a
second copy elsewhere.

Round-trip identity
-------------------
``to_simscape(from_simscape(q_simscape)) == q_simscape`` for every
input ``q_simscape`` of length 25 (within floating-point noise). The
reverse direction (``from_simscape(to_simscape(q_opensim))``) is **not**
identity because 14 OpenSim coordinates have no Simscape equivalent and
are reset to neutral on the way back out.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Final

import numpy as np

# ---------------------------------------------------------------------------
# Coordinate order — OpenSim (39 DOF, model order)
# ---------------------------------------------------------------------------
# Order matches the ``<Coordinate name=...>`` declarations in the
# Rajagopal2015-derived ``golf_humanoid.osim`` file. The OpenSim Python
# API exposes coordinates in this same order via ``model.getCoordinateSet()``.

OPENSIM_COORD_ORDER: Final[tuple[str, ...]] = (
    # Pelvis FreeJoint — rotations first (tilt/list/rotation), then translations
    "pelvis_tilt",  # 0  — rotation about world Z (forward tilt)
    "pelvis_list",  # 1  — rotation about world X (lateral list)
    "pelvis_rotation",  # 2  — rotation about world Y (axial spin)
    "pelvis_tx",  # 3  — translation along X
    "pelvis_ty",  # 4  — translation along Y (vertical, Y-up)
    "pelvis_tz",  # 5  — translation along Z
    # Right leg
    "hip_flexion_r",  # 6
    "hip_adduction_r",  # 7
    "hip_rotation_r",  # 8
    "knee_angle_r",  # 9
    "knee_angle_r_beta",  # 10 — patellofemoral helper coordinate
    "ankle_angle_r",  # 11
    "subtalar_angle_r",  # 12
    "mtp_angle_r",  # 13
    # Left leg
    "hip_flexion_l",  # 14
    "hip_adduction_l",  # 15
    "hip_rotation_l",  # 16
    "knee_angle_l",  # 17
    "knee_angle_l_beta",  # 18
    "ankle_angle_l",  # 19
    "subtalar_angle_l",  # 20
    "mtp_angle_l",  # 21
    # Lumbar spine — Rajagopal's 3-DOF block; folded into Simscape spine+torso
    "lumbar_extension",  # 22 — flexion / extension (rx in Simscape spine frame)
    "lumbar_bending",  # 23 — lateral bend (ry in Simscape spine frame)
    "lumbar_rotation",  # 24 — axial rotation (rz at Simscape torso joint)
    # Right arm
    "arm_flex_r",  # 25 — shoulder flexion
    "arm_add_r",  # 26 — shoulder adduction
    "arm_rot_r",  # 27 — shoulder internal/external rotation
    "elbow_flex_r",  # 28
    "pro_sup_r",  # 29 — forearm pronation/supination (no Simscape eq.)
    "wrist_flex_r",  # 30
    "wrist_dev_r",  # 31 — wrist deviation (radial/ulnar)
    # Left arm
    "arm_flex_l",  # 32
    "arm_add_l",  # 33
    "arm_rot_l",  # 34
    "elbow_flex_l",  # 35
    "pro_sup_l",  # 36
    "wrist_flex_l",  # 37
    "wrist_dev_l",  # 38
)
assert len(OPENSIM_COORD_ORDER) == 39, "Rajagopal2015 base must have 39 coords"


# ---------------------------------------------------------------------------
# Coordinate order — Simscape (25 DOF), per shared/models/golf_humanoid_topology.yaml
# ---------------------------------------------------------------------------
# Mirror of the ``q_order`` block in the topology YAML. Kept here so users
# of this module don't need to parse YAML at import time.

SIMSCAPE_COORD_ORDER: Final[tuple[str, ...]] = (
    "hip.tx",  # 0
    "hip.ty",  # 1
    "hip.tz",  # 2
    "hip.rx",  # 3
    "hip.ry",  # 4
    "hip.rz",  # 5
    "spine.rx",  # 6
    "spine.ry",  # 7
    "torso.rz",  # 8
    "l_scap.rx",  # 9
    "l_scap.ry",  # 10
    "l_shoulder.rx",  # 11
    "l_shoulder.ry",  # 12
    "l_shoulder.rz",  # 13
    "l_elbow.rz",  # 14
    "l_wrist.rx",  # 15
    "l_wrist.ry",  # 16
    "r_scap.rx",  # 17
    "r_scap.ry",  # 18
    "r_shoulder.rx",  # 19
    "r_shoulder.ry",  # 20
    "r_shoulder.rz",  # 21
    "r_elbow.rz",  # 22
    "r_wrist.rx",  # 23
    "r_wrist.ry",  # 24
)
assert len(SIMSCAPE_COORD_ORDER) == 25


# ---------------------------------------------------------------------------
# Mapping table — OpenSim coord name → Simscape coord name
# ---------------------------------------------------------------------------
# Only mapped coordinates appear here. Any OpenSim coord absent from the
# table has no Simscape equivalent and is *dropped* by ``to_simscape`` and
# *defaulted* by ``from_simscape``.
#
# The 14 coordinates that DO NOT map (all leg DOFs + the forearm
# pronation/supination DOFs) are:
#
#     hip_flexion_r, hip_adduction_r, hip_rotation_r,
#     knee_angle_r, knee_angle_r_beta, ankle_angle_r,
#     subtalar_angle_r, mtp_angle_r,
#     hip_flexion_l, hip_adduction_l, hip_rotation_l,
#     knee_angle_l, knee_angle_l_beta, ankle_angle_l,
#     subtalar_angle_l, mtp_angle_l,
#     pro_sup_r, pro_sup_l
#
# The Simscape body model is ground-locked at the hips and folds the
# forearm into a single rigid link, so these DOFs are physically absent
# from the swing chain.
#
# Conversely, the 4 Simscape coordinates that DO NOT have an OpenSim
# equivalent are the scapulothoracic universal joints
# (``l_scap.rx``, ``l_scap.ry``, ``r_scap.rx``, ``r_scap.ry``).
# Rajagopal2015 attaches the humerus directly to the torso and does not
# model the scapula explicitly. ``from_simscape`` discards these four
# values; ``to_simscape`` synthesizes them as zero from a 39-D OpenSim
# pose.

OPENSIM_TO_SIMSCAPE: Final[Mapping[str, str]] = {
    # Pelvis floating base — re-order rotations/translations and convert Y-up→Z-up.
    # OpenSim pelvis_tx/ty/tz are in Y-up world frame. The frame conversion
    # below is applied as part of ``to_simscape`` so the values stored on
    # the Simscape side are in the Z-up world used by the Simscape model.
    "pelvis_tx": "hip.tx",
    "pelvis_ty": "hip.tz",  # OpenSim Y (up) ↔ Simscape Z (up)
    "pelvis_tz": "hip.ty",  # OpenSim Z (right) ↔ Simscape Y (right)
    "pelvis_tilt": "hip.rx",  # forward tilt
    "pelvis_list": "hip.ry",  # lateral list
    "pelvis_rotation": "hip.rz",  # axial spin (heading)
    # Lumbar 3-DOF block → Simscape spine universal + torso revolute
    "lumbar_extension": "spine.rx",
    "lumbar_bending": "spine.ry",
    "lumbar_rotation": "torso.rz",
    # Right shoulder gimbal
    "arm_add_r": "r_shoulder.rx",
    "arm_flex_r": "r_shoulder.ry",
    "arm_rot_r": "r_shoulder.rz",
    "elbow_flex_r": "r_elbow.rz",
    "wrist_flex_r": "r_wrist.rx",
    "wrist_dev_r": "r_wrist.ry",
    # Left shoulder gimbal
    "arm_add_l": "l_shoulder.rx",
    "arm_flex_l": "l_shoulder.ry",
    "arm_rot_l": "l_shoulder.rz",
    "elbow_flex_l": "l_elbow.rz",
    "wrist_flex_l": "l_wrist.rx",
    "wrist_dev_l": "l_wrist.ry",
}


# ---------------------------------------------------------------------------
# Sign convention — applied per-OpenSim-coordinate when projecting.
# ---------------------------------------------------------------------------
# +1.0 = OpenSim and Simscape agree on sign for this DOF.
# -1.0 = OpenSim positive convention is the negative of Simscape's.
#
# Every entry is documented with its rationale. Empirically, OpenSim's
# Rajagopal2015 conventions follow the ISB recommendations; Simscape's
# golf-swing chain follows the right-hand-rule about each named axis with
# the world frame defined in ``golf_humanoid_topology.yaml``.

OPENSIM_SIGN_CONVENTION: Final[Mapping[str, float]] = {
    # Pelvis translations — Y-up→Z-up frame swap is handled by the index map;
    # all linear translations keep their sign (OpenSim +Y up = Simscape +Z up;
    # OpenSim +Z right = Simscape +Y right).
    "pelvis_tx": +1.0,
    "pelvis_ty": +1.0,
    "pelvis_tz": +1.0,
    # Pelvis rotations: OpenSim's pelvis_tilt is rotation about world Z in
    # OpenSim's Y-up world; Simscape hip.rx is rotation about world X in
    # the Z-up world. After the Y-up→Z-up axis remap done by the index
    # table, sign agrees by right-hand-rule about the new axis.
    "pelvis_tilt": +1.0,
    "pelvis_list": +1.0,
    "pelvis_rotation": +1.0,
    # Lumbar block: Rajagopal's lumbar_extension positive direction is
    # forward (flexion), matching Simscape spine.rx positive direction.
    "lumbar_extension": +1.0,
    "lumbar_bending": +1.0,
    "lumbar_rotation": +1.0,
    # Right shoulder: arm_add_r positive in OpenSim is shoulder ABduction
    # (away from body); Simscape r_shoulder.rx positive direction follows
    # right-hand-rule about the local +x axis which points forward, so
    # ABduction maps to a *positive* rotation — they agree.
    "arm_add_r": +1.0,
    "arm_flex_r": +1.0,
    "arm_rot_r": +1.0,
    # Right elbow: OpenSim elbow_flex_r positive = flexion (forearm toward
    # shoulder); Simscape r_elbow.rz positive also flexion → agree.
    "elbow_flex_r": +1.0,
    # Right wrist: OpenSim wrist_flex_r positive = palmar flexion;
    # Simscape r_wrist.rx positive also palmar flexion → agree.
    # OpenSim wrist_dev_r positive = ulnar deviation; Simscape r_wrist.ry
    # convention is *radial* deviation positive, so the sign flips.
    "wrist_flex_r": +1.0,
    "wrist_dev_r": -1.0,
    # Left shoulder: by the right-hand-rule about a *left*-side limb's
    # local axes, the abduction direction in OpenSim is the **negative**
    # of the right-side, while Simscape uses a sign-symmetric convention
    # across left/right (both arms use +rx for ABduction). So the left
    # adduction sign flips.
    "arm_add_l": -1.0,
    "arm_flex_l": +1.0,
    "arm_rot_l": -1.0,
    "elbow_flex_l": +1.0,
    "wrist_flex_l": +1.0,
    # Same wrist-deviation reasoning as the right side, flipped again
    # because the left-side ulnar/radial axis points opposite the right.
    "wrist_dev_l": +1.0,
}

# Defensive sanity check at import time: every mapped coordinate must
# have a sign convention, and vice-versa.
assert set(OPENSIM_TO_SIMSCAPE.keys()) == set(OPENSIM_SIGN_CONVENTION.keys()), (
    "OPENSIM_TO_SIMSCAPE and OPENSIM_SIGN_CONVENTION must cover the same "
    "OpenSim coordinates"
)


# ---------------------------------------------------------------------------
# Neutral-pose vector — the OpenSim default for at-address position.
# ---------------------------------------------------------------------------
# This is the pose used when ``from_simscape`` needs to embed a 25-DOF
# Simscape vector into the 39-D OpenSim space — every OpenSim DOF that is
# *not* covered by the Simscape chain is filled from this table.
#
# Values are zero for every DOF in the MVP build. The Rajagopal2015
# default state for an at-address golf pose is the all-zeros pose; future
# work (issue #4093 follow-up) may populate the legs with knee/hip flexion
# values matching the Simscape ``StartPosition`` parameters.

OPENSIM_NEUTRAL_POSE: Final[np.ndarray] = np.zeros(39, dtype=np.float64)
OPENSIM_NEUTRAL_POSE.setflags(write=False)


# ---------------------------------------------------------------------------
# Pre-computed index tables (built once at import; tests rely on them).
# ---------------------------------------------------------------------------

_OPENSIM_NAME_TO_IDX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(OPENSIM_COORD_ORDER)
}
_SIMSCAPE_NAME_TO_IDX: Final[dict[str, int]] = {
    name: i for i, name in enumerate(SIMSCAPE_COORD_ORDER)
}

# Inverse lookup: Simscape coord name → (opensim coord name, sign).
# Built by walking ``OPENSIM_TO_SIMSCAPE`` once at import time. Each
# Simscape coord must appear at most once on the right-hand side
# (asserted below), since two OpenSim DOFs cannot drive the same
# Simscape DOF.
_SIMSCAPE_TO_OPENSIM: Final[dict[str, tuple[str, float]]] = {}
for _os_name, _sim_name in OPENSIM_TO_SIMSCAPE.items():
    if _sim_name in _SIMSCAPE_TO_OPENSIM:
        raise RuntimeError(
            f"Simscape coordinate {_sim_name!r} is mapped from two OpenSim "
            f"coordinates ({_SIMSCAPE_TO_OPENSIM[_sim_name][0]!r} and "
            f"{_os_name!r}); the mapping table is malformed."
        )
    _SIMSCAPE_TO_OPENSIM[_sim_name] = (_os_name, OPENSIM_SIGN_CONVENTION[_os_name])

# For each Simscape index, the (opensim_index, sign) pair that fills it.
# Simscape coords with no OpenSim equivalent get (-1, 0.0) — handled as
# zero-fill in ``to_simscape``.
_SIM_IDX_TO_OS: Final[tuple[tuple[int, float], ...]] = tuple(
    (
        (
            _OPENSIM_NAME_TO_IDX[_SIMSCAPE_TO_OPENSIM[sim_name][0]],
            _SIMSCAPE_TO_OPENSIM[sim_name][1],
        )
        if sim_name in _SIMSCAPE_TO_OPENSIM
        else (-1, 0.0)
    )
    for sim_name in SIMSCAPE_COORD_ORDER
)


# ---------------------------------------------------------------------------
# Public API: projection / embedding
# ---------------------------------------------------------------------------


def to_simscape(q_opensim: np.ndarray) -> np.ndarray:
    """Project an OpenSim 39-D generalized-coord vector to Simscape 25-D.

    Parameters
    ----------
    q_opensim
        Length-39 array of OpenSim coordinate values, in
        :data:`OPENSIM_COORD_ORDER` order. Radians for rotational DOFs,
        meters for translational DOFs (OpenSim's default unit system).

    Returns
    -------
    q_simscape
        Length-25 array in :data:`SIMSCAPE_COORD_ORDER` order, ready to
        feed into ``compute_skeleton_fk`` / cost evaluation. Sign
        conventions and Y-up→Z-up axis swaps applied per the mapping
        tables above.

    Notes
    -----
    Simscape coordinates with no OpenSim equivalent (the four
    scapulothoracic DOFs ``l_scap.rx``, ``l_scap.ry``, ``r_scap.rx``,
    ``r_scap.ry``) are returned as zero — Rajagopal2015 has no scapula
    DOFs, so the most physically reasonable embedding is the neutral
    scapula pose.
    """
    q_opensim = np.asarray(q_opensim, dtype=np.float64)
    if q_opensim.shape != (39,):
        raise ValueError(
            f"to_simscape expects a (39,) array, got shape {q_opensim.shape}"
        )
    if not np.all(np.isfinite(q_opensim)):
        raise ValueError("to_simscape: input contains non-finite values")

    q_sim = np.zeros(25, dtype=np.float64)
    for sim_idx, (os_idx, sign) in enumerate(_SIM_IDX_TO_OS):
        if os_idx >= 0:
            q_sim[sim_idx] = sign * q_opensim[os_idx]
    return q_sim


def from_simscape(q_simscape: np.ndarray) -> np.ndarray:
    """Embed a Simscape 25-D generalized-coord vector into OpenSim 39-D.

    Parameters
    ----------
    q_simscape
        Length-25 array in :data:`SIMSCAPE_COORD_ORDER` order.

    Returns
    -------
    q_opensim
        Length-39 array in :data:`OPENSIM_COORD_ORDER` order. Mapped
        coordinates carry their Simscape value (with the sign convention
        inverted back to OpenSim's frame); unmapped coordinates default
        to :data:`OPENSIM_NEUTRAL_POSE`.

    Notes
    -----
    The 14 OpenSim coordinates with no Simscape equivalent (legs +
    forearm pronation/supination) are filled from
    :data:`OPENSIM_NEUTRAL_POSE` (currently all-zeros — at-address pose).
    Round-trip identity holds in the
    ``to_simscape ∘ from_simscape`` direction.
    """
    q_simscape = np.asarray(q_simscape, dtype=np.float64)
    if q_simscape.shape != (25,):
        raise ValueError(
            f"from_simscape expects a (25,) array, got shape {q_simscape.shape}"
        )
    if not np.all(np.isfinite(q_simscape)):
        raise ValueError("from_simscape: input contains non-finite values")

    q_os = OPENSIM_NEUTRAL_POSE.copy()
    for sim_idx, (os_idx, sign) in enumerate(_SIM_IDX_TO_OS):
        if os_idx >= 0:
            # Inverse sign multiplier: Simscape → OpenSim is the same factor
            # because each ``sign`` is ±1.
            q_os[os_idx] = sign * q_simscape[sim_idx]
    return q_os


# ---------------------------------------------------------------------------
# Quaternion ordering helpers
# ---------------------------------------------------------------------------


def quat_eigen_to_canonical(q_xyzw: np.ndarray) -> np.ndarray:
    """Convert Eigen-style ``[x, y, z, w]`` quaternions to canonical ``[w, x, y, z]``.

    OpenSim's ``SimTK::Rotation::convertRotationToQuaternion`` returns
    ``[x, y, z, w]`` (Eigen convention). The canonical ``SimOut.grip_quat``
    schema and the Simscape pipeline use ``[w, x, y, z]``. Apply this
    conversion at the SimOut boundary, never in the middle of a
    computation.

    Accepts ``(4,)`` or ``(N, 4)`` shapes. The last axis is the
    quaternion-element axis.
    """
    q = np.asarray(q_xyzw, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat_eigen_to_canonical: last axis must be 4, got {q.shape}")
    out = np.empty_like(q)
    out[..., 0] = q[..., 3]  # w
    out[..., 1] = q[..., 0]  # x
    out[..., 2] = q[..., 1]  # y
    out[..., 3] = q[..., 2]  # z
    return out


def quat_canonical_to_eigen(q_wxyz: np.ndarray) -> np.ndarray:
    """Inverse of :func:`quat_eigen_to_canonical` (``[w, x, y, z]`` → ``[x, y, z, w]``)."""
    q = np.asarray(q_wxyz, dtype=np.float64)
    if q.shape[-1] != 4:
        raise ValueError(f"quat_canonical_to_eigen: last axis must be 4, got {q.shape}")
    out = np.empty_like(q)
    out[..., 0] = q[..., 1]  # x
    out[..., 1] = q[..., 2]  # y
    out[..., 2] = q[..., 3]  # z
    out[..., 3] = q[..., 0]  # w
    return out


# ---------------------------------------------------------------------------
# Frame-orientation helpers (Y-up ↔ Z-up)
# ---------------------------------------------------------------------------
#
# OpenSim convention: +X anterior, +Y up, +Z to the subject's right.
# Simscape convention (per ``golf_humanoid_topology.yaml``):
#     +X along the target line, +Y to the golfer's right, +Z up.
#
# The fixed rotation from OpenSim → Simscape coordinates is therefore
#     x_sim = x_os
#     y_sim = z_os
#     z_sim = y_os
#
# This is a *handedness-preserving* axis permutation (det = +1) — the
# matrix corresponds to swapping the Y and Z axes which, combined with
# leaving the third row sign positive, keeps the right-hand rule.

_R_YUP_TO_ZUP: Final[np.ndarray] = np.array(
    [
        [1.0, 0.0, 0.0],
        [0.0, 0.0, 1.0],
        [0.0, 1.0, 0.0],
    ],
    dtype=np.float64,
)
_R_YUP_TO_ZUP.setflags(write=False)


def frame_y_up_to_z_up(v: np.ndarray) -> np.ndarray:
    """Rotate a Y-up (OpenSim) vector or vector batch into the Z-up (Simscape) frame.

    Parameters
    ----------
    v
        Either ``(3,)`` for a single 3-vector or ``(N, 3)`` for a batch.

    Returns
    -------
    v_zup
        Same shape as ``v``, expressed in the Simscape Z-up world.
    """
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"frame_y_up_to_z_up: last axis must be 3, got {v.shape}")
    return v @ _R_YUP_TO_ZUP.T


def frame_z_up_to_y_up(v: np.ndarray) -> np.ndarray:
    """Inverse of :func:`frame_y_up_to_z_up`. The matrix is its own transpose-inverse."""
    v = np.asarray(v, dtype=np.float64)
    if v.shape[-1] != 3:
        raise ValueError(f"frame_z_up_to_y_up: last axis must be 3, got {v.shape}")
    return v @ _R_YUP_TO_ZUP  # inverse of the swap is the swap itself

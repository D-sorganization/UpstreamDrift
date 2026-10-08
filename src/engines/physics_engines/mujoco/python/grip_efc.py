"""Grip wrench extraction from MuJoCo weld constraint forces (GCV-8, #11714).

Shared by the MuJoCo and MyoSuite engines: both hold the club with two weld
equalities (``grip_weld_r`` / ``grip_weld_l``).  After ``mj_forward`` the
equality rows are selected with ``efc_type == mjCNSTR_EQUALITY`` and
``efc_id == eq_id``; their generalised force ``J_k^T efc_force_k`` is mapped
to a world-frame wrench ``(F, tau)`` on the club at the club-side anchor with
the club-point Jacobians, so MuJoCo's internal row scaling (``torquescale``,
quaternion error map) never enters the result.

Sign (binding, ADR-0052): wrench exerted by the hand ON THE CLUB.  MuJoCo
defines the weld residual as ``obj1 - obj2`` so the generalised force is
``(J_2 - J_1)^T w`` for the wrench ``w`` on object 2; when the club is object
1 the sign flips.  The sign is pinned by the lift tests.
"""

from __future__ import annotations

from collections.abc import Mapping

import mujoco
import numpy as np

from src.shared.python.biomechanics.grip_extraction import (
    hand_from_arrays,
    unavailable_analysis,
)
from src.shared.python.biomechanics.grip_wrench import (
    GripAnalysis,
    HandWrench,
    analyze_grip,
)

__all__ = [
    "DEFAULT_GRIP_WELDS",
    "club_body_of_weld",
    "dense_efc_jacobian",
    "grip_analysis_from_efc",
    "weld_wrench_on_club",
]

#: Hand side -> equality name (the MyoSuite scene and the fixtures).
DEFAULT_GRIP_WELDS: Mapping[str, str] = {"L": "grip_weld_l", "R": "grip_weld_r"}


def dense_efc_jacobian(model: mujoco.MjModel, data: mujoco.MjData) -> np.ndarray:
    """Dense ``(nefc, nv)`` constraint Jacobian for dense or sparse layouts."""
    nefc, nv = int(data.nefc), int(model.nv)
    flat = np.asarray(data.efc_J, dtype=np.float64)
    if flat.size == nefc * nv:
        return flat.reshape(nefc, nv)
    dense = np.zeros((nefc, nv))
    mujoco.mju_sparse2dense(
        dense, flat, data.efc_J_rownnz, data.efc_J_rowadr, data.efc_J_colind
    )
    return dense


def _eq_body(model: mujoco.MjModel, eq_id: int, which: int) -> tuple[int, int | None]:
    """Body id of equality object ``which`` (1 or 2) and its site id if any."""
    obj = int(model.eq_obj1id[eq_id] if which == 1 else model.eq_obj2id[eq_id])
    if int(model.eq_objtype[eq_id]) == int(mujoco.mjtObj.mjOBJ_SITE):
        return int(model.site_bodyid[obj]), obj
    return obj, None


def _check_weld(model: mujoco.MjModel, eq_id: int) -> None:
    if not 0 <= eq_id < model.neq:
        raise ValueError(f"equality id {eq_id} out of range [0, {model.neq})")
    if int(model.eq_type[eq_id]) != int(mujoco.mjtEq.mjEQ_WELD):
        raise ValueError(f"equality {eq_id} is not a weld equality")


def club_body_of_weld(model: mujoco.MjModel, eq_id: int) -> int:
    """Club body of a grip weld by the repo convention (object 2 is the club)."""
    _check_weld(model, eq_id)
    return _eq_body(model, eq_id, 2)[0]


def weld_wrench_on_club(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    eq_id: int,
    club_body_id: int,
) -> tuple[np.ndarray, np.ndarray]:
    """World-frame wrench ``(F, tau)`` exerted on the club by one weld.

    Call after ``mj_forward`` (or ``mj_step``): ``efc_force`` must be current.

    Returns:
        ``(wrench, point)``: the 6-vector ``[F, tau]`` and the club-side anchor
        (site position, else the club body origin) it acts at.

    Raises:
        ValueError: for a non-weld/unknown equality, an inactive weld, a club
            that is neither weld object, or a rank-deficient weld Jacobian.
    """
    _check_weld(model, eq_id)
    body1, _ = _eq_body(model, eq_id, 1)
    body2, site2 = _eq_body(model, eq_id, 2)
    if club_body_id not in (body1, body2):
        raise ValueError(
            f"club body {club_body_id} is not an object of equality {eq_id}"
        )
    rows = np.flatnonzero(
        (data.efc_type == int(mujoco.mjtConstraint.mjCNSTR_EQUALITY))
        & (data.efc_id == eq_id)
    )
    if rows.size == 0:
        raise ValueError(f"equality {eq_id} is inactive (no constraint rows)")
    qfrc = dense_efc_jacobian(model, data)[rows].T @ data.efc_force[rows]

    point = np.array(data.site_xpos[site2] if site2 is not None else data.xpos[body2])
    j1 = np.zeros((6, model.nv))
    j2 = np.zeros((6, model.nv))
    mujoco.mj_jac(model, data, j1[:3], j1[3:], point, body1)
    mujoco.mj_jac(model, data, j2[:3], j2[3:], point, body2)
    jdiff = j2 - j1
    if np.linalg.matrix_rank(jdiff) < 6:
        raise ValueError(f"weld {eq_id} Jacobian is rank deficient at the anchor")
    wrench_on_obj2, *_ = np.linalg.lstsq(jdiff.T, qfrc, rcond=None)
    wrench = wrench_on_obj2 if club_body_id == body2 else -wrench_on_obj2
    return wrench, point


def _hand_from_weld(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    side: str,
    name: str,
    club_body_id: int | None,
) -> tuple[HandWrench | None, str]:
    eq_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_EQUALITY, name)
    if eq_id < 0:
        return None, f"equality {name!r} not found"
    club = club_body_id if club_body_id is not None else club_body_of_weld(model, eq_id)
    try:
        wrench, point = weld_wrench_on_club(model, data, eq_id, club)
    except ValueError as exc:
        return None, str(exc)
    return hand_from_arrays(side, point, wrench[:3], wrench[3:]), ""


def grip_analysis_from_efc(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    *,
    welds: Mapping[str, str] = DEFAULT_GRIP_WELDS,
    club_body_id: int | None = None,
) -> GripAnalysis:
    """Per-hand grip analysis from the weld ``efc_force`` rows.

    Args:
        welds: hand side (``"L"``/``"R"``) to equality name.
        club_body_id: club body; by default object 2 of each weld.

    Postconditions: ``split_method == "efc_force"`` when both hands resolve;
    otherwise the missing hand is ``None`` and ``unavailable_reason`` says why
    (a weld that is absent or inactive is unavailable, never zero).
    """
    if set(welds) != {"L", "R"}:
        raise ValueError("welds must map exactly the sides 'L' and 'R'")
    hands: dict[str, HandWrench | None] = {}
    reasons: list[str] = []
    for side in ("L", "R"):
        hand, why = _hand_from_weld(model, data, side, welds[side], club_body_id)
        hands[side] = hand
        if why:
            reasons.append(f"{side}: {why}")
    if hands["L"] is None and hands["R"] is None:
        return unavailable_analysis("; ".join(reasons))
    result = analyze_grip(hands["L"], hands["R"], split_method="efc_force")
    if reasons:
        return GripAnalysis(
            **{
                **result.__dict__,
                "unavailable_reason": "; ".join(
                    filter(None, [result.unavailable_reason, *reasons])
                ),
            }
        )
    return result

"""Result of a contact-grip run, engine-agnostic (issue #11739, OSV-7 phase 3).

A :class:`ContactRun` is the shared :class:`GripKineticsSeries` (per-hand wrench
on the club, so every bushing analysis and plot applies) plus the quantities
only a distributed contact has: per-pad normal forces, the squeeze, and the
hand-to-club slip (displacement along the grip axis and rotation about it).
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from src.shared.python.grip_contact.parity import GripKineticsSeries

SIDES = ("L", "R")

__all__ = ["ContactRun", "slip_from_frames"]


@dataclass(frozen=True)
class ContactRun:
    """A contact-grip run: the shared series plus contact-only outputs.

    ``normal_force_n[side]`` is ``(n, pads)``; ``roll_slip_rad`` and
    ``axial_slip_m`` are the hand-frame to club-frame rotation about and
    displacement along the grip axis; ``squeeze_n`` is the total pad normal
    force of a hand.
    """

    series: GripKineticsSeries
    normal_force_n: Mapping[str, np.ndarray]
    roll_slip_rad: Mapping[str, np.ndarray]
    axial_slip_m: Mapping[str, np.ndarray]

    @property
    def squeeze_n(self) -> dict[str, np.ndarray]:
        """Total pad normal force of each hand."""
        return {s: np.asarray(self.normal_force_n[s]).sum(axis=1) for s in SIDES}

    def save_npz(self, path: Path) -> None:
        """Write the series and the contact outputs (``<path>`` and ``.contact.npz``)."""
        self.series.save_npz(path)
        extra: dict[str, Any] = {}
        for s in SIDES:
            extra[f"normal_force_{s}_n"] = np.asarray(self.normal_force_n[s])
            extra[f"roll_slip_{s}_rad"] = np.asarray(self.roll_slip_rad[s])
            extra[f"axial_slip_{s}_m"] = np.asarray(self.axial_slip_m[s])
        np.savez_compressed(path.with_suffix(".contact.npz"), **extra)

    @classmethod
    def load_npz(cls, path: Path) -> ContactRun:
        """Read a run written by :meth:`save_npz`."""
        series = GripKineticsSeries.load_npz(path)
        with np.load(path.with_suffix(".contact.npz"), allow_pickle=False) as data:
            return cls(
                series,
                {s: np.asarray(data[f"normal_force_{s}_n"]) for s in SIDES},
                {s: np.asarray(data[f"roll_slip_{s}_rad"]) for s in SIDES},
                {s: np.asarray(data[f"axial_slip_{s}_m"]) for s in SIDES},
            )


def slip_from_frames(
    hand_pose: Mapping[str, tuple[np.ndarray, np.ndarray]],
    club_pose: Mapping[str, tuple[np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Axial slip and roll slip of each hand from ``(R, p)`` frame series.

    The axial slip is the x component of ``R_hand^T (p_club - p_hand)`` (the
    grip axis is the frame x axis); the roll slip is the rotation of
    ``R_hand^T R_club`` about x.
    """
    axial, roll = {}, {}
    for s in SIDES:
        r_h, p_h = (np.asarray(a, float) for a in hand_pose[s])
        r_c, p_c = (np.asarray(a, float) for a in club_pose[s])
        axial[s] = np.einsum("nji,nj->ni", r_h, p_c - p_h)[:, 0]
        rel = np.einsum("nji,njk->nik", r_h, r_c)
        roll[s] = np.arctan2(rel[:, 2, 1] - rel[:, 1, 2], rel[:, 1, 1] + rel[:, 2, 2])
    return axial, roll

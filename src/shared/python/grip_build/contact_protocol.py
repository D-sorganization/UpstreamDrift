"""Swappable contact-distance models for the grip builder (HRG-GB2)."""

from __future__ import annotations

from typing import Protocol, runtime_checkable

import numpy as np

from src.shared.python.grip_build.club_sdf import ClubSDF
from src.shared.python.grip_build.geometry import capsule_capsule_signed_distance


@runtime_checkable
class ContactDistanceModel(Protocol):
    """Geometry-only contact clearance queries."""

    name: str

    def club_clearance(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        radius_m: float,
        club: ClubSDF,
    ) -> float:
        """Signed clearance to the club (negative = penetrating soft tolerance band)."""

    def finger_clearance(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        radius_m: float,
        other_p0: np.ndarray,
        other_p1: np.ndarray,
        other_radius_m: float,
    ) -> float:
        """Signed clearance between two finger capsules."""


class DefaultCapsuleContactModel:
    """Default capsule-to-club and capsule-to-capsule distances."""

    name = "default_capsule"

    def club_clearance(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        radius_m: float,
        club: ClubSDF,
    ) -> float:
        return club.signed_distance_capsule(p0, p1, radius_m)

    def finger_clearance(
        self,
        p0: np.ndarray,
        p1: np.ndarray,
        radius_m: float,
        other_p0: np.ndarray,
        other_p1: np.ndarray,
        other_radius_m: float,
    ) -> float:
        return capsule_capsule_signed_distance(
            p0, p1, radius_m, other_p0, other_p1, other_radius_m
        )

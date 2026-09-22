"""Named golf plausibility priors for club-only matching (CO-02 #10606).

Priors regularize unobserved body/control directions. They are named assumptions
from reviewed geometry or literature, never measured body or force truth.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numpy.typing import NDArray

from src.shared.python.contracts import postcondition, precondition

PRIOR_SCHEMA = "club-plausibility-priors/1.0.0"


@dataclass(frozen=True)
class PriorAssumption:
    """A single named, reviewable prior assumption."""

    name: str
    source: str
    description: str
    is_measured_truth: bool = False

    def __post_init__(self) -> None:
        if not self.name:
            raise ValueError("PriorAssumption.name must be non-empty")
        if not self.source:
            raise ValueError("PriorAssumption.source must be non-empty")
        if self.is_measured_truth:
            raise ValueError(
                "plausibility priors must not claim measured truth "
                f"(assumption={self.name!r})"
            )

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "source": self.source,
            "description": self.description,
            "is_measured_truth": self.is_measured_truth,
        }


@dataclass(frozen=True)
class GolfPlausibilityPriors:
    """Frozen golf-pose/effort/contact regularization weights and limits."""

    assumptions: tuple[PriorAssumption, ...]
    max_joint_rate_rad_s: float
    max_joint_accel_rad_s2: float
    grip_closure_tol_m: float
    effort_regularization_weight: float
    posture_regularization_weight: float
    schema: str = PRIOR_SCHEMA

    def __post_init__(self) -> None:
        if not self.assumptions:
            raise ValueError("GolfPlausibilityPriors.assumptions must be non-empty")
        for field_name, value in (
            ("max_joint_rate_rad_s", self.max_joint_rate_rad_s),
            ("max_joint_accel_rad_s2", self.max_joint_accel_rad_s2),
            ("grip_closure_tol_m", self.grip_closure_tol_m),
            ("effort_regularization_weight", self.effort_regularization_weight),
            ("posture_regularization_weight", self.posture_regularization_weight),
        ):
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{field_name} must be finite and >= 0")

    @classmethod
    def default(cls) -> GolfPlausibilityPriors:
        """Reviewed default priors for club-only candidate ranking."""
        assumptions = (
            PriorAssumption(
                name="fixed_anthropometry_and_club_geometry",
                source="club_models.py catalog + CO-01 club_calibration",
                description=(
                    "Subject segment lengths and club length/type stay fixed for a "
                    "fit; geometry is not free to absorb measurement residual."
                ),
            ),
            PriorAssumption(
                name="joint_rate_and_acceleration_bounds",
                source="reviewed golf ROM literature envelope (named assumption)",
                description=(
                    "Joint rates and accelerations are softly bounded; values are "
                    "regularizers, not measured kinematics."
                ),
            ),
            PriorAssumption(
                name="bilateral_grip_closure",
                source="motion_matching.acceptance.AcceptanceGates.max_closure_residual_m",
                description="Hands remain welded to the grip within closure tolerance.",
            ),
            PriorAssumption(
                name="stance_contact_and_nonpenetration",
                source="AcceptanceGates max_penetration_m / support polygon gates",
                description=(
                    "Foot/ground contact and non-penetration apply when the model "
                    "exposes contact; absent contact state fails closed."
                ),
            ),
            PriorAssumption(
                name="posture_and_effort_regularization",
                source="tour-baselines qualification + effort prior convention",
                description=(
                    "Unobserved null-space motion is pulled toward a golf posture "
                    "prior and low effort; score cannot override measured residual."
                ),
            ),
        )
        return cls(
            assumptions=assumptions,
            max_joint_rate_rad_s=35.0,
            max_joint_accel_rad_s2=800.0,
            grip_closure_tol_m=0.005,
            effort_regularization_weight=0.25,
            posture_regularization_weight=0.35,
        )

    @precondition(
        lambda self, joint_rates_rad_s, joint_accels_rad_s2, grip_closure_m, effort: (
            np.isfinite(grip_closure_m) and np.isfinite(effort)
        ),
        "grip_closure_m and effort must be finite",
    )
    @postcondition(lambda result: 0.0 <= result <= 1.0, "prior score in [0, 1]")
    def score_posture(
        self,
        joint_rates_rad_s: NDArray[np.floating],
        joint_accels_rad_s2: NDArray[np.floating],
        grip_closure_m: float,
        effort: float,
    ) -> float:
        """Return a unitless prior plausibility score in ``[0, 1]``.

        Higher is more plausible under named assumptions. This score never
        constitutes measured body or force evidence.
        """
        rates = np.asarray(joint_rates_rad_s, dtype=np.float64)
        accels = np.asarray(joint_accels_rad_s2, dtype=np.float64)
        if rates.size == 0 or accels.size == 0:
            raise ValueError("joint rate and accel arrays must be non-empty")
        if not np.all(np.isfinite(rates)) or not np.all(np.isfinite(accels)):
            raise ValueError("joint rates and accels must be finite")
        if effort < 0.0:
            raise ValueError("effort must be >= 0")

        rate_pen = float(
            np.mean(np.clip(np.abs(rates) / self.max_joint_rate_rad_s, 0, 2))
        )
        accel_pen = float(
            np.mean(np.clip(np.abs(accels) / self.max_joint_accel_rad_s2, 0, 2))
        )
        closure_pen = float(
            np.clip(abs(grip_closure_m) / max(self.grip_closure_tol_m, 1e-12), 0, 2)
        )
        effort_pen = float(np.clip(effort, 0, 2))
        raw = (
            self.posture_regularization_weight * (rate_pen + accel_pen) / 2.0
            + self.effort_regularization_weight * effort_pen
            + (
                1.0
                - self.posture_regularization_weight
                - self.effort_regularization_weight
            )
            * closure_pen
        )
        return float(np.clip(1.0 - 0.5 * raw, 0.0, 1.0))

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "assumptions": [a.as_dict() for a in self.assumptions],
            "max_joint_rate_rad_s": self.max_joint_rate_rad_s,
            "max_joint_accel_rad_s2": self.max_joint_accel_rad_s2,
            "grip_closure_tol_m": self.grip_closure_tol_m,
            "effort_regularization_weight": self.effort_regularization_weight,
            "posture_regularization_weight": self.posture_regularization_weight,
        }

"""Versioned qualification profiles and numeric gates for tour baselines (TB-02 #10587).

Separates authoritative full-body humanoid gates (G1, G2, G3) from educational reduced-model
baselines (double pendulum, triple pendulum, upper body).
Enforces:
- Missing native replay evidence strictly rejects scientific qualification.
- Synthetic test data strictly rejects product promotion.
- Frozen numeric gates tied to modeled kinematic and dynamic degrees of freedom.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from src.shared.python.contracts import postcondition, precondition
from src.shared.python.tour_baselines.metrics import (
    ProductPromotionStatus,
    ScientificQualificationStatus,
    TourFitMetrics,
)
from src.shared.python.tour_baselines.models import ModelClass
from src.shared.python.tour_baselines.packages import NativeReplayEvidence


class GateStatus(str, Enum):
    """Evaluation status of an individual qualification gate."""

    PASSED = "passed"
    FAILED = "failed"
    MISSING = "missing"


@dataclass(frozen=True)
class GateVerdict:
    """Outcome and measurement for an individual gate."""

    name: str
    status: GateStatus
    threshold: float
    measured: float | None
    unit: str
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "status": self.status.value,
            "threshold": self.threshold,
            "measured": self.measured,
            "unit": self.unit,
            "reason": self.reason,
        }


@dataclass(frozen=True)
class QualificationVerdict:
    """Comprehensive qualification decision for a baseline candidate."""

    is_qualified: bool
    scientific_status: ScientificQualificationStatus
    product_status: ProductPromotionStatus
    gate_version: str
    gates: tuple[GateVerdict, ...]
    reason: str = ""

    def as_dict(self) -> dict[str, Any]:
        return {
            "is_qualified": self.is_qualified,
            "scientific_status": self.scientific_status.value,
            "product_status": self.product_status.value,
            "gate_version": self.gate_version,
            "gates": [g.as_dict() for g in self.gates],
            "reason": self.reason,
        }


class QualificationProfile(ABC):
    """Abstract baseline qualification profile with frozen numeric gates."""

    gate_version: str = "tour-qualification/1.0.0"

    @property
    @abstractmethod
    def model_class(self) -> ModelClass:
        """Declared model class governed by this profile."""
        ...

    @property
    @abstractmethod
    def thresholds(self) -> dict[str, float]:
        """Frozen numeric thresholds for this qualification profile."""
        ...

    @abstractmethod
    def evaluate_gates(self, metrics: TourFitMetrics) -> list[GateVerdict]:
        """Evaluate numeric gates against computed fit metrics."""
        ...


class FullBodyAuthoritativeProfile(QualificationProfile):
    """Authoritative full-body multi-segment ground-support profile (G1/G2/G3)."""

    @property
    def model_class(self) -> ModelClass:
        return ModelClass.FULL_BODY_MECH

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "max_whole_marker_rmse_m": 0.060,  # G3 driver limit
            "max_clubhead_rmse_m": 0.100,
            "min_coverage_fraction": 0.95,
        }

    def evaluate_gates(self, metrics: TourFitMetrics) -> list[GateVerdict]:
        gates: list[GateVerdict] = []
        # Whole marker RMSE
        th_rmse = self.thresholds["max_whole_marker_rmse_m"]
        pass_rmse = metrics.whole_marker_rmse_m <= th_rmse
        gates.append(
            GateVerdict(
                name="whole_marker_rmse",
                status=GateStatus.PASSED if pass_rmse else GateStatus.FAILED,
                threshold=th_rmse,
                measured=metrics.whole_marker_rmse_m,
                unit="m",
                reason="" if pass_rmse else f"Exceeds whole RMSE limit of {th_rmse} m",
            )
        )
        # Clubhead RMSE
        th_club = self.thresholds["max_clubhead_rmse_m"]
        pass_club = metrics.endpoint_clubhead_rmse_m <= th_club
        gates.append(
            GateVerdict(
                name="clubhead_rmse",
                status=GateStatus.PASSED if pass_club else GateStatus.FAILED,
                threshold=th_club,
                measured=metrics.endpoint_clubhead_rmse_m,
                unit="m",
                reason="" if pass_club else f"Exceeds clubhead limit of {th_club} m",
            )
        )
        # Coverage fraction
        th_cov = self.thresholds["min_coverage_fraction"]
        pass_cov = metrics.coverage_fraction >= th_cov
        gates.append(
            GateVerdict(
                name="coverage_fraction",
                status=GateStatus.PASSED if pass_cov else GateStatus.FAILED,
                threshold=th_cov,
                measured=metrics.coverage_fraction,
                unit="fraction",
                reason="" if pass_cov else f"Below coverage limit of {th_cov}",
            )
        )
        return gates


class DoublePendulumPlanarProfile(QualificationProfile):
    """Planar 2-DoF shoulder/wrist swing-plane qualification profile."""

    @property
    def model_class(self) -> ModelClass:
        return ModelClass.DOUBLE_PENDULUM_PLANAR

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "max_in_plane_clubhead_rmse_m": 0.120,  # 120 mm in-plane clubhead gate
            "max_whole_marker_rmse_m": 0.150,
            "min_coverage_fraction": 0.90,
        }

    def evaluate_gates(self, metrics: TourFitMetrics) -> list[GateVerdict]:
        gates: list[GateVerdict] = []
        # In-plane clubhead RMSE
        th_head = self.thresholds["max_in_plane_clubhead_rmse_m"]
        measured_head = metrics.endpoint_clubhead_rmse_m
        pass_head = measured_head <= th_head
        gates.append(
            GateVerdict(
                name="in_plane_clubhead_rmse",
                status=GateStatus.PASSED if pass_head else GateStatus.FAILED,
                threshold=th_head,
                measured=measured_head,
                unit="m",
                reason=(
                    ""
                    if pass_head
                    else f"Clubhead error {measured_head:.4f} m > {th_head} m"
                ),
            )
        )
        # Whole marker RMSE
        th_whole = self.thresholds["max_whole_marker_rmse_m"]
        pass_whole = metrics.whole_marker_rmse_m <= th_whole
        gates.append(
            GateVerdict(
                name="whole_marker_rmse",
                status=GateStatus.PASSED if pass_whole else GateStatus.FAILED,
                threshold=th_whole,
                measured=metrics.whole_marker_rmse_m,
                unit="m",
                reason=(
                    ""
                    if pass_whole
                    else f"Whole error {metrics.whole_marker_rmse_m:.4f} m > {th_whole} m"
                ),
            )
        )
        # Coverage fraction
        th_cov = self.thresholds["min_coverage_fraction"]
        pass_cov = metrics.coverage_fraction >= th_cov
        gates.append(
            GateVerdict(
                name="coverage_fraction",
                status=GateStatus.PASSED if pass_cov else GateStatus.FAILED,
                threshold=th_cov,
                measured=metrics.coverage_fraction,
                unit="fraction",
                reason=(
                    ""
                    if pass_cov
                    else f"Coverage {metrics.coverage_fraction:.3f} < {th_cov}"
                ),
            )
        )
        return gates


class TriplePendulumPlanarProfile(QualificationProfile):
    """Planar 3-DoF trunk/arm/wrist swing-plane qualification profile."""

    @property
    def model_class(self) -> ModelClass:
        return ModelClass.TRIPLE_PENDULUM_PLANAR

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "max_in_plane_clubhead_rmse_m": 0.085,  # 85 mm in-plane clubhead gate
            "max_whole_marker_rmse_m": 0.110,
            "min_coverage_fraction": 0.90,
        }

    def evaluate_gates(self, metrics: TourFitMetrics) -> list[GateVerdict]:
        gates: list[GateVerdict] = []
        th_head = self.thresholds["max_in_plane_clubhead_rmse_m"]
        measured_head = metrics.endpoint_clubhead_rmse_m
        pass_head = measured_head <= th_head
        gates.append(
            GateVerdict(
                name="in_plane_clubhead_rmse",
                status=GateStatus.PASSED if pass_head else GateStatus.FAILED,
                threshold=th_head,
                measured=measured_head,
                unit="m",
                reason=(
                    ""
                    if pass_head
                    else f"Clubhead error {measured_head:.4f} m > {th_head} m"
                ),
            )
        )
        th_whole = self.thresholds["max_whole_marker_rmse_m"]
        pass_whole = metrics.whole_marker_rmse_m <= th_whole
        gates.append(
            GateVerdict(
                name="whole_marker_rmse",
                status=GateStatus.PASSED if pass_whole else GateStatus.FAILED,
                threshold=th_whole,
                measured=metrics.whole_marker_rmse_m,
                unit="m",
                reason=(
                    ""
                    if pass_whole
                    else f"Whole error {metrics.whole_marker_rmse_m:.4f} m > {th_whole} m"
                ),
            )
        )
        th_cov = self.thresholds["min_coverage_fraction"]
        pass_cov = metrics.coverage_fraction >= th_cov
        gates.append(
            GateVerdict(
                name="coverage_fraction",
                status=GateStatus.PASSED if pass_cov else GateStatus.FAILED,
                threshold=th_cov,
                measured=metrics.coverage_fraction,
                unit="fraction",
                reason=(
                    ""
                    if pass_cov
                    else f"Coverage {metrics.coverage_fraction:.3f} < {th_cov}"
                ),
            )
        )
        return gates


class UpperBodyGolferProfile(QualificationProfile):
    """3D 11-DoF upper-body kinematic/torque-driven qualification profile."""

    @property
    def model_class(self) -> ModelClass:
        return ModelClass.UPPER_BODY_GOLFER_3D

    @property
    def thresholds(self) -> dict[str, float]:
        return {
            "max_upper_body_rmse_m": 0.050,  # 50 mm upper body 3D marker RMSE
            "max_clubhead_rmse_m": 0.075,
            "min_coverage_fraction": 0.90,
        }

    def evaluate_gates(self, metrics: TourFitMetrics) -> list[GateVerdict]:
        gates: list[GateVerdict] = []
        th_upper = self.thresholds["max_upper_body_rmse_m"]
        pass_upper = metrics.whole_marker_rmse_m <= th_upper
        gates.append(
            GateVerdict(
                name="upper_body_rmse",
                status=GateStatus.PASSED if pass_upper else GateStatus.FAILED,
                threshold=th_upper,
                measured=metrics.whole_marker_rmse_m,
                unit="m",
                reason=(
                    ""
                    if pass_upper
                    else f"Upper-body error {metrics.whole_marker_rmse_m:.4f} m > {th_upper} m"
                ),
            )
        )
        th_head = self.thresholds["max_clubhead_rmse_m"]
        pass_head = metrics.endpoint_clubhead_rmse_m <= th_head
        gates.append(
            GateVerdict(
                name="clubhead_rmse",
                status=GateStatus.PASSED if pass_head else GateStatus.FAILED,
                threshold=th_head,
                measured=metrics.endpoint_clubhead_rmse_m,
                unit="m",
                reason=(
                    ""
                    if pass_head
                    else f"Clubhead error {metrics.endpoint_clubhead_rmse_m:.4f} m > {th_head} m"
                ),
            )
        )
        th_cov = self.thresholds["min_coverage_fraction"]
        pass_cov = metrics.coverage_fraction >= th_cov
        gates.append(
            GateVerdict(
                name="coverage_fraction",
                status=GateStatus.PASSED if pass_cov else GateStatus.FAILED,
                threshold=th_cov,
                measured=metrics.coverage_fraction,
                unit="fraction",
                reason=(
                    ""
                    if pass_cov
                    else f"Coverage {metrics.coverage_fraction:.3f} < {th_cov}"
                ),
            )
        )
        return gates


def evaluate_qualification(
    profile: QualificationProfile,
    metrics: TourFitMetrics,
    replay: NativeReplayEvidence,
    *,
    is_synthetic_test_data: bool = False,
) -> QualificationVerdict:
    """Evaluate qualification profile, enforcing native replay and non-synthetic rules."""
    gates = tuple(profile.evaluate_gates(metrics))
    all_gates_passed = all(g.status == GateStatus.PASSED for g in gates)

    if not replay.verified_reproduced:
        return QualificationVerdict(
            is_qualified=False,
            scientific_status=ScientificQualificationStatus.UNQUALIFIED,
            product_status=ProductPromotionStatus.REJECTED,
            gate_version=profile.gate_version,
            gates=gates,
            reason="Missing native replay evidence; candidate cannot qualify without verification",
        )

    if not all_gates_passed:
        failed_names = [g.name for g in gates if g.status != GateStatus.PASSED]
        return QualificationVerdict(
            is_qualified=False,
            scientific_status=ScientificQualificationStatus.UNQUALIFIED,
            product_status=ProductPromotionStatus.REJECTED,
            gate_version=profile.gate_version,
            gates=gates,
            reason=f"Failed quantitative gates: {', '.join(failed_names)}",
        )

    # Scientific qualification passes
    if is_synthetic_test_data:
        return QualificationVerdict(
            is_qualified=True,
            scientific_status=ScientificQualificationStatus.QUALIFIED,
            product_status=ProductPromotionStatus.REJECTED,
            gate_version=profile.gate_version,
            gates=gates,
            reason="Synthetic test data cannot be promoted as a tour baseline",
        )

    return QualificationVerdict(
        is_qualified=True,
        scientific_status=ScientificQualificationStatus.QUALIFIED,
        product_status=ProductPromotionStatus.PROMOTED,
        gate_version=profile.gate_version,
        gates=gates,
        reason="All quantitative gates passed and native replay verified",
    )

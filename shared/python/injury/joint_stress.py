"""
Joint Stress Analysis Module

Analyzes stress and loading on all major joints involved in the golf swing,
not just the spine. Each joint has specific injury mechanisms and risk factors.

Target Joints:
- Hip (lead and trail): Internal rotation, impingement risk
- Shoulder: Rotator cuff loading, labral stress
- Elbow: Medial epicondylitis (golfer's elbow) risk
- Wrist: Ulnar deviation stress, TFCC loading
- Knee: Valgus/varus stress during rotation

References:
- McHardy et al. (2006) Golf injuries: a review of the literature
- Gosheger et al. (2003) Injuries and overuse syndromes in golf
- Theriault & Lachance (1998) Golf injuries: an overview
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class JointSide(Enum):
    """Which side of the body (for bilateral joints)."""

    LEAD = "lead"  # Front foot side (left for right-handed)
    TRAIL = "trail"  # Back foot side (right for right-handed)
    BOTH = "both"


class StressType(Enum):
    """Types of joint stress."""

    COMPRESSION = "compression"
    TENSION = "tension"
    SHEAR = "shear"
    TORSION = "torsion"
    IMPINGEMENT = "impingement"


@dataclass
class JointStressResult:
    """Results for a single joint."""

    joint_name: str
    side: JointSide

    # Time series data
    time: np.ndarray = field(default_factory=lambda: np.array([]))

    # Stress components
    compression: np.ndarray = field(default_factory=lambda: np.array([]))
    tension: np.ndarray = field(default_factory=lambda: np.array([]))
    shear: np.ndarray = field(default_factory=lambda: np.array([]))
    torsion: np.ndarray = field(default_factory=lambda: np.array([]))

    # Peak values
    peak_compression: float = 0.0
    peak_shear: float = 0.0
    peak_torsion: float = 0.0

    # ROM utilization (% of maximum ROM used)
    rom_utilization: float = 0.0

    # Risk flags
    impingement_risk: bool = False
    overload_risk: bool = False

    # Risk score (0-100)
    risk_score: float = 0.0


@dataclass
class HipStressMetrics:
    """Hip-specific stress metrics."""

    internal_rotation_max: float = 0.0  # degrees
    external_rotation_max: float = 0.0
    flexion_max: float = 0.0
    extension_max: float = 0.0

    # Femoroacetabular impingement (FAI) risk indicators
    cam_impingement_risk: float = 0.0  # 0-100
    pincer_impingement_risk: float = 0.0

    # Lead vs trail hip asymmetry
    rotation_asymmetry: float = 0.0


@dataclass
class ShoulderStressMetrics:
    """Shoulder-specific stress metrics."""

    # Rotator cuff loading
    supraspinatus_load: float = 0.0
    infraspinatus_load: float = 0.0
    subscapularis_load: float = 0.0

    # Labral stress
    anterior_labral_stress: float = 0.0
    posterior_labral_stress: float = 0.0

    # Impingement indicators
    subacromial_impingement_risk: float = 0.0
    internal_impingement_risk: float = 0.0


@dataclass
class ElbowStressMetrics:
    """Elbow-specific stress metrics (golfer's elbow risk)."""

    # Medial epicondyle loading (golfer's elbow)
    medial_epicondyle_stress: float = 0.0
    valgus_torque_max: float = 0.0  # Nm

    # Lateral epicondyle (tennis elbow - less common in golf)
    lateral_epicondyle_stress: float = 0.0

    # Risk score for medial epicondylitis
    golfers_elbow_risk: float = 0.0  # 0-100


@dataclass
class WristStressMetrics:
    """Wrist-specific stress metrics."""

    # Ulnar deviation stress
    ulnar_deviation_max: float = 0.0  # degrees
    radial_deviation_max: float = 0.0

    # TFCC (Triangular Fibrocartilage Complex) loading
    tfcc_compression: float = 0.0
    tfcc_shear: float = 0.0

    # Hook of hamate stress (common fracture site in golf)
    hamate_stress: float = 0.0

    # Risk scores
    tfcc_injury_risk: float = 0.0
    hamate_fracture_risk: float = 0.0


class JointStressAnalyzer:
    """
    Comprehensive joint stress analysis for the golf swing.

    Analyzes loading patterns on all major joints to identify injury risks
    beyond just the lumbar spine. Each joint has specific biomechanical
    considerations based on golf-specific injury literature.

    Example:
        >>> analyzer = JointStressAnalyzer(body_weight=80.0, handedness='right')
        >>> results = analyzer.analyze_all_joints(joint_angles, joint_velocities, joint_torques, time)
        >>> for joint, result in results.items():
        ...     print(f"{joint}: Risk score = {result.risk_score:.0f}")
    """

    # Joint-specific injury thresholds (from literature)
    HIP_INTERNAL_ROTATION_LIMIT = 45.0  # degrees
    SHOULDER_ELEVATION_LIMIT = 170.0  # degrees
    ELBOW_VALGUS_TORQUE_LIMIT = 35.0  # Nm
    WRIST_ULNAR_DEVIATION_LIMIT = 35.0  # degrees

    def __init__(
        self,
        body_weight: float,
        handedness: str = "right",
        height: float | None = None,
    ):
        """
        Initialize the joint stress analyzer.

        Args:
            body_weight: Body weight in kg
            handedness: 'right' or 'left' handed golfer
            height: Height in m (optional)
        """
        self.body_weight = body_weight
        self.body_weight_N = body_weight * 9.81
        self.handedness = handedness
        self.height = height

        # Lead/trail side mapping
        self.lead_side = "left" if handedness == "right" else "right"
        self.trail_side = "right" if handedness == "right" else "left"

    def analyze_all_joints(
        self,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
    ) -> dict[str, JointStressResult]:
        """
        Analyze stress on all major joints.

        Args:
            joint_angles: Dictionary of joint angle time series
            joint_velocities: Dictionary of joint velocity time series
            joint_torques: Dictionary of joint torque time series
            time: Time array

        Returns:
            Dictionary mapping joint names to JointStressResult
        """
        results = {}

        # Analyze each joint pair
        results["hip_lead"] = self.analyze_hip(
            joint_angles, joint_velocities, joint_torques, time, JointSide.LEAD
        )
        results["hip_trail"] = self.analyze_hip(
            joint_angles, joint_velocities, joint_torques, time, JointSide.TRAIL
        )

        results["shoulder_lead"] = self.analyze_shoulder(
            joint_angles, joint_velocities, joint_torques, time, JointSide.LEAD
        )
        results["shoulder_trail"] = self.analyze_shoulder(
            joint_angles, joint_velocities, joint_torques, time, JointSide.TRAIL
        )

        results["elbow_lead"] = self.analyze_elbow(
            joint_angles, joint_velocities, joint_torques, time, JointSide.LEAD
        )
        results["elbow_trail"] = self.analyze_elbow(
            joint_angles, joint_velocities, joint_torques, time, JointSide.TRAIL
        )

        results["wrist_lead"] = self.analyze_wrist(
            joint_angles, joint_velocities, joint_torques, time, JointSide.LEAD
        )
        results["wrist_trail"] = self.analyze_wrist(
            joint_angles, joint_velocities, joint_torques, time, JointSide.TRAIL
        )

        return results

    def analyze_hip(
        self,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
        side: JointSide,
    ) -> JointStressResult:
        """Analyze hip joint stress."""
        result = JointStressResult(joint_name="hip", side=side, time=time)
        n_frames = len(time)

        # Get angle data (use generic if side-specific not available)
        prefix = "hip_lead" if side == JointSide.LEAD else "hip_trail"
        rotation = joint_angles.get(
            f"{prefix}_rotation", joint_angles.get("hip_rotation", np.zeros(n_frames))
        )
        flexion = joint_angles.get(
            f"{prefix}_flexion", joint_angles.get("hip_flexion", np.zeros(n_frames))
        )

        # Compute internal rotation range
        rotation_deg = np.degrees(rotation)
        max_internal = np.max(rotation_deg)
        max_external = np.min(rotation_deg)

        # Hip impingement risk increases with high internal rotation + flexion
        flexion_deg = np.degrees(flexion)
        impingement_indicator = max_internal + np.max(flexion_deg) * 0.5

        # Risk assessment
        if impingement_indicator > 100:
            result.impingement_risk = True
            result.risk_score = min(100, impingement_indicator - 50)
        else:
            result.risk_score = max(0, impingement_indicator - 50)

        # Lead hip typically has higher internal rotation demand
        if side == JointSide.LEAD:
            result.risk_score *= 1.2

        result.rom_utilization = min(100, abs(max_internal - max_external) / 90 * 100)

        return result

    def analyze_shoulder(
        self,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
        side: JointSide,
    ) -> JointStressResult:
        """Analyze shoulder joint stress."""
        result = JointStressResult(joint_name="shoulder", side=side, time=time)
        n_frames = len(time)

        # Get angle and torque data
        horizontal = joint_angles.get("shoulder_horizontal", np.zeros(n_frames))
        joint_angles.get("shoulder_vertical", np.zeros(n_frames))

        # Get torques
        torque = joint_torques.get("shoulder_horizontal", np.zeros(n_frames))

        # Compute shear and torsion from velocities and torques
        velocity = joint_velocities.get("shoulder_horizontal", np.zeros(n_frames))

        # Peak values
        result.peak_torsion = np.max(np.abs(torque))

        # Rotator cuff loading estimate (simplified)
        # High-velocity rotation with high torque indicates high RC loading
        rc_loading = np.abs(velocity) * np.abs(torque) / 100

        # Risk assessment
        max_rc_loading = np.max(rc_loading)
        if max_rc_loading > 50:
            result.overload_risk = True

        # Trail shoulder has higher stress in most swings
        multiplier = 1.2 if side == JointSide.TRAIL else 1.0
        result.risk_score = min(100, max_rc_loading * multiplier)

        result.rom_utilization = min(
            100, np.max(np.abs(np.degrees(horizontal))) / 180 * 100
        )

        return result

    def analyze_elbow(
        self,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
        side: JointSide,
    ) -> JointStressResult:
        """Analyze elbow joint stress (golfer's elbow risk)."""
        result = JointStressResult(joint_name="elbow", side=side, time=time)
        n_frames = len(time)

        # Get flexion angle and torque
        flexion = joint_angles.get("elbow_flexion", np.zeros(n_frames))
        torque = joint_torques.get("elbow_flexion", np.zeros(n_frames))

        # Wrist flexor loading contributes to medial epicondyle stress
        wrist_torque = joint_torques.get("wrist_cock", np.zeros(n_frames))

        # Valgus torque estimate (simplified)
        # In golf, this relates to the "casting" motion
        valgus_estimate = np.abs(torque) * 0.3 + np.abs(wrist_torque) * 0.5

        result.peak_torsion = np.max(valgus_estimate)

        # Golfer's elbow risk assessment
        # Lead elbow typically at higher risk
        max_valgus = np.max(valgus_estimate)
        if max_valgus > self.ELBOW_VALGUS_TORQUE_LIMIT:
            result.overload_risk = True

        multiplier = 1.3 if side == JointSide.LEAD else 1.0
        result.risk_score = min(
            100, (max_valgus / self.ELBOW_VALGUS_TORQUE_LIMIT) * 50 * multiplier
        )

        result.rom_utilization = min(100, np.max(np.degrees(flexion)) / 145 * 100)

        return result

    def analyze_wrist(
        self,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
        side: JointSide,
    ) -> JointStressResult:
        """Analyze wrist joint stress (TFCC and hamate risk)."""
        result = JointStressResult(joint_name="wrist", side=side, time=time)
        n_frames = len(time)

        # Get wrist angles
        cock = joint_angles.get(
            "wrist_cock", np.zeros(n_frames)
        )  # Ulnar/radial deviation
        rotation = joint_angles.get("wrist_rotation", np.zeros(n_frames))

        # Get velocity for impact stress estimation
        velocity = joint_velocities.get("wrist_cock", np.zeros(n_frames))

        # Ulnar deviation in degrees
        ulnar_dev = np.degrees(cock)
        max_ulnar = np.max(ulnar_dev)

        # TFCC stress increases with ulnar deviation + rotation + impact forces
        tfcc_stress = np.abs(ulnar_dev) * np.abs(np.degrees(rotation)) / 100

        # Hamate stress relates to grip pressure at impact
        # Approximated by high velocity + high deviation
        hamate_stress = np.abs(velocity) * np.abs(ulnar_dev) / 100

        result.peak_shear = np.max(tfcc_stress)

        # Risk assessment
        # Lead wrist typically at higher risk for hamate fracture
        # Trail wrist at higher risk for TFCC injury
        if side == JointSide.LEAD:
            hamate_mult = 1.5
            tfcc_mult = 1.0
        else:
            hamate_mult = 1.0
            tfcc_mult = 1.3

        hamate_risk = np.max(hamate_stress) * hamate_mult
        tfcc_risk = np.max(tfcc_stress) * tfcc_mult

        result.risk_score = min(100, max(hamate_risk, tfcc_risk))

        if max_ulnar > self.WRIST_ULNAR_DEVIATION_LIMIT:
            result.overload_risk = True

        result.rom_utilization = min(100, abs(max_ulnar) / 45 * 100)

        return result

    def get_summary(self, results: dict[str, JointStressResult]) -> dict[str, object]:
        """Get a summary of all joint stress results."""
        summary: dict[str, object] = {
            "highest_risk_joint": "",
            "highest_risk_score": 0.0,
            "joints_at_risk": [],
            "total_risk_score": 0.0,
            "recommendations": [],
        }

        risk_scores = []
        for name, result in results.items():
            risk_scores.append((name, result.risk_score))
            if result.risk_score > float(summary["highest_risk_score"]):  # type: ignore[arg-type]
                summary["highest_risk_score"] = result.risk_score
                summary["highest_risk_joint"] = name
            if result.risk_score > 50:
                assert isinstance(summary["joints_at_risk"], list)
                summary["joints_at_risk"].append(name)
            if result.impingement_risk:
                assert isinstance(summary["recommendations"], list)
                summary["recommendations"].append(
                    f"{name}: Impingement risk detected - consider ROM exercises"
                )
            if result.overload_risk:
                assert isinstance(summary["recommendations"], list)
                summary["recommendations"].append(
                    f"{name}: Overload risk - reduce intensity or modify technique"
                )

        summary["total_risk_score"] = sum(score for _, score in risk_scores) / len(
            risk_scores
        )

        return summary


if __name__ == "__main__":
    # Example usage
    analyzer = JointStressAnalyzer(body_weight=80.0, handedness="right")

    # Generate example data
    time = np.linspace(0, 1.0, 100)
    n = len(time)

    joint_angles = {
        "hip_rotation": np.sin(2 * np.pi * time) * 0.5,
        "hip_flexion": np.cos(2 * np.pi * time) * 0.3,
        "shoulder_horizontal": np.sin(2 * np.pi * time) * 1.2,
        "shoulder_vertical": np.cos(2 * np.pi * time) * 0.8,
        "elbow_flexion": np.sin(2 * np.pi * time) * 1.5 + 1.0,
        "wrist_cock": np.sin(4 * np.pi * time) * 0.5,
        "wrist_rotation": np.cos(4 * np.pi * time) * 0.3,
    }

    dt = time[1] - time[0]
    joint_velocities = {k: np.gradient(v, dt) for k, v in joint_angles.items()}
    joint_torques = {k: 30 * np.gradient(v, dt) for k, v in joint_velocities.items()}

    # Analyze all joints
    results = analyzer.analyze_all_joints(
        joint_angles, joint_velocities, joint_torques, time
    )

    for _name, _result in results.items():
        pass

    summary = analyzer.get_summary(results)
    recommendations = summary["recommendations"]
    assert isinstance(recommendations, list)
    for _rec in recommendations:
        pass

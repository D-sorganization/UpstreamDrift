"""
Spinal Load Analysis Module

Calculates forces on the lumbar spine during the golf swing to assess injury risk.
Based on peer-reviewed biomechanics research on golf-related low back pain.

Key Metrics:
- Axial compression (vertical loading)
- Anterior-posterior shear
- Lateral shear (side bending load)
- Axial torsion (rotational torque)
- X-factor stretch (pelvis-thorax separation)
- Crunch factor (combined lateral bend + rotation)

Risk Thresholds (from literature):
- L4-L5 Compression: Safe <4x BW, Caution 4-6x BW, High Risk >6x BW
- Lateral Shear: Safe <0.5x BW, Caution 0.5-1x BW, High Risk >1x BW
- X-Factor Stretch: Safe <45°, Caution 45-55°, High Risk >55°
- Transition Time: Safe >0.3s, Caution 0.2-0.3s, High Risk <0.2s

References:
- Hosea et al. (1990) measured up to 8x body weight compression
- McHardy & Pollard (2005) identified crunch factor in modern swing
- Lindsay et al. (2002) review of spine loading mechanisms
"""

from dataclasses import dataclass, field
from enum import Enum

import numpy as np


class SpinalRiskLevel(Enum):
    """Risk level categories for spinal loading."""

    SAFE = "safe"
    CAUTION = "caution"
    HIGH_RISK = "high_risk"
    CRITICAL = "critical"


@dataclass
class SpinalSegment:
    """Represents a single spinal segment (vertebral pair)."""

    name: str  # e.g., "L4-L5", "L5-S1"
    compression: np.ndarray = field(default_factory=lambda: np.array([]))  # N
    ap_shear: np.ndarray = field(
        default_factory=lambda: np.array([])
    )  # N (anterior-posterior)
    lateral_shear: np.ndarray = field(default_factory=lambda: np.array([]))  # N
    torsion: np.ndarray = field(default_factory=lambda: np.array([]))  # Nm


@dataclass
class XFactorMetrics:
    """Metrics related to pelvis-thorax separation (X-factor)."""

    x_factor_angle: np.ndarray  # degrees, time series
    x_factor_stretch: float  # maximum separation angle (degrees)
    x_factor_stretch_time: float  # time of maximum stretch (s)
    separation_rate: float  # rate of change at transition (deg/s)
    transition_duration: float  # time from top to impact (s)


@dataclass
class CrunchFactorMetrics:
    """Metrics quantifying the 'crunch factor' (lateral bend during rotation)."""

    lateral_bend_angle: np.ndarray  # degrees, time series
    rotation_angle: np.ndarray  # degrees, time series
    crunch_factor: np.ndarray  # combined metric, time series
    peak_crunch: float  # maximum crunch factor value
    peak_crunch_time: float  # time of peak crunch (s)
    asymmetry_ratio: float  # left vs right side loading ratio


@dataclass
class SpinalLoadResult:
    """Complete spinal load analysis results."""

    # Time array
    time: np.ndarray

    # Per-segment loading
    segments: dict[str, SpinalSegment] = field(default_factory=dict)

    # X-factor metrics
    x_factor: XFactorMetrics | None = None

    # Crunch factor metrics
    crunch_factor: CrunchFactorMetrics | None = None

    # Peak values (normalized to body weight)
    peak_compression_bw: float = 0.0  # multiples of body weight
    peak_ap_shear_bw: float = 0.0
    peak_lateral_shear_bw: float = 0.0
    peak_torsion: float = 0.0  # Nm

    # Risk assessment
    compression_risk: SpinalRiskLevel = SpinalRiskLevel.SAFE
    shear_risk: SpinalRiskLevel = SpinalRiskLevel.SAFE
    x_factor_risk: SpinalRiskLevel = SpinalRiskLevel.SAFE
    overall_risk: SpinalRiskLevel = SpinalRiskLevel.SAFE

    # Cumulative load (for tracking over sessions)
    cumulative_compression_impulse: float = 0.0  # N*s
    cumulative_shear_impulse: float = 0.0  # N*s


class SpinalLoadAnalyzer:
    """
    Analyzes forces on the lumbar spine during the golf swing.

    This analyzer computes compression, shear, and torsion loads on spinal
    segments using inverse dynamics and anatomical models. Results are
    compared against evidence-based injury risk thresholds.

    Example:
        >>> analyzer = SpinalLoadAnalyzer(body_weight=80.0)
        >>> result = analyzer.analyze(joint_angles, joint_velocities, joint_torques)
        >>> print(f"Peak compression: {result.peak_compression_bw:.1f}x body weight")
        >>> print(f"Overall risk: {result.overall_risk.value}")
    """

    # Risk thresholds (based on literature)
    COMPRESSION_SAFE = 4.0  # x body weight
    COMPRESSION_CAUTION = 6.0
    COMPRESSION_HIGH = 8.0

    SHEAR_SAFE = 0.5  # x body weight
    SHEAR_CAUTION = 1.0
    SHEAR_HIGH = 1.5

    X_FACTOR_SAFE = 45.0  # degrees
    X_FACTOR_CAUTION = 55.0
    X_FACTOR_HIGH = 65.0

    TRANSITION_SAFE = 0.30  # seconds
    TRANSITION_CAUTION = 0.20
    TRANSITION_HIGH = 0.15

    def __init__(
        self,
        body_weight: float,
        height: float | None = None,
        trunk_length: float | None = None,
        lumbar_segments: list[str] | None = None,
    ):
        """
        Initialize the spinal load analyzer.

        Args:
            body_weight: Body weight in kg
            height: Height in m (optional, for estimating segment lengths)
            trunk_length: Trunk length in m (optional, overrides height estimate)
            lumbar_segments: List of segment names to analyze (default: L3-L4 to L5-S1)
        """
        self.body_weight = body_weight
        self.body_weight_N = body_weight * 9.81  # Convert to Newtons

        self.height = height
        self.trunk_length = trunk_length or (0.288 * height if height else 0.50)

        self.lumbar_segments = lumbar_segments or ["L3-L4", "L4-L5", "L5-S1"]

        # Segment mass ratios (from anthropometric data)
        self._head_arms_trunk_mass_ratio = 0.678  # HAT segment mass / total mass
        self._lumbar_position_ratios = {
            "L3-L4": 0.40,  # Position along trunk (0 = pelvis, 1 = shoulders)
            "L4-L5": 0.30,
            "L5-S1": 0.20,
        }

    def analyze(
        self,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
        pelvis_angles: np.ndarray | None = None,
        thorax_angles: np.ndarray | None = None,
    ) -> SpinalLoadResult:
        """
        Perform complete spinal load analysis.

        Args:
            joint_angles: Dictionary of joint angle time series (radians)
                Required keys: 'lumbar_flexion', 'lumbar_lateral', 'lumbar_rotation'
            joint_velocities: Dictionary of joint angular velocity time series (rad/s)
            joint_torques: Dictionary of joint torque time series (Nm)
            time: Time array (seconds)
            pelvis_angles: Optional pelvis orientation time series [roll, pitch, yaw]
            thorax_angles: Optional thorax orientation time series [roll, pitch, yaw]

        Returns:
            SpinalLoadResult containing all computed metrics and risk assessments
        """
        result = SpinalLoadResult(time=time)
        n_frames = len(time)

        # Validate input shapes
        for name, arr in joint_angles.items():
            if len(arr) != n_frames:
                raise ValueError(
                    f"Joint angle '{name}' length ({len(arr)}) "
                    f"does not match time length ({n_frames})"
                )
        for name, arr in joint_velocities.items():
            if len(arr) != n_frames:
                raise ValueError(
                    f"Joint velocity '{name}' length ({len(arr)}) "
                    f"does not match time length ({n_frames})"
                )
        for name, arr in joint_torques.items():
            if len(arr) != n_frames:
                raise ValueError(
                    f"Joint torque '{name}' length ({len(arr)}) "
                    f"does not match time length ({n_frames})"
                )

        # Calculate loads on each spinal segment
        for segment_name in self.lumbar_segments:
            segment = self._compute_segment_loads(
                segment_name, joint_angles, joint_velocities, joint_torques, time
            )
            result.segments[segment_name] = segment

        # Calculate X-factor metrics if pelvis/thorax data available
        if pelvis_angles is not None and thorax_angles is not None:
            result.x_factor = self._compute_x_factor(pelvis_angles, thorax_angles, time)

        # Calculate crunch factor
        if "lumbar_lateral" in joint_angles and "lumbar_rotation" in joint_angles:
            result.crunch_factor = self._compute_crunch_factor(
                joint_angles["lumbar_lateral"], joint_angles["lumbar_rotation"], time
            )

        # Extract peak values
        result = self._compute_peak_values(result)

        # Assess risk levels
        result = self._assess_risk(result)

        # Compute cumulative loads
        result = self._compute_cumulative_loads(result, time)

        return result

    def _compute_segment_loads(
        self,
        segment_name: str,
        joint_angles: dict[str, np.ndarray],
        joint_velocities: dict[str, np.ndarray],
        joint_torques: dict[str, np.ndarray],
        time: np.ndarray,
    ) -> SpinalSegment:
        """Compute forces on a single spinal segment."""
        segment = SpinalSegment(name=segment_name)
        n_frames = len(time)

        # Get position ratio for this segment
        pos_ratio = self._lumbar_position_ratios.get(segment_name, 0.3)

        # Mass above this segment (head, arms, upper trunk)
        mass_above = (
            self.body_weight * self._head_arms_trunk_mass_ratio * (1 - pos_ratio)
        )
        weight_above = mass_above * 9.81

        # Initialize arrays
        compression = np.zeros(n_frames)
        ap_shear = np.zeros(n_frames)
        lateral_shear = np.zeros(n_frames)
        torsion = np.zeros(n_frames)

        # Get angle arrays (convert to degrees for calculations, use radians input)
        flexion = joint_angles.get("lumbar_flexion", np.zeros(n_frames))
        lateral = joint_angles.get("lumbar_lateral", np.zeros(n_frames))
        joint_angles.get("lumbar_rotation", np.zeros(n_frames))

        # Get velocity arrays
        joint_velocities.get("lumbar_flexion", np.zeros(n_frames))
        joint_velocities.get("lumbar_lateral", np.zeros(n_frames))
        rotation_vel = joint_velocities.get("lumbar_rotation", np.zeros(n_frames))

        # Get torque arrays
        flexion_torque = joint_torques.get("lumbar_flexion", np.zeros(n_frames))
        joint_torques.get("lumbar_lateral", np.zeros(n_frames))
        rotation_torque = joint_torques.get("lumbar_rotation", np.zeros(n_frames))

        for i in range(n_frames):
            # Compression from body weight (modified by trunk angle)
            gravity_compression = weight_above * np.cos(flexion[i])

            # Additional compression from muscle forces (estimated from torques)
            # Muscle forces create compression as they pull across the joint
            # Approximate muscle force from torque using moment arm (~5cm)
            moment_arm = 0.05  # meters
            muscle_compression = np.abs(flexion_torque[i]) / moment_arm

            compression[i] = gravity_compression + muscle_compression

            # A-P shear from gravity (when trunk flexed forward)
            ap_shear[i] = weight_above * np.sin(flexion[i])

            # Lateral shear from lateral bending
            lateral_shear[i] = weight_above * np.sin(lateral[i])

            # Additional shear from centrifugal effects during rotation
            # F = m * omega^2 * r
            rotation_omega = rotation_vel[i]
            lever_arm = self.trunk_length * pos_ratio
            centrifugal_shear = mass_above * (rotation_omega**2) * lever_arm
            lateral_shear[i] += centrifugal_shear * 0.5  # Partially contributes

            # Torsion from rotation torque
            torsion[i] = np.abs(rotation_torque[i])

        segment.compression = compression
        segment.ap_shear = ap_shear
        segment.lateral_shear = lateral_shear
        segment.torsion = torsion

        return segment

    def _compute_x_factor(
        self,
        pelvis_angles: np.ndarray,
        thorax_angles: np.ndarray,
        time: np.ndarray,
    ) -> XFactorMetrics:
        """
        Compute X-factor (pelvis-thorax separation) metrics.

        The X-factor is the difference in transverse plane rotation between
        the pelvis and thorax. X-factor stretch is the maximum separation
        during the transition from backswing to downswing.
        """
        # Assuming angles are [roll, pitch, yaw] where yaw is rotation
        pelvis_rotation = (
            pelvis_angles[:, 2] if pelvis_angles.ndim > 1 else pelvis_angles
        )
        thorax_rotation = (
            thorax_angles[:, 2] if thorax_angles.ndim > 1 else thorax_angles
        )

        # X-factor is thorax rotation minus pelvis rotation
        x_factor_angle = np.degrees(thorax_rotation - pelvis_rotation)

        # Find peak (maximum separation)
        peak_idx = np.argmax(np.abs(x_factor_angle))
        x_factor_stretch = np.abs(x_factor_angle[peak_idx])
        x_factor_stretch_time = time[peak_idx]

        # Calculate rate of change (derivative)
        dt = np.mean(np.diff(time)) if len(time) > 1 else 0.001
        x_factor_rate = np.gradient(x_factor_angle, dt)
        separation_rate = np.max(np.abs(x_factor_rate))

        # Estimate transition duration (from peak X-factor to impact)
        # Impact is typically when club is vertical (near end of motion)
        # Approximate as time from peak to end
        transition_duration = (
            time[-1] - x_factor_stretch_time if len(time) > peak_idx else 0.25
        )

        return XFactorMetrics(
            x_factor_angle=x_factor_angle,
            x_factor_stretch=x_factor_stretch,
            x_factor_stretch_time=x_factor_stretch_time,
            separation_rate=separation_rate,
            transition_duration=transition_duration,
        )

    def _compute_crunch_factor(
        self,
        lateral_bend: np.ndarray,
        rotation: np.ndarray,
        time: np.ndarray,
    ) -> CrunchFactorMetrics:
        """
        Compute crunch factor metrics.

        The crunch factor quantifies the combined effect of lateral bending
        and rotation, which creates asymmetric loading on the spine. This
        is associated with increased injury risk in the modern golf swing.
        """
        lateral_deg = np.degrees(lateral_bend)
        rotation_deg = np.degrees(rotation)

        # Crunch factor = lateral bend * rotation (simplified model)
        # Higher values indicate more asymmetric loading
        crunch_factor = np.abs(lateral_deg) * np.abs(rotation_deg) / 100.0

        peak_crunch = np.max(crunch_factor)
        peak_idx = np.argmax(crunch_factor)
        peak_crunch_time = time[peak_idx]

        # Asymmetry ratio (compare loading on each side)
        # Positive lateral bend = right side compression (for right-handed)
        right_side_load = np.sum(np.maximum(lateral_deg, 0) ** 2)
        left_side_load = np.sum(np.maximum(-lateral_deg, 0) ** 2)
        asymmetry_ratio = right_side_load / (left_side_load + 1e-6)

        return CrunchFactorMetrics(
            lateral_bend_angle=lateral_deg,
            rotation_angle=rotation_deg,
            crunch_factor=crunch_factor,
            peak_crunch=peak_crunch,
            peak_crunch_time=peak_crunch_time,
            asymmetry_ratio=asymmetry_ratio,
        )

    def _compute_peak_values(self, result: SpinalLoadResult) -> SpinalLoadResult:
        """Extract peak values normalized to body weight."""
        max_compression = 0.0
        max_ap_shear = 0.0
        max_lateral_shear = 0.0
        max_torsion = 0.0

        for segment in result.segments.values():
            if len(segment.compression) > 0:
                max_compression = max(max_compression, np.max(segment.compression))
            if len(segment.ap_shear) > 0:
                max_ap_shear = max(max_ap_shear, np.max(np.abs(segment.ap_shear)))
            if len(segment.lateral_shear) > 0:
                max_lateral_shear = max(
                    max_lateral_shear, np.max(np.abs(segment.lateral_shear))
                )
            if len(segment.torsion) > 0:
                max_torsion = max(max_torsion, np.max(segment.torsion))

        result.peak_compression_bw = max_compression / self.body_weight_N
        result.peak_ap_shear_bw = max_ap_shear / self.body_weight_N
        result.peak_lateral_shear_bw = max_lateral_shear / self.body_weight_N
        result.peak_torsion = max_torsion

        return result

    def _assess_risk(self, result: SpinalLoadResult) -> SpinalLoadResult:
        """Assess risk levels based on computed values."""
        # Compression risk
        if result.peak_compression_bw >= self.COMPRESSION_HIGH:
            result.compression_risk = SpinalRiskLevel.CRITICAL
        elif result.peak_compression_bw >= self.COMPRESSION_CAUTION:
            result.compression_risk = SpinalRiskLevel.HIGH_RISK
        elif result.peak_compression_bw >= self.COMPRESSION_SAFE:
            result.compression_risk = SpinalRiskLevel.CAUTION
        else:
            result.compression_risk = SpinalRiskLevel.SAFE

        # Shear risk (use max of AP and lateral)
        max_shear = max(result.peak_ap_shear_bw, result.peak_lateral_shear_bw)
        if max_shear >= self.SHEAR_HIGH:
            result.shear_risk = SpinalRiskLevel.CRITICAL
        elif max_shear >= self.SHEAR_CAUTION:
            result.shear_risk = SpinalRiskLevel.HIGH_RISK
        elif max_shear >= self.SHEAR_SAFE:
            result.shear_risk = SpinalRiskLevel.CAUTION
        else:
            result.shear_risk = SpinalRiskLevel.SAFE

        # X-factor risk
        if result.x_factor is not None:
            x_stretch = result.x_factor.x_factor_stretch
            trans_time = result.x_factor.transition_duration

            if x_stretch >= self.X_FACTOR_HIGH or trans_time <= self.TRANSITION_HIGH:
                result.x_factor_risk = SpinalRiskLevel.CRITICAL
            elif (
                x_stretch >= self.X_FACTOR_CAUTION
                or trans_time <= self.TRANSITION_CAUTION
            ):
                result.x_factor_risk = SpinalRiskLevel.HIGH_RISK
            elif x_stretch >= self.X_FACTOR_SAFE or trans_time <= self.TRANSITION_SAFE:
                result.x_factor_risk = SpinalRiskLevel.CAUTION
            else:
                result.x_factor_risk = SpinalRiskLevel.SAFE

        # Overall risk (highest of all categories)
        risk_levels = [
            result.compression_risk,
            result.shear_risk,
            result.x_factor_risk,
        ]
        risk_order = [
            SpinalRiskLevel.SAFE,
            SpinalRiskLevel.CAUTION,
            SpinalRiskLevel.HIGH_RISK,
            SpinalRiskLevel.CRITICAL,
        ]
        max_risk_idx = max(risk_order.index(r) for r in risk_levels)
        result.overall_risk = risk_order[max_risk_idx]

        return result

    def _compute_cumulative_loads(
        self, result: SpinalLoadResult, time: np.ndarray
    ) -> SpinalLoadResult:
        """Compute cumulative load impulses for tracking over time."""
        dt = np.mean(np.diff(time)) if len(time) > 1 else 0.001

        total_compression_impulse = 0.0
        total_shear_impulse = 0.0

        for segment in result.segments.values():
            if len(segment.compression) > 0:
                # Handle NumPy 2.0 deprecation of trapz
                if hasattr(np, "trapezoid"):
                    total_compression_impulse += float(
                        np.trapezoid(segment.compression, dx=float(dt))
                    )
                else:
                    trapz_func = getattr(np, "trapz")  # noqa: B009
                    total_compression_impulse += float(
                        trapz_func(segment.compression, dx=float(dt))
                    )
            if len(segment.lateral_shear) > 0:
                # Handle NumPy 2.0 deprecation of trapz
                if hasattr(np, "trapezoid"):
                    total_shear_impulse += float(
                        np.trapezoid(np.abs(segment.lateral_shear), dx=float(dt))
                    )
                else:
                    trapz_func = getattr(np, "trapz")  # noqa: B009
                    total_shear_impulse += float(
                        trapz_func(np.abs(segment.lateral_shear), dx=float(dt))
                    )

        result.cumulative_compression_impulse = total_compression_impulse
        result.cumulative_shear_impulse = total_shear_impulse

        return result

    def get_recommendations(self, result: SpinalLoadResult) -> list[str]:
        """
        Generate recommendations based on analysis results.

        Args:
            result: SpinalLoadResult from analyze()

        Returns:
            List of recommendation strings
        """
        recommendations = []

        if result.overall_risk == SpinalRiskLevel.SAFE:
            recommendations.append(
                "Spinal loads are within safe limits. Maintain current technique."
            )
            return recommendations

        # Compression recommendations
        if result.compression_risk in [
            SpinalRiskLevel.HIGH_RISK,
            SpinalRiskLevel.CRITICAL,
        ]:
            recommendations.append(
                f"High spinal compression detected ({result.peak_compression_bw:.1f}x body weight). "
                "Consider: reducing trunk flexion angle, strengthening core muscles, "
                "or using a more upright posture."
            )

        # Shear recommendations
        if result.shear_risk in [SpinalRiskLevel.HIGH_RISK, SpinalRiskLevel.CRITICAL]:
            recommendations.append(
                f"High lateral shear forces detected ({result.peak_lateral_shear_bw:.1f}x body weight). "
                "Consider: reducing lateral bend during downswing, using a 'stabilized spine' "
                "technique, or widening stance for better stability."
            )

        # X-factor recommendations
        if result.x_factor_risk in [
            SpinalRiskLevel.HIGH_RISK,
            SpinalRiskLevel.CRITICAL,
        ]:
            if (
                result.x_factor
                and result.x_factor.x_factor_stretch > self.X_FACTOR_CAUTION
            ):
                recommendations.append(
                    f"High X-factor stretch ({result.x_factor.x_factor_stretch:.0f}°). "
                    "Consider: allowing more hip turn during backswing, reducing shoulder turn, "
                    "or using a 'classic' swing pattern with less pelvis-thorax separation."
                )
            if (
                result.x_factor
                and result.x_factor.transition_duration < self.TRANSITION_CAUTION
            ):
                recommendations.append(
                    f"Rapid transition ({result.x_factor.transition_duration:.2f}s). "
                    "Consider: slowing the transition from backswing to downswing, "
                    "focusing on rhythm and tempo rather than maximum speed."
                )

        # Crunch factor recommendations
        if result.crunch_factor and result.crunch_factor.peak_crunch > 20:
            recommendations.append(
                f"High crunch factor detected (peak: {result.crunch_factor.peak_crunch:.1f}). "
                "This indicates combined lateral bending and rotation stress. "
                "Consider: maintaining a more neutral spine angle during rotation, "
                "or exploring the 'stack and tilt' technique to reduce lateral movement."
            )

        # Asymmetry recommendations
        if result.crunch_factor and result.crunch_factor.asymmetry_ratio > 3.0:
            recommendations.append(
                f"Significant asymmetric loading (ratio: {result.crunch_factor.asymmetry_ratio:.1f}:1). "
                "Consider: balanced training exercises, counter-rotation stretches, "
                "or practicing left-handed swings to balance muscle development."
            )

        return recommendations


def create_example_analysis() -> tuple[SpinalLoadAnalyzer, SpinalLoadResult]:
    """Create an example analysis for testing and demonstration."""
    # Create analyzer for 80kg golfer
    analyzer = SpinalLoadAnalyzer(body_weight=80.0, height=1.80)

    # Generate example data (1 second swing at 100 Hz)
    time = np.linspace(0, 1.0, 100)

    # Example joint angles (simplified swing pattern)
    # Backswing: increase rotation and slight lateral bend
    # Downswing: rapid rotation with lateral bend
    t_top = 0.4  # Time at top of backswing

    lumbar_flexion = np.where(
        time < t_top, 0.2 * np.sin(np.pi * time / t_top), 0.3 * np.ones_like(time)
    )

    lumbar_lateral = np.where(
        time < t_top,
        0.1 * np.sin(np.pi * time / t_top),  # Slight right bend in backswing
        -0.15 * np.sin(np.pi * (time - t_top) / (1 - t_top)),  # Left bend in downswing
    )

    lumbar_rotation = np.where(
        time < t_top,
        0.8 * np.sin(np.pi * time / t_top),
        -1.2 * np.sin(np.pi * (time - t_top) / (1 - t_top)),
    )

    joint_angles = {
        "lumbar_flexion": lumbar_flexion,
        "lumbar_lateral": lumbar_lateral,
        "lumbar_rotation": lumbar_rotation,
    }

    # Compute velocities (numerical derivative)
    dt = time[1] - time[0]
    joint_velocities = {key: np.gradient(val, dt) for key, val in joint_angles.items()}

    # Estimate torques (simplified: torque proportional to acceleration)
    joint_torques = {
        key: 50 * np.gradient(vel, dt) for key, vel in joint_velocities.items()
    }

    # Pelvis and thorax angles for X-factor
    pelvis_rotation = np.where(
        time < t_top, 0.3 * np.sin(np.pi * time / t_top), -0.5 * np.ones_like(time)
    )

    thorax_rotation = np.where(
        time < t_top,
        0.9 * np.sin(np.pi * time / t_top),
        -1.2 * np.sin(np.pi * (time - t_top) / (1 - t_top)),
    )

    pelvis_angles = np.column_stack(
        [np.zeros_like(time), np.zeros_like(time), pelvis_rotation]
    )
    thorax_angles = np.column_stack(
        [np.zeros_like(time), np.zeros_like(time), thorax_rotation]
    )

    # Run analysis
    result = analyzer.analyze(
        joint_angles=joint_angles,
        joint_velocities=joint_velocities,
        joint_torques=joint_torques,
        time=time,
        pelvis_angles=pelvis_angles,
        thorax_angles=thorax_angles,
    )

    return analyzer, result


if __name__ == "__main__":
    # Run example analysis
    analyzer, result = create_example_analysis()

    if result.x_factor:
        print(  # noqa: T201
            f"X-Factor Stretch: {result.x_factor.x_factor_stretch:.1f} deg"
        )

    if result.crunch_factor:
        print(  # noqa: T201
            f"Peak Crunch Factor: {result.crunch_factor.peak_crunch:.1f}"
        )

    print("\nRecommendations:")  # noqa: T201
    for rec in analyzer.get_recommendations(result):
        print(f"  - {rec}")  # noqa: T201

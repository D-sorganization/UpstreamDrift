"""
Golf Swing Analyzer.

Main analysis engine that processes video and generates
comprehensive swing analysis reports.
"""

import logging
import math
import uuid
from collections.abc import Callable
from datetime import datetime

from src.shared.python.core.contracts import precondition

from .types import (
    BalanceMetrics,
    BodyAngles,
    Landmark,
    PhaseTransition,
    PlaneMetrics,
    PoseFrame,
    PostureMetrics,
    StanceDirection,
    SwingAnalysis,
    SwingIssue,
    SwingPhase,
    SwingPositionMetrics,
    SwingScores,
    SwingType,
    TempoMetrics,
)
from .video_processor import VideoProcessor

logger = logging.getLogger(__name__)


class SwingAnalyzer:
    """
    Complete golf swing analysis engine.

    Analyzes video of golf swings to extract:
    - Swing phases (address, backswing, impact, finish)
    - Body angles and measurements
    - Tempo and timing metrics
    - Balance and weight shift
    - Swing plane analysis
    - Posture evaluation
    - Issues and recommendations

    Usage:
        analyzer = SwingAnalyzer()
        results = analyzer.analyze_video("swing.mp4")
        print(f"Overall Score: {results.scores.overall}")
    """

    # Landmark indices (MediaPipe standard)
    LEFT_SHOULDER = 11
    RIGHT_SHOULDER = 12
    LEFT_HIP = 23
    RIGHT_HIP = 24
    LEFT_ELBOW = 13
    RIGHT_ELBOW = 14
    LEFT_WRIST = 15
    RIGHT_WRIST = 16
    LEFT_KNEE = 25
    RIGHT_KNEE = 26
    LEFT_ANKLE = 27
    RIGHT_ANKLE = 28
    NOSE = 0

    def __init__(
        self,
        min_confidence: float = 0.5,
        smoothing_window: int = 5,
    ) -> None:
        """
        Initialize the swing analyzer.

        Args:
            min_confidence: Minimum pose confidence threshold (0-1).
            smoothing_window: Window size for angle smoothing.
        """
        self.min_confidence = min_confidence
        self.smoothing_window = smoothing_window

    @precondition(
        lambda self,
        video_path,
        stance=StanceDirection.UNKNOWN,
        progress_callback=None: video_path is not None and len(video_path) > 0,
        "Video path must be a non-empty string",
    )
    def analyze_video(
        self,
        video_path: str,
        stance: StanceDirection = StanceDirection.UNKNOWN,
        progress_callback: Callable[[str, float], None] | None = None,
    ) -> SwingAnalysis:
        """
        Analyze a golf swing video.

        Args:
            video_path: Path to the video file.
            stance: Golfer's stance (auto-detected if UNKNOWN).
            progress_callback: Optional callback(stage, progress) for progress.

        Returns:
            Complete SwingAnalysis with all metrics.
        """
        logger.info(f"Analyzing video: {video_path}")

        # Load video
        processor = VideoProcessor(video_path)
        if not processor.is_loaded:
            raise ValueError(f"Failed to load video: {video_path}")

        # Extract poses
        if progress_callback:
            progress_callback("Extracting poses", 0)

        poses = processor.extract_poses(
            min_confidence=self.min_confidence,
            progress_callback=lambda c, t: (
                progress_callback("Extracting poses", c / t * 100)
                if progress_callback
                else None
            ),
        )

        if len(poses) < 10:
            raise ValueError(
                f"Insufficient valid poses ({len(poses)}). "
                "Need at least 10 frames with detected pose."
            )

        processor.close()

        # Run analysis
        return self.analyze_poses(
            poses,
            fps=processor.fps,
            video_id=str(video_path),
            stance=stance,
        )

    @precondition(
        lambda self, poses, fps=30.0, video_id="", stance=StanceDirection.UNKNOWN: poses
        is not None
        and len(poses) >= 10,
        "Must have at least 10 valid pose frames",
    )
    @precondition(
        lambda self, poses, fps=30.0, video_id="", stance=StanceDirection.UNKNOWN: fps
        > 0,
        "FPS must be positive",
    )
    def analyze_poses(
        self,
        poses: list[PoseFrame],
        fps: float = 30.0,
        video_id: str = "",
        stance: StanceDirection = StanceDirection.UNKNOWN,
    ) -> SwingAnalysis:
        """
        Analyze swing from pre-extracted pose data.

        Args:
            poses: List of PoseFrame objects.
            fps: Video frames per second.
            video_id: Identifier for the video.
            stance: Golfer's stance direction.

        Returns:
            Complete SwingAnalysis.
        """
        if len(poses) < 10:
            raise ValueError("Need at least 10 valid pose frames")

        # Auto-detect stance if unknown
        if stance == StanceDirection.UNKNOWN:
            stance = self._detect_stance(poses[0].landmarks)

        # Detect swing phases
        phases = self._detect_phases(poses, fps, stance)

        # Get key frames
        key_frames = self._get_key_frames(phases)

        # Extract metrics at key positions
        key_positions = self._extract_key_positions(poses, key_frames, fps, stance)

        # Calculate tempo
        tempo = self._calculate_tempo(phases)

        # Calculate balance
        balance = self._calculate_balance(poses, key_frames)

        # Calculate plane metrics
        plane = self._calculate_plane(poses, key_frames)

        # Calculate posture
        posture = self._calculate_posture(poses, key_frames, stance)

        # Identify issues
        issues = self._identify_issues(key_positions, tempo, balance, posture)

        # Generate recommendations
        recommendations = self._generate_recommendations(issues)

        # Calculate scores
        scores = self._calculate_scores(tempo, balance, plane, posture, issues)

        return SwingAnalysis(
            session_id=str(uuid.uuid4()),
            video_id=video_id,
            analysis_timestamp=datetime.now().timestamp() * 1000,
            golfer_stance=stance,
            swing_type=SwingType.UNKNOWN,
            total_frames=len(poses),
            fps=fps,
            pose_frames=poses,
            phases=phases,
            address_metrics=key_positions.get("address"),
            top_metrics=key_positions.get("top"),
            impact_metrics=key_positions.get("impact"),
            finish_metrics=key_positions.get("finish"),
            tempo=tempo,
            balance=balance,
            plane=plane,
            posture=posture,
            scores=scores,
            issues=issues,
            recommendations=recommendations,
        )

    def _detect_stance(self, landmarks: list[Landmark]) -> StanceDirection:
        """Detect if golfer is right or left handed based on body orientation."""
        left_shoulder = landmarks[self.LEFT_SHOULDER]
        right_shoulder = landmarks[self.RIGHT_SHOULDER]

        # Calculate shoulder line angle
        dx = right_shoulder.x - left_shoulder.x
        dz = right_shoulder.z - left_shoulder.z
        angle = math.atan2(dz, dx) * (180 / math.pi)

        if angle > 20:
            return StanceDirection.RIGHT_HANDED
        if angle < -20:
            return StanceDirection.LEFT_HANDED
        return StanceDirection.UNKNOWN

    def _calculate_angle(self, a: Landmark, b: Landmark, c: Landmark) -> float:
        """Calculate angle at point B between points A and C."""
        ba = (a.x - b.x, a.y - b.y, a.z - b.z)
        bc = (c.x - b.x, c.y - b.y, c.z - b.z)

        dot = ba[0] * bc[0] + ba[1] * bc[1] + ba[2] * bc[2]
        mag_ba = math.sqrt(ba[0] ** 2 + ba[1] ** 2 + ba[2] ** 2)
        mag_bc = math.sqrt(bc[0] ** 2 + bc[1] ** 2 + bc[2] ** 2)

        if mag_ba == 0 or mag_bc == 0:
            return 0

        cos_angle = max(-1, min(1, dot / (mag_ba * mag_bc)))
        return math.acos(cos_angle) * (180 / math.pi)

    def _calculate_body_angles(
        self, landmarks: list[Landmark], stance: StanceDirection
    ) -> BodyAngles:
        """Calculate all body angles from landmarks."""
        ls = landmarks[self.LEFT_SHOULDER]
        rs = landmarks[self.RIGHT_SHOULDER]
        lh = landmarks[self.LEFT_HIP]
        rh = landmarks[self.RIGHT_HIP]
        le = landmarks[self.LEFT_ELBOW]
        re = landmarks[self.RIGHT_ELBOW]
        lw = landmarks[self.LEFT_WRIST]
        rw = landmarks[self.RIGHT_WRIST]
        lk = landmarks[self.LEFT_KNEE]
        rk = landmarks[self.RIGHT_KNEE]
        la = landmarks[self.LEFT_ANKLE]
        ra = landmarks[self.RIGHT_ANKLE]

        # Midpoints
        shoulder_mid = Landmark(
            (ls.x + rs.x) / 2,
            (ls.y + rs.y) / 2,
            (ls.z + rs.z) / 2,
        )
        hip_mid = Landmark(
            (lh.x + rh.x) / 2,
            (lh.y + rh.y) / 2,
            (lh.z + rh.z) / 2,
        )

        # Spine angle (forward tilt)
        spine_angle = 90 - math.atan2(
            shoulder_mid.y - hip_mid.y,
            abs(shoulder_mid.z - hip_mid.z) + 0.001,
        ) * (180 / math.pi)

        # Rotations (using z-depth)
        shoulder_rotation = math.atan2(rs.z - ls.z, rs.x - ls.x) * (180 / math.pi)
        hip_rotation = math.atan2(rh.z - lh.z, rh.x - lh.x) * (180 / math.pi)

        # X-Factor
        x_factor = abs(shoulder_rotation - hip_rotation)

        # Elbow angles
        left_elbow_angle = self._calculate_angle(ls, le, lw)
        right_elbow_angle = self._calculate_angle(rs, re, rw)

        # Knee flexion
        left_knee_flexion = 180 - self._calculate_angle(lh, lk, la)
        right_knee_flexion = 180 - self._calculate_angle(rh, rk, ra)

        return BodyAngles(
            spine_angle=spine_angle,
            spine_lateral=(ls.y - rs.y) * 100,
            spine_rotation=shoulder_rotation,
            hip_rotation=hip_rotation,
            hip_tilt=(lh.y - rh.y) * 100,
            hip_slide=(hip_mid.x - 0.5) * 100,
            shoulder_rotation=shoulder_rotation,
            shoulder_tilt=(ls.y - rs.y) * 100,
            left_elbow_angle=left_elbow_angle,
            right_elbow_angle=right_elbow_angle,
            left_wrist_angle=0,  # Simplified
            right_wrist_angle=0,
            left_knee_flexion=left_knee_flexion,
            right_knee_flexion=right_knee_flexion,
            x_factor=x_factor,
            x_factor_stretch=x_factor,
        )

    def _detect_phases(
        self,
        poses: list[PoseFrame],
        fps: float,
        stance: StanceDirection,
    ) -> list[PhaseTransition]:
        """Detect swing phases from pose sequence."""
        phases = []
        frame_duration = 1000 / fps

        # Calculate angles for all frames
        angle_history = [
            self._calculate_body_angles(p.landmarks, stance) for p in poses
        ]

        # Find top of backswing (max shoulder rotation)
        max_rotation: float = -999.0
        top_idx = 0
        for i, angles in enumerate(angle_history):
            rotation = angles.shoulder_rotation
            if stance == StanceDirection.RIGHT_HANDED:
                if rotation > max_rotation:
                    max_rotation = rotation
                    top_idx = i
            else:
                if rotation < -max_rotation:  # Inverted for left-handed
                    max_rotation = rotation
                    top_idx = i

        # Find address (minimal rotation at start)
        address_idx = 0
        for i in range(min(top_idx, len(angle_history) // 3)):
            if abs(angle_history[i].shoulder_rotation) < 15:
                address_idx = i
                break

        # Find impact (shoulder near square after top)
        impact_idx = top_idx
        for i in range(top_idx, len(angle_history)):
            if abs(angle_history[i].shoulder_rotation) < 20:
                impact_idx = i
                break

        # Find finish
        finish_idx = len(poses) - 1

        # Build phase transitions
        def add_phase(phase: SwingPhase, start: int, end: int) -> None:
            """Append a phase transition if the frame range is valid."""
            if start < end:
                phases.append(
                    PhaseTransition(
                        phase=phase,
                        start_frame=poses[start].frame_number,
                        end_frame=poses[end].frame_number,
                        duration=(end - start) * frame_duration,
                        confidence=0.8,
                    )
                )

        add_phase(SwingPhase.ADDRESS, 0, address_idx + 1)
        add_phase(SwingPhase.BACKSWING, address_idx, top_idx)
        add_phase(SwingPhase.TOP_OF_BACKSWING, top_idx, top_idx + 2)
        add_phase(SwingPhase.DOWNSWING, top_idx, impact_idx)
        add_phase(SwingPhase.IMPACT, impact_idx, impact_idx + 1)
        add_phase(SwingPhase.FOLLOW_THROUGH, impact_idx, finish_idx)

        return phases

    def _get_key_frames(self, phases: list[PhaseTransition]) -> dict[str, int]:
        """Extract key frame indices from phases."""
        key_frames = {}

        for phase in phases:
            if phase.phase == SwingPhase.ADDRESS:
                key_frames["address"] = phase.start_frame
            elif phase.phase == SwingPhase.TOP_OF_BACKSWING:
                key_frames["top"] = phase.start_frame
            elif phase.phase == SwingPhase.IMPACT:
                key_frames["impact"] = phase.start_frame
            elif phase.phase == SwingPhase.FOLLOW_THROUGH:
                key_frames["finish"] = phase.end_frame

        return key_frames

    def _extract_key_positions(
        self,
        poses: list[PoseFrame],
        key_frames: dict[str, int],
        fps: float,
        stance: StanceDirection,
    ) -> dict[str, SwingPositionMetrics]:
        """Extract metrics at key swing positions."""
        positions = {}

        for name, frame_num in key_frames.items():
            # Find the pose for this frame
            pose = next((p for p in poses if p.frame_number == frame_num), None)
            if pose:
                angles = self._calculate_body_angles(pose.landmarks, stance)
                positions[name] = SwingPositionMetrics(
                    frame_number=frame_num,
                    timestamp=pose.timestamp,
                    angles=angles,
                    confidence=pose.confidence,
                )

        return positions

    def _calculate_tempo(self, phases: list[PhaseTransition]) -> TempoMetrics:
        """Calculate tempo and timing metrics."""
        backswing_dur = sum(
            p.duration
            for p in phases
            if p.phase in [SwingPhase.BACKSWING, SwingPhase.TOP_OF_BACKSWING]
        )
        downswing_dur = sum(
            p.duration
            for p in phases
            if p.phase in [SwingPhase.DOWNSWING, SwingPhase.IMPACT]
        )
        total = backswing_dur + downswing_dur

        ratio = backswing_dur / downswing_dur if downswing_dur > 0 else 0

        # Determine rhythm quality
        if 2.5 <= ratio <= 3.5:
            rhythm = "smooth"
        elif ratio < 2:
            rhythm = "quick"
        elif ratio > 4:
            rhythm = "slow"
        else:
            rhythm = "uneven"

        return TempoMetrics(
            backswing_duration=backswing_dur,
            downswing_duration=downswing_dur,
            total_swing_duration=total,
            tempo_ratio=ratio,
            transition_pause=0,  # Would need more analysis
            rhythm=rhythm,
        )

    def _calculate_balance(
        self,
        poses: list[PoseFrame],
        key_frames: dict[str, int],
    ) -> BalanceMetrics:
        """Calculate balance and weight shift metrics."""

        def get_weight_distribution(landmarks: list[Landmark]) -> tuple[float, float]:
            """Estimate left/right weight distribution from hip and ankle landmarks."""
            lh = landmarks[self.LEFT_HIP]
            rh = landmarks[self.RIGHT_HIP]
            la = landmarks[self.LEFT_ANKLE]
            ra = landmarks[self.RIGHT_ANKLE]

            hip_mid_x = (lh.x + rh.x) / 2
            stance_width = abs(ra.x - la.x)

            if stance_width == 0:
                return 50, 50

            ratio = (hip_mid_x - la.x) / stance_width
            right = min(100, max(0, ratio * 100))
            return 100 - right, right

        metrics = BalanceMetrics()

        for name, frame_num in key_frames.items():
            pose = next((p for p in poses if p.frame_number == frame_num), None)
            if pose:
                left, right = get_weight_distribution(pose.landmarks)
                if name == "address":
                    metrics.address_weight_left = left
                    metrics.address_weight_right = right
                elif name == "top":
                    metrics.top_weight_left = left
                    metrics.top_weight_right = right
                elif name == "impact":
                    metrics.impact_weight_left = left
                    metrics.impact_weight_right = right
                elif name == "finish":
                    metrics.finish_weight_left = left
                    metrics.finish_weight_right = right

        return metrics

    def _calculate_plane(
        self,
        poses: list[PoseFrame],
        key_frames: dict[str, int],
    ) -> PlaneMetrics:
        """Calculate swing plane metrics."""
        # Simplified plane analysis
        return PlaneMetrics(
            backswing_plane_angle=60,
            downswing_plane_angle=55,
            plane_differential=5,
            on_plane=True,
        )

    def _calculate_posture(
        self,
        poses: list[PoseFrame],
        key_frames: dict[str, int],
        stance: StanceDirection,
    ) -> PostureMetrics:
        """Calculate posture metrics."""
        address_pose = next(
            (p for p in poses if p.frame_number == key_frames.get("address")),
            poses[0] if poses else None,
        )

        if not address_pose:
            return PostureMetrics()

        angles = self._calculate_body_angles(address_pose.landmarks, stance)

        # Calculate head stability
        if len(poses) < 2:
            head_stability: float = 100.0
        else:
            address_nose = address_pose.landmarks[self.NOSE]
            max_movement: float = 0.0
            for pose in poses:
                nose = pose.landmarks[self.NOSE]
                dist = math.sqrt(
                    (nose.x - address_nose.x) ** 2 + (nose.y - address_nose.y) ** 2
                )
                max_movement = max(max_movement, dist)
            head_stability = max(0.0, 100.0 - max_movement * 500)

        return PostureMetrics(
            address_spine_angle=angles.spine_angle,
            address_knee_flexion=(angles.left_knee_flexion + angles.right_knee_flexion)
            / 2,
            address_arm_hang="good",
            head_stability=head_stability,
            early_extension=False,
            loss_of_posture=False,
            reverse_spine_tilt=False,
        )

    def _identify_issues(
        self,
        key_positions: dict[str, SwingPositionMetrics],
        tempo: TempoMetrics,
        balance: BalanceMetrics,
        posture: PostureMetrics,
    ) -> list[SwingIssue]:
        """Identify swing faults and issues."""
        issues = []

        # Tempo issues
        if tempo.tempo_ratio < 2:
            issues.append(
                SwingIssue(
                    id=str(uuid.uuid4()),
                    name="Quick Tempo",
                    severity="moderate",
                    phase=SwingPhase.BACKSWING,
                    description="Backswing is too fast relative to downswing",
                    detected_at=0,
                    measurement_value=tempo.tempo_ratio,
                    expected_range=(2.5, 3.5),
                    drill_recommendation="Count 1-2-3 during backswing",
                )
            )

        # Balance issues
        if balance.sway_amount > 15:
            issues.append(
                SwingIssue(
                    id=str(uuid.uuid4()),
                    name="Excessive Sway",
                    severity="major",
                    phase=SwingPhase.BACKSWING,
                    description=f"Lateral hip movement of {balance.sway_amount:.1f}cm",
                    detected_at=0,
                    measurement_value=balance.sway_amount,
                    expected_range=(0, 10),
                    drill_recommendation="Practice with club against trail hip",
                )
            )

        # Posture issues
        if posture.head_stability < 60:
            issues.append(
                SwingIssue(
                    id=str(uuid.uuid4()),
                    name="Head Movement",
                    severity="moderate",
                    phase=SwingPhase.BACKSWING,
                    description="Excessive head movement during swing",
                    detected_at=0,
                    measurement_value=100 - posture.head_stability,
                    expected_range=(0, 40),
                    drill_recommendation="Practice in front of mirror",
                )
            )

        return issues

    def _generate_recommendations(self, issues: list[SwingIssue]) -> list[str]:
        """Generate practice recommendations."""
        recommendations = []

        major_issues = [i for i in issues if i.severity == "major"]
        if major_issues:
            recommendations.append(
                f"Focus on fixing {len(major_issues)} major issue(s): "
                f"{', '.join(i.name for i in major_issues)}"
            )

        for issue in issues[:3]:
            if issue.drill_recommendation:
                recommendations.append(
                    f"For {issue.name}: {issue.drill_recommendation}"
                )

        if not issues:
            recommendations.append("Great swing! Focus on consistency.")

        return recommendations

    def _calculate_scores(
        self,
        tempo: TempoMetrics,
        balance: BalanceMetrics,
        plane: PlaneMetrics,
        posture: PostureMetrics,
        issues: list[SwingIssue],
    ) -> SwingScores:
        """Calculate swing scores (0-100)."""
        # Tempo score
        tempo_dev = abs(tempo.tempo_ratio - 3)
        tempo_score = max(0, 100 - tempo_dev * 20)

        # Balance score
        sway_penalty = min(30, balance.sway_amount * 2)
        balance_score = max(0, 100 - sway_penalty)

        # Plane score
        plane_score = max(0, 100 - plane.plane_differential * 5)

        # Posture score
        posture_score = posture.head_stability

        # Issue penalty
        issue_penalty = (
            len([i for i in issues if i.severity == "major"]) * 10
            + len([i for i in issues if i.severity == "moderate"]) * 5
        )

        # Overall
        components = [tempo_score, balance_score, plane_score, posture_score]
        overall = sum(components) / len(components) - issue_penalty / 2
        overall = max(0, min(100, overall))

        return SwingScores(
            overall=overall,
            tempo=tempo_score,
            balance=balance_score,
            plane=plane_score,
            posture=posture_score,
            rotation=75,  # Placeholder
            timing=tempo_score,
            consistency=80,  # Would need multiple swings
        )

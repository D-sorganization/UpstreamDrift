"""
Type definitions for Golf Swing Video Analysis.

These types mirror the TypeScript definitions used in the web frontend
for consistent data exchange between Python backend and JavaScript frontend.
"""

from dataclasses import dataclass, field
from enum import Enum


class SwingPhase(Enum):
    """Golf swing phases based on professional instruction."""

    ADDRESS = "address"
    TAKEAWAY = "takeaway"
    BACKSWING = "backswing"
    TOP_OF_BACKSWING = "top_of_backswing"
    TRANSITION = "transition"
    DOWNSWING = "downswing"
    IMPACT = "impact"
    FOLLOW_THROUGH = "follow_through"
    FINISH = "finish"
    UNKNOWN = "unknown"


class StanceDirection(Enum):
    """Golfer's stance direction."""

    RIGHT_HANDED = "right_handed"
    LEFT_HANDED = "left_handed"
    UNKNOWN = "unknown"


class SwingType(Enum):
    """Type of golf swing."""

    DRIVER = "driver"
    IRON = "iron"
    WEDGE = "wedge"
    PUTTER = "putter"
    HYBRID = "hybrid"
    UNKNOWN = "unknown"


@dataclass
class Landmark:
    """3D landmark point with visibility confidence."""

    x: float
    y: float
    z: float
    visibility: float = 1.0


@dataclass
class PoseFrame:
    """Complete pose data for a single video frame."""

    frame_number: int
    timestamp: float  # milliseconds
    landmarks: list[Landmark]
    confidence: float


@dataclass
class BodyAngles:
    """All measured body angles in degrees."""

    # Spine angles
    spine_angle: float = 0.0
    spine_lateral: float = 0.0
    spine_rotation: float = 0.0

    # Hip angles
    hip_rotation: float = 0.0
    hip_tilt: float = 0.0
    hip_slide: float = 0.0

    # Shoulder angles
    shoulder_rotation: float = 0.0
    shoulder_tilt: float = 0.0

    # Arm angles
    left_elbow_angle: float = 0.0
    right_elbow_angle: float = 0.0
    left_wrist_angle: float = 0.0
    right_wrist_angle: float = 0.0

    # Knee angles
    left_knee_flexion: float = 0.0
    right_knee_flexion: float = 0.0

    # X-Factor
    x_factor: float = 0.0
    x_factor_stretch: float = 0.0


@dataclass
class BodyVelocities:
    """Velocity measurements."""

    hip_rotational_velocity: float = 0.0  # degrees/second
    shoulder_rotational_velocity: float = 0.0
    hand_speed: float = 0.0  # meters/second
    head_movement: float = 0.0  # cm


@dataclass
class TempoMetrics:
    """Tempo and timing metrics."""

    backswing_duration: float = 0.0  # milliseconds
    downswing_duration: float = 0.0
    total_swing_duration: float = 0.0
    tempo_ratio: float = 0.0  # backswing:downswing
    transition_pause: float = 0.0
    rhythm: str = "smooth"  # smooth, quick, slow, uneven


@dataclass
class BalanceMetrics:
    """Balance and weight shift metrics."""

    address_weight_left: float = 50.0
    address_weight_right: float = 50.0
    top_weight_left: float = 30.0
    top_weight_right: float = 70.0
    impact_weight_left: float = 70.0
    impact_weight_right: float = 30.0
    finish_weight_left: float = 90.0
    finish_weight_right: float = 10.0
    sway_amount: float = 0.0  # cm
    slide_amount: float = 0.0  # cm
    hip_bump: float = 0.0  # cm


@dataclass
class PlaneMetrics:
    """Swing plane analysis metrics."""

    backswing_plane_angle: float = 60.0
    downswing_plane_angle: float = 55.0
    plane_differential: float = 5.0
    on_plane: bool = True
    shaft_angle_at_address: float = 45.0
    shaft_angle_at_top: float = 60.0
    shaft_angle_at_impact: float = 45.0


@dataclass
class PostureMetrics:
    """Posture analysis metrics."""

    address_spine_angle: float = 35.0
    address_knee_flexion: float = 25.0
    address_arm_hang: str = "good"
    head_stability: float = 100.0
    early_extension: bool = False
    loss_of_posture: bool = False
    reverse_spine_tilt: bool = False


@dataclass
class SwingScores:
    """Scoring breakdown (0-100 scale)."""

    overall: float = 0.0
    tempo: float = 0.0
    balance: float = 0.0
    plane: float = 0.0
    posture: float = 0.0
    rotation: float = 0.0
    timing: float = 0.0
    consistency: float = 0.0


@dataclass
class SwingIssue:
    """Identified swing fault."""

    id: str
    name: str
    severity: str  # minor, moderate, major
    phase: SwingPhase
    description: str
    detected_at: int  # frame number
    measurement_value: float
    expected_range: tuple[float, float]
    drill_recommendation: str | None = None


@dataclass
class PhaseTransition:
    """Swing phase timing information."""

    phase: SwingPhase
    start_frame: int
    end_frame: int
    duration: float  # milliseconds
    confidence: float


@dataclass
class SwingPositionMetrics:
    """Metrics at a specific swing position."""

    frame_number: int
    timestamp: float
    angles: BodyAngles
    velocities: BodyVelocities | None = None
    confidence: float = 1.0


@dataclass
class SwingAnalysis:
    """Complete swing analysis result."""

    # Session info
    session_id: str = ""
    video_id: str = ""
    analysis_timestamp: float = 0.0
    golfer_stance: StanceDirection = StanceDirection.UNKNOWN
    swing_type: SwingType = SwingType.UNKNOWN

    # Frame data
    total_frames: int = 0
    fps: float = 30.0
    pose_frames: list[PoseFrame] = field(default_factory=list)

    # Phase detection
    phases: list[PhaseTransition] = field(default_factory=list)

    # Key positions
    address_metrics: SwingPositionMetrics | None = None
    top_metrics: SwingPositionMetrics | None = None
    impact_metrics: SwingPositionMetrics | None = None
    finish_metrics: SwingPositionMetrics | None = None

    # Dynamic metrics
    tempo: TempoMetrics = field(default_factory=TempoMetrics)
    balance: BalanceMetrics = field(default_factory=BalanceMetrics)
    plane: PlaneMetrics = field(default_factory=PlaneMetrics)
    posture: PostureMetrics = field(default_factory=PostureMetrics)

    # Scores and issues
    scores: SwingScores = field(default_factory=SwingScores)
    issues: list[SwingIssue] = field(default_factory=list)
    recommendations: list[str] = field(default_factory=list)

    def to_dict(self) -> dict:
        """Convert to dictionary for JSON serialization."""
        import dataclasses

        def convert(obj) -> Any:
            if dataclasses.is_dataclass(obj):
                return {k: convert(v) for k, v in dataclasses.asdict(obj).items()}  # type: ignore[arg-type]
            if isinstance(obj, Enum):
                return obj.value
            if isinstance(obj, list):
                return [convert(item) for item in obj]
            return obj

        return convert(self)

    @classmethod
    def from_dict(cls, data: dict) -> "SwingAnalysis":
        """Create from dictionary (e.g., from JSON)."""
        # This would need full implementation for production
        return cls(**data)

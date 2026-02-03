# UpstreamDrift Robotics Expansion Proposal

## Strategic Vision: From Golf Biomechanics to Industrial Humanoid Robotics

**Document Version**: 1.0
**Date**: 2026-02-03
**Status**: Proposal for Review

---

## Executive Summary

This proposal outlines an ambitious expansion of UpstreamDrift from a golf biomechanics simulation platform into a comprehensive **Industrial Humanoid Robotics Modeling Suite**. The existing architecture provides an excellent foundation with:

- 5 production-grade physics engines (MuJoCo, Drake, Pinocchio, OpenSim, MyoSuite)
- Advanced kinematics (constraint Jacobians, manipulability analysis, IK solvers)
- 6 control modes (torque, impedance, admittance, hybrid, computed torque, task-space)
- Design-by-Contract architecture with clean protocols
- Modular analysis framework

The proposed additions will transform this into a platform capable of:

- Full humanoid bipedal locomotion simulation
- Whole-body manipulation and control
- Human-robot interaction with safety guarantees
- Real-time robot control interfaces
- Learning-based control and motion synthesis
- Digital twin deployment for industrial applications

---

## Table of Contents

1. [Phase 1: Humanoid Foundation](#phase-1-humanoid-foundation)
2. [Phase 2: Perception and Planning](#phase-2-perception-and-planning)
3. [Phase 3: Learning and Adaptation](#phase-3-learning-and-adaptation)
4. [Phase 4: Industrial Deployment](#phase-4-industrial-deployment)
5. [Phase 5: Advanced Research Capabilities](#phase-5-advanced-research-capabilities)
6. [Architecture Extensions](#architecture-extensions)
7. [Implementation Priorities](#implementation-priorities)

---

## Phase 1: Humanoid Foundation

### 1.1 Bipedal Locomotion Engine

**Purpose**: Enable realistic humanoid walking, running, and dynamic balance.

#### Components

```
src/engines/locomotion/
├── gait_generators/
│   ├── zero_moment_point.py      # ZMP-based walking
│   ├── capture_point.py          # DCM/ICP locomotion
│   ├── hybrid_zero_dynamics.py   # HZD for dynamic gaits
│   └── central_pattern_gen.py    # CPG oscillators
├── balance_controllers/
│   ├── linear_inverted_pendulum.py  # LIPM
│   ├── centroidal_dynamics.py       # Full centroidal model
│   └── whole_body_balance.py        # QP-based balancing
├── footstep_planners/
│   ├── a_star_footstep.py        # Discrete footstep planning
│   ├── rrt_footstep.py           # RRT-based planning
│   └── reactive_stepping.py       # Push recovery
└── terrain_adaptation/
    ├── height_map_processor.py
    ├── slope_compensation.py
    └── stair_climbing.py
```

#### Key Interfaces

```python
@runtime_checkable
class LocomotionEngine(Protocol):
    """Interface for locomotion controllers."""

    def set_velocity_command(self, vx: float, vy: float, omega: float) -> None:
        """Set desired walking velocity (linear + angular)."""

    def set_footstep_plan(self, footsteps: list[FootstepTarget]) -> None:
        """Set explicit footstep sequence."""

    def get_support_polygon(self) -> np.ndarray:
        """Get current support polygon vertices."""

    def get_center_of_pressure(self) -> np.ndarray:
        """Get instantaneous CoP in world frame."""

    def get_capture_point(self) -> np.ndarray:
        """Get divergent component of motion (DCM)."""

    def is_balanced(self) -> bool:
        """Check if robot is in stable balance."""

    def compute_push_recovery(self, impulse: np.ndarray) -> RecoveryAction:
        """Compute reactive stepping/ankle/hip strategy."""
```

#### Gait State Machine

```
STANDING -> [velocity_command] -> STARTING
STARTING -> [first_step_complete] -> WALKING
WALKING -> [zero_velocity] -> STOPPING
STOPPING -> [feet_together] -> STANDING
WALKING -> [external_push] -> PUSH_RECOVERY
PUSH_RECOVERY -> [stabilized] -> WALKING
```

### 1.2 Whole-Body Control Framework

**Purpose**: Unified control of humanoid with prioritized task hierarchies.

#### Hierarchical Quadratic Programming (HQP)

```python
@dataclass
class TaskDescriptor:
    """Describes a control task with priority."""
    name: str
    priority: int  # 0 = highest priority
    task_type: TaskType  # EQUALITY, INEQUALITY, SOFT

    # Task definition: A @ ddq + b = 0 (or inequality)
    jacobian: np.ndarray  # A matrix (task_dim x n_v)
    target: np.ndarray    # Desired acceleration/value
    weight: np.ndarray    # Diagonal weight for soft tasks

    # Bounds for inequality tasks
    lower_bound: np.ndarray | None = None
    upper_bound: np.ndarray | None = None


class WholeBodyController:
    """Whole-body control via hierarchical optimization."""

    def __init__(self, model: PhysicsEngine):
        self.model = model
        self.tasks: list[TaskDescriptor] = []
        self.contact_constraints: list[ContactConstraint] = []

    def add_task(self, task: TaskDescriptor) -> None:
        """Add task to hierarchy."""

    def set_contacts(self, contacts: list[ContactConstraint]) -> None:
        """Set active contact constraints."""

    def solve(self) -> tuple[np.ndarray, np.ndarray, dict]:
        """Solve HQP for optimal (ddq, tau, contact_forces)."""

    def compute_centroidal_momentum_task(
        self,
        desired_momentum: np.ndarray
    ) -> TaskDescriptor:
        """Create task for centroidal momentum tracking."""

    def compute_com_task(
        self,
        desired_com: np.ndarray,
        desired_com_vel: np.ndarray
    ) -> TaskDescriptor:
        """Create task for CoM tracking."""
```

#### Standard Task Hierarchy (Example)

| Priority | Task | Description |
|----------|------|-------------|
| 0 | Contact Constraints | Feet don't slip, forces in friction cone |
| 1 | Dynamics Consistency | M*ddq + h = S*tau + J^T*f |
| 2 | CoM Tracking | Keep CoM over support polygon |
| 3 | Angular Momentum | Regulate whole-body rotation |
| 4 | End-Effector Tasks | Hand position/orientation tracking |
| 5 | Posture | Maintain nominal joint configuration |
| 6 | Joint Limits | Soft bounds to avoid singularities |

### 1.3 Contact Dynamics and Multi-Contact

**Purpose**: Rich contact modeling for manipulation and locomotion.

#### Contact Models

```python
class ContactType(Enum):
    POINT = "point"              # Single point contact
    LINE = "line"                # Edge contact
    PATCH = "patch"              # Surface contact (foot)
    SOFT = "soft"                # Deformable contact


@dataclass
class ContactState:
    """Rich contact state representation."""
    contact_id: int
    body_id: int
    contact_type: ContactType

    # Geometric
    position: np.ndarray         # Contact point in world
    normal: np.ndarray           # Surface normal
    penetration: float           # Penetration depth

    # Forces
    normal_force: float          # Normal force magnitude
    friction_force: np.ndarray   # Tangential friction force
    wrench: np.ndarray           # Full 6D wrench if patch contact

    # Friction cone
    friction_coefficient: float
    is_sliding: bool             # True if at friction limit

    # For patch contacts
    center_of_pressure: np.ndarray | None = None
    contact_polygon: np.ndarray | None = None


class ContactManager:
    """Manages multi-contact scenarios."""

    def detect_contacts(self) -> list[ContactState]:
        """Detect all active contacts."""

    def compute_grasp_matrix(
        self,
        object_frame: np.ndarray,
        contacts: list[ContactState]
    ) -> np.ndarray:
        """Compute grasp matrix G mapping contact forces to object wrench."""

    def check_force_closure(
        self,
        contacts: list[ContactState]
    ) -> tuple[bool, float]:
        """Check if grasp has force closure, return quality metric."""

    def compute_friction_cone_constraints(
        self,
        contacts: list[ContactState],
        linearization_faces: int = 8
    ) -> tuple[np.ndarray, np.ndarray]:
        """Return (A, b) such that A @ f <= b for friction cone."""
```

### 1.4 Force/Torque Sensing Framework

**Purpose**: Enable force-controlled manipulation and compliant interaction.

```python
@dataclass
class ForceTorqueSensorConfig:
    """Configuration for F/T sensor."""
    sensor_id: str
    parent_body: str
    child_body: str  # "world" for base-mounted
    transform: np.ndarray  # Sensor frame relative to parent

    # Noise model
    force_noise_std: float = 0.1     # N
    torque_noise_std: float = 0.01   # Nm

    # Bandwidth (for filtering)
    cutoff_frequency: float = 100.0  # Hz

    # Bias drift
    bias_drift_rate: float = 0.001   # N/s or Nm/s


class ForceTorqueSensor:
    """Simulated F/T sensor with realistic noise model."""

    def __init__(self, config: ForceTorqueSensorConfig):
        self.config = config
        self._bias = np.zeros(6)
        self._filter = LowPassFilter(config.cutoff_frequency)

    def read(self) -> np.ndarray:
        """Get filtered wrench measurement [fx, fy, fz, tx, ty, tz]."""

    def read_raw(self) -> np.ndarray:
        """Get unfiltered measurement with noise."""

    def tare(self) -> None:
        """Zero the sensor (remove current reading as bias)."""

    def get_contact_location(self) -> np.ndarray | None:
        """Estimate single contact location from wrench."""
```

---

## Phase 2: Perception and Planning

### 2.1 Motion Planning Framework

**Purpose**: Collision-free trajectory generation for complex environments.

#### Planning Algorithms

```
src/planning/
├── sampling_based/
│   ├── rrt.py                    # Rapidly-exploring Random Trees
│   ├── rrt_star.py               # Optimal RRT
│   ├── rrt_connect.py            # Bidirectional RRT
│   ├── prm.py                    # Probabilistic Roadmap
│   ├── fmt_star.py               # Fast Marching Tree
│   └── bit_star.py               # Batch Informed Trees
├── optimization_based/
│   ├── chomp.py                  # Covariant Hamiltonian Optimization
│   ├── trajopt.py                # Sequential Convex Optimization
│   ├── stomp.py                  # Stochastic Trajectory Optimization
│   └── gpmp.py                   # Gaussian Process Motion Planning
├── task_space/
│   ├── cartesian_planner.py      # Task-space interpolation
│   ├── screw_motion.py           # SE(3) geodesic planning
│   └── via_point_planner.py      # Via-point trajectories
└── whole_body/
    ├── multi_contact_planner.py  # Contact sequence planning
    ├── manipulation_planner.py   # Pick-place-regrasp
    └── loco_manipulation.py      # Combined locomotion + manipulation
```

#### Planning Interface

```python
@dataclass
class PlanningProblem:
    """Defines a motion planning problem."""
    start_config: np.ndarray
    goal_config: np.ndarray | None = None
    goal_pose: np.ndarray | None = None  # SE(3) for IK-based goals
    goal_region: GoalRegion | None = None  # For task-space goals

    # Constraints
    collision_bodies: list[str] = field(default_factory=list)
    path_constraints: list[Constraint] = field(default_factory=list)

    # Options
    max_planning_time: float = 5.0
    optimization_objective: str = "path_length"  # or "clearance", "smoothness"


@dataclass
class PlanningResult:
    """Result of motion planning."""
    success: bool
    path: np.ndarray | None        # (N, n_q) waypoints
    trajectory: Trajectory | None   # Time-parameterized
    planning_time: float
    iterations: int
    cost: float

    # For analysis
    tree_size: int | None = None
    collision_checks: int | None = None


class MotionPlanner(Protocol):
    """Interface for motion planners."""

    def plan(self, problem: PlanningProblem) -> PlanningResult:
        """Plan a collision-free path."""

    def plan_to_pose(
        self,
        start: np.ndarray,
        goal_pose: np.ndarray,
        end_effector: str
    ) -> PlanningResult:
        """Plan to end-effector pose (IK-based goal)."""
```

### 2.2 Collision Detection and Distance Queries

**Purpose**: Fast, accurate collision checking for planning and control.

```python
class CollisionChecker:
    """Collision detection and distance computation."""

    def __init__(self, model: PhysicsEngine):
        self.model = model
        self._collision_pairs: list[tuple[str, str]] = []
        self._disabled_pairs: set[tuple[str, str]] = set()

    def check_collision(self, q: np.ndarray) -> bool:
        """Check if configuration is in collision."""

    def get_colliding_pairs(
        self,
        q: np.ndarray
    ) -> list[tuple[str, str, ContactInfo]]:
        """Get all colliding body pairs with contact info."""

    def compute_distance(
        self,
        q: np.ndarray,
        body_a: str,
        body_b: str
    ) -> tuple[float, np.ndarray, np.ndarray]:
        """Compute signed distance and closest points."""

    def compute_min_distance(
        self,
        q: np.ndarray
    ) -> tuple[float, str, str, np.ndarray, np.ndarray]:
        """Compute minimum distance over all enabled collision pairs."""

    def compute_distance_jacobian(
        self,
        q: np.ndarray,
        body_a: str,
        body_b: str
    ) -> np.ndarray:
        """Compute Jacobian of distance w.r.t. configuration."""
```

### 2.3 Perception Integration

**Purpose**: Connect simulation to vision systems and perception pipelines.

```python
@dataclass
class CameraConfig:
    """Virtual camera configuration."""
    name: str
    width: int = 640
    height: int = 480
    fov: float = 60.0  # degrees
    near_clip: float = 0.01
    far_clip: float = 100.0

    # Pose relative to parent body
    parent_body: str = "world"
    position: np.ndarray = field(default_factory=lambda: np.zeros(3))
    orientation: np.ndarray = field(default_factory=lambda: np.array([1,0,0,0]))

    # Noise model
    rgb_noise_std: float = 0.0
    depth_noise_std: float = 0.001  # meters


class PerceptionInterface:
    """Interface to perception systems."""

    def __init__(self, engine: PhysicsEngine):
        self.engine = engine
        self.cameras: dict[str, CameraConfig] = {}

    def add_camera(self, config: CameraConfig) -> None:
        """Add a virtual camera to the scene."""

    def get_rgb_image(self, camera_name: str) -> np.ndarray:
        """Get RGB image (H, W, 3) uint8."""

    def get_depth_image(self, camera_name: str) -> np.ndarray:
        """Get depth image (H, W) float32 in meters."""

    def get_segmentation_mask(self, camera_name: str) -> np.ndarray:
        """Get instance segmentation mask (H, W) int32."""

    def get_point_cloud(
        self,
        camera_name: str,
        organized: bool = True
    ) -> np.ndarray:
        """Get point cloud (H, W, 3) or (N, 3) in camera frame."""

    def project_points(
        self,
        camera_name: str,
        points_world: np.ndarray
    ) -> np.ndarray:
        """Project 3D points to image coordinates."""

    def estimate_object_pose(
        self,
        camera_name: str,
        object_id: str,
        method: str = "icp"
    ) -> tuple[np.ndarray, float]:
        """Estimate object pose from depth, return (pose, confidence)."""
```

### 2.4 Scene and Object Management

**Purpose**: Dynamic scene manipulation for manipulation tasks.

```python
@dataclass
class ObjectProperties:
    """Physical properties of an object."""
    name: str
    mesh_path: str | None = None
    primitive: str | None = None  # "box", "sphere", "cylinder"
    dimensions: np.ndarray | None = None

    mass: float = 1.0
    inertia: np.ndarray | None = None  # 3x3 or principal moments
    friction: float = 0.5
    restitution: float = 0.1

    # For visual
    color: np.ndarray = field(default_factory=lambda: np.array([0.5, 0.5, 0.5, 1.0]))


class SceneManager:
    """Manage dynamic objects in the scene."""

    def __init__(self, engine: PhysicsEngine):
        self.engine = engine
        self.objects: dict[str, ObjectProperties] = {}

    def spawn_object(
        self,
        properties: ObjectProperties,
        pose: np.ndarray  # 7D: xyz + quaternion
    ) -> str:
        """Spawn object, return unique ID."""

    def remove_object(self, object_id: str) -> bool:
        """Remove object from scene."""

    def get_object_pose(self, object_id: str) -> np.ndarray:
        """Get object pose (7D)."""

    def set_object_pose(self, object_id: str, pose: np.ndarray) -> None:
        """Teleport object to pose (for reset)."""

    def get_object_velocity(self, object_id: str) -> np.ndarray:
        """Get object twist (6D: linear + angular)."""

    def apply_wrench(
        self,
        object_id: str,
        wrench: np.ndarray,
        point: np.ndarray | None = None
    ) -> None:
        """Apply external wrench to object."""
```

---

## Phase 3: Learning and Adaptation

### 3.1 Reinforcement Learning Integration

**Purpose**: Train controllers using modern RL algorithms.

```python
class RoboticsGymEnv(gymnasium.Env):
    """Gymnasium environment wrapping UpstreamDrift simulation."""

    def __init__(
        self,
        engine_type: EngineType = EngineType.MUJOCO,
        model_path: str | None = None,
        task: TaskConfig | None = None,
        obs_config: ObservationConfig | None = None,
        action_config: ActionConfig | None = None,
        reward_config: RewardConfig | None = None,
    ):
        self.engine = create_engine(engine_type)
        if model_path:
            self.engine.load_from_path(model_path)

        self.task = task or DefaultTask()
        self.obs_config = obs_config or DefaultObservation()
        self.action_config = action_config or TorqueAction()
        self.reward_config = reward_config or DefaultReward()

        # Define spaces
        self.observation_space = self._build_observation_space()
        self.action_space = self._build_action_space()

    def step(self, action: np.ndarray) -> tuple[np.ndarray, float, bool, bool, dict]:
        """Execute action, return (obs, reward, terminated, truncated, info)."""

    def reset(
        self,
        seed: int | None = None,
        options: dict | None = None
    ) -> tuple[np.ndarray, dict]:
        """Reset environment, return (obs, info)."""

    def render(self) -> np.ndarray | None:
        """Render frame for visualization."""


@dataclass
class ObservationConfig:
    """Configure observation space."""
    include_joint_pos: bool = True
    include_joint_vel: bool = True
    include_joint_torque: bool = False
    include_ee_pos: bool = False
    include_ee_vel: bool = False
    include_contact_forces: bool = False
    include_imu: bool = False
    include_privileged: bool = False  # True for teacher policies

    # Noise models
    position_noise_std: float = 0.0
    velocity_noise_std: float = 0.0

    # History
    history_length: int = 1  # Stack previous observations


@dataclass
class RewardConfig:
    """Configure reward function."""
    task_reward_weight: float = 1.0
    energy_penalty_weight: float = 0.001
    smoothness_penalty_weight: float = 0.0001
    contact_penalty_weight: float = 0.0

    # Shaping
    use_potential_shaping: bool = False
    alive_bonus: float = 0.0


# Pre-built task environments
class HumanoidWalkEnv(RoboticsGymEnv):
    """Humanoid walking task."""
    pass

class HumanoidStandEnv(RoboticsGymEnv):
    """Humanoid standing/balance task."""
    pass

class ManipulationPickPlaceEnv(RoboticsGymEnv):
    """Pick and place manipulation."""
    pass

class DualArmManipulationEnv(RoboticsGymEnv):
    """Bimanual manipulation tasks."""
    pass
```

### 3.2 Imitation Learning Framework

**Purpose**: Learn from demonstrations and motion capture.

```python
@dataclass
class Demonstration:
    """A single demonstration trajectory."""
    timestamps: np.ndarray          # (T,)
    joint_positions: np.ndarray     # (T, n_q)
    joint_velocities: np.ndarray    # (T, n_v)
    actions: np.ndarray | None      # (T, n_u) if available

    # Optional
    end_effector_poses: np.ndarray | None = None  # (T, 7)
    contact_states: list[list[ContactState]] | None = None

    # Metadata
    task_id: str | None = None
    success: bool = True
    source: str = "teleoperation"  # or "motion_capture", "optimization"


class DemonstrationDataset:
    """Dataset of demonstration trajectories."""

    def __init__(self, demonstrations: list[Demonstration] | None = None):
        self.demonstrations = demonstrations or []

    def add(self, demo: Demonstration) -> None:
        """Add demonstration to dataset."""

    def save(self, path: str) -> None:
        """Save dataset to disk."""

    @classmethod
    def load(cls, path: str) -> "DemonstrationDataset":
        """Load dataset from disk."""

    def to_transitions(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Convert to (states, actions, next_states) for BC."""

    def augment(
        self,
        noise_std: float = 0.01,
        num_augmentations: int = 5
    ) -> "DemonstrationDataset":
        """Augment demonstrations with noise."""


class ImitationLearner:
    """Base class for imitation learning algorithms."""

    def __init__(
        self,
        observation_space: gymnasium.Space,
        action_space: gymnasium.Space,
        device: str = "cuda"
    ):
        self.observation_space = observation_space
        self.action_space = action_space
        self.device = device

    def train(
        self,
        dataset: DemonstrationDataset,
        epochs: int = 100,
        batch_size: int = 256
    ) -> dict[str, list[float]]:
        """Train policy on demonstrations."""

    def predict(self, observation: np.ndarray) -> np.ndarray:
        """Predict action for observation."""

    def save(self, path: str) -> None:
        """Save trained policy."""

    def load(self, path: str) -> None:
        """Load trained policy."""


class BehaviorCloning(ImitationLearner):
    """Supervised learning from demonstrations."""
    pass

class DAgger(ImitationLearner):
    """Dataset Aggregation with expert queries."""

    def train_online(
        self,
        env: RoboticsGymEnv,
        expert: Callable[[np.ndarray], np.ndarray],
        iterations: int = 10,
        trajectories_per_iter: int = 10
    ) -> dict:
        """Online training with expert intervention."""

class GAIL(ImitationLearner):
    """Generative Adversarial Imitation Learning."""
    pass
```

### 3.3 Motion Retargeting

**Purpose**: Transfer motion between different embodiments.

```python
class MotionRetargeter:
    """Retarget motion between skeleton types."""

    def __init__(
        self,
        source_skeleton: SkeletonConfig,
        target_skeleton: SkeletonConfig
    ):
        self.source = source_skeleton
        self.target = target_skeleton
        self._joint_mapping = self._compute_joint_mapping()

    def retarget(
        self,
        source_motion: np.ndarray,  # (T, n_source)
        method: str = "optimization"  # or "direct", "ik"
    ) -> np.ndarray:
        """Retarget motion to target skeleton."""

    def retarget_from_mocap(
        self,
        marker_positions: np.ndarray,  # (T, n_markers, 3)
        marker_names: list[str]
    ) -> np.ndarray:
        """Retarget from motion capture markers."""


@dataclass
class SkeletonConfig:
    """Skeleton configuration for retargeting."""
    joint_names: list[str]
    parent_indices: list[int]
    joint_offsets: np.ndarray  # (n_joints, 3) T-pose offsets
    joint_axes: np.ndarray     # (n_joints, 3) rotation axes

    # Semantic labels for mapping
    semantic_labels: dict[str, str] = field(default_factory=dict)
    # e.g., {"left_shoulder": "shoulder_L", "right_hip": "hip_R"}
```

### 3.4 Sim-to-Real Transfer

**Purpose**: Prepare learned policies for real robot deployment.

```python
@dataclass
class DomainRandomizationConfig:
    """Configuration for domain randomization."""

    # Dynamics
    mass_range: tuple[float, float] = (0.8, 1.2)  # multiplier
    friction_range: tuple[float, float] = (0.5, 1.5)
    damping_range: tuple[float, float] = (0.8, 1.2)
    motor_strength_range: tuple[float, float] = (0.9, 1.1)

    # Delays and noise
    action_delay_range: tuple[int, int] = (0, 3)  # timesteps
    observation_delay_range: tuple[int, int] = (0, 2)
    observation_noise_std: float = 0.01
    action_noise_std: float = 0.01

    # Environment
    gravity_range: tuple[float, float] = (9.5, 10.1)
    floor_friction_range: tuple[float, float] = (0.5, 1.5)


class DomainRandomizer:
    """Apply domain randomization to simulation."""

    def __init__(
        self,
        engine: PhysicsEngine,
        config: DomainRandomizationConfig
    ):
        self.engine = engine
        self.config = config

    def randomize(self, seed: int | None = None) -> dict[str, float]:
        """Randomize simulation parameters, return applied values."""

    def reset_to_nominal(self) -> None:
        """Reset to nominal (non-randomized) parameters."""


class SystemIdentifier:
    """Identify real robot parameters from data."""

    def __init__(self, model: PhysicsEngine):
        self.model = model

    def identify_from_trajectories(
        self,
        trajectories: list[Demonstration],
        params_to_identify: list[str]
    ) -> dict[str, float]:
        """Identify parameters from real robot trajectories."""

    def compute_reality_gap(
        self,
        sim_trajectory: np.ndarray,
        real_trajectory: np.ndarray
    ) -> float:
        """Quantify sim-to-real gap."""
```

---

## Phase 4: Industrial Deployment

### 4.1 Real-Time Control Interface

**Purpose**: Interface with real robot hardware.

```python
class RealTimeController:
    """Real-time control loop manager."""

    def __init__(
        self,
        control_frequency: float = 1000.0,  # Hz
        communication_type: str = "ethercat"  # or "ros2", "udp"
    ):
        self.dt = 1.0 / control_frequency
        self.comm_type = communication_type

    def connect(self, robot_config: RobotConfig) -> bool:
        """Connect to real robot."""

    def disconnect(self) -> None:
        """Safely disconnect."""

    def set_control_callback(
        self,
        callback: Callable[[RobotState], ControlCommand]
    ) -> None:
        """Set the control callback (called at control_frequency)."""

    def start(self) -> None:
        """Start real-time control loop."""

    def stop(self) -> None:
        """Stop control loop."""

    def get_timing_stats(self) -> TimingStatistics:
        """Get control loop timing statistics."""


@dataclass
class RobotState:
    """State received from real robot."""
    timestamp: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    joint_torques: np.ndarray

    # Optional sensors
    ft_wrenches: dict[str, np.ndarray] | None = None
    imu_data: IMUReading | None = None
    contact_states: list[bool] | None = None


@dataclass
class ControlCommand:
    """Command sent to real robot."""
    timestamp: float
    mode: ControlMode

    # Depending on mode:
    position_targets: np.ndarray | None = None
    velocity_targets: np.ndarray | None = None
    torque_commands: np.ndarray | None = None

    # Feed-forward
    feedforward_torque: np.ndarray | None = None

    # Impedance parameters (if supported)
    stiffness: np.ndarray | None = None
    damping: np.ndarray | None = None
```

### 4.2 Digital Twin Framework

**Purpose**: Synchronized simulation for monitoring and prediction.

```python
class DigitalTwin:
    """Digital twin of a real robot."""

    def __init__(
        self,
        sim_engine: PhysicsEngine,
        real_interface: RealTimeController
    ):
        self.sim = sim_engine
        self.real = real_interface
        self._state_estimator = StateEstimator()

    def synchronize(self) -> float:
        """Sync simulation state with real robot, return sync error."""

    def predict(self, horizon: float, control_sequence: np.ndarray) -> np.ndarray:
        """Predict future trajectory given control sequence."""

    def detect_anomaly(self) -> AnomalyReport | None:
        """Detect discrepancy between simulation and real."""

    def get_estimated_contacts(self) -> list[ContactState]:
        """Estimate contact states from force measurements."""

    def compute_virtual_forces(self) -> np.ndarray:
        """Compute forces that would explain state discrepancy."""


@dataclass
class AnomalyReport:
    """Report of detected anomaly."""
    timestamp: float
    anomaly_type: str  # "collision", "slip", "stuck", "model_mismatch"
    severity: float    # 0.0 to 1.0
    affected_joints: list[int]
    recommended_action: str
```

### 4.3 Safety System

**Purpose**: Ensure safe operation in human environments.

```python
class SafetyMonitor:
    """Real-time safety monitoring."""

    def __init__(self, robot_config: RobotConfig):
        self.config = robot_config
        self._limits = SafetyLimits.from_config(robot_config)
        self._human_detector = HumanDetector()

    def check_state(self, state: RobotState) -> SafetyStatus:
        """Check if current state is safe."""

    def check_command(self, command: ControlCommand) -> SafetyStatus:
        """Check if command would result in safe state."""

    def compute_safe_command(
        self,
        desired: ControlCommand,
        state: RobotState
    ) -> ControlCommand:
        """Modify command to ensure safety."""

    def get_stopping_distance(
        self,
        state: RobotState,
        body: str
    ) -> float:
        """Compute minimum stopping distance for a body."""

    def set_speed_override(self, factor: float) -> None:
        """Set speed reduction factor (0.0 to 1.0)."""


@dataclass
class SafetyLimits:
    """Safety limits for robot operation."""
    max_joint_velocity: np.ndarray
    max_joint_torque: np.ndarray
    max_cartesian_velocity: float  # m/s
    max_cartesian_force: float     # N

    # Workspace limits
    workspace_bounds: np.ndarray   # (6,) [x_min, x_max, y_min, ...]
    forbidden_zones: list[np.ndarray]  # List of (6,) boxes

    # Human interaction
    max_contact_force: float = 150.0  # N (ISO 10218-1)
    max_pressure: float = 110.0       # N/cm^2


class CollisionAvoidance:
    """Real-time collision avoidance."""

    def __init__(
        self,
        robot_model: PhysicsEngine,
        safety_distance: float = 0.1  # meters
    ):
        self.model = robot_model
        self.safety_distance = safety_distance
        self._obstacles: list[Obstacle] = []

    def add_obstacle(self, obstacle: Obstacle) -> None:
        """Add obstacle to collision checking."""

    def update_human_position(self, human_state: HumanState) -> None:
        """Update detected human position."""

    def compute_repulsive_field(
        self,
        state: RobotState
    ) -> np.ndarray:
        """Compute artificial potential field repulsion."""

    def check_path_clearance(
        self,
        trajectory: np.ndarray,
        min_distance: float | None = None
    ) -> tuple[bool, float]:
        """Check if trajectory maintains clearance."""
```

### 4.4 Teleoperation System

**Purpose**: Enable human-in-the-loop control and demonstration collection.

```python
class TeleoperationInterface:
    """Interface for robot teleoperation."""

    def __init__(
        self,
        robot: PhysicsEngine,
        input_device: InputDevice
    ):
        self.robot = robot
        self.input = input_device
        self._scaling = 1.0
        self._clutch_engaged = True

    def set_control_mode(self, mode: TeleoperationMode) -> None:
        """Set teleoperation mode (position, velocity, wrench)."""

    def set_workspace_mapping(
        self,
        leader_frame: np.ndarray,
        follower_frame: np.ndarray,
        scaling: float = 1.0
    ) -> None:
        """Configure workspace mapping between leader and follower."""

    def update(self) -> ControlCommand:
        """Process input and generate control command."""

    def get_haptic_feedback(self) -> np.ndarray:
        """Get force feedback for haptic device."""

    def start_demonstration_recording(self) -> None:
        """Begin recording demonstration."""

    def stop_demonstration_recording(self) -> Demonstration:
        """Stop recording and return demonstration."""


class InputDevice(Protocol):
    """Protocol for teleoperation input devices."""

    def get_pose(self) -> np.ndarray:
        """Get current device pose (7D)."""

    def get_twist(self) -> np.ndarray:
        """Get current device velocity (6D)."""

    def get_gripper_state(self) -> float:
        """Get gripper command (0.0 = closed, 1.0 = open)."""

    def set_force_feedback(self, wrench: np.ndarray) -> None:
        """Set haptic force feedback (6D)."""

    def get_buttons(self) -> dict[str, bool]:
        """Get button states."""


# Supported input devices
class SpaceMouseInput(InputDevice):
    """3Dconnexion SpaceMouse."""
    pass

class VRControllerInput(InputDevice):
    """VR controller (Oculus, Vive, etc.)."""
    pass

class HapticDeviceInput(InputDevice):
    """Haptic devices (Phantom, Sigma.7)."""
    pass

class KeyboardMouseInput(InputDevice):
    """Keyboard + mouse fallback."""
    pass
```

---

## Phase 5: Advanced Research Capabilities

### 5.1 Model Predictive Control (MPC)

**Purpose**: Optimal control with constraints and predictions.

```python
class ModelPredictiveController:
    """Nonlinear Model Predictive Control."""

    def __init__(
        self,
        model: PhysicsEngine,
        horizon: int = 20,
        dt: float = 0.01
    ):
        self.model = model
        self.horizon = horizon
        self.dt = dt
        self._solver = self._create_solver()

    def set_cost_function(self, cost: CostFunction) -> None:
        """Set the running and terminal cost."""

    def set_constraints(self, constraints: list[Constraint]) -> None:
        """Set state and control constraints."""

    def solve(
        self,
        initial_state: np.ndarray,
        reference_trajectory: np.ndarray | None = None
    ) -> MPCResult:
        """Solve MPC problem, return optimal trajectory and control."""

    def get_first_control(self) -> np.ndarray:
        """Get first control input to apply (receding horizon)."""


@dataclass
class CostFunction:
    """MPC cost function specification."""

    # Running cost: l(x, u) = x^T Q x + u^T R u + q^T x + r^T u
    Q: np.ndarray  # State cost matrix
    R: np.ndarray  # Control cost matrix
    q: np.ndarray | None = None  # Linear state cost
    r: np.ndarray | None = None  # Linear control cost

    # Terminal cost: V(x) = x^T P x + p^T x
    P: np.ndarray | None = None
    p: np.ndarray | None = None

    # Reference tracking
    x_ref: np.ndarray | None = None  # Reference trajectory
    u_ref: np.ndarray | None = None  # Reference controls


class CentroidalMPC(ModelPredictiveController):
    """MPC using centroidal dynamics (for locomotion)."""
    pass

class WholeBodyMPC(ModelPredictiveController):
    """MPC using full rigid-body dynamics."""
    pass
```

### 5.2 Differentiable Physics

**Purpose**: Enable gradient-based optimization through physics.

```python
class DifferentiableEngine:
    """Differentiable physics simulation."""

    def __init__(self, engine: PhysicsEngine):
        self.engine = engine
        self._autodiff_backend = "jax"  # or "torch", "drake"

    def simulate_trajectory(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,  # (T, n_u)
        dt: float
    ) -> np.ndarray:
        """Forward simulation, returns state trajectory (T+1, n_q+n_v)."""

    def compute_gradient(
        self,
        initial_state: np.ndarray,
        controls: np.ndarray,
        loss_fn: Callable[[np.ndarray], float]
    ) -> np.ndarray:
        """Compute gradient of loss w.r.t. controls."""

    def optimize_trajectory(
        self,
        initial_state: np.ndarray,
        goal_state: np.ndarray,
        horizon: int,
        method: str = "adam"
    ) -> tuple[np.ndarray, np.ndarray]:
        """Optimize trajectory to reach goal, return (states, controls)."""


class ContactDifferentiableEngine(DifferentiableEngine):
    """Differentiable simulation through contact."""

    def __init__(self, engine: PhysicsEngine, contact_method: str = "smoothed"):
        super().__init__(engine)
        self.contact_method = contact_method  # "smoothed", "randomized", "stochastic"
```

### 5.3 Deformable Object Simulation

**Purpose**: Simulate soft objects, cables, and cloth.

```python
class DeformableObject:
    """Base class for deformable objects."""

    def __init__(self, mesh: np.ndarray, material: MaterialProperties):
        self.mesh = mesh
        self.material = material

    def get_node_positions(self) -> np.ndarray:
        """Get current node positions (N, 3)."""

    def get_node_velocities(self) -> np.ndarray:
        """Get current node velocities (N, 3)."""

    def apply_external_force(
        self,
        node_indices: np.ndarray,
        forces: np.ndarray
    ) -> None:
        """Apply forces to specific nodes."""


class SoftBody(DeformableObject):
    """Volumetric soft body (FEM)."""
    pass

class Cable(DeformableObject):
    """1D deformable cable/rope."""

    def get_length(self) -> float:
        """Get current cable length."""

    def get_tension(self) -> float:
        """Get cable tension."""


class Cloth(DeformableObject):
    """2D deformable cloth/fabric."""

    def attach_to_body(
        self,
        body_id: str,
        attachment_nodes: np.ndarray
    ) -> None:
        """Attach cloth nodes to rigid body."""


@dataclass
class MaterialProperties:
    """Material properties for deformable objects."""
    youngs_modulus: float = 1e6      # Pa
    poisson_ratio: float = 0.3
    density: float = 1000.0          # kg/m^3
    damping: float = 0.01

    # For cloth
    bending_stiffness: float | None = None
    shear_stiffness: float | None = None
```

### 5.4 Multi-Robot Coordination

**Purpose**: Simulate and control multiple robots.

```python
class MultiRobotSystem:
    """Manage multiple robots in shared environment."""

    def __init__(self):
        self.robots: dict[str, PhysicsEngine] = {}
        self._coordinator = TaskCoordinator()

    def add_robot(
        self,
        robot_id: str,
        engine: PhysicsEngine,
        base_pose: np.ndarray
    ) -> None:
        """Add robot to the system."""

    def remove_robot(self, robot_id: str) -> None:
        """Remove robot from system."""

    def step_all(self, dt: float) -> None:
        """Step all robots synchronously."""

    def check_inter_robot_collision(self) -> list[tuple[str, str]]:
        """Check for collisions between robots."""

    def allocate_tasks(
        self,
        tasks: list[Task]
    ) -> dict[str, list[Task]]:
        """Allocate tasks to robots."""


class FormationController:
    """Control robot formations."""

    def __init__(self, robots: list[str], formation: FormationConfig):
        self.robots = robots
        self.formation = formation

    def compute_formation_control(
        self,
        leader_pose: np.ndarray
    ) -> dict[str, np.ndarray]:
        """Compute control for each robot to maintain formation."""

    def set_formation(self, formation: FormationConfig) -> None:
        """Change formation."""


class CooperativeManipulation:
    """Coordinated manipulation by multiple robots."""

    def __init__(self, robots: list[PhysicsEngine], object_model: str):
        self.robots = robots
        self._object = object_model

    def compute_load_sharing(
        self,
        desired_object_wrench: np.ndarray
    ) -> list[np.ndarray]:
        """Compute wrench distribution among robots."""

    def plan_cooperative_motion(
        self,
        object_goal_pose: np.ndarray
    ) -> list[np.ndarray]:
        """Plan coordinated motion trajectories."""
```

---

## Architecture Extensions

### New Protocol Extensions

```python
# Extend PhysicsEngine protocol for humanoid capabilities
@runtime_checkable
class HumanoidEngine(PhysicsEngine, Protocol):
    """Extended protocol for humanoid robots."""

    @abstractmethod
    def get_com_position(self) -> np.ndarray:
        """Get center of mass position [3]."""

    @abstractmethod
    def get_com_velocity(self) -> np.ndarray:
        """Get center of mass velocity [3]."""

    @abstractmethod
    def compute_centroidal_momentum(self) -> np.ndarray:
        """Compute 6D centroidal momentum [linear, angular]."""

    @abstractmethod
    def compute_centroidal_momentum_matrix(self) -> np.ndarray:
        """Compute CMM: h = A(q) @ v."""

    @abstractmethod
    def get_foot_contacts(self) -> dict[str, ContactState]:
        """Get contact state for each foot."""


@runtime_checkable
class ManipulationEngine(PhysicsEngine, Protocol):
    """Extended protocol for manipulation robots."""

    @abstractmethod
    def get_end_effector_pose(self, ee_name: str) -> np.ndarray:
        """Get end-effector pose (7D: xyz + quat)."""

    @abstractmethod
    def get_end_effector_velocity(self, ee_name: str) -> np.ndarray:
        """Get end-effector twist (6D)."""

    @abstractmethod
    def compute_ee_jacobian(self, ee_name: str) -> np.ndarray:
        """Compute 6xn end-effector Jacobian."""

    @abstractmethod
    def solve_ik(
        self,
        ee_name: str,
        target_pose: np.ndarray,
        seed: np.ndarray | None = None
    ) -> tuple[np.ndarray, bool]:
        """Solve inverse kinematics, return (q, success)."""

    @abstractmethod
    def get_gripper_state(self, gripper_name: str) -> GripperState:
        """Get gripper state (position, force, object detected)."""
```

### New Analysis Mixins

```python
class LocomotionMetricsMixin:
    """Analysis metrics for locomotion."""

    def compute_cost_of_transport(
        self,
        distance: float,
        energy: float,
        mass: float
    ) -> float:
        """Compute dimensionless CoT = E / (m * g * d)."""

    def compute_froude_number(
        self,
        velocity: float,
        leg_length: float
    ) -> float:
        """Compute Froude number Fr = v^2 / (g * L)."""

    def compute_step_width(
        self,
        left_foot_positions: np.ndarray,
        right_foot_positions: np.ndarray
    ) -> float:
        """Compute average step width."""

    def compute_gait_symmetry(
        self,
        left_phase_durations: np.ndarray,
        right_phase_durations: np.ndarray
    ) -> float:
        """Compute gait symmetry index (1.0 = perfect symmetry)."""


class ManipulationMetricsMixin:
    """Analysis metrics for manipulation."""

    def compute_task_completion_time(
        self,
        task_start: float,
        task_end: float
    ) -> float:
        """Compute task completion time."""

    def compute_path_efficiency(
        self,
        actual_path: np.ndarray,
        optimal_path: np.ndarray
    ) -> float:
        """Compute path length ratio (optimal/actual)."""

    def compute_smoothness(
        self,
        trajectory: np.ndarray,
        dt: float
    ) -> float:
        """Compute trajectory smoothness (integrated jerk)."""

    def compute_grasp_quality(
        self,
        contact_points: np.ndarray,
        contact_normals: np.ndarray,
        friction: float
    ) -> float:
        """Compute grasp quality metric."""
```

---

## Implementation Priorities

### Tier 1: Foundation (Months 1-3)

| Component | Priority | Effort | Dependencies |
|-----------|----------|--------|--------------|
| Whole-Body Control (HQP) | Critical | High | None |
| Bipedal Locomotion (LIPM) | Critical | High | WBC |
| Contact Manager | Critical | Medium | None |
| F/T Sensor Framework | High | Low | None |
| Collision Checker | High | Medium | None |

### Tier 2: Planning & Perception (Months 4-6)

| Component | Priority | Effort | Dependencies |
|-----------|----------|--------|--------------|
| RRT/RRT* Planners | Critical | Medium | Collision |
| CHOMP/TrajOpt | High | High | Collision |
| Perception Interface | High | Medium | None |
| Scene Manager | Medium | Low | None |

### Tier 3: Learning (Months 7-9)

| Component | Priority | Effort | Dependencies |
|-----------|----------|--------|--------------|
| Gymnasium Environments | Critical | Medium | None |
| Domain Randomization | High | Low | Gym Envs |
| Behavior Cloning | High | Medium | Gym Envs |
| Motion Retargeting | Medium | High | None |

### Tier 4: Industrial (Months 10-12)

| Component | Priority | Effort | Dependencies |
|-----------|----------|--------|--------------|
| Real-Time Interface | Critical | High | None |
| Digital Twin | High | High | RT Interface |
| Safety System | Critical | Medium | Collision |
| Teleoperation | High | Medium | RT Interface |

### Tier 5: Research (Ongoing)

| Component | Priority | Effort | Dependencies |
|-----------|----------|--------|--------------|
| MPC Framework | High | High | WBC |
| Differentiable Physics | Medium | Very High | None |
| Deformable Objects | Low | Very High | None |
| Multi-Robot | Medium | High | Scene Manager |

---

## Recommended Humanoid Models

### Open-Source Models for Integration

| Model | DOF | Features | License |
|-------|-----|----------|---------|
| **Unitree H1** | 19 | Industrial humanoid, torque control | Apache 2.0 |
| **Unitree G1** | 23 | Compact humanoid, dexterous hands | Apache 2.0 |
| **Boston Dynamics Atlas** | 28 | High-performance locomotion | Research only |
| **Agility Digit** | 30 | Commercial humanoid | Contact |
| **NVIDIA Isaac Humanoid** | 25+ | Simulation optimized | Apache 2.0 |
| **MuJoCo Humanoid** | 21 | Standard benchmark | Apache 2.0 |
| **MyoSuite Full Body** | 52 | Muscle-actuated | Apache 2.0 |

### Recommended Arms for Manipulation

| Model | DOF | Payload | Features |
|-------|-----|---------|----------|
| **Franka Emika** | 7 | 3kg | Torque sensing, impedance control |
| **KUKA iiwa** | 7 | 7/14kg | Industrial, certified |
| **Universal Robots** | 6 | 3-16kg | Collaborative, easy programming |
| **Kinova Gen3** | 7 | 4kg | Lightweight, vision integrated |

---

## Technology Stack Recommendations

### Physics Backends

- **Primary**: MuJoCo 3.x (fastest, best contact)
- **Optimization**: Drake (trajectory optimization, mathematical programming)
- **Real-time**: Pinocchio (analytical derivatives, C++ performance)
- **Biomechanics**: OpenSim/MyoSuite (muscle models, validation)

### Learning Frameworks

- **RL**: Stable-Baselines3, CleanRL, RSL-RL (for locomotion)
- **Imitation**: robomimic, diffusion_policy
- **Differentiable**: Brax, DiffTaichi, Drake's autodiff

### Real-Time Middleware

- **ROS 2 Humble/Iron**: Standard robotics middleware
- **EtherCAT**: Industrial real-time communication
- **ZeroMQ**: High-performance IPC

### Visualization

- **MuJoCo Viewer**: Native 3D visualization
- **Meshcat**: Web-based 3D visualization
- **RViz2**: ROS 2 visualization
- **Rerun**: Modern ML visualization

---

## Conclusion

This proposal outlines a comprehensive path from the current golf biomechanics platform to a full-featured industrial humanoid robotics suite. The existing architecture's strengths (multi-engine support, clean protocols, advanced kinematics/control) provide an excellent foundation.

The phased approach allows for incremental value delivery while building toward the ambitious goal of a unified platform for:

- Humanoid locomotion research
- Industrial manipulation
- Human-robot collaboration
- Learning-based control
- Digital twin deployment

**Estimated Total Effort**: 12-18 months with dedicated team
**Recommended Team Size**: 3-5 engineers + 1-2 researchers

---

## Appendix A: File Structure Preview

```
src/
├── engines/
│   ├── physics_engines/     # Existing
│   ├── locomotion/          # NEW: Gait generation, balance
│   ├── manipulation/        # NEW: Grasping, object handling
│   └── perception/          # NEW: Vision, sensing
├── control/
│   ├── whole_body/          # NEW: HQP, WBC
│   ├── mpc/                  # NEW: Model predictive control
│   └── safety/              # NEW: Safety monitoring
├── planning/
│   ├── motion/              # NEW: Motion planners
│   ├── task/                # NEW: Task planners
│   └── footstep/            # NEW: Footstep planners
├── learning/
│   ├── rl/                  # NEW: RL environments
│   ├── imitation/           # NEW: IL algorithms
│   └── sim2real/            # NEW: Transfer methods
├── deployment/
│   ├── realtime/            # NEW: RT control
│   ├── digital_twin/        # NEW: Twin framework
│   └── teleoperation/       # NEW: Teleop interfaces
└── shared/
    ├── python/              # Existing: interfaces, analysis
    ├── models/              # Existing + NEW: humanoid URDFs
    └── configs/             # NEW: Robot configs
```

---

*Document prepared for UpstreamDrift strategic planning.*

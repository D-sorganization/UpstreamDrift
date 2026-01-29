# Claude Code Instructions: RL-Based Golf Swing Correction Implementation

## Project Context

**Repository**: `Golf_Modeling_Suite` (https://github.com/D-sorganization/Golf_Modeling_Suite)

**Objective**: Implement the reinforcement learning-based golf swing correction framework from Lee et al. (2026) "Reinforcement Learning-Based Golf Swing Correction Framework Incorporating Temporal Rhythm and Kinematic Stability" (Sensors, 26, 392).

**Developer Context**: The user is relatively new to coding but capable of implementing advanced control schemes. Provide detailed explanations, use type hints extensively, include comprehensive docstrings, and flag potential pitfalls.

---

## Phase 1: Project Setup and Module Structure

### Task 1.1: Create Module Directory Structure

Create the following directory structure under `src/`:

```
src/
└── correction/
    ├── __init__.py
    ├── environment.py
    ├── rewards/
    │   ├── __init__.py
    │   ├── pose_accuracy.py
    │   ├── improvement_rate.py
    │   ├── hip_stability.py
    │   └── velocity_dtw.py
    ├── preprocessing/
    │   ├── __init__.py
    │   ├── temporal_alignment.py
    │   ├── spatial_normalization.py
    │   └── phase_detection.py
    ├── agents/
    │   ├── __init__.py
    │   └── ppo_corrector.py
    ├── data/
    │   ├── __init__.py
    │   └── pose_extraction.py
    └── evaluation/
        ├── __init__.py
        ├── metrics.py
        └── visualization.py
```

Also create:
```
examples/
└── correction/
    ├── train_corrector.py
    ├── evaluate_correction.py
    └── extract_keypoints.py

tests/
└── correction/
    ├── __init__.py
    ├── test_environment.py
    ├── test_preprocessing.py
    ├── test_rewards.py
    └── conftest.py

data/
└── swing_sequences/
    ├── expert/
    │   └── .gitkeep
    └── amateur/
        └── .gitkeep
```

### Task 1.2: Update Dependencies

Add the following to `requirements.txt`:

```
# RL and ML
stable-baselines3>=2.0.0
gymnasium>=0.29.0
tensorboard>=2.15.0

# Pose estimation
mediapipe>=0.10.0
opencv-python>=4.8.0

# Signal processing
dtaidistance>=2.3.0
scipy>=1.11.0

# Already in project (verify present)
numpy>=1.24.0
matplotlib>=3.7.0
```

Add to `environment.yml` under dependencies:

```yaml
  - stable-baselines3>=2.0
  - gymnasium>=0.29
  - mediapipe>=0.10
  - dtaidistance>=2.3
  - tensorboard>=2.15
  - opencv>=4.8
```

### Task 1.3: Create Module `__init__.py` Files

**`src/correction/__init__.py`**:
```python
"""
RL-based Golf Swing Correction Module.

Implements the framework from Lee et al. (2026) for correcting amateur
golf swings toward expert patterns using Proximal Policy Optimization
with temporal rhythm and kinematic stability rewards.

Main components:
- environment: Gymnasium environment for swing correction MDP
- preprocessing: Temporal alignment and spatial normalization
- rewards: Multi-term reward function components
- agents: PPO policy wrapper and training utilities
- evaluation: Metrics and visualization tools
"""

from .environment import GolfSwingCorrectionEnv

__all__ = ["GolfSwingCorrectionEnv"]
__version__ = "0.1.0"
```

---

## Phase 2: Core Environment Implementation

### Task 2.1: Implement Base Environment

**File**: `src/correction/environment.py`

**Requirements**:
1. Implement `GolfSwingCorrectionEnv` as a Gymnasium environment
2. State space: Flattened normalized joint coordinates `(N_joints * 2,)`
3. Action space: Continuous correction deltas per joint coordinate
4. Transition dynamics: `s_{t+1} = s_t + a_t` (deterministic)
5. Episode terminates at final frame of swing sequence
6. Multi-term reward function combining:
   - Pose accuracy (L2 distance to expert)
   - Improvement rate (change in L2 from previous frame)
   - Hip alignment stability
   - Velocity-based rhythm consistency

**Key equations from paper**:
- Pose accuracy reward: `R_L2(t) = -α * (L2(t))^η` (Equation 21)
- Improvement reward: `R_ΔL2(t) = β * tanh(k * (L2(t-1) - L2(t)))` (Equation 22)
- Hip alignment: `R_hip(t) = -λ * (||h_L^u - h_L^e|| + ||h_R^u - h_R^e||)` (Equation 23)
- Velocity-DTW: `R_VDTW(t) = -γ * DTW(V^u, V^e)` (Equation 25)

**Default hyperparameters** (from paper Table 7, setting S0):
- `pose_accuracy_alpha`: 1.0
- `pose_accuracy_eta`: 2.0
- `improvement_beta`: 0.5
- `improvement_k`: 10.0
- `hip_lambda`: 0.3
- `rhythm_gamma`: 0.7

**Implementation notes**:
- Use `gymnasium` not deprecated `gym`
- Clip actions to prevent unrealistic corrections (default max magnitude: 0.1)
- Clip corrected coordinates to [-1, 1] after applying actions
- Store previous L2 error for improvement reward calculation
- Joint importance weights: higher for wrists/shoulders (1.5), moderate for hips (1.2), standard for others (1.0)

**Include methods**:
- `reset(seed, options) -> (observation, info)`
- `step(action) -> (observation, reward, terminated, truncated, info)`
- `_get_observation() -> np.ndarray`
- `_compute_l2_error(frame) -> float`
- `_compute_reward() -> float`
- `_compute_hip_reward(frame) -> float`
- `_compute_velocity_reward(frame) -> float`
- `get_corrected_sequence() -> np.ndarray`
- `render()` (optional, can use existing MuJoCo/Meshcat visualization)

---

### Task 2.2: Implement Individual Reward Components

Create separate files for each reward component for modularity and testing.

**File**: `src/correction/rewards/pose_accuracy.py`

Implement:
```python
def compute_weighted_l2_error(
    corrected_pose: np.ndarray,  # Shape: (N_joints, 2)
    expert_pose: np.ndarray,     # Shape: (N_joints, 2)
    joint_weights: np.ndarray,   # Shape: (N_joints,)
) -> float:
    """Compute weighted mean L2 error between poses."""

def pose_accuracy_reward(
    l2_error: float,
    alpha: float = 1.0,
    eta: float = 2.0,
) -> float:
    """R_L2 = -α * (L2)^η"""
```

**File**: `src/correction/rewards/improvement_rate.py`

Implement:
```python
def improvement_rate_reward(
    previous_l2: float,
    current_l2: float,
    beta: float = 0.5,
    k: float = 10.0,
) -> float:
    """R_ΔL2 = β * tanh(k * (L2_prev - L2_curr))"""
```

**File**: `src/correction/rewards/hip_stability.py`

Implement:
```python
def hip_alignment_reward(
    corrected_hips: np.ndarray,  # Shape: (2, 2) for left/right hip (x,y)
    expert_hips: np.ndarray,
    lambda_weight: float = 0.3,
) -> float:
    """R_hip = -λ * (||h_L^u - h_L^e|| + ||h_R^u - h_R^e||)"""
```

**File**: `src/correction/rewards/velocity_dtw.py`

Implement:
```python
def compute_velocity_sequence(positions: np.ndarray) -> np.ndarray:
    """v_i(t) = p_i(t) - p_i(t-1), returns shape (T-1, N_joints, 2)"""

def velocity_dtw_distance(
    seq1_positions: np.ndarray,
    seq2_positions: np.ndarray,
    per_joint: bool = False,
) -> float | np.ndarray:
    """
    Full Velocity-DTW distance using dtaidistance library.
    Operates on velocity magnitude for each joint.
    """

def per_frame_velocity_reward(
    corrected_vel: np.ndarray,  # Shape: (N_joints, 2)
    expert_vel: np.ndarray,
    gamma: float = 0.7,
) -> float:
    """
    Per-frame approximation for use during training.
    Full DTW is expensive; this provides per-step signal.
    """
```

**File**: `src/correction/rewards/__init__.py`
```python
from .pose_accuracy import compute_weighted_l2_error, pose_accuracy_reward
from .improvement_rate import improvement_rate_reward
from .hip_stability import hip_alignment_reward
from .velocity_dtw import (
    compute_velocity_sequence,
    velocity_dtw_distance,
    per_frame_velocity_reward,
)

__all__ = [
    "compute_weighted_l2_error",
    "pose_accuracy_reward",
    "improvement_rate_reward",
    "hip_alignment_reward",
    "compute_velocity_sequence",
    "velocity_dtw_distance",
    "per_frame_velocity_reward",
]
```

---

## Phase 3: Preprocessing Pipeline

### Task 3.1: Implement Temporal Alignment

**File**: `src/correction/preprocessing/temporal_alignment.py`

**Requirements**:
1. Phase-wise interpolation to align user sequence to expert frame count
2. Uses linear interpolation within each phase (Equations 3-4 from paper)
3. Prevents global temporal distortion by aligning phase-by-phase

**Implement**:
```python
def phase_wise_alignment(
    user_sequence: np.ndarray,      # Shape: (T_user, N_joints, 2)
    expert_sequence: np.ndarray,    # Shape: (T_expert, N_joints, 2)
    user_phase_boundaries: list[int],
    expert_phase_boundaries: list[int],
) -> tuple[np.ndarray, np.ndarray]:
    """
    Align user sequence to expert phase-by-phase.
    
    Returns aligned_user (same length as expert) and expert (unchanged).
    """

def _interpolate_phase(
    phase_data: np.ndarray,  # Shape: (T_phase, N_joints, 2)
    target_frames: int,
) -> np.ndarray:
    """
    Linear interpolation using normalized time τ ∈ [0, 1].
    Uses scipy.interpolate.interp1d.
    """
```

### Task 3.2: Implement Spatial Normalization

**File**: `src/correction/preprocessing/spatial_normalization.py`

**Requirements**:
1. Hip-centered coordinate system (Equation 5-6)
2. Scale by torso length (shoulder center to hip center) (Equations 7-9)
3. Clip to [-1, 1] range (Equation 10)

**Implement**:
```python
def hip_centered_normalization(
    sequence: np.ndarray,           # Shape: (T, N_joints, 2)
    joint_names: list[str],
    left_hip_name: str = "left_hip",
    right_hip_name: str = "right_hip",
    left_shoulder_name: str = "left_shoulder",
    right_shoulder_name: str = "right_shoulder",
    clip_range: float = 1.0,
) -> np.ndarray:
    """
    Apply hip-centered spatial normalization per frame.
    
    Steps:
    1. Compute hip center: c_hip = 0.5 * (h_L + h_R)
    2. Compute shoulder center: c_shoulder = 0.5 * (s_L + s_R)
    3. Compute scale: S = ||c_shoulder - c_hip||
    4. For each joint: p'' = (p - c_hip) / S
    5. Clip to [-clip_range, clip_range]
    """

def denormalize_sequence(
    normalized_sequence: np.ndarray,
    original_sequence: np.ndarray,
    joint_names: list[str],
    # ... same joint name params
) -> np.ndarray:
    """
    Reverse normalization for visualization.
    Useful for displaying corrected poses in original coordinate system.
    """
```

### Task 3.3: Implement Phase Detection

**File**: `src/correction/preprocessing/phase_detection.py`

**Requirements**:
1. Detect 8 canonical golf swing phases: address, takeaway, half, top, down_half, impact, follow_through, finish
2. Uses velocity-based features (movement start, velocity reversal at top, max speed at impact)
3. Fallback to uniform distribution if detection fails

**Implement**:
```python
SWING_PHASES = [
    "address",
    "takeaway",
    "half",
    "top",
    "down_half",
    "impact",
    "follow_through",
    "finish",
]

def detect_swing_phases(
    sequence: np.ndarray,       # Shape: (T, N_joints, 2)
    joint_names: list[str],
    fps: float = 30.0,
) -> list[int]:
    """
    Detect phase boundaries from keypoint sequence.
    
    Key events detected:
    - Movement start (address → takeaway): speed exceeds threshold
    - Top of backswing: horizontal velocity sign change
    - Impact: maximum speed
    - Finish: speed drops below threshold
    
    Returns list of frame indices: [0, t1, t2, ..., T]
    """

def _detect_movement_start(speed: np.ndarray, threshold_ratio: float = 0.02) -> int:
    """Find first frame where speed exceeds threshold."""

def _detect_top_of_backswing(velocities: np.ndarray, movement_start: int) -> int:
    """Find velocity reversal (sign change in horizontal velocity)."""

def _detect_impact(speed: np.ndarray) -> int:
    """Find frame with maximum speed."""

def _interpolate_phase_boundaries(
    T: int,
    address_end: int,
    top: int,
    impact: int,
    finish: int,
) -> list[int]:
    """Create 8 phase boundaries from key events."""
```

**File**: `src/correction/preprocessing/__init__.py`
```python
from .temporal_alignment import phase_wise_alignment
from .spatial_normalization import hip_centered_normalization, denormalize_sequence
from .phase_detection import detect_swing_phases, SWING_PHASES

__all__ = [
    "phase_wise_alignment",
    "hip_centered_normalization",
    "denormalize_sequence",
    "detect_swing_phases",
    "SWING_PHASES",
]
```

---

## Phase 4: Pose Extraction from Video

### Task 4.1: Implement BlazePose Extraction

**File**: `src/correction/data/pose_extraction.py`

**Requirements**:
1. Extract 2D keypoints from video using MediaPipe BlazePose
2. Map MediaPipe landmarks to paper's joint names
3. Handle missing detections with linear interpolation
4. Save extracted sequences as .npy files

**MediaPipe landmark indices to use**:
```python
BLAZEPOSE_TO_PAPER = {
    15: "left_wrist",
    16: "right_wrist",
    11: "left_shoulder",
    12: "right_shoulder",
    23: "left_hip",
    24: "right_hip",
    25: "left_knee",
    26: "right_knee",
    8: "right_ear",  # Head proxy
}
```

**Implement**:
```python
def extract_keypoints_from_video(
    video_path: str | Path,
    output_path: str | Path | None = None,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, list[str]]:
    """
    Extract 2D keypoints from golf swing video.
    
    Returns:
        keypoints: Shape (T, N_joints, 2), normalized coords [0, 1]
        joint_names: List of joint names in order
    """

def _interpolate_missing_frames(
    frames: list[np.ndarray | None],
    n_joints: int,
) -> np.ndarray:
    """Linear interpolation for frames with missing detections."""

def batch_extract_keypoints(
    video_dir: str | Path,
    output_dir: str | Path,
    **kwargs,
) -> dict[str, Path]:
    """
    Process all videos in directory.
    Returns mapping of video name to output path.
    """
```

---

## Phase 5: PPO Agent and Training

### Task 5.1: Implement PPO Wrapper

**File**: `src/correction/agents/ppo_corrector.py`

**Requirements**:
1. Wrapper around Stable-Baselines3 PPO with paper's hyperparameters
2. Network architecture: 3 fully connected layers (256, 256, 128) for both actor and critic
3. Training configuration from Section 4.1.2

**Default hyperparameters**:
```python
PPO_CONFIG = {
    "learning_rate": 3e-4,
    "n_steps": 512,           # Rollout horizon
    "batch_size": 2048,
    "n_epochs": 10,
    "gamma": 0.99,
    "gae_lambda": 0.95,
    "clip_range": 0.2,
    "ent_coef": 0.01,
    "vf_coef": 0.5,
    "max_grad_norm": 0.5,
    "policy_kwargs": {
        "net_arch": {
            "pi": [256, 256, 128],
            "vf": [256, 256, 128],
        }
    },
}
```

**Implement**:
```python
class SwingCorrectorAgent:
    """
    PPO-based swing correction agent.
    
    Wraps Stable-Baselines3 PPO with paper's configuration.
    """
    
    def __init__(
        self,
        env: GolfSwingCorrectionEnv,
        config: dict | None = None,
        tensorboard_log: str | None = None,
    ):
        """Initialize PPO agent with environment."""
    
    def train(
        self,
        total_timesteps: int = 3_000_000,
        callback: BaseCallback | None = None,
        progress_bar: bool = True,
    ) -> None:
        """Train the agent."""
    
    def save(self, path: str | Path) -> None:
        """Save model checkpoint."""
    
    def load(self, path: str | Path) -> None:
        """Load model checkpoint."""
    
    def predict(
        self,
        observation: np.ndarray,
        deterministic: bool = True,
    ) -> np.ndarray:
        """Get correction action for observation."""
    
    def correct_sequence(
        self,
        user_sequence: np.ndarray,
        expert_sequence: np.ndarray,
        joint_names: list[str],
    ) -> np.ndarray:
        """
        Correct entire swing sequence.
        
        Returns corrected sequence same shape as input.
        """
```

### Task 5.2: Create Training Script

**File**: `examples/correction/train_corrector.py`

**Requirements**:
1. Command-line interface with argparse
2. Load and preprocess data
3. Create vectorized environments (SubprocVecEnv for parallel training)
4. Set up callbacks: CheckpointCallback, EvalCallback
5. TensorBoard logging
6. Save final model

**CLI arguments**:
```
--user-data: Path to user swing sequence (.npy)
--expert-data: Path to expert swing sequence (.npy)
--total-timesteps: Training steps (default: 3,000,000)
--n-envs: Number of parallel environments (default: 4)
--run-name: Name for logging/checkpoints
--checkpoint-freq: Checkpoint save frequency (default: 10000)
--eval-freq: Evaluation frequency (default: 5000)
```

---

## Phase 6: Evaluation and Visualization

### Task 6.1: Implement Evaluation Metrics

**File**: `src/correction/evaluation/metrics.py`

**Implement**:
```python
def compute_correction_metrics(
    original_sequence: np.ndarray,
    corrected_sequence: np.ndarray,
    expert_sequence: np.ndarray,
    joint_names: list[str],
) -> dict[str, float]:
    """
    Compute comprehensive metrics matching paper's Tables 1-5.
    
    Returns dict with keys:
    - l2_original_{joint}, l2_corrected_{joint}, l2_improvement_{joint}
    - l2_original_avg, l2_corrected_avg, l2_improvement_avg
    - vdtw_original_{joint}, vdtw_corrected_{joint}, vdtw_improvement_{joint}
    - vdtw_improvement_avg
    - hip_drift
    """

def compute_phase_wise_metrics(
    original_sequence: np.ndarray,
    corrected_sequence: np.ndarray,
    expert_sequence: np.ndarray,
    phase_boundaries: list[int],
    joint_names: list[str],
) -> dict[str, dict[str, float]]:
    """
    Compute metrics per swing phase (Tables 2, 4 in paper).
    
    Returns nested dict: phase_name -> metric_name -> value
    """

def statistical_validation(
    original_errors: np.ndarray,
    corrected_errors: np.ndarray,
) -> dict[str, float]:
    """
    Paired t-test and effect size (Section 4.8).
    
    Returns:
    - p_value
    - cohens_d (effect size)
    - is_significant (p < 0.05)
    """
```

### Task 6.2: Implement Visualization

**File**: `src/correction/evaluation/visualization.py`

**Implement**:
```python
def plot_joint_trajectories(
    user_sequence: np.ndarray,
    expert_sequence: np.ndarray,
    corrected_sequence: np.ndarray,
    joint_names: list[str],
    joint_to_plot: str,
    phase_boundaries: list[int] | None = None,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """
    Plot X and Y trajectories for single joint (Figures 4-7 in paper).
    
    Shows user (original), expert, and corrected overlaid.
    Optionally shows phase boundaries as vertical lines.
    """

def plot_velocity_profiles(
    user_sequence: np.ndarray,
    expert_sequence: np.ndarray,
    corrected_sequence: np.ndarray,
    joint_names: list[str],
    joint_to_plot: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot velocity magnitude over time (Figure 3 style)."""

def plot_l2_error_over_time(
    original_sequence: np.ndarray,
    corrected_sequence: np.ndarray,
    expert_sequence: np.ndarray,
    joint_names: list[str],
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot per-joint L2 error before/after correction (Figure 2 style)."""

def plot_training_curves(
    tensorboard_log_dir: str | Path,
    save_path: str | Path | None = None,
) -> plt.Figure:
    """Plot training reward and loss curves from TensorBoard logs."""

def create_correction_report(
    metrics: dict[str, float],
    figures: list[plt.Figure],
    output_path: str | Path,
) -> None:
    """Generate HTML report summarizing correction results."""
```

### Task 6.3: Create Evaluation Script

**File**: `examples/correction/evaluate_correction.py`

**Requirements**:
1. Load trained model
2. Run correction on test sequences
3. Compute all metrics
4. Generate visualizations
5. Save results to JSON and create report

**CLI arguments**:
```
--model-path: Path to trained model
--user-data: Path to user swing sequence
--expert-data: Path to expert swing sequence
--output-dir: Directory for results
--generate-report: Create HTML report (flag)
```

---

## Phase 7: Testing

### Task 7.1: Environment Tests

**File**: `tests/correction/test_environment.py`

**Test cases**:
```python
def test_reset_returns_correct_shape():
    """Observation shape matches (N_joints * 2,)"""

def test_step_advances_frame():
    """Frame counter increments after step"""

def test_episode_terminates_at_final_frame():
    """Terminated=True when current_frame >= T"""

def test_action_applies_correction():
    """Corrected pose changes by action amount"""

def test_correction_clipped_to_valid_range():
    """Coordinates stay in [-1, 1] after correction"""

def test_zero_action_gives_zero_improvement_reward():
    """No correction = no improvement reward"""

def test_optimal_correction_reduces_l2():
    """Moving toward expert reduces L2 error"""

def test_reward_components_have_correct_signs():
    """L2 and hip penalties are negative, improvement can be positive"""
```

### Task 7.2: Preprocessing Tests

**File**: `tests/correction/test_preprocessing.py`

**Test cases**:
```python
def test_phase_alignment_matches_expert_length():
    """Aligned user sequence has same length as expert"""

def test_phase_alignment_preserves_phase_count():
    """Number of phases unchanged after alignment"""

def test_hip_normalization_centers_at_origin():
    """Hip center is approximately (0, 0) after normalization"""

def test_normalization_clips_to_range():
    """All coordinates in [-1, 1]"""

def test_phase_detection_returns_monotonic_boundaries():
    """Phase boundaries strictly increasing"""

def test_phase_detection_covers_full_sequence():
    """First boundary is 0, last is T"""
```

### Task 7.3: Reward Tests

**File**: `tests/correction/test_rewards.py`

**Test cases**:
```python
def test_l2_reward_negative_for_nonzero_error():
    """Pose accuracy reward is negative when error > 0"""

def test_l2_reward_zero_for_perfect_match():
    """Pose accuracy reward is 0 when L2 = 0"""

def test_improvement_reward_positive_when_error_decreases():
    """ΔL2 reward > 0 when current L2 < previous L2"""

def test_improvement_reward_bounded_by_tanh():
    """ΔL2 reward in (-β, β) range"""

def test_hip_reward_zero_for_perfect_alignment():
    """Hip reward is 0 when hips match expert exactly"""

def test_velocity_dtw_symmetric():
    """DTW(A, B) == DTW(B, A)"""

def test_velocity_dtw_zero_for_identical_sequences():
    """DTW distance is 0 for identical velocity profiles"""
```

### Task 7.4: Test Fixtures

**File**: `tests/correction/conftest.py`

```python
import pytest
import numpy as np

@pytest.fixture
def joint_names():
    """Standard 9 joints used in paper."""
    return [
        "left_wrist", "right_wrist",
        "left_shoulder", "right_shoulder",
        "left_hip", "right_hip",
        "left_knee", "right_knee",
        "right_ear",
    ]

@pytest.fixture
def synthetic_sequences(joint_names):
    """
    Generate synthetic user/expert sequences for testing.
    Expert: smooth sinusoidal motion
    User: expert + noise + phase offset
    """
    T, N = 60, len(joint_names)
    t = np.linspace(0, 2*np.pi, T)
    
    expert = np.zeros((T, N, 2))
    for j in range(N):
        expert[:, j, 0] = 0.5 * np.sin(t + j * 0.1)
        expert[:, j, 1] = 0.3 * np.cos(t + j * 0.1)
    
    user = expert + np.random.randn(T, N, 2) * 0.1
    
    return user, expert

@pytest.fixture
def simple_env(synthetic_sequences, joint_names):
    """Create environment with synthetic data."""
    from src.correction.environment import GolfSwingCorrectionEnv
    user, expert = synthetic_sequences
    return GolfSwingCorrectionEnv(
        user_sequence=user,
        expert_sequence=expert,
        joint_names=joint_names,
    )
```

---

## Phase 8: Integration with Existing Engines

### Task 8.1: Drake Integration (Optional Extension)

**File**: `src/correction/integrations/drake_integration.py`

**Purpose**: Use Drake trajectory optimization to generate expert trajectories or refine RL-corrected trajectories.

```python
def generate_expert_trajectory_drake(
    plant_path: str | Path,
    initial_pose: np.ndarray,
    final_pose: np.ndarray,
    duration: float = 2.0,
    num_samples: int = 60,
) -> np.ndarray:
    """
    Generate biomechanically optimal trajectory using Drake.
    
    Can be used as expert reference for RL training.
    """

def refine_corrected_trajectory(
    corrected_trajectory: np.ndarray,
    plant_path: str | Path,
    smoothness_weight: float = 0.1,
) -> np.ndarray:
    """
    Use trajectory optimization to smooth RL-corrected trajectory.
    
    Removes any artifacts while preserving correction intent.
    """
```

### Task 8.2: MuJoCo/MyoSuite Integration (Optional Extension)

**File**: `src/correction/integrations/mujoco_integration.py`

**Purpose**: Add muscle activation penalty to reward function.

```python
class MuscleAwareCorrectionEnv(GolfSwingCorrectionEnv):
    """
    Extended environment with muscle dynamics validation.
    
    Adds penalty for corrections requiring implausible muscle activations.
    """
    
    def __init__(
        self,
        mujoco_model_path: str | Path,
        muscle_penalty_weight: float = 0.2,
        *args, **kwargs
    ):
        """Load MuJoCo model and initialize muscle validation."""
    
    def _compute_muscle_activation_cost(
        self,
        current_pose: np.ndarray,
        next_pose: np.ndarray,
    ) -> float:
        """
        Estimate muscle activation cost for transition.
        
        Uses inverse dynamics to get required torques,
        then estimates activation via torque-squared proxy.
        """
```

### Task 8.3: Pinocchio Integration (Optional Extension)

**File**: `src/correction/integrations/pinocchio_integration.py`

**Purpose**: Fast forward/inverse kinematics for keypoint ↔ joint angle conversion.

```python
class PinocchioDynamics:
    """
    Use Pinocchio for efficient FK/IK computations.
    """
    
    def __init__(self, urdf_path: str | Path):
        """Load URDF model."""
    
    def keypoints_to_joint_angles(
        self,
        keypoints: np.ndarray,  # Shape: (N_joints, 2) or (N_joints, 3)
        q_init: np.ndarray | None = None,
    ) -> np.ndarray:
        """Inverse kinematics to get joint configuration."""
    
    def joint_angles_to_keypoints(
        self,
        q: np.ndarray,
    ) -> np.ndarray:
        """Forward kinematics to get keypoint positions."""
```

---

## Phase 9: Documentation

### Task 9.1: Module Documentation

Create comprehensive docstrings for all public functions following NumPy style.

### Task 9.2: User Guide

**File**: `docs/correction/user_guide.md`

Sections:
1. Overview of RL-based swing correction
2. Installation and setup
3. Extracting keypoints from video
4. Training a correction model
5. Evaluating results
6. Integrating with other engines
7. Troubleshooting

### Task 9.3: API Reference

**File**: `docs/correction/api_reference.md`

Auto-generated from docstrings using pdoc or sphinx.

---

## Implementation Order

**Recommended sequence**:

1. **Week 1**: Phase 1 (setup) + Phase 2 (environment) + Phase 7.1 (env tests)
2. **Week 2**: Phase 3 (preprocessing) + Phase 7.2 (preprocessing tests)
3. **Week 3**: Phase 4 (pose extraction) + Phase 5 (training)
4. **Week 4**: Phase 6 (evaluation) + Phase 7.3-7.4 (remaining tests)
5. **Week 5+**: Phase 8 (integrations) + Phase 9 (documentation)

---

## Code Quality Requirements

1. **Type hints**: All function signatures must have type hints
2. **Docstrings**: NumPy style for all public functions
3. **Testing**: Minimum 80% coverage for core modules
4. **Linting**: Must pass `ruff check` and `black --check`
5. **Type checking**: Must pass `mypy --strict` (or project's mypy.ini settings)

---

## Key Files Reference

| File | Purpose | Priority |
|------|---------|----------|
| `src/correction/environment.py` | Core Gym environment | Critical |
| `src/correction/rewards/velocity_dtw.py` | Paper's key contribution | Critical |
| `src/correction/preprocessing/temporal_alignment.py` | Phase-wise alignment | Critical |
| `src/correction/preprocessing/spatial_normalization.py` | Hip-centered coords | Critical |
| `src/correction/preprocessing/phase_detection.py` | Swing phase detection | High |
| `src/correction/data/pose_extraction.py` | Video to keypoints | High |
| `src/correction/agents/ppo_corrector.py` | PPO wrapper | High |
| `examples/correction/train_corrector.py` | Training script | High |
| `src/correction/evaluation/metrics.py` | Evaluation metrics | Medium |
| `src/correction/evaluation/visualization.py` | Plotting utilities | Medium |

---

## Error Handling Notes

Common issues to handle:

1. **Empty/short sequences**: Raise `ValueError` if sequence < 10 frames
2. **Missing joints**: Raise `ValueError` listing missing required joints
3. **Mismatched lengths**: After alignment, verify user.shape[0] == expert.shape[0]
4. **Division by zero**: In normalization when torso length ≈ 0
5. **MediaPipe failures**: Interpolate missing frames, warn if >20% missing
6. **DTW on empty sequences**: Return 0 distance with warning

---

## Performance Considerations

1. **DTW computation**: Full Velocity-DTW is O(n²). Use per-frame approximation during training, full DTW only for evaluation.
2. **Vectorized environments**: Use `SubprocVecEnv` for parallel training (4-8 envs recommended)
3. **NumPy operations**: Avoid Python loops over frames/joints; use vectorized operations
4. **Memory**: Store only current episode's corrected sequence, not full training history

---

## Reference Paper Equations

For implementation reference:

- **Eq. 3-4**: Phase interpolation (temporal alignment)
- **Eq. 5-6**: Hip centering (spatial normalization)
- **Eq. 7-9**: Body scale normalization
- **Eq. 10**: Coordinate clipping
- **Eq. 11**: State representation
- **Eq. 12-13**: MDP state/action spaces
- **Eq. 14**: State transition dynamics
- **Eq. 15**: RL objective
- **Eq. 16-17**: PPO clipped objective
- **Eq. 18**: Advantage estimation
- **Eq. 19**: Total reward function
- **Eq. 20-21**: Pose accuracy reward
- **Eq. 22**: Improvement rate reward
- **Eq. 23**: Hip alignment reward
- **Eq. 24-25**: Velocity-DTW reward

---

## Final Checklist

Before considering implementation complete:

- [ ] All tests pass (`pytest tests/correction/`)
- [ ] Linting passes (`ruff check src/correction/`)
- [ ] Type checking passes (`mypy src/correction/`)
- [ ] Can train on synthetic data and see reward increase
- [ ] Can extract keypoints from sample video
- [ ] Can generate evaluation report
- [ ] Documentation is complete
- [ ] Example scripts run without errors

# Chapter 24: Motion Retargeting

The motion retargeting module in `src/learning/retargeting/retargeter.py`
transfers motion data between skeletal embodiments with different
topologies, bone lengths, and joint conventions.

---

## 24.1 Skeleton Configuration

### 24.1.1 SkeletonConfig

The `SkeletonConfig` dataclass describes the kinematic structure of a
skeleton:

| Field             | Type             | Description                          |
| ----------------- | ---------------- | ------------------------------------ | ------------------------------------------ |
| `name`            | `str`            | Skeleton identifier                  |
| `joint_names`     | `list[str]`      | Ordered list of joint names          |
| `parent_indices`  | `list[int]`      | Parent index per joint (-1 for root) |
| `joint_offsets`   | `NDArray (n, 3)` | T-pose offset from parent in metres  |
| `joint_axes`      | `NDArray (n, 3)  | None`                                | Rotation axis per joint (default z)        |
| `joint_limits`    | `NDArray (n, 2)  | None`                                | [min, max] angle limits (default $\pm\pi$) |
| `semantic_labels` | `dict[str, str]` | Maps semantic names to joint names   |
| `end_effectors`   | `list[str]`      | Names of end-effector joints         |

Post-initialisation validation ensures that `parent_indices` and
`joint_offsets` have the same length as `joint_names`. Missing
`joint_axes` default to the z-axis; missing `joint_limits` default to
$[-\pi, \pi]$.

### 24.1.2 Utility Methods

| Method                           | Description                              |
| -------------------------------- | ---------------------------------------- |
| `n_joints` (property)            | Number of joints                         |
| `get_joint_index(name)`          | Name to index lookup (raises if missing) |
| `get_semantic_joint(semantic)`   | Semantic label to joint name             |
| `get_kinematic_chain(end_joint)` | Root-to-end joint name list              |

### 24.1.3 Standard Humanoid Configuration

`SkeletonConfig.create_humanoid()` returns a standard 22-joint humanoid
skeleton suitable for whole-body motion:

```
Spine chain (6 joints):  pelvis -> spine_1 -> spine_2 -> spine_3 -> neck -> head
Left leg (4 joints):     pelvis -> left_hip -> left_knee -> left_ankle -> left_foot
Right leg (4 joints):    pelvis -> right_hip -> right_knee -> right_ankle -> right_foot
Left arm (4 joints):     spine_3 -> left_shoulder -> left_elbow -> left_wrist -> left_hand
Right arm (4 joints):    spine_3 -> right_shoulder -> right_elbow -> right_wrist -> right_hand
```

All 22 joints have semantic labels (e.g., `"left_shoulder"` maps to
joint `"left_shoulder"`), and five end effectors are designated: `head`,
`left_hand`, `right_hand`, `left_foot`, `right_foot`.

T-pose offsets use approximate human proportions in metres (e.g.,
upper leg length 0.4 m, forearm length 0.25 m).

### 24.1.4 Kinematic Chain Extraction

`get_kinematic_chain(end_joint)` traverses the parent hierarchy from
`end_joint` to the root and returns the full chain:

```python
skel = SkeletonConfig.create_humanoid()
chain = skel.get_kinematic_chain("left_hand")
# ['pelvis', 'spine_1', 'spine_2', 'spine_3',
#  'left_shoulder', 'left_elbow', 'left_wrist', 'left_hand']
```

---

## 24.2 Retargeting Methods

The `MotionRetargeter` class handles motion transfer between a source
and target skeleton.

### 24.2.1 Initialisation

```python
retargeter = MotionRetargeter(source_skeleton, target_skeleton)
```

During construction, two mappings are computed:

1. **Joint mapping** -- Established via semantic labels. For each
   semantic name in the source skeleton, the corresponding target joint
   is looked up. Only joints with matching semantic labels are mapped.

2. **Scale factors** -- The ratio of target bone length to source bone
   length for each mapped joint:

   $$s_j = \frac{\| \text{offset}_{\text{target},j} \|}{\| \text{offset}_{\text{source},j} \|}$$

   Scale factors are used for positional retargeting (motion capture).

### 24.2.2 Direct Mapping

```python
target_motion = retargeter.retarget(source_motion, method="direct")
```

The simplest retargeting method: source joint angles are copied directly
to the corresponding target joints. For each mapped pair
$(s_j, t_j)$:

$$q_{t_j}(t) = q_{s_j}(t)$$

After copying, joint limits of the target skeleton are enforced by
clipping.

This method works well when the source and target skeletons have similar
topologies and proportions.

### 24.2.3 Optimisation-Based Retargeting

```python
target_motion = retargeter.retarget(source_motion, method="optimization")
```

For each frame, the target joint angles are optimised to minimise the
squared error between source and target end-effector positions:

$$\min_{\mathbf{q}} \sum_{e \in \text{EE}} \| \text{FK}_e(\mathbf{q}) - \mathbf{p}_e^{\text{source}} \|^2$$

where $\text{FK}_e(\mathbf{q})$ is the forward kinematics of end
effector $e$ and $\mathbf{p}_e^{\text{source}}$ is the source
end-effector position.

The optimisation uses gradient descent with numerical gradients
($\epsilon = 10^{-4}$, step size 0.01, up to 50 iterations per frame).
The initial guess is the direct-mapping result.

Forward kinematics is computed by traversing the kinematic chain,
applying z-axis rotations at each joint:

$$R(\theta) = \begin{bmatrix} \cos\theta & -\sin\theta & 0 \\ \sin\theta & \cos\theta & 0 \\ 0 & 0 & 1 \end{bmatrix}$$

$$p_e = R(\theta_{n}) \cdots R(\theta_1) \, \delta_1 + \cdots + \delta_n$$

Joint limits are enforced at each iteration.

### 24.2.4 IK-Based Retargeting

```python
target_motion = retargeter.retarget(source_motion, method="ik")
```

Currently delegates to the optimisation-based method. The architecture
is designed to allow integration with engine-specific IK solvers (e.g.,
MuJoCo or Pinocchio IK) in the future.

---

## 24.3 Motion Capture Integration

### 24.3.1 `retarget_from_mocap`

```python
target_motion = retargeter.retarget_from_mocap(
    marker_positions,        # (T, n_markers, 3)
    marker_names,            # list of marker names
    marker_to_joint_mapping, # optional dict
)
```

This method bridges from marker-based motion capture data to joint
angles on the target skeleton:

1. **Marker-to-joint mapping.** If no explicit mapping is provided,
   `_infer_marker_mapping()` uses standard C3D naming conventions:

   | Marker Name | Semantic Joint |
   | ----------- | -------------- |
   | LSHO        | left_shoulder  |
   | RSHO        | right_shoulder |
   | LELB        | left_elbow     |
   | RELB        | right_elbow    |
   | LWRI        | left_wrist     |
   | RWRI        | right_wrist    |
   | LHIP        | left_hip       |
   | RHIP        | right_hip      |
   | LKNE        | left_knee      |
   | RKNE        | right_knee     |
   | LANK        | left_ankle     |
   | RANK        | right_ankle    |

2. **Position extraction.** For each frame, marker positions are
   associated with the target joints via the mapping.

3. **Analytical IK.** For two-link chains, joint angles are computed
   using the **law of cosines**:

   $$\cos\theta = \frac{a^2 + b^2 - d^2}{2ab}$$

   where $a$ and $b$ are the bone lengths of the two links and $d$ is
   the distance to the target position. The joint angle is then:

   $$\theta = \arccos\left(\text{clip}\left(\frac{a^2 + b^2 - d^2}{2ab + \epsilon}, -1, 1\right)\right)$$

   The small $\epsilon = 10^{-6}$ prevents division by zero.

### 24.3.2 Mapping Visualisation

`visualize_mapping()` returns a human-readable text summary:

```
Motion Retargeting: source_humanoid -> target_robot
==================================================

Joint Mapping:
  left_elbow           -> left_elbow           (scale: 0.83)
  left_shoulder        -> left_shoulder        (scale: 1.00)
  ...

Mapped joints: 18
Source joints: 22
Target joints: 20
```

---

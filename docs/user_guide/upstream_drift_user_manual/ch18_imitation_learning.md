# Chapter 18: Imitation Learning

The `learning/imitation/` module provides tools for learning robot control
policies from expert demonstrations. It includes a flexible demonstration dataset
format, and three imitation learning algorithms: Behavior Cloning (BC), Dataset
Aggregation (DAgger), and Generative Adversarial Imitation Learning (GAIL).
All implementations use pure NumPy for portability, with no dependency on
PyTorch or TensorFlow.

Source files referenced in this chapter:

- `src/learning/imitation/dataset.py`
- `src/learning/imitation/learners.py`

---

## 18.1 Demonstration Dataset

### 18.1.1 The Demonstration Dataclass

The `Demonstration` dataclass in `src/learning/imitation/dataset.py` represents a
single recorded trajectory of robot motion.

**Fields:**

| Field                | Type                       | Default           | Description                               |
| -------------------- | -------------------------- | ----------------- | ----------------------------------------- | ------------------------------ |
| `timestamps`         | `NDArray` shape `(T,)`     | required          | Time values for each frame in seconds.    |
| `joint_positions`    | `NDArray` shape `(T, n_q)` | required          | Joint positions at each timestep.         |
| `joint_velocities`   | `NDArray` shape `(T, n_v)` | required          | Joint velocities at each timestep.        |
| `actions`            | `NDArray` shape `(T, n_u)` | `None`            | Control actions applied at each timestep. |
| `end_effector_poses` | `NDArray` shape `(T, 7)`   | `None`            | EE poses [x,y,z,qw,qx,qy,qz].             |
| `contact_states`     | `list[list[dict]]`         | `None`            | Contact info per timestep.                |
| `task_id`            | `str                       | None`             | `None`                                    | Task identifier for filtering. |
| `success`            | `bool`                     | `True`            | Whether the demonstration was successful. |
| `source`             | `str`                      | `"teleoperation"` | Source of the demonstration.              |
| `metadata`           | `dict[str, Any]`           | `{}`              | Arbitrary additional metadata.            |

**Validation.** The `__post_init__` method enforces that `timestamps`,
`joint_positions`, and `joint_velocities` all share the same first dimension
$T$, raising `ValueError` if they differ.

**Properties:**

- `duration` -- Total time span: `timestamps[-1] - timestamps[0]`.
- `n_frames` -- Number of frames: `len(timestamps)`.
- `n_joints` -- Number of joints: `joint_positions.shape[1]`.

**Frame access.** The `get_frame(idx)` method returns a dictionary with
the data at a single timestep, including the action and end-effector pose
if available.

**Subsampling.** The `subsample(factor)` method returns a new `Demonstration`
keeping every $k$-th frame, useful for adapting high-frequency recordings to
lower control frequencies.

**Serialization.** The `to_dict()` method converts all arrays to nested Python
lists for JSON serialization. The `from_dict(data)` class method reconstructs a
`Demonstration` from a dictionary, converting lists back to NumPy arrays.

### 18.1.2 The DemonstrationDataset Class

`DemonstrationDataset` is a container for multiple demonstrations with methods
for filtering, conversion, augmentation, and I/O.

**Construction.**

```python
dataset = DemonstrationDataset()           # Empty
dataset = DemonstrationDataset([demo1, demo2])  # From list
```

**Standard container interface.** The class supports `len()`, indexing with
`[]`, and iteration, making it usable in standard Python loops and comprehensions.

**Adding demonstrations.**

- `add(demo)` -- Append a single `Demonstration`.
- `extend(demos)` -- Append a list of `Demonstration` objects.

**Filtering.**

- `filter_successful()` -- Returns a new `DemonstrationDataset` containing only
  demonstrations with `success=True`.
- `filter_by_task(task_id)` -- Returns a new dataset containing only
  demonstrations whose `task_id` matches the given string.

**Aggregate properties.**

- `total_frames` -- Sum of `n_frames` across all demonstrations.
- `total_transitions` -- Sum of `max(0, n_frames - 1)` across all
  demonstrations. Each demonstration of length $T$ contributes $T - 1$
  state transitions.

**Conversion to training data.** Two methods convert the dataset into NumPy
arrays suitable for supervised learning:

1. `to_transitions()` returns a tuple `(states, actions, next_states)` where
   each state is the concatenation of joint positions and velocities
   $s = [q, \dot{q}]$. Each demonstration of $T$ frames produces $T - 1$
   transitions. Demonstrations without actions are skipped.

2. `to_state_action_pairs()` returns `(states, actions)`, discarding the
   next-state information. This is the primary input format for behavior
   cloning.

**Data augmentation.** The `augment` method creates additional training data by
adding Gaussian noise to joint positions and velocities:

$$
\tilde{q} = q + \mathcal{N}(0, \sigma^2 I)
$$

$$
\tilde{\dot{q}} = \dot{q} + \mathcal{N}(0, \sigma^2 I)
$$

where $\sigma$ is the `noise_std` parameter. For each original demonstration,
`num_augmentations` noisy copies are created (default 5). The original
demonstration is preserved. Augmented demonstrations have their `source` field
appended with `"_augmented"` and their metadata includes `{"augmented": True}`.

Actions are NOT perturbed during augmentation -- only the state trajectory is
modified. This teaches the policy to recover from states slightly off the
demonstrated trajectory, improving robustness.

```python
augmented = dataset.augment(noise_std=0.01, num_augmentations=5)
# If dataset had 10 demos, augmented has 10 * (1 + 5) = 60 demos
```

**Sampling.** The `sample(n)` method returns a random subset of `n`
demonstrations, useful for creating mini-datasets for quick experiments.

**Persistence.** The `save(path)` and `load(path)` class method serialize the
dataset to and from JSON format. The file includes a version string and the
count of demonstrations for quick inspection.

**Statistics.** The `get_statistics()` method computes and returns:

- `n_demonstrations` -- Number of demonstrations.
- `total_frames` -- Total frame count.
- `total_transitions` -- Total transition count.
- `success_rate` -- Fraction of successful demonstrations.
- `mean_duration` -- Average demonstration length in seconds.
- `position_mean`, `position_std` -- Per-joint mean and standard deviation of
  positions across all frames.
- `velocity_mean`, `velocity_std` -- Per-joint mean and standard deviation of
  velocities across all frames.

These statistics are essential for normalizing inputs before training.

---

## 18.2 Behavior Cloning

Behavior cloning (BC) is the simplest imitation learning approach: it frames
the problem as supervised regression from states to actions. The
`BehaviorCloning` class in `src/learning/imitation/learners.py` implements BC
with a NumPy-based MLP.

### 18.2.1 TrainingConfig

All imitation learners share a common `TrainingConfig` dataclass:

| Field           | Type        | Default      | Description                         |
| --------------- | ----------- | ------------ | ----------------------------------- |
| `epochs`        | `int`       | `100`        | Number of training epochs.          |
| `batch_size`    | `int`       | `256`        | Mini-batch size.                    |
| `learning_rate` | `float`     | `1e-3`       | Learning rate for SGD.              |
| `weight_decay`  | `float`     | `1e-5`       | L2 regularization coefficient.      |
| `hidden_sizes`  | `list[int]` | `[256, 256]` | Hidden layer dimensions.            |
| `activation`    | `str`       | `"relu"`     | Activation function name.           |
| `dropout`       | `float`     | `0.0`        | Dropout probability (not yet used). |

### 18.2.2 Network Architecture

The policy is a fully-connected multi-layer perceptron (MLP) stored as a list
of dictionaries, where each dictionary contains a weight matrix `W` and a bias
vector `b`. The network has `len(hidden_sizes)` hidden layers plus one output
layer.

For a network with hidden sizes $[h_1, h_2]$, the architecture is:

$$
\text{Input } (d_{obs}) \xrightarrow{W_1, b_1} h_1 \xrightarrow{\text{ReLU}} \xrightarrow{W_2, b_2} h_2 \xrightarrow{\text{ReLU}} \xrightarrow{W_3, b_3} \text{Output } (d_{act})
$$

Weight initialization uses small random Gaussian values (`np.random.randn * 0.01`)
and zero biases.

### 18.2.3 Forward Pass

The `_forward` method computes the network output:

$$
h_0 = x
$$

$$
h_l = \text{ReLU}(W_l \, h_{l-1} + b_l) \quad \text{for } l = 1, \ldots, L-1
$$

$$
\hat{a} = W_L \, h_{L-1} + b_L
$$

where $x$ is the observation, $L$ is the total number of layers, and ReLU is
defined as $\text{ReLU}(z) = \max(0, z)$. The output layer has no activation
function (linear output), which is appropriate for continuous action prediction.

### 18.2.4 Loss Function

The loss is the mean squared error (MSE) between predicted and expert actions:

$$
\mathcal{L} = \frac{1}{N} \sum_{i=1}^{N} \|\hat{a}_i - a_i^*\|^2
$$

where $N$ is the batch size, $\hat{a}_i$ is the predicted action, and $a_i^*$
is the expert action. This is implemented as:

```python
np.mean((predictions - actions) ** 2)
```

### 18.2.5 Backpropagation

The `_backward` method implements manual backpropagation through the network.
The algorithm proceeds as follows:

**Forward pass with caching.** All intermediate activations
$h_0, h_1, \ldots, h_L$ are stored for use in the backward pass.

**Output gradient.** The gradient of the MSE loss with respect to the output is:

$$
\delta_L = \frac{2}{N}(\hat{a} - a^*)
$$

**Backward recursion.** For each layer $l$ from $L$ down to 1:

1. Compute weight gradient:

$$
\frac{\partial \mathcal{L}}{\partial W_l} = h_{l-1}^T \, \delta_l
$$

2. Compute bias gradient:

$$
\frac{\partial \mathcal{L}}{\partial b_l} = \sum_{i} \delta_l^{(i)}
$$

where the sum is over the batch dimension.

3. Propagate error to previous layer (for $l > 1$):

$$
\delta_{l-1} = (\delta_l \, W_l^T) \odot \mathbb{1}[h_{l-1} > 0]
$$

The term $\mathbb{1}[h_{l-1} > 0]$ is the ReLU gradient, which passes the
error through only for units that were active (positive) during the forward
pass.

### 18.2.6 Training Loop

The `train` method implements the full training procedure:

1. **Data preparation.** Calls `dataset.to_state_action_pairs()` to get
   `(observations, actions)` arrays. Splits into training and validation sets
   based on `validation_split` (default 10%).

2. **Epoch loop.** For each epoch:
   a. Shuffle training data with a random permutation.
   b. Iterate over mini-batches of size `config.batch_size`.
   c. For each mini-batch, compute gradients via `_backward` and update weights.

3. **Weight update with decay.** The update rule is mini-batch SGD with L2
   weight decay (also known as L2 regularization or ridge regression):

$$
W \leftarrow W - \eta \left(\frac{\partial \mathcal{L}}{\partial W} + \lambda W\right)
$$

$$
b \leftarrow b - \eta \, \frac{\partial \mathcal{L}}{\partial b}
$$

where $\eta$ is the learning rate and $\lambda$ is the weight decay
coefficient. Note that weight decay is applied to weights only, not biases,
which is standard practice to avoid biasing the network toward zero output.

4. **Metrics.** Training and validation loss are recorded at each epoch and
   returned as the training history dictionary.

### 18.2.7 Inference

The `predict` method runs a single forward pass:

```python
action = bc.predict(observation, deterministic=True)
```

If the input is 1-dimensional (a single observation), it is reshaped to a
batch of size 1, processed, and the output is flattened back to 1D.

### 18.2.8 Persistence

The `save` and `load` methods serialize the policy weights and configuration to
a NumPy `.npz` file. This includes the observation and action dimensions, the
training configuration, and all layer weights and biases.

### 18.2.9 Complete Example

```python
from src.learning.imitation.dataset import DemonstrationDataset
from src.learning.imitation.learners import BehaviorCloning, TrainingConfig

# Load demonstrations
dataset = DemonstrationDataset.load("demonstrations.json")

# Filter and augment
dataset = dataset.filter_successful()
dataset = dataset.augment(noise_std=0.01, num_augmentations=5)

# Configure and train
config = TrainingConfig(
    epochs=200,
    batch_size=128,
    learning_rate=1e-3,
    weight_decay=1e-5,
    hidden_sizes=[256, 256, 128],
)
bc = BehaviorCloning(
    observation_dim=42,
    action_dim=7,
    config=config,
)
history = bc.train(dataset, validation_split=0.15)

# Evaluate
obs = env.reset()[0]
action = bc.predict(obs)

# Save
bc.save("bc_policy.npz")
```

---

## 18.3 DAgger (Dataset Aggregation)

DAgger addresses the distribution shift problem inherent in behavior cloning.
In BC, the policy is trained on expert states, but at test time it visits states
generated by its own (imperfect) actions. DAgger iteratively collects on-policy
data, labels it with expert actions, and retrains.

### 18.3.1 Algorithm Overview

The DAgger algorithm proceeds as follows:

1. **Initialize.** Train an initial policy $\pi_0$ via behavior cloning on the
   expert dataset $\mathcal{D}_0$.

2. **For each iteration** $i = 1, \ldots, N$:

   a. Compute a mixing parameter $\beta_i$ that determines the probability of
   using the expert policy for execution.

   b. Collect trajectories using the mixed policy:

$$
\pi_{mix} = \beta_i \, \pi_{expert} + (1 - \beta_i) \, \pi_{learner}
$$

At each timestep, with probability $\beta_i$ the expert action is executed,
otherwise the learned policy's action is executed.

c. Regardless of which action was executed, label every visited state with the
expert's action: $a^* = \pi_{expert}(s)$.

d. Aggregate the new data into the dataset: $\mathcal{D}_{i} = \mathcal{D}_{i-1} \cup \mathcal{D}_{new}$.

e. Retrain the policy on the aggregated dataset.

### 18.3.2 Beta Schedules

The `train_online` method supports two $\beta$ schedules:

- **Linear decay** (`beta_schedule="linear"`):

$$
\beta_i = 1 - \frac{i}{N}
$$

This starts with pure expert execution and linearly transitions to pure
policy execution.

- **Exponential decay** (any other value):

$$
\beta_i = 0.5^i
$$

This decays more rapidly, reaching $\beta = 0.0625$ by iteration 4.

### 18.3.3 Implementation

The `DAgger` class wraps a `BehaviorCloning` instance internally. The workflow
is:

```python
from src.learning.imitation.learners import DAgger, TrainingConfig

config = TrainingConfig(epochs=50, batch_size=256)
dagger = DAgger(
    observation_dim=42,
    action_dim=7,
    config=config,
)

# Phase 1: Offline training on initial dataset
dagger.train(initial_dataset)

# Phase 2: Online training with expert queries
def expert_policy(obs):
    """Expert policy callable."""
    return compute_expert_action(obs)

results = dagger.train_online(
    env=gym_env,
    expert=expert_policy,
    iterations=10,
    trajectories_per_iter=10,
    max_steps=500,
    beta_schedule="linear",
)
```

The `train_online` method returns a dictionary with:

- `iteration_rewards` -- Mean trajectory reward at each iteration.
- `dataset_size` -- Growing dataset size at each iteration.

### 18.3.4 Trajectory Collection

At each DAgger iteration, `trajectories_per_iter` trajectories are collected.
For each trajectory:

1. The environment is reset.
2. At each timestep:
   - Both the learner's action and the expert's action are computed.
   - The expert action is always stored in the demonstration (for labeling).
   - The executed action is chosen based on $\beta$: expert with probability
     $\beta$, learner otherwise.
3. The trajectory is stored as a `Demonstration` with `source="dagger"`.
4. All new demonstrations are added to the aggregated dataset.
5. The BC policy is retrained on the full aggregated dataset.

---

## 18.4 GAIL (Generative Adversarial Imitation Learning)

GAIL frames imitation learning as a game between a policy (generator) and a
discriminator. The discriminator learns to distinguish expert trajectories from
policy trajectories, while the policy learns to fool the discriminator.

### 18.4.1 Network Architecture

GAIL maintains two neural networks:

1. **Policy network** $\pi_\theta(s)$: Maps states to actions. Architecture is
   identical to the BC policy (MLP with ReLU activations and linear output).

2. **Discriminator network** $D_\phi(s, a)$: Maps state-action pairs to a
   probability in $[0, 1]$. The input is the concatenation of state and action
   vectors. The architecture is an MLP with ReLU hidden layers and a sigmoid
   output:

$$
D_\phi(s, a) = \sigma\bigl(W_L \, h_{L-1} + b_L\bigr)
$$

where $\sigma(x) = \frac{1}{1 + e^{-x}}$ is the sigmoid function.

### 18.4.2 Discriminator Training

The discriminator is trained to output values close to 1 for expert data and
close to 0 for policy data. The discriminator loss is the binary cross-entropy:

$$
\mathcal{L}_D = -\mathbb{E}_{(s,a) \sim \pi_{expert}}\bigl[\log D_\phi(s, a)\bigr] - \mathbb{E}_{(s,a) \sim \pi_\theta}\bigl[\log(1 - D_\phi(s, a))\bigr]
$$

In practice this is computed as:

```python
disc_loss = -np.mean(np.log(expert_preds + eps) + np.log(1 - policy_preds + eps))
```

where `eps = 1e-8` prevents numerical overflow in the logarithm.

### 18.4.3 Policy Reward

Instead of using the environment's reward function, GAIL defines the policy
reward as:

$$
r(s, a) = -\log\bigl(1 - D_\phi(s, a)\bigr)
$$

This reward is high when the discriminator believes the state-action pair is
from the expert (i.e., $D_\phi(s, a)$ is close to 1), and low when the
discriminator recognizes it as policy-generated.

The `get_reward` method computes this reward for a single state-action pair:

```python
gail = GAIL(observation_dim=42, action_dim=7)
reward = gail.get_reward(state, action)
```

### 18.4.4 Policy Update

In the full GAIL algorithm, the policy is updated using a reinforcement learning
algorithm (typically PPO or TRPO) with the discriminator reward replacing the
environment reward. The current UpstreamDrift implementation provides a
simplified training loop suitable for initial experiments; full RL integration
should use the RL environments from Chapter 17 with the GAIL reward.

### 18.4.5 Training

```python
from src.learning.imitation.learners import GAIL, TrainingConfig

config = TrainingConfig(
    epochs=500,
    batch_size=256,
    learning_rate=1e-4,
    hidden_sizes=[256, 256],
)
gail = GAIL(
    observation_dim=42,
    action_dim=7,
    config=config,
)

# Train on expert demonstrations
history = gail.train(dataset)
# history contains 'discriminator_loss' and 'policy_loss'

# Use as reward function in RL
reward = gail.get_reward(state, action)

# Predict action
action = gail.predict(observation, deterministic=False)
```

When `deterministic=False`, Gaussian noise with standard deviation 0.1 is added
to the action for exploration.

### 18.4.6 Persistence

GAIL saves both the policy and discriminator networks via `save(path)` and
restores them with `load(path)`, using NumPy `.npz` format.

---

# Issue: [MyoSuite] Perturbation Analysis: Core Module Implementation

## Labels
`perturbation-analysis`, `physics-engine`, `myosuite`, `phase-1`

## Summary

Implement perturbation analysis for the MyoSuite physics engine following the unified
`PerturbationAnalyzer` protocol. MyoSuite's gym-style RL environment interface and
muscle-driven dynamics enable evaluation of **learned control policies** for perturbation
robustness — answering whether an RL-trained movement strategy is more consistent than
a hand-tuned one.

## Motivation

MyoSuite wraps MuJoCo with musculotendon dynamics in a gym-compatible interface. This
uniquely enables:
- Evaluating RL-trained policies for sensitivity to observation noise and muscle noise
- Comparing RL policies trained with different objectives for robustness
- Testing whether curriculum-learning or domain-randomization during training produces
  more robust policies
- Studying muscle synergy decomposition under perturbation

For golf swing analysis: train multiple RL policies for the golf swing task, then use
perturbation analysis to determine which policy produces the most consistent clubhead
speed — providing a principled way to select among policies beyond just reward.

## Requirements

### Core Protocol Implementation
- [ ] Create `src/engines/physics_engines/myosuite/python/perturbation/analyzer.py`
- [ ] Implement `MyoSuitePerturbationAnalyzer` class conforming to `PerturbationAnalyzer` protocol
- [ ] Use `MyoSuitePhysicsEngine` for simulation
- [ ] Support MyoSuite environment loading via gym interface

### Action/Control Profile Handling
- [ ] Accept control profiles as:
  - Polynomial coefficients per muscle (for parity with pendulum)
  - Time-series muscle activation arrays a(t) ∈ [0,1]^n_muscles
  - Callable policy `π(obs) → action` (for RL policy evaluation)
  - Pre-recorded action sequences from demonstrations
- [ ] Map muscle names to MyoSuite action indices
- [ ] Validate action dimensions match environment action space

### Perturbation Modes
- [ ] **Action perturbation** (primary):
  - Additive noise on muscle activations (clamped to [0, 1])
  - Multiplicative noise on muscle activations
  - Applied at each `env.step()` call
- [ ] **Observation perturbation** (RL-specific):
  - Add noise to policy observations: `obs_perturbed = obs + amp × noise`
  - Tests how observation uncertainty affects policy output
  - Relevant for real-world deployment where sensors are noisy
- [ ] **Parameter perturbation** (model-level):
  - Perturb muscle strength (max isometric force)
  - Perturb tendon stiffness
  - Tests robustness to model uncertainty / inter-subject variability

### Simulation Loop
- [ ] Reset environment via `env.reset()` before each trial
- [ ] Set initial state if possible (via `env.sim.data.qpos/qvel`)
- [ ] Step via `env.step(action)` for each timestep
- [ ] Record trajectory: observations, actions, rewards, done flags
- [ ] Record internal MuJoCo state: `env.sim.data.qpos`, `.qvel`, `.ctrl`
- [ ] Handle simulation failures (early termination, divergence)
- [ ] Support both fixed-length and variable-length episodes

### Metric Extraction
- [ ] Compute all mandatory metrics per §4.2 of guidelines
- [ ] Use `env.sim.data.xpos[body_id]` for end-effector position
- [ ] Add MyoSuite-specific optional metrics:
  - `total_reward` — cumulative reward over episode
  - `episode_length` — steps before termination
  - `muscle_activation_mean` — average activation per muscle
  - `muscle_activation_smoothness` — jerk metric on activation signals
  - `muscle_fatigue_index` — sustained high activation indicator
  - `synergy_decomposition` — NMF of activation matrix (n_muscles × n_timesteps)
  - `policy_entropy` — action distribution entropy (if stochastic policy)

### RL Policy Evaluation Mode
- [ ] Accept pre-trained policy as `policy_fn(obs) → action`
- [ ] Apply perturbation to observations (tests sensor robustness)
- [ ] Apply perturbation to actions (tests actuator robustness)
- [ ] Compare multiple policies on same perturbation config
- [ ] Report which policy achieves best RS while maintaining reward threshold

### Statistics & Reporting
- [ ] Use shared `MetricStatistics` and `variability_summary()` from shared module
- [ ] Compute Robustness Score
- [ ] Compute reward-conditioned RS (RS only for trials achieving minimum reward)
- [ ] JSON export per schema in guidelines §8.1
- [ ] Include muscle and RL-specific statistics in export

### Comparison
- [ ] Implement `compare_profiles()` for two action profiles or policies
- [ ] Compare open-loop (recorded actions) vs closed-loop (policy) robustness
- [ ] Mann-Whitney U test per metric
- [ ] ComparisonReport with reward-robustness trade-off analysis
- [ ] Pareto frontier: reward vs robustness score across policies

### Testing
- [ ] Unit test: zero-amplitude → identical results (CV=0)
- [ ] Unit test: seed reproducibility
- [ ] Unit test: monotonicity (amplitude ↑ → CV ↑)
- [ ] Unit test: muscle activation clamping to action space bounds
- [ ] Unit test: observation perturbation passes through to policy correctly
- [ ] Integration test: full batch on myoElbowPose or myoHandPose task
- [ ] Integration test: comparison of two action profiles
- [ ] Validation test: match pendulum engine on equivalent 2-DOF (joint torque mode)

## Acceptance Criteria

- `MyoSuitePerturbationAnalyzer` passes protocol type check
- Action, observation, and parameter perturbation modes all functional
- Muscle activations always clamped to environment action space bounds
- All mandatory metrics computed correctly
- RL policy evaluation mode works with standard stable-baselines3 policies
- JSON export validates against schema with muscle/RL extensions
- Joint-level results on simple model match pendulum engine within tolerance
- All tests pass

## Parity Checklist
- [ ] Implements `PerturbationAnalyzer` protocol
- [ ] Supports white, pink, and brown noise
- [ ] Supports additive and multiplicative perturbation
- [ ] Reports all mandatory metrics (§4.2 of guidelines)
- [ ] Uses `PerturbationConfig` dataclass for configuration
- [ ] Returns `PerturbationSummary` with `MetricStatistics`
- [ ] Reproducible with seed parameter
- [ ] Batch runner with progress reporting
- [ ] Unit tests with synthetic known-sensitivity cases
- [ ] Design by Contract: pre/postconditions documented
- [ ] JSON export compatible with cross-engine comparison schema

## Dependencies
- Issue #006: Pendulum reference implementation (for shared utilities)
- Issue #010: OpenSim perturbation (shared muscle-level patterns)
- `src/shared/python/perturbation/` shared module must exist

## References
- Guidelines: `docs/perturbation_analysis_parity_guidelines.md`
- MyoSuite engine: `src/engines/physics_engines/myosuite/python/myosuite_physics_engine.py`
- MyoSuite muscle analysis: `src/engines/physics_engines/myosuite/python/muscle_analysis.py`
- Engine protocol: `src/shared/python/engine_core/interfaces.py`
- MyoSuite documentation: https://myosuite.readthedocs.io/

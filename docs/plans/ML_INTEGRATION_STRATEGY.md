# ML Integration Strategy

## Objective

To outline the boundaries and integrations between UpstreamDrift (the core simulation product) and MLProjects (the training ecosystem).

## Problem

Currently, our monolith is bloating due to the mixing of end-user features (simulation, API, visualization) with internal R&D features (RL training loops, large datasets, heavyweight ML dependencies). We need to decouple them.

## The Strategy

1. **Docker Profiles `core` vs `training` (Resolves #1558)**
   UpstreamDrift now publishes tiers.
   - `upstream-drift:runtime` (Core) – strictly physics, API, and fast visualization.
   - `upstream-drift:training` (ML) – built ON TOP of runtime, adding `ray`, `stable-baselines3`, `gymnasium`.

2. **Decoupled Repositories (Resolves #1561)**
   `UpstreamDrift` consumes models. `MLProjects` produces models.

### Reference Workflow

- **Data Engineering:** Extract states or physics logic out of `UpstreamDrift` to a lightweight dataset.
- **Training:** Work in `MLProjects` using the `mlprojects/pytorch-gpu:2026.02` ecosystem to crunch data and train humanoid controllers via RL/Imitation.
- **Artifact Export:** Save the converged model strictly as an ONNX file or TorchScript `.pt` file within the `MLProjects` output registry.
- **Artifact Import:** In `UpstreamDrift`, load the frozen `.onnx` or `.pt` model using the standalone `onnxruntime` or basic `torch` evaluation module. Since inference dependencies are extremely small compared to full training environments, UpstreamDrift's core footprint remains small.

If a user *must* train directly inside `UpstreamDrift` (e.g., visual humanoid RL loop), they simply execute `docker compose --profile training up` or build the `training` target in the `Dockerfile`.

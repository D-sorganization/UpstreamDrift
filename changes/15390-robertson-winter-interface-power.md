---
issue: 15390
summary: "Robertson–Winter interface power (F·v + M·ω), per-segment balance residual, MuJoCo adapter"
dl_state: "in_review"
next_step: "CI on PR for #15390."
branch: "cursor/rw-interface-power-15390-a314"
---

Add engine-agnostic interface-power contracts and endpoint-power helper; MuJoCo
computes equivalent joint reaction power (planar m a − m g method) with
dE/dt residual per segment. PowerFlowAnalyzer exposes
``compute_interface_power_frame`` beside existing τ·q̇ joint power; kinematic
sequence planned-note resolved. Segment KE globes (#15389) remain kinematic, not
flow.
# Issue: Joint Torque Input / Control Interface Dashboard (Shared)

## Summary

Create a shared dashboard interface for joint torque inputs and control strategy
management. The control approach inputs (torque profiles, PD gains, trajectory
tracking parameters) must be visible and adjustable by the user. Currently these
exist in code (`set_control`, `WholeBodyController`, etc.) but are not exposed
to the user through any dashboard or API surface.

## Motivation

Users cannot see or adjust control inputs during simulation. The `set_control(u)`
interface exists on all engines, and the `WholeBodyController` provides sophisticated
control, but these are code-level abstractions with no UI visibility.

## Requirements

- [ ] Shared control panel component (API + UI)
- [ ] Display current joint torques per-joint with names
- [ ] Allow manual torque input override per joint
- [ ] Support control strategy selection (direct torque, PD, WBC)
- [ ] Display control strategy parameters (gains, weights)
- [ ] Real-time torque visualization during simulation
- [ ] API endpoints for control configuration
- [ ] Integration with existing simulation flow

## Acceptance Criteria

- All joint torques visible in dashboard during simulation
- User can select and configure control strategies
- API endpoints for reading/writing control parameters
- Works across all physics engines

## Labels

`enhancement`, `dashboard`, `control`, `shared`

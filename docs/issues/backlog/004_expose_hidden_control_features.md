# Issue: Expose Hidden Control Features in Code

## Summary

Many control features exist in the codebase but are not visible to users through
the dashboard or API. These include: drift-control decomposition (Section F),
counterfactual experiments (ZTCF/ZVCF - Section G), induced acceleration analysis,
whole-body control tasks, and contact force computation. All must be surfaced.

## Motivation

As development proceeds, features get added in code but not exposed to users.
This creates hidden functionality that risks being lost. Following the Pragmatic
Programmer principle of "no broken windows," we must maintain feature parity
between code capabilities and user-facing interfaces.

## Features to Expose

- [ ] Drift vs control acceleration decomposition (Section F)
- [ ] ZTCF (Zero-Torque Counterfactual) computation
- [ ] ZVCF (Zero-Velocity Counterfactual) computation
- [ ] Induced acceleration analysis per actuator
- [ ] Whole-body controller task management
- [ ] Flexible shaft properties (Section B5)
- [ ] Contact force computation
- [ ] Mass matrix / bias forces / gravity computation
- [ ] Jacobian computation per body

## Acceptance Criteria

- All features listed above accessible via API endpoints
- Control features registry with discoverability
- Documentation of each exposed feature
- Tests for each endpoint

## Labels

`enhancement`, `api`, `feature-exposure`

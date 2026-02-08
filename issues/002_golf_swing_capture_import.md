# Issue: Golf Swing Capture Import for Reinforcement Learning

## Summary

Create an easy-to-use interface for importing golf swing motion capture data (C3D, CSV,
JSON) as reference trajectories for reinforcement learning training. The importer should
integrate with the existing `DemonstrationDataset` and `MotionRetargeter` modules.

## Motivation

Real-world golf swing captures provide ground-truth trajectories for imitation learning
and reinforcement learning reward shaping. We need a streamlined pipeline from raw
capture data to RL-ready demonstration datasets.

## Requirements

- [ ] Import C3D motion capture files (golf swing recordings)
- [ ] Import CSV/JSON kinematic time-series data
- [ ] Convert marker data to joint angles via inverse kinematics
- [ ] Integrate with `DemonstrationDataset` for imitation learning
- [ ] Provide trajectory-matching reward for RL environments
- [ ] Easy CLI and API interface for import workflows
- [ ] Validate imported data against model DOFs

## Acceptance Criteria

- Can import standard C3D golf swing files
- Produces valid `DemonstrationDataset` compatible output
- API endpoint for upload and conversion
- Unit tests covering import, conversion, validation

## Labels

`enhancement`, `machine-learning`, `motion-capture`

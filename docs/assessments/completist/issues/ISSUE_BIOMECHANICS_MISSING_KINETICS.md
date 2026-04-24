---
title: Missing Kinetic Metrics in Biomechanical Analysis
labels: physics-gap, high
assignees: biomechanics-team
---

## Description

The `kinematic_sequence.py` module currently analyzes the timing and velocity of body segments (kinematics) but completely omits kinetic metrics (forces, torques, powers) that are essential for explaining _why_ the sequence occurs.

Explicit TODOs include:

# High Priority Issue: Heuristic Gear Effect Model

**Status:** Open
**Priority:** High
**Labels:** physics-gap, high
**Date Identified:** 2026-01-22
**Author:** PHYSICS AUDITOR

## Description
The gear effect (spin generation from off-center impact) is modeled in `shared/python/impact_model.py` using a linear heuristic function: `horizontal_spin = -gear_factor * h_offset * speed * h_scale`.

This is a phenomenological approximation that lacks a rigorous physics derivation based on the clubhead's Center of Gravity (CG) location relative to the face and the resulting torque/rotation of the clubhead during the collision interval.

## Impact
- **Forgiveness Analysis:** Cannot accurately model the benefit of specific clubhead designs (e.g., MOI optimization, CG placement).
- **Trajectory:** Errors in side spin calculation for mishits.

## Affected Files
- `shared/python/impact_model.py`

## Recommended Fix
1.  Replace the heuristic function with a full rigid-body collision model that accounts for the clubhead's inertia tensor and CG offset.
2.  Compute the clubhead rotation induced by the impact force and the resulting relative velocity at the contact point.

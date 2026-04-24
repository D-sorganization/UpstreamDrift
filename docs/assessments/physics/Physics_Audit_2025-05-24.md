# Physics Audit Report - 2025-05-24

**Auditor:** Jules (Physics Auditor Agent)
**Date:** 2025-05-24
**Focus Area:** Physics Fidelity, Mathematical Correctness, Biomechanics Accuracy

## Executive Summary

The physics engine demonstrates a solid foundation in basic projectile motion and rigid body dynamics but suffers from critical oversimplifications in impact mechanics and ground reaction force modeling. The implementation of shaft dynamics lacks essential torsional components required for accurate dispersion analysis. Additionally, kinematic sequence scoring presents a high patent infringement risk.

- **Overall Physics Fidelity Score:** 6/10
- **Critical Issues Count:** 2 (Impact Model, GRF Fallback)
- **High Priority Gaps:** 2 (Shaft Torsion, Patent Risk)
- **Confidence in Results:** Medium (simulation is directionally correct but lacks precision for pro-level analysis)

## Findings by Category

### 1. Mathematical Correctness

**Finding 1.1: Simplified Scalar Effective Mass in Impact Model**

- **File:** `src/shared/python/physics/impact_model.py` (Line 158)
- **Issue:** The `RigidBodyImpactModel` calculates effective mass as `1 / (1/m + r^2/I)`. This scalar approximation ignores the full 3D inertia tensor of the clubhead and the directional component of the impact force relative to the center of gravity (CG).
- **Expected Physics:** $J = (M^{-1} + (r \times n)^T I^{-1} (r \times n))^{-1} (1+e) v_{rel}$
- **Actual Implementation:** Scalar approximation.
- **Impact:** Significantly inaccurate ball speed and spin rates for off-center hits (gear effect), leading to incorrect carry distances and dispersion patterns.
- **Recommended Fix:** Implement full 3D impulse-momentum equations using the inertia tensor.

**Finding 1.2: Incorrect GRF Fallback Calculation**

- **File:** `src/shared/python/physics/ground_reaction_forces.py` (Line 385)
- **Issue:** When contact data is unavailable, the fallback mechanism sums the scalar weight ($W=mg$) of bodies.
- **Expected Physics:** Dynamic Ground Reaction Force $F_{GRF} = m(g + a_{com})$.
- **Actual Implementation:** `total_force[2] += abs(np.sum(g))` (Static weight only).
- **Impact:** Underestimates peak forces during dynamic swings (which can exceed 2-3x body weight), rendering biomechanical analysis invalid for power transfer.
- **Recommended Fix:** Implement inverse dynamics to estimate required GRF from body accelerations.

### 2. Physical Plausibility

**Finding 2.1: Missing Shaft Torsional Dynamics**

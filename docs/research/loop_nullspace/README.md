# Closed-Loop Null-Space Exploration (NSX) Research Reference

## Overview

This directory contains the standalone mathematical specification, symbolic derivations, golden test fixtures, and per-child contract appendix for **Epic #12593 (`NSX: Closed-Loop Null-Space Exploration`)**, delivered under **#12549 (`NSX-00`)**.

## Contents

- [`loop_nullspace.tex`](loop_nullspace.tex): Editable standalone LaTeX mathematical reference covering:
  1. General allocation equation $A x = b$ (R2) and the four structural dimension counts.
  2. Zero-side three-case diagnostic analysis (`CONSISTENT_UNIQUE`, `CONSISTENT_NONUNIQUE`, `INCONSISTENT(residual)`) (R3).
  3. Exact invertible $6 \times 6$ grasp-map kernel at the canonical grip midpoint frame (R4).
  4. Bare-club Newton–Euler equations with welded hand solids (`LHand`, `LHandStandoff`, `RHand`, totaling $1.1082\,\mathrm{kg}$) partitioned from the composite $1.4212\,\mathrm{kg}$ `Clubface Vector` body leaving $0.313\,\mathrm{kg}$ bare club (NSX-01).
  5. Six allocation-invariant quantities with scaled residual validity gates (R5).
  6. The $c_+ / c_-$ asymmetric gross-power identity and strict reduced-space curvature condition (R6).
  7. Constrained Zero-Torque and Zero-Velocity Counterfactuals (ZTCF/ZVCF) via the KKT projector (R9).
  8. Static versus trajectory-coupled feasible envelopes, Chebyshev center/margin, and inclusion-minimal conflict sets (IIS).
  9. Reaction-restriction sensitivity (`WELD6` $\to$ `SHAFT_FREE5` $\to$ `BALL3`) versus true closure alternatives.
  10. Complete citation verification audit table (R13).
  11. Explicit scientific non-claims.
- [`contract_appendix.md`](contract_appendix.md): Per-child implementation contract appendix covering `NSX-01` through `NSX-62` (exact module paths, signatures, Design-by-Contract preconditions/postconditions, and named golden fixtures).
- [`derivations/generate_goldens.py`](derivations/generate_goldens.py): Deterministic SymPy symbolic-to-numeric derivation generator.
- [`derivations/golden_fixtures.json`](derivations/golden_fixtures.json): Checked-in golden values consumed by unit tests across `NSX-10` through `NSX-62`.

## Regeneration and Verification

```bash
python3 docs/research/loop_nullspace/derivations/generate_goldens.py
python3 -m pytest tests/unit/loop_nullspace/test_golden_derivations.py -q
```

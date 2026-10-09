# Legacy Impact Model

This package exposes the existing rigid, compliant and finite-time impact APIs,
including the additive `compute_gear_effect_spin` heuristic used by
`ImpactSolverAPI`. Import public functions from the package facade.

The gear helper uses toe-positive/high-positive contact coordinates for the
documented right-handed frame: X forward, Y left, Z up. Toe adds positive-Z spin;
the native spin-cross-velocity Magnus direction is then left. Its magnitude
scales are assumptions, and the additive helper does not solve a coupled
rotational collision.

The calculation-level [Direction Reference](../../../../../docs/research/shot_pattern_analysis/phase2/review_followups/LEGACY_GEAR_SIGN_REFERENCE.md)
and editable [LaTeX Source](../../../../../docs/research/shot_pattern_analysis/phase2/review_followups/LEGACY_GEAR_SIGN_REFERENCE.tex)
record the sign correction, units, regression evidence and limitations.
The separate [Matched-Delivery Design](../../../../../docs/research/shot_pattern_analysis/phase2/DESIGN.md)
documents the coupled-contact research pathway; that pathway does not add this
legacy gear-spin term.

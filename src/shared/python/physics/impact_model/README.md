# Legacy Impact Model

This package exposes the existing rigid, compliant and finite-time impact APIs,
including the additive `compute_gear_effect_spin` heuristic used by
`ImpactSolverAPI`. Import public functions from the package facade.

The gear helper uses toe-positive/high-positive contact coordinates for the
documented right-handed frame: X forward, Y left, Z up. Toe adds positive-Z spin;
the native spin-cross-velocity Magnus direction is then left (draw for a
right-hander); heel is the reverse (fade). Its magnitude scales are heuristic
assumptions and unqualified, and the additive helper does not solve a coupled
rotational collision. Historical outputs produced before this sign correction
are not rewritten.

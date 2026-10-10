# Reviewed Driver Curve-Control Bounds

The narrow reporting repair accepts the predeclared Driver 5° fixed-target curve-1 original control versus curve-2 supplemental speed-bound control. It requires the exact ordered protocols, club, lean, curve levels, original/default face bounds, supplemental ±6° face bounds, speed factors 0.75–1.25 versus 0.75–1.30, and equal non-null target requests. All other source, numerical-projection, nominal-target, seed, input-pairing and final-byte checks remain in force.

The valid full-API regression failed before the change; ten invalid variants remained rejected. Thirty focused tests pass, as do Ruff, scoped Mypy and the repository file-size gate. All 384 production-selected pairs pass the source/config/nominal preflight: 56 lean comparisons, 164 curve comparisons and 164 shaft-SD comparisons. That preflight intentionally stops before row-level statistics and is not statistical acceptance of all pairs.

An actual saved 10,000-shot Driver 5° combined-noise SD2 comparison passes the complete public pair API with 9,940 common-six supported draws. No new physical simulation was run. The JSON receipt binds the source, tests, executed scripts, logs and saved result. Current preflight source has only formatting/comment changes from its preserved executed copy, verified by normalized AST equality.

This repairs a reporting incompatibility with already reviewed supplemental bounds; it does not retrospectively alter the original protocol or claim that curve alone causes the resulting interaction. Final figure and report qualification remain pending.

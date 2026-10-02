# Necromatcher Saved-Spline Restart and Recipe Retention

Tracked by #11235 and #11234 in draft PR #11240. This procedure changes research
execution and repeatability; historical motion and dynamics remain unqualified.

## Two Explicit Starting Methods

`sampled_parent` retains the previous behavior: selected parent poses seed a new
knot grid. Its initialization policy can explicitly author bounded knot positions
and zero slopes. The resulting trajectory can differ from the saved parent.

`preserved_spline` copies the exact physical Hermite coefficients and original
knot clock. It requires `operation=fit` and strict initialization. It neither
resamples the parent nor repeats authored initialization. Sampling choices may
change within the saved source interval, but must retain both endpoints and the
original knot count. No extrapolation or silent clock conversion is permitted.

## Identity and Admission

The immutable public `ImageSplineStart` stores knots, physical coefficients,
native coordinate order, free-coordinate order, native definition identity and a
SHA-256 of canonical little-endian float64 coefficient bytes. The native
definition identity hashes the exact serialized definition bytes used by the
compiled plant; it differs from the exported XML asset hash.
Serialization uses `json.dumps(definition, allow_nan=False).encode("utf-8")`
with Python's default separators and retained dictionary order, matching the
native binding. It is not a sorted or whitespace-independent JSON digest.

`preserved_fit_spline` recalls declared snapshots or explicitly constructs a
snapshot from complete legacy records. Missing legacy coefficients make restart
unavailable. Partial records, changed hashes, coordinate orders, model identities
or disagreements between declared and legacy coefficients reject admission.
Declared snapshots also require the corresponding original knot, coefficient
and coordinate fields; a snapshot alone is an incomplete parent record.
The worker additionally validates the compiled plant, requested knot count,
exact source endpoints and all stored parent poses against the saved spline.

Each result retains `initial_spline`, its coefficient hash and
`initialization_source`. The final `spline_start` snapshot supports the next
restart. A strict restart has no authored-initialization receipt. Optimizer
execution, convergence and scientific acceptance remain separate facts.

## Native and Web Workflow

1. Recall a saved fit and inspect its retained range/contact recipe.
2. Select **Resume Saved Spline** when available. Keep both endpoint frames,
   positive coordinate prior scales and the displayed original knot count.
3. Enter a new version identity and explicit evaluation/wall budgets.
4. Start the research refit. Inspect durable status and rejection reasons.
5. Verify the saved starting hash against the parent, freshly recomputed initial
   pixel RMS, final image/constraint diagnostics and original-footage overlay.

The shared plan supplies authoritative `baseline_config` from original fit
evidence, falling back to recorded request configuration for legacy versions.
Both forms retain all constraints, interior fractions, bounds and initialization
policy while editing visible scalar weights. The API accepts nested `config`
through the canonical decoder. Legacy flattened scalar requests remain supported;
mixing explicit legacy scalars with a nested recipe rejects rather than silently
choosing a precedence.
The conflicting scalar keys are `max_iterations`, `prior_weight`,
`smoothness_weight` and `closure_weight`. Frame selection, visibility and wall
budget remain outer job options. Conflict admission occurs in the API request
wrapper before canonical configuration decoding.

Within a job, original pixel evidence and its pose prior remain unchanged. A new
job uses the selected parent's first pose as its prior seed; an exact trajectory
restart does not imply the same prior as a previous job. Compare initial pixel
RMS only with matching frame selection and visibility weighting.

## Validation and Remaining Work

Regression tests check exact source/midpoint positions, velocities and
accelerations, trap resampling/reauthoring, reject changed identity/clock/order,
and verify complete recipe transport in both hosts. Actual Tiger/Hogan restart
receipts must retain their own committed producer before numerical claims.
The V8 constant two-heel contact hypothesis remains inadequate across
follow-through/walking. Reviewed phase intervals, time-varying contact,
camera/landmark evidence and independent dynamics remain open.

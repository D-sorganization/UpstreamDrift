# Historical Spline Constraint Probes

## Scope and Acceptance

Progresses #11235 and epic #11232. Historical source frames carry image evidence;
interior spline evaluations are authored constraint probes and must never become
new observations, confidence entries or source-frame identities.

1. Preserve default image-fitting behavior with probes disabled.
2. Validate explicit interior fractions in (0, 1), with finite positive scales
   and nonnegative weights in the public native constraint options.
3. Evaluate the canonical trajectory at the sorted union of source PTS and
   configured fractions of each spline interval. Select original PTS rows for
   image, prior, speed and RMS; apply constraint residuals to the union.
4. Obtain geometry through the public matching plant/IK interface. Chain native
   constraint pose derivatives through canonical spline bases. Unsupported
   requested capabilities fail explicitly.
5. Add red-first coverage for interior closure failures, observation-count
   conservation, analytic coefficient derivatives, contact row dimensions,
   invalid options and unchanged default fits.
6. Assess whole-cubic coordinate extrema independently. Finite nonlinear probes
   report tested times and worst residuals; they do not certify all-time grip or
   ground feasibility. Hard bound enforcement and image-fidelity acceptance
   remain separate subsequent work.

## Evidence Boundary

Source clock derivatives do not establish historical physical timing or efforts.
No candidate becomes accepted through optimizer convergence alone. New whole-
track runs must preserve input identities, options, numerical budgets, residual
metrics, feasibility assessments and rejection reasons. This plan is not an
execution receipt.

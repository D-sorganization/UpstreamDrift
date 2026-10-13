# Wedge Nominal Search-Bound Review

The accepted PW zero-lean doubled-curvature cases are excluded by an arbitrary
face-search box, not by physical non-airborne behavior. This limits the study's
answer to the requested curve-magnitude comparison across clubs.

## Existing Evidence and Independent Probe

The accepted preflight solved the mild PW draw with face −0.129520714°, path
+1.370479286°, and speed 32.011468770 m/s. Its doubled draw search instead
required face in [0°, 6°]; the optimizer could not cross zero and left a
0.351838 m residual. The fade is mirrored. Thus the old reference-centered
box encoded an unnecessary sign constraint precisely for the strong case.

A diagnostic changed only the face search box to [−6°, +6°]. It retained the
77.402477249 m raw target, speed limits [24, 40] m/s, face-minus-path ±3°,
0.05 m acceptance tolerance, primary contact/flight resolution, and all physics:

| Pattern | Face (°) | Path (°) | Speed (m/s) | Raw Residual (m) | Fine Replay Residual (m) |
| --- | ---: | ---: | ---: | ---: | ---: |
| Draw | −0.260459639 | +2.739540361 | 32.045914261 | 1.14e−13 | 0.000211396 |
| Fade | +0.260459639 | −2.739540361 | 32.045914261 | 1.14e−13 | 0.000211396 |

Both solves took seven optimizer evaluations. Fine replay used 512 contact
steps and 0.005 s flight step. There was no post-flight aiming operation.
This is deterministic feasibility evidence, not Monte Carlo output or empirical
qualification. `probe_wedge_match_bounds.json` binds execution and script hashes.

## Recommendation Before Further Implementation

Preserve the accepted matrices and receipts. Describe their PW cases as
“unavailable under the declared nominal search bounds”; avoid interpreting them
as physically infeasible or as evidence that strong wedge curves cannot match.
The current report is honest only with this qualification, but its missing
zero-lean wedge curve-strength comparison leaves the user's scientific scope
partly unanswered.

After the frozen run, predeclare a separately versioned supplement using a
common symmetric face-search box across curve strengths, with unchanged speed
bounds, target, physics, noise pairing, and numerical tolerance. First replay
all existing nominal solutions and diagnose every previously bounded failure;
then run only newly supported Monte Carlo cells, retaining the old matrix as
the original protocol. For the confirmed within-lean PW gap this affects 18
cells (nine profiles × two twist SDs, zero lean, doubled curvature). The
fixed-target control also needs a separate feasibility review: saved preflight
has additional bounded failures for driver lean 5° and PW lean 5°. This probe
makes no claim about their cause or reachability.

A broader face search is an experimental-design revision, not permission to
relax residual tolerances, widen speed bounds, overwrite unavailable rows, or
restart active workers. Root decision and a new source-bound protocol precede
any supplemental population run.

Reproduce from the repository root:

```bash
PYTHONPATH=. OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/review_followups/probe_wedge_match_bounds.py
```

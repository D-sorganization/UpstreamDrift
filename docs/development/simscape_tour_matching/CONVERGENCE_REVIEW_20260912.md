# Matching Convergence Review and Execution Gates

## Current Conclusion

The goal remains open. There is useful implementation and diagnostic progress,
but no accepted full-swing open-loop sixth-order match. Do not describe the
remaining work as merely running more computations. Read the latest HANDOFF
section before older chronological entries and plans.

## Why Progress Has Stalled

1. The optimized object has repeatedly differed from the deliverable. Static
   pose fits and feedback tracking can follow the capture without producing a
   time-only polynomial that reproduces it. Run45 feedback RMS is 35.2 mm, while
   its time-only replay is 346.5 mm and compressed sextic replay is 655.5 mm.
   Several actuator profiles have relative L2 approximation error above 0.94.
   These numbers are different experiments, not accepted matches.
2. A real gimbal singularity and an independent representation branch bug were
   found. Run41 failed near shoulder middle angle -pi/2. Run49 used the other
   Euler branch despite identical orientation, changing the native effort map.
   Branch preservation improves run50 marker parity from 234 mm to 0.016 mm,
   but velocity and closure gates still fail. Quaternions alone do not resolve
   a singular original actuator map.
3. Numerical qualification is incomplete. Run51 first adaptive level completes
   0.85 s with 1.267 micrometre maximum marker disagreement and good closure,
   but native velocity disagreement 0.00502 exceeds the 0.0001 gate. Its second
   level fails at a floating-point output boundary. No refinement conclusion is
   available. This prototype took 75.7 s versus scalar 3.62 s; there is currently
   no demonstrated speed advantage from the Python quaternion integrator.
4. Expensive diagnostic and representation work has not yet been converted into
   a dynamically feasible trajectory/control optimization. Repeating tightly
   bounded static-node shooting or fitting an arbitrary feedback torque record
   is not supported by current evidence.

## Ordered Work for the Next Agent

Run52 update: the boundary fix passes20 generic tests and88 real-runtime/shared
tests (one optional skip). Both replay levels now complete. Finest marker error
is1.17068e-6 m and native velocity error0.00465279; both gates still fail.
The two adaptive levels differ by9.60936e-8 m in marker position. Before another
same-tolerance run, inspect physical angular velocity and chart conditioning,
then independently refine the scalar reference and adaptive tolerances within
declared budgets. The first item's boundary-fix action below is now complete.

1. Fix run51's output-boundary failure with a regression test and retain genuine
   step-underflow rejection. Re-run a bounded identical-input comparison using
   a new immutable runtime/run directory. Preserve every existing parity gate.
   Separate model/frame/actuator errors from integration error; compare physical
   angular velocity as well as ill-conditioned native rates. Do not remove the
   native-rate gate simply because physical marker errors are small.
2. Keep the qualified scalar native model as the matching baseline while the
   alternate model is being qualified. Benchmark actual cost per useful fit
   improvement before changing optimizer engines. Do not block all optimization
   on implementing alternate representations in every engine.
3. Prepare a single bounded trajectory/control co-optimization experiment.
   Reuse existing native constrained forward dynamics and profile providers.
   Start from integrated feasible states rather than independent static poses;
   permit node motion and penalize continuity defects. Keep original initial
   state, model identity, observation masks and native actuator routing explicit.
   Optimize a normalized-time degree-six basis jointly with trajectory states;
   if piecewise cubic controls are used for initialization, retain continuity
   constraints and treat sextic conversion as a new optimization, not a guarantee.
4. First test the difficult transition interval while preserving the earlier
   prefix through residuals and global initial-state replay. Compare objective,
   continuity defect, closure, effort and derivative conditioning against the
   frozen seed. Stop a non-improving bounded experiment for diagnosis; do not
   automatically increase the budget or loosen acceptance. State the budget and
   quantitative improvement/defect targets before launch.
5. Extend only a successful formulation to all 654 frames through 1.813888889 s.
   Final acceptance requires one uninterrupted original-state forward replay of
   the global degree-six native efforts, all valid C3D marker residuals and
   loop-closure checks, then independent MATLAB R2025b replay. Supply overlay,
   residual-by-time/marker plots, coefficients, units, time basis and exact hashes.
6. Qualify MuJoCo and Drake alternate builders independently using the same
   canonical providers and physical specification. Existing conversion adapters
   do not prove alternate engine dynamics equivalence. OpenSim remains a staged
   implementation/qualification effort; it is not an accepted equivalent model.

## Representation Contract

Use pose_interchange as the shared provider for named frames, axis sequence,
Hamilton quaternion order, native branch/winding, rates, convective accelerations
and virtual-work-preserving effort transforms. Preserve two-DOF joints as two
DOF. A native torque polynomial transforms at the actual current state into
spherical moments; those moments need not remain polynomial. Reject singular
inverse maps explicitly. Fixed-frame transport is implemented; moving-frame
acceleration transport remains outstanding. See REPRESENTATION_HANDOFF.md and
PINOCCHIO_MANIFOLD_HANDOFF.md for executable interfaces and qualification scope.

## Handoff Discipline

Give lower-cost agents one numbered gate at a time, with immutable inputs,
owned files, explicit budget and pass/fail checks. Require TDD for new behavior,
finite/shape/unit/frame contracts, shared providers and narrow dependencies.
Record terminal status even on failure; a partial report is not a completed run.
Update HANDOFF and DEVELOPMENT_LOG in every implementation commit. Never
promote feedback tracking, pointwise parity or a short prefix to full acceptance.

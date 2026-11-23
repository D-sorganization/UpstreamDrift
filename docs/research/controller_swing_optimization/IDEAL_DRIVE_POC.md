# Ideal Drive Trace Prototype

This research branch tests recording the ideal generalized source effort that
reaches the native MuJoCo KKT solve. Each accepted RK4 stage records separate
source and impact generalized-force components after the native passive-shaft
mask. Root source channels are zeroed by the held-step integrator before the
solve. Contact, passive, bias and weld forces are not relabeled as drive.

The native dynamics path applies direct generalized inputs and does not model
actuator saturation or a joint-to-muscle transmission. The trace is labeled
`ideal_direct_generalized_no_saturation`; it is not a measurement of motor or
anatomical torque and must not be presented as post-saturation evidence. This
prototype does not certify anatomical mappings, between-stage limits, contact
or physical feasibility. Unsupported engines fail closed when this evidence
is requested, and speculative search stages are not committed.

The focused tests compare instrumented and ordinary rollouts, including a
native MuJoCo chain-shaft KKT solve, and check passive masking, source/impact
separation, earlier trial exclusion, inconsistent component rejection and
first-step failure rollback. Re-run against the delivery runtime and commit.

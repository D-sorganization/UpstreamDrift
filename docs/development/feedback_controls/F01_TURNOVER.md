# F01 Comparison Contract Turnover (#11785)

F01 extends the existing engine-model inventory rather than creating another
engine catalog. The ledger is now `engine-model-inventory/1.1.0`; the OpenSim
native-humanoid package explicitly lists its torque and legacy eight-name muscle
target. The latter is not executable support unless native readback finds
muscles; see F09 child #12148. `FeedbackComparisonRegistry` derives all required rows and admits
only evidence with current model/provider source identity and bound state,
physics, contact, integrator, input-channel schema, policy, time-grid, input,
horizon, and channel
identities. Runtime availability and qualification remain unverified.
Within-engine and same-input admission rejects truncated horizons. Observation
accuracy requires a distinct observation-grid digest, separate from the input
time-grid hash; F09 must provide that digest from the native scoring clock.

The new contract is a prerequisite to numerical gates, not their result.
`same-input-bundle/v1` remains intact for its existing Euclidean use but is
inadmissible here because its loader only checks the spec hash. Tools T01
supplies `experiment-replay/1.0.0` with canonical state-schema, execution-policy,
time-grid and applied-input digests, ordered channels, full native state, and a
time-only player. F09 should create native execution receipts and score comparison
levels. F08/F10 require own-contact continuous muscle excitation replay;
shared rigid-body, externally forced, or restarted 50 ms evidence is distinct.

The test sequence began with a failing import of the F01 module, then passed
the admission negatives and legacy inventory tests. The local Tools pinned
submodule was populated from an existing local Tools checkout at commit
`3678409fc51024150ab28970b72e3b468935f345`; no submodule pointer changed.
Private capture content and measured results were not added to this PR.

Next integration steps: map T01 exported bundle fields into
`ComparisonEvidence`; bind native provider runtime/version and actual
loaded-model hash; have F09 produce independent full-horizon replay receipts
with contact/force evidence; only then compare numerical match and runtime
against the four equal-budget baselines. The manual calculation registry
remains blocked pending full inventory and qualification.

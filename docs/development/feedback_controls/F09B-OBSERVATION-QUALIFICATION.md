# F09b Native Observation Scoring

Issue #11837 extends the F09 comparison workflow from positions-only sampling
to receipt-backed native observation scoring. It consumes the authoritative
F01 registry and the validated Tools T01 `ExperimentReplayBundle` public type;
it does not define another model inventory, replay format, numerical gate, or
integrator.

`NativeObservationCase` binds one F01 comparison row to the actual T01 replay
bundle, a complete native marker-position output, the exact observed-position
clock and validity mask, and the existing native acceptance receipt/gates.
Before the F01 replay admission call, the adapter checks T01 and F01 model,
provider, state-schema, input-channel order and schema, exact input values and
grid, execution-policy digest, replay mode, timebase, and horizon. The content
identity adds T01 `integrity.initial_state_sha256`; it never infers initial
state identity from a schema digest or partial coordinate vector. Position
and tangent-velocity dimensions are checked separately, so a valid manifold
schema with `nq != nv` remains representable.

Observation scoring reuses F09a's positions-only sampler and canonical replay
metrics, pelvis-yaw metric, and `acceptance.evaluate`. It preserves the exact
observation times and separate native-output and observation-clock hashes.
The receipt also carries the initial-state payload hash, criterion IDs and
digest, output/observation/alignment identities, and existing gate verdict.
The selected capture and a digest of the native acceptance receipt are bound
without copying private receipt paths into the public summary.
Changed initial state, applied input, provider, state schema, or policy cannot
reuse a stale source identity.

Reports include every required F01 registry row and the required six-engine
set. Missing evidence and unavailable or unqualified rows remain visible;
scores do not promote F01's qualification field. A report is qualified only
when all required engines and rows are present, each F01 row is already
qualified, and every existing numerical gate passes. No capture thresholds
are introduced, and synthetic tests establish only software-contract behavior.

The public replay-bundle facade is pinned through `vendor/ud-tools`; the
`extend_sidekick_lab_path` seam exposes the Tools-owned `sidekick.lab.mocap`
package without modifying its vendored source. This child does not provide
private capture evidence, native physics results, muscle qualification, or a
cross-engine acceptance claim. F09 remains open for native consumers and
qualification evidence.

Validation:

```powershell
python3 -m pytest tests/unit/engines/test_feedback_observation_qualification.py -q -n 2 --tools-mode vendored
python -m ruff check src/engines/feedback_observation_qualification.py tests/unit/engines/test_feedback_observation_qualification.py
python -m ruff format --check src/engines/feedback_observation_qualification.py tests/unit/engines/test_feedback_observation_qualification.py
```

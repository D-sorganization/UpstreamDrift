# UpstreamDrift Product Review and Board Issue Proposals

**Review Date:** 2026-09-28

**Disposition:** Proposed for expert review; no implementation or scientific approval.

**Scope:** UpstreamDrift, its consumed Tools revision, desktop and web workflows, API execution, scientific claims, and release evidence.

## Decision Brief

The highest priority is trustworthy results. A professional interface must not report a successful simulation with no recorded samples, a landing before ground contact, or a verified neural result without verification evidence. These defects deserve attention before cosmetic redesign or speculative acceleration.

This review proposes **12 bounded issue briefs**, with one immediate P0 candidate. Several should extend existing issues rather than create duplicates. It also identifies two existing programs that the Board should retain: installed-product acceptance and independent physical validation. The Board should approve scope, owners, and acceptance criteria before implementation agents claim work.

| ID  | Priority | Proposed GitHub Issue Title                                                      | Primary Owner                                 | Evidence                                | Disposition                                              |
| --- | -------- | -------------------------------------------------------------------------------- | --------------------------------------------- | --------------------------------------- | -------------------------------------------------------- |
| R01 | P0       | Start the REST Simulation Recorder and Reject Empty Successful Results           | UpstreamDrift API                             | Executed service probe                  | New candidate; deduplicate before filing                 |
| R02 | P1       | Isolate Engine, Recorder, and Analysis State per Simulation Run                  | UpstreamDrift API                             | Controlled concurrency probe and source | New candidate; related #1488                             |
| R03 | P1       | Bound Simulation Work and Preserve Cancellable Jobs Under Load                   | UpstreamDrift API; Tools flight provider      | Request and eviction probes; source     | Extend capacity work where applicable                    |
| R04 | P1       | Propagate Flight Termination Before Reporting Landing Metrics                    | Tools flight; UpstreamDrift flight/API/UI     | Executed in both model families         | Paired provider/consumer issues                          |
| R05 | P1       | Refuse Neural Promotion Without Sufficient Comparable Evidence                   | UpstreamDrift neural motion                   | Executed public gate probe              | Follow-up to #10960 / #10625                             |
| R06 | P1       | Make Neural Verification Badges and Published Claims Fail Closed                 | UpstreamDrift UI and scientific documentation | Source and contradictory artifacts      | Follow-up to #10960 / #10626                             |
| R07 | P1       | Wire Neural Matching Controls to Executed Requests or Mark Them Unavailable      | UpstreamDrift motion-matching UI              | Complete widget-to-request trace        | Extend #10626                                            |
| R08 | P1       | Bind Matcher Process Events to Their Run and Handle Failed Starts                | UpstreamDrift desktop                         | Executed QProcess probe; source         | New candidate                                            |
| R09 | P1       | Preserve Actionable Error and Partial-Result States Across the Simulation API    | UpstreamDrift API and clients                 | Executed error probe; source            | Extend #5911 / #8009 without re-filing their fixed scope |
| R10 | P1       | Keep Ball-Flight Results Attached to Their Input Snapshot and Model Provenance   | UpstreamDrift web                             | Source trace                            | Follow-up to #7456 / #8978 / #9352                       |
| R11 | P1       | Report the Actual Integrated Horizon Consistently Across REST and WebSocket Runs | UpstreamDrift simulation                      | Exact arithmetic and source trace       | New candidate                                            |
| R12 | P2       | Define and Validate Neural Data-Efficiency Claims Statistically                  | UpstreamDrift neural motion                   | Executed counterexample                 | Follow-up to #10625 / #10960                             |

Priority means impact, not an assertion that every deployment exposes every path. P0 is an ordinary supported workflow returning materially false success. P1 covers result integrity, failure recovery, and resource exhaustion. P2 covers a misleading secondary research metric. Confidence is separate from priority; the briefs identify what was executed and what still needs native or human validation.

## Review Boundary and Reproducibility

| Repository or Artifact | Reviewed Revision                          | Meaning                                                      |
| ---------------------- | ------------------------------------------ | ------------------------------------------------------------ |
| UpstreamDrift          | `599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309` | Fetched `origin/main`; clean tracked source at review start  |
| Consumed Tools         | `3678409fc51024150ab28970b72e3b468935f345` | Exact `vendor/ud-tools` gitlink and checkout                 |
| Sibling Tools          | `c0d2a938fd99f627185571e87eda1dd26feeb0d7` | Local comparison only; not claimed to be current remote main |
| Runner_Dashboard       | `49fc655681e7e8017412ea49a349c80a8b2830d4` | Local expert-panel contract and presets                      |
| Review Branch          | `docs/product-review-20260928`             | Documentation only; implementation unchanged                 |

The sibling Tools comparison showed no differences from the consumed pin in `swing_sim/flight/{models,types,registry}.py` or the reviewed `impact_interval` tree. R04 is therefore not merely a stale consumer pin relative to that sibling checkout. Fix shared provider code in Tools, then consume a reviewed published pin in UpstreamDrift; do not patch the vendor tree.

`agent_context status` reported a matched Tools runtime and checkout, five current registered boundary reviews, and 52 registered source files across 12 components. That is useful integrity evidence, but it does not cover every runtime path. The feature-parity registry has 46 entries; its tests passed. Neither fact proves controls execute their advertised behavior.

This is a broad, risk-based source and executable-contract assessment, not exhaustive verification of every engine or a visual usability certification. Windows Python 3.13 was available for the probes; the documented release profiles are Python 3.11/3.12. No MATLAB, native full-swing engine qualification, GPU benchmark, camera trial, packaged desktop journey, screen-reader trial, or measured athlete validation was performed. MATLAB acceptance must use **R2025b** per repository policy. The initial direct imports lacked the source-layout `PYTHONPATH`; the successful probes used the paths shown below. That setup failure is not presented as a product defect.

### Coverage and Existing Strengths

| Area                | Reviewed Boundaries                                                                                                     | Result and Remaining Limit                                                |
| ------------------- | ----------------------------------------------------------------------------------------------------------------------- | ------------------------------------------------------------------------- |
| Desktop UI/UX       | Matcher controls, request builder, worker lifecycle, result badges, tab routing; launcher truthfulness                  | Concrete R06–R08; no visual or accessibility conformance claim            |
| Web UI/UX           | App routing, error boundaries, API client, BallFlight inputs/results/imports                                            | Lazy routes and route recovery exist; R10 addresses result interpretation |
| Errors and Recovery | REST route/service boundary, persistence, task lifetime, Qt process failures                                            | R03, R08, R09; existing error helpers should be reused                    |
| Performance         | Worker offload, flight API event loop, request/sample bounds, task capacity, benchmark gates                            | R02–R05 and R12; no invented latency or speedup measurements              |
| Numerical Semantics | Recorder lifecycle, simulation clocks, flight events and metric definitions                                             | R01, R04, R11; full engine accuracy remains unqualified here              |
| Scientific Claims   | Neural promotion, measured-baseline contract, diagnostic receipts, deferred-validation catalog                          | R05, R06, R12; physical evidence remains a separate authority             |
| Shared Tools        | Gitlink/Python/Rust pins, agent-context boundary, public flight facade, ODE result/metrics, impact-interval termination | Pin checks passed; R04 crosses the actual provider/consumer boundary      |
| Wiring and Release  | UI → request → service → result; feature registry; readiness ledger and installed-product blocker                       | Registrations exist; R01/R07 show why behavior must be tested             |

Do not re-file the old missing-vendor-init finding (#9407), lazy engine-discovery finding (#8934), or WebSocket per-step event-loop finding (#8936) as if unchanged. The current source initializes the vendor in the documented install path, discovers engine availability through metadata, and batches WebSocket physics in `anyio.to_thread.run_sync`. Route splitting and error boundaries also exist. The flight families intentionally retain different coefficient models under ADR-0047; merging them is not the proposed fix.

### Backlog Reconciliation

Local authorities consulted: `docs/audits/2026-08-21-adversarial-integration-review.md`, `docs/audits/2026-08-21-uiux-performance-review.md`, `src/config/industrial_readiness.json`, `src/config/feature_parity.json`, `docs/development/DEVELOPMENT_LOG.md`, and `docs/development/planning/catalog.json`. The local `issues.json` and `open_issues.json` snapshots cover much older issue numbers and cannot establish current absence of duplicates.

Live GitHub reads were restored by selecting the valid stored account after initial CLI authentication failures. On 2026-09-28, all 42 open UpstreamDrift issue titles and seven open PR titles were reviewed; #10960 was confirmed closed on 2026-09-27. Its remediation is already reflected in the reviewed source. R05/R06/R12 are follow-up counterexamples, not a request to repeat the closed audit. The open Fleet Critic PR [#10977](https://github.com/D-sorganization/UpstreamDrift/pull/10977) discusses related neural evidence; reconcile its older observations against this snapshot before combining recommendations.

Existing open authorities include UpstreamDrift #10380 (professional release), #10375 (independent validation), #10382 (observable outputs), and #9417 (artifacts). A focused Tools open-issue search identified [#4260](https://github.com/D-sorganization/Tools/issues/4260) (flight contract) and [#4267](https://github.com/D-sorganization/Tools/issues/4267) (landing/bounce/roll) as parent programs for R04. This bounded title/body reconciliation is not an exhaustive search of every historical issue or active branch. **“New candidate” still requires a current implementation-issue and lease check before filing.** This review changes documentation only and claims none of those implementation issues.

## Issue Briefs

Each brief can become an issue body after the disposition is confirmed. Suggested common classification: `panel-review`, `tier:strong`; use repository-supported priority/domain labels rather than assuming new labels exist. Delegate small implementation slices only after the design and acceptance contract are agreed.

### R01 — Start the REST Simulation Recorder and Reject Empty Successful Results

**Priority / Confidence:** P0 / reproduced with the real service and real recorder, using a minimal fake stepping engine.

**Owner / Effort:** UpstreamDrift API / small to medium.

**Scope:** Synchronous and background REST simulations, recorded outputs, and their result contracts.

**Problem and Evidence:** [SimulationService][ud-simulation] creates `GenericPhysicsRecorder(engine)` but never calls `start()`. Its loop calls `record_step()` even when recording is false. The real [recording mixin][ud-recording] immediately returns in that state. `_extract_simulation_data` produces a nonempty dictionary of empty series, satisfying the response model's weak nonempty-dictionary check. A three-step run returned:

```text
success=True; frames=3; times=[]; engine_reads=0; recorded=0
```

This is a broken execution-to-analysis/export connection, not merely a missing display field. The 82-test service/parity batch passed; recorder mocks in `tests/unit/api/test_simulation_service.py` do not expose this lifecycle mismatch.

**Proposed Work:** Start and stop recording in a guaranteed lifecycle; define whether frames includes the initial sample; validate required channels and aligned nonempty time/state arrays before success. Preserve actual control inputs in the recording rather than relying on the recorder's default zero control. Surface truncation when capacity is reached.

**Acceptance Criteria:**

- A deterministic fake engine plus the real recorder produces the expected initial and stepped samples, nonzero commanded controls, and aligned time/state/control lengths through both REST paths.
- A required-channel extraction failure or zero recorded samples cannot return successful simulation status.
- Buffer exhaustion is explicit, and requested/executed/retained sample counts cannot silently disagree.
- One installed MuJoCo smoke journey runs → records → analyzes → exports → reloads with matching run identity and values; engine-free tests remain available separately.

**Board Decision:** Treat as an immediate supported-path blocker. Do not close it on mock-only tests or because the process exits successfully. Coordinate with R02/R09/R11 so their contracts use the same run result.

### R02 — Isolate Engine, Recorder, and Analysis State per Simulation Run

**Priority / Confidence:** P1 / deterministic interleaving reproduction; real-engine collision not executed.

**Owner / Effort:** UpstreamDrift API / medium to large.

**Related Work:** #1488; R01, R03, R09.

**Problem and Evidence:** The server places one simulation service and engine manager in app state. [`_prepare_engine`][ud-simulation] loads through that mutable manager and then reads its active engine. Concurrent `anyio` worker calls have no per-run ownership guard. The [manager][ud-engine] resets and replaces `active_physics_engine`. A barrier-backed test manager makes the interleaving explicit: requests for `mujoco` and `drake` both received the `drake` engine. Shared `_stats`, `_active_recorder`, `_last_recorder`, and biomechanics bindings also carry last-writer state. WebSocket speed/stats use the same service state. `TaskManager.engine_semaphore` is declared but has no caller in `src/api` beyond its definition.

**Proposed Work:** Choose an explicit ownership contract. A single-active-run desktop API with busy/queued responses is a valid first containment option. Multi-run support requires per-run engine instances and immutable run-addressed recorders/results. Do not “fix” this by only adding a four-slot semaphore around the same mutable manager.

**Acceptance Criteria:**

- A deterministic barrier test proves two overlapping preparations cannot acquire each other's engine/model/configuration.
- Overlapping REST/REST and REST/WebSocket requests either receive an explicit busy/queue response or independent run IDs, clocks, controls, results, and recording state.
- Analysis/export references a selected run ID; completion of another run cannot change the analyzed dataset.
- Engine cleanup occurs once after its owner finishes or cancels; one run cannot unload another's engine.

**Tradeoff for the Board:** Serialization is smaller and safer for a local single-user product; independent worker processes support scale and hard timeouts but add resource and recovery complexity. Decide the supported concurrency profile before choosing the implementation.

### R03 — Bound Simulation Work and Preserve Cancellable Jobs Under Load

**Priority / Confidence:** P1 / input and eviction counterexamples executed; resource exhaustion deliberately not attempted.

**Owner / Effort:** UpstreamDrift API, with Tools flight limits / medium.

**Related Work:** #6948, #1488, #6992; depends on R02 ownership decision.

**Problem and Evidence:** [SimulationRequest][ud-requests] accepts `duration=300, timestep=1e-6`, giving **300,000,000 steps** despite bounded individual fields. [BallFlightSimulationRequest][ud-flight-api] accepts `time_step_s=1e-12`; its default ten-second horizon describes **10 trillion potential sample times** before the actual event truncates the run. Both flight implementations allocate `np.arange` from an unchecked horizon/sample-interval ratio. The flight API's `async def` calls `_simulate_one` synchronously, including ODE solving and sampling, on the event loop. [TaskManager][ud-tasks] evicts the oldest task without checking whether it is running: a `max_tasks=1` probe removed the running record when a second pending record was inserted. Expiry likewise does not distinguish active work. The REST stepping loop has no cancellation/deadline check or progress updates within it.

**Proposed Work:** Validate aggregate step/sample/memory budgets before work begins; offload flight computation; admit jobs only within explicit capacity; retain active state independently of terminal-result eviction. Reuse existing cancellation contracts, including the Tools flight cancellation hook. Keep output sampling distinct from integrator tolerance/steps.

**Acceptance Criteria:**

- Excessive ratios, nonfinite inputs, and over-budget model batches fail before engine creation or large allocation, with an actionable limit response.
- A heartbeat/health request remains responsive while an intentionally slow injected flight calculation runs; choose the release-profile latency budget from measurements, not an invented universal threshold.
- Under full capacity, new work is rejected or queued; active jobs remain queryable and cancellable, and terminal data is bounded.
- Cancellation and deadlines stop actual computation within a documented bound, not merely stop client polling. Native non-cooperative engines have a containment policy.
- Progress reports executed work; cancellation and persistence failures have distinct terminal outcomes.

**Performance Evidence Required:** Before/after wall time, event-loop lag, peak memory, queue wait and cancellation latency, including rejected jobs and export/serialization costs. Do not claim Rust/GPU acceleration from algorithm names alone.

### R04 — Propagate Flight Termination Before Reporting Landing Metrics

**Priority / Confidence:** P1 / reproduced in both consumed model families.

**Owner / Effort:** Tools `swing_sim.flight` plus UpstreamDrift flight/API/UI / medium.

**Related Work:** ADR-0047, #8978, #9352; separate from already-implemented impact-interval separation handling.

**Problem and Evidence:** Both [UD flight integration][ud-flight] and [Tools flight integration][tools-flight] integrate to a terminal ground event **or** a time cap, then calculate metrics from the last retained point. They do not gate landing metrics on `status`, `success`, or a ground event. At 70 m/s, 0.3 rad launch angle and `max_time=0.1`, both returned:

```text
last height = 2.050773716461735 m
reported carry = 6.575110473940198 m
reported landing angle = -17.4520240434372 deg
```

The ball is airborne and ascending. These are partial-trajectory quantities being named as landing metrics. Source inspection also shows a failed solver's partial output can reach the same metric builder; solver-failure injection remains a required regression test. SciPy distinguishes interval completion, event completion and integration failure; `success=True` alone does not establish landing. See the [SciPy result contract](https://docs.scipy.org/doc/scipy/reference/generated/scipy.integrate.solve_ivp.html).

**Proposed Work:** Return explicit `landed`, `time_limit`, `solver_failed`, and `cancelled` semantics with terminal-event evidence and actual horizon. Keep useful partial trajectories; make carry/landing metrics unavailable unless their defining event occurred. Preserve these states in trajectory interchange, imports, inverse-solver objectives, and UI exports.

**Acceptance Criteria:**

- Time-cap-while-ascending, normal landing, negative launch, solver failure and cancellation fixtures have distinct results.
- A partial trajectory cannot score as a completed carry/landing target or silently enter a qualified dataset.
- The consumed Tools facade and UD API preserve the same termination meaning without forcing their aerodynamic coefficients to be identical.
- Tools provider PR lands first; UpstreamDrift pins the published revision and runs an actual provider-to-API-to-viewer contract test.

**Board Decision:** Approve coordinated provider/consumer work. Reuse the fail-closed termination pattern already present in Tools `impact_interval`; do not claim all Tools physics lacks termination handling.

### R05 — Refuse Neural Promotion Without Sufficient Comparable Evidence

**Priority / Confidence:** P1 / reproduced through the public benchmark API.

**Owner / Effort:** UpstreamDrift neural motion / medium.

**Disposition:** Create a bounded follow-up linked to closed #10960 and the #10625 benchmark program; reconcile with open critic PR #10977.

**Problem and Evidence:** [The promotion gate][ud-promotion] requires speedup and non-regression in p95 latency and acceptance rate. It does not require any attempted or accepted validation examples. With one latency sample for each method (1.0 s versus 0.1 s) and `AcceptedMatchMetrics(0,0,0,0,0,0,0)` for both, it returns `PromotionDecision.PROMOTED` and a production-qualification recommendation. [MeasuredBaseline][ud-benchmark-types] checks fields for `None`, not measurement ancestry or comparability. This is not the older candidate-derived baseline defect: the current runner already requires an explicit baseline.

**Proposed Work:** Separate “comparison arithmetic computed” from “promotion eligible.” Require nonempty measured validation, sufficient independent trials under a predeclared policy, matching model/task/quality tolerances, data/split/checkpoint identities, native replay evidence and hardware/runtime provenance. Insufficient evidence should be `UNQUALIFIED`/`RESEARCH_ONLY`, not a vacuous pass.

**Acceptance Criteria:**

- Zero attempts, zero accepted examples, missing native verification, incomparable datasets and missing measurement provenance cannot promote.
- A one-sample speed comparison cannot be labeled production qualification without an explicitly approved exception policy.
- Promotion evidence records the paired cohort, failures, seeds, distribution/uncertainty, quality thresholds and complete timing boundary.
- A positive fixture with real sufficient evidence promotes; forged field labels or a digest of arbitrary numbers do not substitute for source receipts.

**Board Decision:** The scientific/statistical panel must set sample sufficiency and non-inferiority policy. Avoid selecting arbitrary counts solely to make tests pass.

### R06 — Make Neural Verification Badges and Published Claims Fail Closed

**Priority / Confidence:** P1 / source-proven UI fallback and artifact contradiction; visual interaction not executed.

**Owner / Effort:** UpstreamDrift UI and research documentation / small to medium.

**Disposition:** Create a bounded follow-up linked to closed #10960 and #10626; reconcile with critic PR #10977 and preserve historical evidence.

**Problem and Evidence:** [Matcher result rendering][ud-matcher] treats any nonempty `neural_inference` dictionary without `status` as `NEURAL_ACCEPTED`, defaults preview to false, and displays **VERIFIED**. When the next result has no neural metadata, that badge is not reset in this path. Separately, `docs/plans/neural_motion_matching/evidence/nm10_benchmark_speed_efficiency_receipt.json` is explicitly `DIAGNOSTIC` with a note that baseline/cost estimates were unmeasured. Its [human-readable report][ud-nm-report] still publishes `PROMOTED`, 4.52–5.15× speedups and break-even numbers as achieved results. The native checkpoint verifier now correctly raises `NotImplementedError` without real weights/rollout; the report must reflect that boundary.

**Proposed Work:** Render qualification only from a validated, run-bound verdict. Missing/malformed metadata is unknown/unverified. Clear badges on a new run and when new results omit neural evidence. Generate human-readable claims from the same disposition-bearing artifact, preserving historical data with an explicit diagnostic banner.

**Acceptance Criteria:**

- A dictionary containing timing only cannot display VERIFIED; preview and classical fallback remain distinguishable.
- Load a verified result, then a classical/unverified result: no stale verification or confidence survives.
- Missing verification receipt, mismatched run/model/checkpoint, and contradictory statuses fail closed.
- Markdown summaries, nested JSON promotions and handoff claims cannot advertise qualification when the governing receipt is diagnostic.

**Board Decision:** Separate UI truthfulness repairs from commissioning expensive new training. The former is implementable now; the latter needs R05 evidence and explicit resources.

### R07 — Wire Neural Matching Controls to Executed Requests or Mark Them Unavailable

**Priority / Confidence:** P1 / complete source trace; no native neural inference executed.

**Owner / Effort:** UpstreamDrift motion-matching UI / medium.

**Disposition:** Follow-up to #10626, dependent on R05/R06 qualification semantics.

**Problem and Evidence:** [The matcher][ud-matcher] creates “Classical Only / Neural Preview / Neural Verified,” a hard-coded model selector, and “Allow classical fallback.” Its `request()` reads none of those widgets. [`MatchRequest` and its command builder][ud-match-request] contain no corresponding fields. Existing GUI tests assert the controls exist and call badge updates directly; that does not prove the controls affect a run.

**Proposed Work:** Carry mode, model/checkpoint identity and fallback choice through the public request/CLI/service boundary to the existing verified-inference orchestrator. Populate availability from qualified registry state. Until executable, show a clear unavailable/research explanation and the next action rather than an effective-looking control.

**Acceptance Criteria:**

- Changing each control changes the captured execution request, or explicitly refuses an unsupported selection.
- Preview cannot be promoted to verified; disabling fallback is honored; checkpoint mismatch and missing runtime return named reasons.
- A UI interaction test drives the same controller/request path as Run, then checks the returned run-bound disposition; construction-only tests are insufficient.
- Desktop/web parity records reflect actual supported behavior and any intentional platform gap.

**Board Decision:** Choose the minimum supported neural journey before adding more model choices. Preserve access to classical matching.

### R08 — Bind Matcher Process Events to Their Run and Handle Failed Starts

**Priority / Confidence:** P1 / failed-start QCoreApplication probe reproduced; tab-routing defect traced in source.

**Owner / Effort:** UpstreamDrift desktop / small to medium.

**Problem and Evidence:** [`RunWorker`][ud-matcher] connects `QProcess.finished` but not `errorOccurred`. Starting a nonexistent executable and servicing the event loop produced `is_running=True`, no terminal event, and `ProcessState.NotRunning`. This can leave the interface in a running state indefinitely. `_on_output` and `_on_finished` route by `tabs.currentIndex()`, not by the tab/run that started the process. Switching tabs while work runs can send its logs/results to the wrong workflow; indices 3 and 4 have no corresponding worker-output branch.

**Proposed Work:** Use an immutable run context with owning panel, request, output directory and run ID. Handle failed start, nonzero exit, crash and cancellation through one idempotent finalizer. Preserve diagnostics, restore controls, and dispose each completed process. Do not interpret a successful process exit as scientific acceptance.

**Acceptance Criteria:**

- Missing executable, denied executable, crash, cancellation and normal completion each emit exactly one terminal result and restore controls.
- Start in Matching, switch to each other tab, and complete: output and status remain attached to the initiating run.
- Repeated start/stop does not leak process objects or duplicate completion callbacks.
- Failure identifies the failing stage and a recovery action, retaining user inputs and useful logs.

Qt provides a specific failed-start error signal; see the [QProcess contract](https://doc.qt.io/qt-6/qprocess.html). Reuse existing process/worker helpers where they fit rather than inventing another lifecycle system.

### R09 — Preserve Actionable Error and Partial-Result States Across the Simulation API

**Priority / Confidence:** P1 / service error payload reproduced; persistence/extraction paths source-reviewed.

**Owner / Effort:** UpstreamDrift API and clients / medium.

**Related Work:** #5911, #8009, #8871; depends on R01/R02 run identity.

**Problem and Evidence:** [`run_simulation`][ud-simulation] catches domain/value/runtime errors and returns `success=False` with empty data and no error field. The route returns that normally, bypassing its own intended exception-to-HTTP mapping. An injected `ValueError('invalid control dimensions')` yielded only `success`, `duration`, `frames`, `data`, `analysis_results`, and `export_paths`. Background execution embeds the same reasonless failure. Persistence failures return an empty export list while the run remains successful; required-series extraction can log and return partial data. Scientific computation, data completeness, analysis and saving are different outcomes but are not represented explicitly.

**Proposed Work:** Define a typed outcome with stable error code, safe message, stage, run/correlation ID, retry guidance, and separate calculation/analysis/persistence status. Preserve server tracebacks without exposing local paths or internals. Reuse the existing domain hierarchy and API error infrastructure; avoid a second exception framework.

**Acceptance Criteria:**

- Invalid input, unavailable engine, numerical failure, cancellation, missing data and disk-full export are distinguishable in both REST modes and clients.
- HTTP status and payload semantics agree; background jobs retain a machine-readable failure reason.
- A completed calculation with failed saving remains recoverable in memory and offers explicit retry/export; it does not imply the file exists.
- Required analysis failure cannot silently look like complete analysis; optional channel absence is separately labeled.
- Tests exercise real service-to-route behavior, not only a mocked service raising what production catches.

### R10 — Keep Ball-Flight Results Attached to Their Input Snapshot and Model Provenance

**Priority / Confidence:** P1 / source-confirmed state relationship; no visual or assistive-technology audit performed.

**Owner / Effort:** UpstreamDrift web / medium.

**Related Work:** #7456, #8978, #9352; depend on R04 terminal-result schema.

**Problem and Evidence:** [BallFlight][ud-ballflight-ui] stores editable inputs separately from results and replaces results only after a successful request. Old results remain when inputs change or a later request fails, with no committed input snapshot or result-staleness state in this path. The result interface omits the coefficient metadata returned by the API. Imported curves do carry family/digest provenance; computed curves need equivalent inspectable identity. A professional scientific interface must make clear which inputs produced the visible numbers.

**Proposed Work:** Preserve prior results as useful history, labeled with their immutable submitted input/model/version/units snapshot. Indicate changed inputs and “previous result” on failure; distinguish queued/running/complete/partial/cancelled. Display or expose coefficient-set and qualification metadata. Make status changes available to assistive technology, not only color/spinners.

**Acceptance Criteria:**

- Run A, edit inputs for B, then fail B: A remains explicitly labeled with A's inputs and provenance; it cannot appear to represent B.
- Out-of-order completion cannot overwrite the active run's result; cancel/unmount does not imply backend cancellation unless it occurred.
- Exported and imported records preserve family, coefficients or parameter digest, input snapshot, source revision, units/frame and termination.
- Keyboard users can run, inspect status, recover from error and compare results without losing focus or inputs; status announcements follow [WCAG status-message guidance](https://www.w3.org/WAI/WCAG22/Understanding/status-messages.html).

**Board Decision:** Approve a reusable run/result presentation contract, beginning with this page and matcher. Do not start a broad visual rewrite before these semantics are stable.

### R11 — Report the Actual Integrated Horizon Consistently Across REST and WebSocket Runs

**Priority / Confidence:** P1 / source-proven arithmetic; native engine timing not run.

**Owner / Effort:** UpstreamDrift simulation / small to medium.

**Related Work:** R01/R04; preserve engine-specific supported step contracts.

**Problem and Evidence:** [REST][ud-simulation] uses `int(duration / timestep)` full steps but returns `duration=request.duration`. [WebSocket][ud-ws] uses `ceil(duration / timestep)` full steps, then clamps the reported clock with `min(duration, frame*timestep)`. For 0.025 s requested at 0.01 s, REST steps to 0.02 s and reports 0.025 s; WebSocket steps to 0.03 s and labels the end 0.025 s. This misaligns state and time, affecting derivatives, replay, event comparison and provenance even after recording is fixed.

**Proposed Work:** Define one endpoint-inclusive/exclusive sampling policy. Either take a supported remainder step, reject incompatible horizons, or report the actual executed horizon explicitly. Never relabel a state with an earlier requested time. Separate requested duration, integrated duration, step count and retained sample count.

**Acceptance Criteria:**

- Non-divisible, divisible, sub-step and floating-point-boundary durations produce truthful clocks in REST, WebSocket, recorder and export.
- A deterministic engine recording every `dt` proves the reported final state time equals the sum of executed steps.
- Cross-engine tests state which backends accept variable final steps; no silent per-backend reinterpretation.
- The state timestamp remains consistent through trajectory interchange and live analysis.

### R12 — Define and Validate Neural Data-Efficiency Claims Statistically

**Priority / Confidence:** P2 / executed numerical counterexample; no new learning experiment.

**Owner / Effort:** UpstreamDrift neural motion and statistical reviewer / medium.

**Disposition:** Use the #10625 benchmark program and link the closed #10960 audit; avoid a competing benchmark program.

**Problem and Evidence:** [`evaluate_data_efficiency`][ud-efficiency] documents monotone budgets but accepts `[100, 1]`. Equal curves `[0.5,0.5]` produce `active_superiority_confirmed=True`. The “sample efficiency multiplier” is the ratio of arithmetic mean acceptance values, not a ratio of sample budgets at matched performance. Budget spacing is ignored. [The report][ud-nm-report] then interprets this quantity as fewer episodes to equivalent accuracy. Those are different estimands.

**Proposed Work:** Have the panel choose the scientific claim first: descriptive acceptance gain, budget-weighted learning-curve area, or budget-to-target efficiency. Rename existing descriptive quantities honestly. If claiming superiority, require independent repeated runs and predeclared uncertainty/decision rules; retain inconclusive outcomes.

**Acceptance Criteria:**

- Unsorted/nonpositive/duplicate budgets are rejected or handled under an explicit documented policy.
- Equal curves cannot assert strict superiority; zero-reference and unattained-target cases remain undefined/inconclusive rather than defaulting to 1× success.
- Analytically constructed curves distinguish an acceptance-rate ratio from sample-budget savings.
- Report generation uses the chosen estimand and propagates diagnostic status; seed-level receipts and uncertainty support any real superiority claim.

## Existing Programs to Retain, Not Duplicate

### Installed Product and Cross-Repository Acceptance — Existing #9417 / #10380 / #9539

The industrial readiness ledger correctly remains blocked on installable-artifact journeys. Keep this as a release gate, with paired provider/consumer evidence. The current 52-file context map and source-tree tests are not a substitute for the shipped wheel/desktop bundle resolving the intended Tools implementation.

Extend the existing acceptance plan with four journeys: install → select supported model → simulate → record/analyze → export/reload; captured data → calibrated matching → inspect qualified/unqualified results; impact → Tools flight → trajectory interchange → another viewer; and missing runtime/disk-full/interrupted job → recovery. Run against the exact packaged provider pin, outside the source checkout and test `PYTHONPATH`, with mocks disabled. Include source/runtime identity, platform/profile, build digest and screenshots or recordings. Test module presence and UI construction separately from scientific accuracy.

Global test bootstrap currently repairs import failures with synthetic packages and may substitute Qt mocks. That supports headless unit testing, but installed-product acceptance must prove it did not take those fallbacks. This review's initial source-layout import failure is setup evidence only, not proof the installed artifact is broken.

### Independent Scientific Validation — Existing Deferred Board Plans

Retain the existing deferred catalog, including DV-10375 (identifiability/uncertainty), DV-10382 (observable golf outputs and error budgets), DV-9546 (impact qualification), DV-9619 (markerless calibration), DV-9613 (camera soak), and DV-9700 (impact/acoustics research). Do not reopen deferred physical work merely because measurements remain unavailable.

Board decisions should specify claim, domain, independent instrument/reference, calibration uncertainty, held-out subjects/trials, repeatability, observability, thresholds and responsible reviewer before activation. Marker-fit error is not an impact-force or injury-risk validation. Conservation checks and cross-engine agreement establish useful numerical consistency, not agreement with physical reality. Software correctness, model validation, human usability and release approval must remain separate records.

## Recommended Sequence and Board Questions

1. **Contain false results:** R01 first; contain shared-state overlap under R02. Keep unqualified neural claims visibly unqualified under R06.
2. **Make outcomes truthful:** R04, R09 and R11 define termination, error, sample/time and persistence semantics.
3. **Wire and recover:** R07/R08/R10 connect actual choices, owning runs and recoverable UI states.
4. **Measure and bound:** R03 establishes resource/cancellation limits; R05/R12 establish defensible performance and efficiency evidence.
5. **Qualify the delivered product:** complete the existing artifact journeys; commission physical/human validation only through the existing Board plans.

R04 requires a Tools provider PR followed by a consumer pin and integration PR. R02's concurrency decision affects R03 and R09. R05/R06 qualification semantics constrain R07; a runnable selector alone is not neural verification. UI work can proceed with explicit unavailable states while native qualification is outstanding.

### RunnerDashboard Panel Brief

Runner_Dashboard's current `PanelCreateRequest` supports 3–4 read-only experts, 1–6 rounds, and `debate` or `brainstorm`; the topic limit is 4,000 characters. Use the report/PR as reading material, with the short brief below as the topic. The three legacy `panel-opinion` tiers are a different workflow; do not confuse their stance format with the interactive expert-panel API. No panel was dispatched and no paid review run was authorized by this document.

**Suggested Seats:** Scientific/Numerical Reviewer; Product and Accessibility Reviewer; Runtime/Performance Engineer; Integration and Release Reviewer. Map seats to enabled providers and available models at dispatch time. The existing moderator should synthesize disagreements rather than treating seat count as scientific evidence.

> Review the UpstreamDrift 2026-09-28 product assessment at the attached PR, including its exact Tools pin. Challenge R01–R12 and distinguish executed evidence, source-inferred defects, and unperformed native/physical validation. For each item decide accept, modify, merge into an existing issue, defer, or reject; give confidence, owner, prerequisites, a minimum implementation slice and falsifiable acceptance evidence. Prioritize false successful results, model/time identity, and unqualified scientific claims before UI polish or acceleration. Decide whether the supported API is single-active-run or truly concurrent. Preserve the two legitimate flight families and existing deferred physical-validation plans. Do not infer current issue status from historical snapshots or issue closure from test-file presence. End with an ordered, capacity-bounded implementation queue and unresolved decisions. This review requests no code changes or release/scientific approval.

**Moderator Output:** `ID | Disposition | Existing/New Issue | Priority | Owner | Dependencies | Acceptance Evidence | Dissent`. Record which findings were disproved or superseded. Publish implementation issues only after live deduplication and lease checks. Architectural/cross-repo direction should use Repository_Management's formal Board proposal intake; this PR is the evidence packet, not an alternative decision authority.

## Validation Record

### Executed Checks

Governance follow-up: the central development-log validator reports pre-existing duplicate entries, missing metadata and size/WIP ceiling violations on the reviewed base. The new PR #11080 entry adds no entry-level validator finding; existing unrelated entries were preserved. SPEC changelog validation and the repository documentation hooks passed.

The following checks ran in the isolated review worktree at the UpstreamDrift revision above and exact initialized Tools pin:

```powershell
$env:PYTEST_DISABLE_PLUGIN_AUTOLOAD='1'
$env:QT_QPA_PLATFORM='offscreen'
python3 -m pytest tests/unit/neural_motion/test_benchmark_nm10.py tests/unit/api/test_task_manager_expiry_throttle.py tests/unit/scripts/test_check_tools_pins.py -o addopts='' -q -p pytest_timeout --timeout=60
# 26 passed; 9 warnings; 55.38 s
python3 -m pytest tests/unit/api/test_simulation_service.py tests/unit/api/test_simulation_service_async.py tests/config/feature_parity -o addopts='' -q -p anyio.pytest_plugin -p pytest_timeout --timeout=60
# 82 passed; 12 warnings; 11.89 s
```

These are 108 focused passing tests, not a full suite or coverage-gate result. Warnings concerned package identity/deprecation and missing suite markers. No application Python or TypeScript source was changed. Additional document/policy checks are recorded in the PR validation summary.

### Reproduction Fragments

Run these from the reviewed checkout with its dependencies and exact vendor pin. For source-layout probes on Windows:

```powershell
$env:PYTHONPATH='src;vendor/ud-tools/src;vendor/ud-tools/src/shared/python;vendor/ud-tools/src/python/src'
```

**R01: Real Service and Recorder, Fake Engine and Non-Writing Output Sink**

```python
from pathlib import Path
from types import SimpleNamespace
from src.api.models.requests import SimulationRequest
from src.api.services.simulation_service import SimulationService

class Engine:
    def __init__(self):
        self.t = 0.0
        self.reads = 0

    def step(self, dt):
        self.t += dt

    def get_full_state(self):
        self.reads += 1
        raise AssertionError("A started recorder would read engine state")

engine = Engine()
manager = SimpleNamespace(_load_engine=lambda _: None,
    get_active_physics_engine=lambda: engine)
sink = SimpleNamespace(save_simulation_results=lambda *a, **kw: Path("not-written.json"))
service = SimulationService(manager, output_manager=sink)
result = service._run_simulation_sync(SimulationRequest(
    engine_type="mujoco", duration=0.03, timestep=0.01))
print(result.success, result.frames, result.data["times"], engine.reads,
      service.active_recorder.current_idx)
# True 3 [] 0 0
```

**R03/R05/R12: Safe Contract Counterexamples — Do Not Execute the Huge Simulation**

```python
from src.api.models.requests import SimulationRequest
from src.api.routes.ball_flight import BallFlightSimulationRequest
from src.api.task_manager import TaskManager
from src.shared.python.neural_motion.benchmark import (
    LatencySummary, AcceptedMatchMetrics, evaluate_promotion_gate,
    evaluate_data_efficiency,
)
r = SimulationRequest(engine_type="mujoco", duration=300, timestep=1e-6)
print(int(r.duration / r.timestep))  # 300000000; validation only
f = BallFlightSimulationRequest(time_step_s=1e-12)
print(f.max_time_s / f.time_step_s)  # 10000000000000.0; validation only
t = TaskManager(max_tasks=1)
t.set("active", {"status": "running"})
t.set("next", {"status": "pending"})
print(t.get("active"))  # None
m = AcceptedMatchMetrics(0, 0, 0, 0, 0, 0, 0)
print(evaluate_promotion_gate(LatencySummary.from_samples([1.0]),
    LatencySummary.from_samples([0.1]), m, m).decision)  # PROMOTED
print(evaluate_data_efficiency([100, 1], [0.5, 0.5], [0.5, 0.5]))
# accepts reversed budgets; active_superiority_confirmed=True
```

**R04: Airborne Endpoints Reported as Landing in Both Families**

```python
from src.shared.python.physics.flight_models import (
    WaterlooPennerModel, UnifiedLaunchConditions,
)
from shared.python.swing_sim.flight import (
    FlightModelRegistry, FlightModelType, LaunchConditions,
)
ud = WaterlooPennerModel().simulate(
    UnifiedLaunchConditions(ball_speed=70, launch_angle=0.3), max_time=0.1)
provider = FlightModelRegistry.get_model(FlightModelType.WATERLOO_PENNER)
tools = provider.simulate(
    LaunchConditions(ball_speed=70, launch_angle=0.3), max_time=0.1)
for result in (ud, tools):
    print(result.trajectory[-1].position[2], result.carry_distance,
          result.landing_angle)
# both: 2.050773716461735 6.575110473940198 -17.4520240434372
```

**R08: Failed Start Leaves the Worker Running**

```python
from PyQt6.QtCore import QCoreApplication, QTimer
from src.tools.motion_matching.gui import RunWorker
app = QCoreApplication([])
worker = RunWorker()
terminal = []
worker.finished.connect(terminal.append)
worker.start([["C:/codex-audit-missing-executable-20260928.exe"]])
QTimer.singleShot(250, app.quit)
app.exec()
print(worker.is_running(), terminal, worker._process.state())
# True [] ProcessState.NotRunning
```

The QProcess probe launches no GUI window. For R02, an injected manager assigned its active engine then waited on a two-party `threading.Barrier` inside `_load_engine`; two concurrent calls to the real `_prepare_engine` returned `[('mujoco', 'drake'), ('drake', 'drake')]`. This is a controlled interleaving test, not a claim that a particular native engine race was observed. For R09, replacing only `_run_simulation_sync` with a function raising `ValueError` demonstrated the reasonless failure response described above.

## Source Permalinks

All code links below are fixed to reviewed commits. Symbols and line anchors identify the evidence; later implementations must recheck current code.

[ud-simulation]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/api/services/simulation_service.py#L296-L565
[ud-recording]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/shared/python/dashboard/_recorder_recording.py#L21-L52
[ud-engine]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/shared/python/engine_core/engine_manager.py#L338-L388
[ud-tasks]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/api/task_manager.py#L105-L170
[ud-requests]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/api/models/requests.py#L41-L162
[ud-flight-api]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/api/routes/ball_flight.py#L48-L235
[ud-flight]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/shared/python/physics/flight_models.py#L315-L395
[tools-flight]: https://github.com/D-sorganization/Tools/blob/3678409fc51024150ab28970b72e3b468935f345/src/shared/python/swing_sim/flight/models.py#L144-L237
[ud-promotion]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/shared/python/neural_motion/benchmark/gates.py#L20-L88
[ud-benchmark-types]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/shared/python/neural_motion/benchmark/types.py#L147-L257
[ud-matcher]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/tools/motion_matching/gui.py
[ud-match-request]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/tools/motion_matching/pipeline.py
[ud-nm-report]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/docs/plans/neural_motion_matching/benchmark_accepted_speed.md
[ud-ballflight-ui]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/ui/src/pages/BallFlight.tsx#L69-L246
[ud-ws]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/api/routes/simulation_ws.py#L657-L743
[ud-efficiency]: https://github.com/D-sorganization/UpstreamDrift/blob/599cce5d5cb7d6972dfa2ca4e5770ea09cfaa309/src/shared/python/neural_motion/benchmark/efficiency.py

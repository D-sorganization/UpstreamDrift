# Mobile Workflow and Runtime Boundary for Issue #14266

## Current Evidence

- The React web/Tauri client defines task routes in
  [`routes.ts`](../../ui/src/routes.ts) and groups tools into five primary
  workspaces in [`workspaceNavigation.ts`](../../ui/src/types/workspaceNavigation.ts).
- Simulation work already has a server-side job pattern. The BunkerShot3D
  client submits a job, polls on a bounded backoff schedule, supports cancel,
  and rejects stale results in [`useBunkerJob.ts`](../../ui/src/pages/useBunkerJob.ts).
- The existing 390 px layout guard checks grid breakpoints on three pages; it
  does not exercise an end-to-end mobile workflow or job resume behavior
  ([`responsiveLayout.test.ts`](../../ui/src/pages/responsiveLayout.test.ts)).

## Proposed Mobile Workflow

Treat mobile as a responsive client to the existing server, not a device-side
simulation runtime. The first useful path is:

1. Open a primary workspace and select a supported tool.
2. Submit a simulation through the existing authenticated service.
3. Keep the server-issued job identifier while the job runs; show pending,
   completed, failed, and cancelled states, and let the user return after
   switching apps or losing connectivity.
4. Show the completed result and its authorized artifact reference through the
   existing service boundary.

Keep native iOS/Android packaging, offline simulation, background execution,
and a new request/result API out of this slice. The later API-contract child
[#14269](https://github.com/D-sorganization/UpstreamDrift/issues/14269) is
explicitly conditional on a confirmed external-client need.

## Runtime and Falsification Criteria

- All physics execution remains server-side; the mobile client only submits,
  observes, cancels, and displays jobs through existing authenticated routes.
- A job may outlive a foreground browser session. On resume, the client must
  recover the same job state without duplicate submission or showing a stale
  result. Network loss must leave a clear retryable status.
- Qualify touch layout at 390 px first, matching the existing layout guard;
  add narrower and wider viewport cases when the supported device range is
  agreed. Do not infer a simulation-duration limit from viewport tests.
- The falsifying flow is one server job that completes while the client is
  backgrounded, followed by correct result retrieval on return. Also verify
  failure and cancellation states, and that a timed-out poll does not submit a
  second job.

This discovery defines the mobile workflow boundary. It does not authorize
deployment, a new listener, a new API contract, or changes to simulation
execution.

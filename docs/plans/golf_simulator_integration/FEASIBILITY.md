# Feasibility and Evidence

## Executive Assessment

**Feasible with implementation work:** sending model-derived shots to GSPro
and showing flight on its licensed Windows display. **Not yet established:**
scientifically qualified contact-to-shot generation for every engine, native
GSPro model animation, or closed-loop autonomous golf using GSPro course state.

This review separates source capability, design inference and live evidence.
The first source survey used `12216ed464a17ca593b716770f6afd59235e531b`;
the relevant physics, impact, visualization and realtime source paths were
unchanged against the planning baseline
`395d3de876d8f87b7b00785ef2f430d0e9387ce7`. Agent-context tooling was unavailable
in the survey environment; conclusions use direct source inspection instead
of asserting a fresh generated context review. Existing tests were inspected,
not executed as proof of a GSPro integration.

## GSPro Evidence

The vendor documents Open Connect as bidirectional JSON over a local TCP
socket at port 921, with shot/status input and response/player information.
It names success, player-update and failure codes. The documented surface
does not describe mesh/animation injection, course geometry, complete ball
state, flight-completion events or aim-control commands.
Source: [GSPro Open Connect v1](https://gsprogolf.com/GSProConnectV1.html),
reviewed 2026-09-15.

Important specification gaps remain: precise speed-unit behavior, signed spin
conventions, framing, heartbeat timing, response correlation, duplicate-shot
handling and reconnect behavior. These require version-specific observations
or vendor clarification before a production profile can be qualified. The
example payload alone is not sufficient evidence for those semantics.

GSPro's own site offers community integration through its API and an OpenAPI
license choice, and describes Windows support and Unity course-design tools.
The latter do not establish a runtime avatar SDK. Check the installed license
mode through the application; do not assume installation establishes an active
license or compatible connector entitlement.
Source: [GSPro Product and Integration Information](https://gsprogolf.com/).

### Local Observation

Read-only inspection on 2026-09-15 found:

- `C:/GSProV1/GSPLauncher.exe`.
- `C:/GSProV1/Core/GSP/GSPro.exe`.
- `C:/GSProV1/Core/GSPC/GSPconnect.exe`.
- No matching GSPro/GSPconnect process or listening port 921 was observed.
- The GSPro executable's PE file version was `2018.2.8.11407744`; this resembles
  Unity build metadata and is **not accepted as the installed GSPro release**.

No license keys, account files or private configuration contents were read.
No application was launched and no test shot was transmitted. Runtime version,
license mode, connector selection and real shot behavior remain unverified.

## Reusable UpstreamDrift Components

Paths below are relative to the repository root, and identify existing code.

| Concern             | Existing Public Surface                                                                            | Integration Use and Limitation                                                                       |
| ------------------- | -------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------- |
| Swing to Launch     | `src/shared/python/physics/__init__.py`: `SwingBallFlightPipeline`, `SwingState`, `PipelineResult` | Extract launch/post-impact data through public results; do not call private derivation helpers       |
| Launch Values       | `src/shared/python/physics/ball_launch_conditions.py`: `LaunchConditions`                          | Speed m/s, angles radians, spin RPM plus axis; additional boundary validation required               |
| Impact              | `src/shared/python/physics/impact_model/__init__.py`                                               | Public pre/post state, solver, event recorder; preserve full state before claiming qualification     |
| Engine Sources      | `src/shared/python/physics/swing_state_providers.py`                                               | Manual and MuJoCo sources exist; Drake/Pinocchio providers are placeholders                          |
| MuJoCo Example      | `src/shared/python/physics/mujoco_swing_source.py`                                                 | Scripted dynamics with peak-speed sample; not a verified ball-contact sample                         |
| Trace               | `src/shared/python/simulation_backends/protocol.py`: `Trace`                                       | Time/state/control/contact data for immutable replay association                                     |
| Viewport            | `src/shared/python/visualization/__init__.py`                                                      | Renderer selection, overlay payload, Rerun render/export; adapt existing presentation infrastructure |
| Existing Viewer     | `src/tools/tour_matching_viewer/`                                                                  | Reuse model replay UI where suitable; do not build another skeleton or FK solver                     |
| Unreal              | `src/unreal_integration/__init__.py`                                                               | Existing streaming/skeleton concepts; separate product, no GSPro interoperability implied            |
| Presentation Events | `src/shared/python/realtime/__init__.py`                                                           | Best-effort IPC for visualization; cannot serve as authoritative shot delivery                       |
| Analytics Import    | `src/tools/launch_monitor_model/__init__.py`                                                       | Delegates to Tools-owned launch-monitor analytics; import support is not an outbound adapter         |

### Qualification Defects to Resolve

1. The pipeline constructs a rich `PreImpactState`, then `_solve_impact()` uses
   a simplified solver API that does not forward club angular velocity, loft
   and MOI. `ImpactSolverAPI.solve_impact()` reconstructs angular velocity as
   zero. Tests must prove actual field influence before claiming preservation.
2. A peak-speed sample need not coincide with club–ball impact. Exported shots
   need a detected/contact-qualified event or an explicit demonstration label.
3. A provider registry entry does not prove a working engine. Qualify one engine
   first; keep unsupported engines unavailable. Simscape acceptance uses
   MATLAB R2025b explicitly, per the repository's existing requirement.
4. `LaunchConditions` is frozen but contains a mutable NumPy axis. Snapshot
   arrays/tuples at the outbound boundary and reject nonfinite/invalid values.
5. Launch-monitor analytics uses rad/s for spin, while `LaunchConditions` uses
   RPM. Convert once through a named boundary, with independently derived tests.
6. Existing model fitting and closure acceptance are not complete scientific
   ball-impact validation. Track software, contact and scientific status separately.

### Existing Tests to Reuse

- `tests/unit/physics/test_swing_ball_flight_pipeline_5337.py`
- `tests/unit/physics/test_swing_state_providers.py`
- `tests/unit/physics/test_mujoco_swing_source.py`
- `tests/unit/impact_model/test_public_api_parity.py`
- `tests/unit/physics/test_impact_friction_axis_and_gear_offset.py`
- `tests/unit/visualization/test_viewport.py`
- `tests/unit/visualization/test_rerun_renderer.py`
- `tests/unit/launch_monitor/test_importer.py`
- `tests/shared/realtime/` and `tests/unit/realtime/`

These tests inform reuse and regression coverage. None establishes a live
GSPro client or licensed end-to-end acceptance.

## Approach Evaluation

Criteria: Python/API fit, reuse, maintainability, observable delivery failure,
simulator independence, real-time presentation and public protocol maturity.

| Approach                            | Gains                                                                      | Costs and Limits                                                                       | Decision                                                     |
| ----------------------------------- | -------------------------------------------------------------------------- | -------------------------------------------------------------------------------------- | ------------------------------------------------------------ |
| Direct Python Adapter Behind a Port | Small dependency surface; ownership of failure semantics; easy local tests | We own sparse-protocol characterization and maintenance                                | Selected for GSPro baseline                                  |
| Existing Community Relay            | Existing monitor connectivity and vendor-neutral event stream              | Additional process/version/dependency boundary; behavior still needs conformance tests | Evaluate as optional adapter; do not make it the core domain |
| Local Reference Renderer First      | Full control over golfer, ball trajectory and test fixtures                | Does not supply licensed GSPro courses or verify commercial behavior                   | Implement alongside GSPro as a second destination            |

The direct design follows the repository's existing injected protocols.
Python's [asyncio Streams Documentation](https://docs.python.org/3/library/asyncio-stream.html)
provides the transport primitives; draining a writer is flow control rather
than an application acknowledgment. Standard-library reuse does not solve
vendor message framing or transactional ambiguity automatically.

[flighthook](https://github.com/divotmaker/flighthook) is an optional reuse
candidate: its maintainers describe a beta relay with REST/WebSocket events
and a GSPro destination. Its Flight Relay Protocol may be useful for external
interchange. Evaluate a pinned version, license and replay/delivery behavior;
it does not establish another commercial simulator destination. Do not copy
its code or adopt its evolving schema as UpstreamDrift's internal model.

## Other Simulators

| Destination                            | Current Evidence                                                         | Action                                                                                     |
| -------------------------------------- | ------------------------------------------------------------------------ | ------------------------------------------------------------------------------------------ |
| Local UpstreamDrift Reference          | Existing flight and rendering primitives                                 | First non-GSPro adapter; useful without a subscription                                     |
| GSPro                                  | Public shot-input documentation                                          | Primary commercial target; qualify installed version                                       |
| E6 CONNECT                             | Vendor release notes show multiple tracking integrations and SDK updates | Obtain authorized integration documentation and entitlement before promising an adapter    |
| Creative Golf                          | Vendor lists supported launch-monitor integrations                       | Obtain a supported SDK/protocol and verify licensing; device compatibility is insufficient |
| FSX, TrackMan, Awesome Golf and Others | No integration contract established in this review                       | Discovery only; no support claim or implementation commitment                              |

Sources: [E6 Release Notes](https://connect.e6golf.com/update/e6-connect/),
[Creative Golf Devices](https://creativegolf.com/devices/).
These sources establish vendor integrations, not public APIs available to us.
Switching must be capability-aware: a backend may accept shots without exposing
course state, animation or returned trajectories.

## Recommendation

Proceed with the core epic. Keep model→launch, shot delivery and visualization
as separate contracts. Gate GSPro unit/sign behavior and contact qualification
before delegating mechanical codec work. Build a useful companion experience
while obtaining vendor answers for native animation and autonomous rounds.
No commercial purchase or vendor communication was performed in this review.

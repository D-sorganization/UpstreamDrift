# Golf Simulator Integration Acceptance Runbook

## Status and Scope

This is the acceptance procedure for [Epic #10188](https://github.com/D-sorganization/UpstreamDrift/issues/10188) and child issue GS-09 ([#10198](https://github.com/D-sorganization/UpstreamDrift/issues/10198)). For the comprehensive topology support matrix, hardware prerequisites, capability boundaries, and disconnect recovery playbooks, see [SUPPORT_MATRIX.md](SUPPORT_MATRIX.md). The corresponding opt-in live qualification test suite is implemented in `tests/integration/golf_simulator/test_live_acceptance.py`. Execute implementation-dependent steps only after the relevant child issue supplies the named feature and test entry points. Record unavailable checks as **BLOCKED**, never as passed.

The baseline experience shows the model swing in a companion view and sends a qualified shot to a licensed GSPro host for ball flight. Native model/avatar insertion into GSPro and automated course-state control remain research gates: no supported SDK for those capabilities has been established. A shot acknowledgment does not establish landing, scoring, readiness for the next shot, or completion of the displayed flight.

Implementation starts with the worker's vendor-independent GS-01 domain contract ([#10190](https://github.com/D-sorganization/UpstreamDrift/issues/10190)). GS-02 codec work ([#10191](https://github.com/D-sorganization/UpstreamDrift/issues/10191)) waits for the lead's GS-00 protocol-profile freeze ([#10189](https://github.com/D-sorganization/UpstreamDrift/issues/10189)). The lead retains GS-03 delivery semantics ([#10192](https://github.com/D-sorganization/UpstreamDrift/issues/10192)) and GS-04 impact qualification ([#10193](https://github.com/D-sorganization/UpstreamDrift/issues/10193)). This runbook supplies acceptance criteria; it does not authorize treating future test filenames as implemented features.

## Environment, Version, and License Evidence

Before a live session, record:

1. UpstreamDrift commit, branch, governing issues, Python version, dependency lock/pin identities, selected physics provider and simulator adapter version.
2. Windows version, GPU/driver, display resolution and topology on both the model host and GSPro host, if different.
3. GSPro application release from its own version/about surface and the selected connection mode. Record a screenshot or operator observation with a timestamp.
4. Operator confirmation that the target machine has a valid subscription/license and can open the practice range. Never capture license keys, credentials, activation files or account secrets.
5. Endpoint host/port and how the operator verified the listener. Do not infer readiness merely from an installed executable or an open TCP port.
6. The vendor documentation revision and the approved protocol-characterization report used by this adapter. Record unit, handedness, spin, framing and response-correlation decisions explicitly.

Initial discovery on the user's machine found `C:/GSProV1/Core/GSP/GSPro.exe` and `GSPconnect.exe`. The PE file version `2018.2.8.11407744` is Unity metadata and is **not reliable evidence of the GSPro application release**. No GSPro process or port 921 listener was observed during that discovery. These are point-in-time observations, not current readiness checks. This planning task did not launch or change GSPro.

## No-License Fake-Server CI Suite

All default automated checks must run without GSPro, paid simulator accounts, GUI windows, GPU drivers or a network outside the test process. Tests use dependency injection for transport, clocks and presentation events. Loopback integration tests bind an ephemeral port; unit tests use in-memory doubles. Never contact the configured live endpoint from default CI.

The following are **proposed future test files, not existing runnable commands**:

| Proposed Test File                                              | Required Evidence                                                                                                                                                      |
| --------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `tests/unit/golf_simulator/test_shot_contract.py`               | Finite values, explicit frame/units, immutable shot identity, source provenance, valid spin representation and invalid-input rejection before I/O.                     |
| `tests/unit/golf_simulator/test_gspro_codec.py`                 | Independently derived golden wire fixtures; exact key/number encoding; unit and sign conversion; unsupported optional fields omitted honestly.                         |
| `tests/unit/golf_simulator/test_session_state.py`               | Operator arm/disarm, single in-flight shot, cancellation, bounded queue and timeout behavior, disconnect classification and no automatic replay of ambiguous delivery. |
| `tests/integration/golf_simulator/test_fake_gspro_peer.py`      | Partial reads/writes, coalesced messages, malformed/oversized input, orderly and abrupt closes, delayed/unsolicited responses, and protocol-defined correlation only.  |
| `tests/unit/golf_simulator/test_replay_coordinator.py`          | Shared trace/shot identity, one transmission at the selected impact event, seek/replay without resubmission, and renderer failure isolation.                           |
| `tests/integration/golf_simulator/test_provider_conformance.py` | Common adapter lifecycle, honest capability declarations, local fallback and explicit rejection of unsupported features.                                               |

The fake peer must implement the characterized framing and response semantics; it must not invent stronger acknowledgments or idempotency than GSPro provides. Do not assume newline framing, one `recv` per JSON object, response order, echoed shot identity, heartbeat support or a fixed heartbeat interval without evidence.

For each new behavior, preserve TDD evidence: failing test on the pre-change implementation, minimal change, passing test and the relevant regression suite. Tests asserting only that implementation output equals itself do not establish conversion accuracy.

### Engineering Contract Checks

- **DbC:** reject invalid data and invalid state transitions before side effects; validate inbound data at the boundary; ensure cleanup is idempotent and emitted shot identity/provenance remains stable. Validation must remain active in production, including optimized Python execution.
- **LoD:** UI calls the orchestration facade; orchestration calls the simulator adapter; the adapter owns protocol and transport. No GUI socket calls, private physics helper imports or chained access through engine internals.
- **DRY:** reuse public launch/impact types, canonical conversion authorities, existing rendering and provenance components. Keep reusable delivery state outside vendor codecs. Do not duplicate the inbound Tools-owned launch-monitor implementation.
- **TDD:** test negative paths, independent conversion fixtures and actual state effects. Optional engine or renderer tests must report meaningful skips without changing the no-license suite's meaning.

## Windows Host Topology

### Single Host

Run UpstreamDrift and licensed GSPro on the same Windows machine. Prefer a verified loopback endpoint. Show the companion model on a second display, beside GSPro in windowed mode, or in an operator-selected presentation layout. Verify each window's visibility before the shot; focus and display composition must not trigger a swing or submit a shot.

### Separate Model and Simulator Hosts

Run the model and companion renderer on the compute host and GSPro on the licensed Windows host. Keep the GSPro adapter on the licensed host and connect to the vendor's unauthenticated local socket there. The remote model uses an authenticated, encrypted UpstreamDrift control connection to that host. Do not expose the vendor socket directly to the LAN or treat it as the authenticated application bridge.

Record hostname/IP, actual listener binding, application bridge reachability/version, and scoped Windows Firewall rules. Verify unauthorized control requests are rejected before reaching the adapter, and verify the vendor port remains local. Do not expose the vendor endpoint publicly. Record clock offsets for cross-host visual timing; do not compare uncalibrated wall-clock timestamps as latency evidence.

## GSPro Range Setup and Operator Flow

1. The operator launches licensed GSPro, enters a practice range and selects the documented connection mode appropriate to the installed version. Record the exact UI choices; do not infer or invent menu names.
2. Verify the simulator is accepting the intended source and that no physical launch monitor or other bridge can submit competing shots. Record the source arrangement and restore it after testing.
3. Start the implemented adapter in a disarmed state. Verify it identifies the chosen endpoint and clearly distinguishes connected, operator-ready and armed states. A generic response or open socket must not silently arm delivery.
4. Load a named deterministic fixture with known launch values. Display source, units, shot identity and intended destination for operator review.
5. The operator confirms the range is ready and explicitly arms one shot. Run the swing/replay once; submit at the designated impact event only.
6. Observe flight on the simulator display. Record protocol response separately from visible ball flight and its completion. Disarm after the single shot. Additional shots require the implemented readiness policy and operator authorization; never invent a ready signal from elapsed time.
7. Save sanitized evidence and restore the original source/display configuration if the session changed it.

## Independent Unit and Sign Calibration

Do not use the connector's own output as the expected-value oracle. Agree the conversion table and acceptance tolerances before running tests, based on official documentation plus controlled observations of the actual GSPro release where documentation is incomplete.

Use manual synthetic shots first to separate protocol calibration from scientific model accuracy. Include straight shots, equal-magnitude left/right launch, backspin, each sidespin direction, zero spin and low-speed putts if that capability is supported. Record input values, hand calculation, encoded payload, reported GSPro values and observed direction. A visual direction observation verifies a sign, not exact magnitude.

UpstreamDrift flight inputs use m/s, launch angles in radians and spin magnitude in RPM. `SwingState` declares +x forward, +y left and +z up. The canonical inbound launch-monitor analytics layer instead uses angular speed in rad/s. Capture which public type is the input; never apply conversion twice or infer units from a similar field name.

For the GSPro destination, qualify speed-unit interpretation, horizontal-angle sign, spin-axis versus back/side-spin representation, handedness behavior and rounding. Lock the selected representation and golden fixtures to the characterized protocol profile. A nonzero spin vector must have a meaningful axis; zero-spin behavior must be explicit. Treat unsupported orientation information as a declared limitation, not a silent projection with scientific claims.

For engine-derived shots, additionally prove the extraction event is actual qualified contact and that angular velocity, loft, inertia and frame transformations are preserved as required. The existing MuJoCo reference source selects peak speed, and the existing simplified impact route does not preserve all `SwingState` fields. A demonstration using those paths must carry its limitation label until the governing qualification work passes.

## One-Shot and Replay Visual Receipts

Capture one evidence bundle per accepted shot containing:

- Source trace identifier/hash, selected impact time, provenance and model/control settings.
- Canonical launch input, adapter profile, sanitized exact outbound message and local transmission/response timeline.
- Companion view recording showing the model swing through the designated impact event.
- GSPro recording or screenshots showing the corresponding shot and flight, with enough context to match the test case and exclude another source.
- Operator observation of visible latency and synchronization, with a stated measurement method and previously agreed tolerance.

Pass only if one operator action causes one submission and one observed shot. Seek, pause, repeat the animation and reopen the evidence bundle: none may resend the physical simulator shot. Test a renderer failure independently; presentation must not cause a duplicate delivery or silently change the shot.

The two views may display different trajectories because GSPro owns its ball-flight and course physics. Label any UpstreamDrift trajectory as a separate model prediction; do not claim imported full-trajectory playback or numerical agreement without a separate validated capability.

## Ambiguous Disconnect and Recovery

Inject a disconnect at each meaningful transport boundary: before a send begins, during a partial send, after bytes are written, and before a qualifying response is received. A send error or missing response after transmission starts can leave delivery ambiguous. Transport write success alone is not application acceptance.

For ambiguous delivery, require the following behavior:

1. Preserve the original shot identity and evidence; enter an explicit unknown-delivery state and disarm.
2. Do not automatically retry the shot on timeout, reconnect, process restart, UI refresh or replay. A local journal prevents accidental resends but cannot establish remote exactly-once processing.
3. Let the operator inspect the GSPro display/history available in that release and choose a documented recovery outcome. Record the observation and decision.
4. Reconnect without sending the pending shot. If the operator chooses a new attempt, require explicit authorization and a separately tracked attempt according to the approved identity policy.

Test clear rejection and confirmed acceptance separately from ambiguity. If response correlation is unsupported or inconclusive, the UI must report that limit instead of assigning a response to a shot merely because it arrived nearby in time. An arbitrary 200 cannot establish acceptance, including with only one pending shot; require characterized attribution to that shot. A failed write after an attempt begins remains potentially delivered even if the client cannot measure how many bytes reached the peer.

## Support Matrix Gates

| Capability                            | Promotion Gate                                                                                                                                                     |
| ------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| No-License Local Adapter/Fake Peer    | Default CI contracts, codec fixtures and transport failure tests pass; no paid software dependency.                                                                |
| GSPro Local Shot Submission           | Exact release/profile recorded; units/signs/framing/responses characterized; operator-arm and one-shot range evidence pass.                                        |
| GSPro Remote Host                     | Local acceptance plus verified network/bridge topology, access controls, disconnect recovery and cross-host timing evidence.                                       |
| Model-Derived Shots                   | Contact extraction and input-field preservation tests pass for each named engine/model pair; scientific provenance retained.                                       |
| Companion Model Replay                | Trace-to-impact synchronization and no-resend replay tests pass on the named renderer/display arrangement.                                                         |
| Native GSPro Avatar                   | Supported vendor extension mechanism and permission established, then separate rig/animation integration acceptance. Remains unavailable until then.               |
| Course State or Autonomous Round Play | Documented supported state/control interfaces, truthful capability detection and dedicated closed-loop acceptance. Shot submission alone does not qualify.         |
| Another Simulator Product             | Lawful supported interface established; provider conformance suite and that product's own live/version/license acceptance pass. No automatic equivalence to GSPro. |

## Risk Register

| Risk                                                  | Required Control or Evidence                                                               |
| ----------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Sparse or changing vendor protocol                    | Versioned characterization and fixtures; unsupported profiles fail visibly.                |
| Duplicate shots after a connection failure            | Unknown-delivery state, disarm, no automatic replay and operator recovery evidence.        |
| Incorrect spin/launch direction or units              | Independent calibration cases and recorded conversion authority.                           |
| Peak-speed demonstration labeled as contact physics   | Provenance labels and per-engine qualification gate.                                       |
| Companion timing mistaken for embedded avatar support | Separate capability labels and recordings of the actual arrangement.                       |
| Remote endpoint exposure                              | Verified private topology and scoped access; authenticated custom bridge where applicable. |
| Expired license or incompatible GSPro release         | Operator license/readiness evidence and version-specific support matrix.                   |

## Pass/Fail Evidence Template

Copy this template into the implementation issue's acceptance record or approved artifact location. Do not place secrets in the record.

```text
Epic / Child Issue:
Operator / UTC Time:
Repository Commit / Adapter Version:
Model / Engine / Trace Hash / Provenance:
GSPro Application Release / Protocol Profile:
License Ready (Operator Confirmation, No Keys):
Windows / GPU / Displays / Host Topology:
Documentation and Characterization Evidence:
Test Case / Preconditions / Approved Tolerances:
Canonical Input / Independent Expected Values:
Payload and Sanitized Transport Evidence:
Shot Identity / Attempt Identity / Impact Time:
Observed Protocol Outcome:
Observed Simulator Outcome:
Companion and Simulator Visual Receipts:
Replay and Duplicate-Prevention Result:
Disconnect and Recovery Result:
Exact Automated Commands and Outcomes:
TDD Red/Green Evidence:
Result: PASS / FAIL / BLOCKED
Limitations / Unavailable Capabilities / Skips:
Follow-Up Issue and Next Action:
```

An overall acceptance report must enumerate the tested matrix rows and versions. A passed fake-server suite does not pass a licensed live session; a single live range shot does not pass unattended rounds, native avatars or all physics engines.

# ADR-0047: Trajectory Visualization — Shared Wire, Preserved Viewers

- Status: Accepted
- Date: 2026-08-30
- Decision Makers: repo owner (accepted 2026-08-30)
- Related Issues/PRs: launcher tiles `shot_tracer`, `rate_of_closure`; `ui/src/pages/BallFlight.tsx`; `src/api/routes/ball_flight.py`; Tools#4800 P8 (#4820/#4852)

## Context

Ball-flight trajectories are visualized in three places, powered by **two
independent flight-model families**:

| Surface                                                            | Runtime               | Physics behind it                                                                                                                                                           |
| ------------------------------------------------------------------ | --------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Shot Tracer tile (`_shot_tracer_gui.py`, pyqtgraph/OpenGL)         | Qt                    | UD `shared/python/physics/flight_models.py` — Waterloo-Penner, MacDonald-Hanzely, unified launch conditions, aero-coefficient registry                                      |
| BallFlight web page (`/ball-flight`) + `api/routes/ball_flight.py` | React                 | same UD family, via the API's model enumeration                                                                                                                             |
| Impact Explorer Flight Explorer + 3D playback                      | Qt + React (vendored) | Tools `swing_sim.flight` — aerodynamics, capability evaluator, ground transfer, Rust fast path — plus the P8 shared playback transport (golden-pinned sample→frame mapping) |

The two model families are _both legitimate research assets_: UD's implements
named published models for comparison; Tools' is the simulation-suite
authority wired into impact, ground, and putting. Unlike putting (P9) there
is **no interchange wire between them** — a Shot Tracer trajectory cannot be
replayed in the Impact Explorer's 3D playback, and an Impact Explorer flight
cannot be overlaid against the Waterloo-Penner curve in Shot Tracer, even
though comparing models is Shot Tracer's stated purpose.

Constraint set by the owner: **no functionality may be deleted or limited.**

## Decision

Integrate at the **trajectory-record level**, not the viewer level. All
three surfaces survive unchanged in identity; what they gain is the ability
to read each other's output.

1. **One trajectory interchange record.** Tools already ships the versioned,
   fail-closed `swing_sim.delivery_trajectory/1` posture and the P8 playback
   transport consumes retained samples. Define
   `ball_flight_trajectory/1` in the same idiom (time-stamped samples,
   declared frame, model provenance, byte-deterministic) as the _export
   format of every flight producer_:
   - UD `flight_models.FlightResult` gains `to_trajectory_record()` /
     adapter (runtime-free, like P9's).
   - Tools `swing_sim.flight` results likewise (largely already true —
     retained samples drive P8 playback today).
2. **Every viewer learns to read the record.** Shot Tracer's multi-model
   comparison accepts imported records alongside its native models (its
   purpose — comparison — finally spans both families); the BallFlight page
   likewise via the API; the Impact Explorer's 3D playback replays any
   record through the P8 transport (deterministic frames from samples is
   already its contract).
3. **No viewer is merged into another.** Shot Tracer stays the quick
   compare-models tool; BallFlight stays the web-native page; the Impact
   Explorer stays the full-suite context. Their tile descriptions state the
   relationship. Consolidating the _rendering components_ (e.g., one Qt 3D
   trajectory widget) is desirable engineering but explicitly deferred —
   it is refactoring behind stable surfaces and needs no decision here.
4. **Model families stay separate and named.** As in ADR-0045, provenance
   travels in the record; a Waterloo-Penner curve and a `swing_sim` capability
   flight can sit on the same axes because each is labeled, never because
   they were forced through one implementation.

## Alternatives Considered

1. **Retire Shot Tracer into the Impact Explorer.** Rejected: deletes the
   lightweight compare-published-models workflow and its OpenGL view;
   violates the constraint.
2. **Port UD's named models into `swing_sim.flight`.** Attractive later
   (one registry), but it is a physics-porting project with validation
   burden; the record-level integration delivers the user value (cross
   viewing/comparison) without it. Revisit after the record exists.
3. **Do nothing.** Rejected: three viewers that cannot share data is the
   navigation-level confusion the launcher review flagged; users reasonably
   expect a trajectory to be portable between tiles of the same app.

## Consequences

- Positive: cross-family comparison becomes possible for the first time;
  every producer gains replay in the best viewer (P8 playback); adapters are
  runtime-free and testable exactly like P9's.
- Negative: one more versioned wire to maintain; three import surfaces to
  wire up (each small).
- Follow-ups (own issues, sized after approval): H1 the record + UD adapter
  with cross-family gates (analytic: identical launch conditions produce
  same-order carry; provenance mandatory); H2 Shot Tracer import; H3
  BallFlight page import; H4 Impact Explorer playback import.

## Validation

- H1's gates mirror P9's posture: byte-deterministic round trip, refusal of
  unknown fields, documented (not reconciled) model differences.
- A record produced by each family must replay in all three viewers in CI
  (headless probe for Qt surfaces, vitest for web).
- No existing viewer test is deleted or weakened.

# Open-FP Force Plate Integration

## Decision and Delivery Status

**Epic:** [#13061](https://github.com/D-sorganization/UpstreamDrift/issues/13061); children in [BACKLOG.md](BACKLOG.md). In review: OF-01, OF-02, OF-05, OF-11 and OF-12 (PRs #13074, #13075, #13078, #13077, #13079).
**Status:** contract, reader, readings and the null-space screen are in
review; LAN client, UI, live bridge and hardware qualification remain.
**Reviewed:** 2026-10-17, against `open-flight/open-fp` commit
`53d682ce70b34bd2d85918b210ebdc81937e33f8` and UpstreamDrift commit
`aa0c176b`.

[open-fp](https://github.com/open-flight/open-fp) is an open-source pair of
golf force plates: one 305 x 305 mm aluminium plate per foot, four 50 kg
half-bridge load cells per plate, eight ADS1220 24-bit ADCs and an Arduino
UNO Q. The board runs a Python pipeline and serves a LAN web UI with live
weight shift, swing capture, replay and CSV export. It is the force-plate
companion to the OpenFlight launch monitor.

**Recommendation: integrate, as a vertical-only ground reaction data source
behind a file and network boundary.** open-fp is the first affordable,
documented source of measured per-foot vertical force and centre of pressure
(CoP) that UpstreamDrift users can own. UpstreamDrift already has a ground
reaction core (`src/shared/python/biomechanics/ground_reaction.py`), a
ground reaction plot tile and contact-wrench machinery that can use it. The
integration must, however, respect three hard limits stated below.

## What Open-FP Provides

| Output                                      | Units, Rate          | Notes                                                                  |
| ------------------------------------------- | -------------------- | ---------------------------------------------------------------------- |
| Per-corner force, 8 channels (L1-L4, R1-R4) | N, 200 Hz            | 1 kHz per channel on the MCU, 5-sample mean                            |
| Per-foot vertical force `fz_L`, `fz_R`      | N                    | Sum of the plate's four cells                                          |
| Per-foot and combined CoP                   | mm, stance frame     | NaN when the plate carries less than 20 N                              |
| Lead weight percentage                      | %                    | Lead foot chosen from handedness (RH/LH)                               |
| Force events                                | s from capture start | motion start, max trail, transition, max lead, finish, trigger, impact |
| Swing metrics                               | %, mm, ms, %BW       | Shift timing, CoP range and speed, peak Fz                             |

Interfaces: a swing library on disk (`<id>.json` metadata, events and
metrics plus `<id>.npz` arrays), a REST API (`/api/swings`,
`/api/swings/{id}`, `/api/swings/{id}.csv`, `/api/events/shot`, ...) and a
30 Hz Socket.IO `live` stream. Details and field-level mapping are in
[ARCHITECTURE.md](ARCHITECTURE.md).

## Hard Limits

1. **Licence: AGPL-3.0.** UpstreamDrift is MIT. Do not vendor, import, pip
   depend on, or copy code from open-fp. Integrate only through its
   documented file formats and network API, with readers written
   independently in UpstreamDrift. This is the same arm's-length boundary
   used for OpenCap (ADR-0053) and the AGPL `pymupdf` ledger row. It is an
   engineering policy, not legal advice; the owner should confirm it.
2. **Vertical force only.** No shear (Fx, Fy) and no free moment (Mz). The
   UpstreamDrift ground reaction contracts are 3-D; the unobserved
   components must be carried as NaN with an observability mask, never as
   zero. Moments about the plate origin can be derived from CoP and Fz.
   open-fp therefore does not satisfy the "bilateral 6-axis force plates"
   prerequisite recorded in `docs/development/planning/DV-12554.md`.
3. **Hardware is not yet qualified.** At the reviewed commit, four of eight
   cells read on the bench; production firmware, end-to-end calibration
   (M4: total force within 1 %, CoP within 10 mm) and swing capture on real
   plates (M6) are not done. Until a device passes both M4 and M6 on
   production firmware, as recorded in deferred-validation record
   [DV-13061](../../development/planning/DV-13061.md), imported data is **unqualified measurement** and may be shown
   and compared, but must not serve as acceptance evidence for any model.

## Integration Value, Ranked

| Rank | Use in UpstreamDrift                                                                | Needs                                                      |
| ---- | ----------------------------------------------------------------------------------- | ---------------------------------------------------------- |
| 1    | Import and plot measured vertical GRF, CoP and lead % in the ground reaction tile   | File reader, frame mapping                                 |
| 2    | Overlay measured against model-predicted per-foot Fz and CoP for a matched swing    | Time alignment, event mapping                              |
| 3    | Weight-shift section in swing reports and tour-baseline comparisons                 | Units in %BW, provenance labels                            |
| 4    | Pull swings from the board over the LAN                                             | Read-only HTTP client                                      |
| 5    | Live weight-shift stream into the realtime IPC layer                                | Socket.IO bridge, optional extra                           |
| 6    | Research: measured vertical wrench as a nominal target in contact-wrench allocation | Qualified hardware, sync error budget                      |
| 7    | Eliminate candidate swing solutions in the closed-loop null space (OF-11, OF-12)    | In review (PRs #13077, #13079); unqualified until DV-13061 |

## Reading Order

1. [Architecture and Contracts](ARCHITECTURE.md): licence boundary, data
   mapping, frames, timing and module placement.
2. [Delivery Backlog](BACKLOG.md): proposed child issues, dependencies,
   acceptance and upstream requests.

This directory is a proposal. It does not replace the engineering design
manual, and no row here is acceptance evidence.

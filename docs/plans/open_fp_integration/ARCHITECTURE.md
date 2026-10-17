# Open-FP Integration Architecture and Contracts

All open-fp facts below were read from `open-flight/open-fp` at commit
`53d682ce70b34bd2d85918b210ebdc81937e33f8` (`docs/PROJECT_INIT.md` sections
5 and 9, and the `app/python/open_fp/` modules). open-fp is a young project
whose formats carry no version number yet, so every reader must pin the
commit it was written against and fail closed on anything else.

## 1. Licence Boundary

| Allowed                                                                    | Not Allowed                                                    |
| -------------------------------------------------------------------------- | -------------------------------------------------------------- |
| Read `<id>.json`, `<id>.npz` and CSV exports produced by an open-fp device | `import open_fp`, or adding open-fp to any dependency extra    |
| Call the open-fp REST API and Socket.IO stream as a network client         | Copying open-fp source, tests or fixtures into this repo       |
| Write UpstreamDrift readers from the documented format                     | Running open-fp's `tools/sim/synth.py` inside UpstreamDrift CI |
| Generate UpstreamDrift's own synthetic fixtures in the documented format   | Shipping open-fp firmware or web assets with UpstreamDrift     |
| Contribute format improvements upstream (under their AGPL)                 | Treating an upstream format change as silently compatible      |

No `docs/legal/licenses.md` row is needed while there is no dependency. If a
Socket.IO client is added for the live bridge (section 7), that client
library gets its own ledger row; `python-socketio` is MIT.

## 2. Source Data Contract (Pinned)

### Swing Metadata `<id>.json`

| Key             | Type                         | Meaning                                                                                                   |
| --------------- | ---------------------------- | --------------------------------------------------------------------------------------------------------- |
| `id`            | str                          | `YYYYMMDD-HHMMSS-mmm[-k]`, **board local time**, at capture finalization                                  |
| `created`       | str                          | Naive local ISO time, seconds resolution, at finalization                                                 |
| `golfer`        | `"RH"` or `"LH"`             | Maps plate L/R to lead/trail                                                                              |
| `bw_n`          | float                        | Body weight in N (quiet-stance capture or median fallback)                                                |
| `t0_host`       | float                        | **Board monotonic clock** time of `t[0]`; not wall-clock                                                  |
| `events`        | dict[str, float or null]     | Seconds from `t[0]`: `motion_start`, `max_trail`, `transition`, `max_lead`, `finish`, `impact`, `trigger` |
| `metrics`       | dict                         | Derived metrics (see open-fp `swing.py::analyze`)                                                         |
| `impact_source` | `"shot"`, `"manual"` or null | Whether impact came from a launch-monitor event                                                           |
| `path`          | list                         | 60-point decimated combined CoP for thumbnails; ignore                                                    |

### Swing Arrays `<id>.npz`

| Key                       | Shape  | Units                                      |
| ------------------------- | ------ | ------------------------------------------ |
| `t`                       | (n,)   | s from capture start                       |
| `forces`                  | (n, 8) | N, order L1 L2 L3 L4 R1 R2 R3 R4           |
| `fz_L`, `fz_R`            | (n,)   | N                                          |
| `cop_L`, `cop_R`, `cop_C` | (n, 2) | mm, stance frame, NaN below 20 N per plate |
| `lead_pct`                | (n,)   | %                                          |

The CSV export (`GET /api/swings/{id}.csv`) has the same arrays as columns
(`t_s, lead_pct, fz_L_n, fz_R_n, cop_{L,R,C}_{x,y}_mm, L1_n ... R4_n`), with
empty cells for NaN, but **drops events, metrics, body weight and
handedness**. Prefer JSON plus NPZ; accept CSV only with user-supplied
handedness and body weight.

### Gaps in the Source Contract

These are not defects in open-fp's own use, but they block rigorous use in
UpstreamDrift. Each is an upstream request in [BACKLOG.md](BACKLOG.md)
(OF-00):

1. **No schema version.** Readers must detect structure and pin a commit.
2. **No absolute time anchor.** `t0_host` is monotonic; `created` is local,
   naive and second-resolution, and is the finalization time rather than
   `t[0]`. A stored swing cannot be aligned to video, mocap or a launch
   monitor to better than seconds without an in-swing anchor.
3. **No geometry snapshot.** Plate spacing and cell positions live in the
   board's config, not in the swing. CoP is already in the stance frame, so
   this matters only for the raw `forces` channels and for placing the
   plates in a capture volume.
4. **No calibration provenance.** The swing does not record which
   calibration (gains, date, residuals) produced its forces.

## 3. Target Contract in UpstreamDrift

Add one small, immutable domain record rather than stretching the 6-axis
types. Proposed shape (names are proposals for OF-01):

```python
@dataclass(frozen=True)
class VerticalPlateTrial:
    times_s: NDArray            # (T,), strictly increasing
    fz_n: Mapping[str, NDArray]  # "lead", "trail" -> (T,)
    cop_m: Mapping[str, NDArray]  # "lead", "trail", "net" -> (T, 2), NaN allowed
    corner_forces_n: NDArray | None  # (T, 8) raw channels, diagnostic only
    body_weight_n: float
    handedness: Literal["RH", "LH"]
    force_events_s: Mapping[str, float | None]  # open-fp names, not swing phases
    provenance: PlateProvenance  # device, source commit, file digests, calibration id or None
    time_anchor: TimeAnchor | None  # absolute anchor and its uncertainty, when known
```

Postconditions checked in `__post_init__`: equal lengths, finite times,
non-negative force within a declared capacity (8 x 50 kg), CoP NaN wherever
that plate's Fz is below the source threshold, handedness valid.

Conversion to the existing core, `to_ground_reaction_series(trial, frame)`,
fills `GroundReactionSeries` from `biomechanics/ground_reaction.py`:

- `force_n[label] = (NaN, NaN, Fz)`: shear is **unobserved**, not zero.
- `cop_m[label] = (x, y, z_surface)` after the frame mapping in section 4.
- `free_moment_nm[label]` is NaN (unobserved).
- `in_contact` follows `Fz > COP_MIN_FZ_N`, but CoP is never recomputed below
  open-fp's own 20 N threshold.
- An explicit observability record (`{"fx": False, "fy": False, "fz": True,
"cop": True, "free_moment": False}`) travels with the series so plots and
  comparisons show only observed components.

Do not route open-fp data through `force_plate_stitching.CombinedForcePlateProcessor`:
it requires Fx..Mz columns and defaults CoP to the origin below threshold,
which conflicts with the NaN rule.

## 4. Frames, Units and Signs

open-fp stance frame: millimetres, origin midway between plate centres, +x
from plate L toward plate R, +y toward the ball, golfer facing +y, implicitly
z up (right-handed). Forces are reported positive under load.

| Quantity    | open-fp                              | UpstreamDrift canonical                      | Conversion                                                                                                               |
| ----------- | ------------------------------------ | -------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------ |
| Length      | mm                                   | m                                            | x 0.001                                                                                                                  |
| Vertical    | Fz > 0 under load                    | ground-on-foot force, +z up                  | same sign                                                                                                                |
| CoP height  | not reported                         | z of contact plane                           | declared plate surface height (about 22 mm above floor) or 0 for a plate-local frame                                     |
| Lead/trail  | RH: L = lead; LH: R = lead           | per-foot labels                              | from `golfer`                                                                                                            |
| Target axis | RH target is toward -x; LH toward +x | golfer-local frame used by the matched model | **decision for OF-03**: confirm the model's target axis and apply a handedness-dependent rotation; never mirror silently |

Frame mapping lives in one function with golden tests for both handednesses
and a point load at each corner. A stance-frame-to-world placement (where
the plates sit in a mocap volume) is a separate, optional calibration input.

## 5. Events

open-fp labels **force events**, deliberately not swing positions. Keep them
in their own namespace (`force.max_trail`, `force.transition`, ...). Do not
map `max_trail` to `top`, or `transition` to any club position.
`build_ground_reaction_plot_series(..., events=...)` already accepts an
arbitrary mapping; the plot legend must show the namespace. Only `impact`
(when `impact_source == "shot"` or `"manual"`) may be offered as a candidate
alignment anchor with model `SwingEvents.impact`.

## 6. Time Alignment

No clock-alignment module exists in UpstreamDrift. Add a small
`align_plate_trial(trial, reference)` service that returns an offset **and**
its uncertainty, choosing the best available anchor:

| Anchor                                                                                     | Expected Error                                                    | Availability                 |
| ------------------------------------------------------------------------------------------ | ----------------------------------------------------------------- | ---------------------------- |
| Absolute UTC anchor in the swing file                                                      | NTP-limited, about 5-20 ms over Wi-Fi                             | after upstream OF-00 lands   |
| Launch-monitor impact (`impact_source == "shot"`)                                          | 5-20 ms (open-fp's own estimate)                                  | when OpenFlight is connected |
| Cross-correlation of measured and model vertical GRF (`signal_toolkit.compute_time_shift`) | depends on model quality; circular if used to validate that model | always, with care            |
| Manual operator offset                                                                     | operator-declared                                                 | always                       |

Rules: record which anchor was used; never use cross-correlation against a
model and then report agreement with that same model as validation; resample
open-fp's 200 Hz (host-mapped, not guaranteed uniform) times onto the
comparison grid with an explicit method; refuse comparisons of sub-10 ms
timing metrics when the alignment uncertainty exceeds them.

## 7. Ingestion Paths

1. **File import (first).** A reader for a folder of `<id>.json` and
   `<id>.npz` pairs. Load NPZ with `numpy.load(path, allow_pickle=False)`,
   check keys and dtypes, cap file size, and validate the id pattern before
   building any path. Expose it through the existing
   `motion_pipeline/sources` style (`supports`, `metadata`, `load`) but as a
   **plate-trial** source: the CIR (`CanonicalObservationFrame`) requires
   markers or keypoints, so open-fp data must not be forced into it.
2. **LAN pull.** A read-only HTTP client for `GET /api/swings` and
   `GET /api/swings/{id}` with timeouts, response size limits and full
   validation. open-fp's API has no authentication, so treat responses as
   untrusted input and never call its mutating endpoints (`DELETE`,
   `PUT /api/config`, calibration) from UpstreamDrift.
3. **Live bridge (optional extra).** Subscribe to the Socket.IO `live`
   messages (30 Hz batches of 200 Hz samples) and republish on a realtime
   channel such as `force_plate/open_fp/live`, registered with
   `realtime.api.register_channel` and hinted `"high"` with
   `channels.register_channel_hint`. Keep the client behind a named extra
   (for example `force-plate-live`) and lazy imports.
4. **Shot event push (later, opt-in).** UpstreamDrift may `POST
/api/events/shot` only if the operator enables it and the impact time
   has a declared uncertainty; open-fp already accepts this from OpenFlight.

## 8. Module Placement

Following `AGENTS.md` section C (data used by two or more engines goes in
`src/shared/python/<topic>/`):

```
src/shared/python/force_plates/
    __init__.py            public facade
    vertical_trial.py      VerticalPlateTrial, PlateProvenance, TimeAnchor
    conversion.py          to_ground_reaction_series, frame mapping
    alignment.py           align_plate_trial
    open_fp/
        reader.py          file reader (json + npz, csv)
        client.py          read-only HTTP client
        live.py            Socket.IO bridge (optional extra)
tests/unit/force_plates/   mirrors the above; synthetic fixtures written by our own generator
```

Open question for the owner: the generic vertical-plate record could later
move to Tools (`sidekick.lab.bio`, next to the C3D force-plate code) if a
second repository needs it. Start in UpstreamDrift, which is the only
consumer, and move it only when a second consumer appears.

## 9. Plates as Constraints on Swing-Matching Solutions

In review in OF-11 (#13067, PR #13077) and OF-12 (#13070, PR #13079); not yet
on `main`. Those PRs add the calculation-level reference to
`docs/research/loop_nullspace/loop_nullspace.tex` (`sec:lowerbody`): the
sensing-plane rows and their exact rank goldens, the family-wise screen,
clock-offset profiling, assumptions and reproduction commands. Until they
merge, this section is the design statement, not a reference.

- **Sensing-plane lever.** Vertical cells balance moments about the plane
  where the plate's horizontal reactions act, a depth `h` below the foot. A
  reported CoP therefore mixes vertical load with shear times `h`; the
  observation rows model this exactly with a declared depth and never assign
  shear a value. With `h = 0` shear stays entirely free. On the spatial
  fixture, `h = 2 cm` reduces the shear rank from 1 to 0 (point feet) and
  from 5 to 3 (two-point feet). Section 3's `(NaN, NaN, Fz)` display view is
  unchanged: shear is still never _reported_.
- **Two roles.** Most plate rows are already fixed by the rigid-body balance,
  so they **falsify** candidates (motion plus model) rather than narrow the
  family. The rest **restrict** internal contact-load directions. Neither
  role selects a unique solution.
- **Family-wise elimination.** Bonferroni coverage over every tested row,
  the clock offset profiled as one common-mode unknown, and the joint
  feasible-set verdict deciding. A correct candidate is eliminated with
  probability at most `alpha` under the stated conditions (true loads in the
  family, declared Gaussian sigma, bias budgeted at the same `k`, true offset
  on the grid). A profiled shift with no data makes the screen inconclusive
  rather than eliminating, and a declared sensing-depth sigma requires the
  friction-cone hypothesis, because its bound holds only inside the cones.

## 10. Engineering Contract Checks

- **DbC:** validate at the file and network boundary; reject unknown
  structure; keep validation active under `python -O` (raise, do not
  `assert`).
- **LoD:** UI calls the force-plate facade; the facade calls the reader or
  client; nothing outside `open_fp/` knows open-fp key names.
- **DRY:** reuse `GroundReactionSeries`, `build_ground_reaction_plot_series`,
  `realtime` and `compute_time_shift`; do not add a fourth GRF type.
- **TDD:** independent fixtures (hand-computed point loads, both
  handednesses, NaN CoP, missing impact, truncated and malicious files)
  written before the reader.

# feat(starting-pose-matcher): source-toggle UI — choose Club, Club+Ball, Body, or any combination

## Why

The matcher today loads ONE source (the Wiffle xlsx). The user wants to drive the same view with **any combination of three independent target sources**:

- **Club** (`ClubTarget`) — loaded from xlsx, c3d, or .mat.
- **Club + ball** (`ClubBallTarget`) — adds a ball-impact boundary condition.
- **Body** (`BodyTarget`) — full-body anatomical markers from C3D.

The toggle must be **multi-select**: the user can show only the club, only the body, or both at once. The cost-function consumer downstream is responsible for using whichever target subset is active.

## What to build

A new "Data sources" panel in `src/tools/starting_pose_matcher/gui.py`:

### UI

```
┌─ Data sources ───────────────────────────────────────┐
│ ☑ Club            [ Browse… ] club_swing_data.mat   │
│   ◯ Club only    ◉ Club + ball                       │
│                                                       │
│ ☑ Body markers    [ Browse… ] mocap_session.c3d     │
│   default-marker-set ▾                                │
│                                                       │
│ Time alignment:  ◉ Impact-aligned   ◯ Address-aligned│
│ Sample rate:     [1000  ] Hz   Duration: [0.300] s   │
└──────────────────────────────────────────────────────┘
```

- File pickers route through `load_club_target` (auto-dispatch xlsx / c3d / mat) and `load_body_target` (auto-dispatch c3d).
- The **Club + ball** radio toggles between `ClubTarget` and `ClubBallTarget` for the same underlying file (the latter built by `extract_ball_impact_from_clubtarget` from the club issue).
- `AlignOptions(sample_rate_hz, simulation_time_s, time_alignment)` is shared across both sources, ensuring shared timegrid (when both are loaded, body loader receives the club's `impact_source` so they share the clock).
- The **default-marker-set** combo lists named subsets (e.g. "Anatomical 28", "Lower body only", "Upper body only", "All markers") wired to `BodyTarget.marker_names` filtering.
- Validation: when both sources are loaded, their `time` arrays MUST match exactly (use `numpy.array_equal`); on mismatch, surface a `QMessageBox.warning`.

### Cost / matching consumer

Refactor the existing `core.py` "active target" handling to support **multiple targets**: a `MultiSourceTarget` dataclass:

```python
@dataclass(frozen=True)
class MultiSourceTarget:
    club: ClubTarget | ClubBallTarget | None
    body: BodyTarget | None

    def shared_time(self) -> np.ndarray: ...
    def has_club(self) -> bool: ...
    def has_body(self) -> bool: ...
```

Cost-function code currently using `ClubTarget` directly accepts `MultiSourceTarget` and reads only the slots it needs. (The actual cost-function refactor is small and should land in this PR; the cost terms stay the same — they just dispatch on `t.has_*()`.)

### Generic naming

UI strings: "Club", "Body markers", "Ball impact", "Anatomical 28". No source names. The file-picker dialog's filter string is also generic: `"Mocap club data (*.xlsx *.xlsm *.mat *.c3d)"`, `"Mocap body data (*.c3d)"`.

## Acceptance criteria

- [ ] `MultiSourceTarget` dataclass added in `src/shared/python/motion_matching/multi_source_target.py`.
- [ ] Source-toggle panel UI added to `gui.py`.
- [ ] File pickers correctly route to `load_club_target` and `load_body_target`.
- [ ] Toggling Club ↔ Club+Ball without re-loading produces a `ClubBallTarget` from the cached `ClubTarget`.
- [ ] Loading body and club from the same C3D shares the timegrid by construction (impact_source plumbed through).
- [ ] Shared `AlignOptions` controls both loaders.
- [ ] Mismatched timegrids yield a clear warning, not a crash.
- [ ] Existing matcher save/load session JSON now includes `data_sources` block; old sessions still load.
- [ ] Headless smoke test loads a `BodyTarget` + `ClubTarget` from the same C3D and verifies `MultiSourceTarget.shared_time()` returns equal arrays.

## Files touched

- New: `src/shared/python/motion_matching/multi_source_target.py`
- Edit: `src/tools/starting_pose_matcher/gui.py` (add panel)
- Edit: `src/tools/starting_pose_matcher/core.py` (route MultiSourceTarget into cost helpers)
- Edit: `src/tools/starting_pose_matcher/session_schema.py` (schema bump)
- Edit: `src/shared/python/motion_matching/__init__.py`
- New: `tests/unit/tools/starting_pose_matcher/test_source_toggle.py`
- New: `tests/unit/motion_matching/test_multi_source_target.py`

## Sequencing

Depends on: `BodyTarget`, body C3D loader, `.mat` loader, `ClubBallTarget` issues. Should land before the animated preview lands — animated preview reads `MultiSourceTarget` to decide which layers to draw.

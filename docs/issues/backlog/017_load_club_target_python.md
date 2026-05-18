# Issue: Implement Python ClubTarget Dataclass and Excel/C3D Loaders

## Summary

Implement the Python mirror of the club-target loader: a `ClubTarget` frozen
dataclass plus `load_club_target_excel`, `load_club_target_c3d`, and
`synthesize_target_from_coefficients` functions. Reuse `mocap_data_loader.py`
and `c3d_reader.py` rather than reimplementing parsers.

## Motivation

See `motion_matching/shared/CLUB_IK_SPEC.md` §"Function signatures" Python
mirror block. Options 2, 3, and 4 all consume `ClubTarget` from Python.
Mirroring the MATLAB API keeps the contracts in lockstep and makes
cross-implementation comparison sane.

## Dependencies

None (the synthesizer wraps a callable, which can be a Python-side stub for
testing or the #036/#037 Simscape adapter once available — guard with an
optional import).

## File targets

- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\club_target.py` (`ClubTarget`, `SourceProvenance`, `AlignOptions`)
- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\loaders.py` (`load_club_target_excel`, `load_club_target_c3d`, `synthesize_target_from_coefficients`)
- `C:\Users\diete\Repositories\UpstreamDrift\src\shared\python\motion_matching\time_alignment.py` (impact detection, resampling, SLERP)
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\test_club_target.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\test_loaders.py`
- `C:\Users\diete\Repositories\UpstreamDrift\tests\motion_matching\test_time_alignment.py`

## Public API

Verbatim from `CLUB_IK_SPEC.md`:

```python
from dataclasses import dataclass
from pathlib import Path
import numpy as np

@dataclass(frozen=True)
class SourceProvenance:
    filename: str
    format: str             # "excel" | "c3d" | "synthetic"
    subject_id: str
    trial_id: str
    sha256: str

@dataclass(frozen=True)
class ClubTarget:
    time: np.ndarray            # (N,) float64
    butt: np.ndarray            # (N,3) float64
    clubhead: np.ndarray        # (N,3) float64
    club_quat: np.ndarray       # (N,4) float64, [w,x,y,z]
    impact_idx: int
    source: SourceProvenance


@dataclass(frozen=True)
class AlignOptions:
    sample_rate_hz: float = 1000.0
    simulation_time_s: float = 0.3
    time_alignment: str = "impact"   # "impact" | "address" | "none"
    impact_target_t_s: float = 0.25  # where the measured impact lands on the sim grid


def load_club_target_excel(path: Path, sheet: str, opts: AlignOptions) -> ClubTarget: ...
def load_club_target_c3d(path: Path, opts: AlignOptions) -> ClubTarget: ...
def synthesize_target_from_coefficients(theta: np.ndarray, opts: AlignOptions) -> ClubTarget: ...
```

## Required tests (TDD)

- `test_excel_loader_returns_canonical_clubtarget_for_TW_ProV1`
- `test_excel_loader_converts_inches_to_metres`
- `test_excel_loader_normalizes_quaternion_sign_w_nonnegative`
- `test_excel_loader_resamples_to_1000_hz`
- `test_excel_loader_detects_impact_at_max_clubhead_speed`
- `test_excel_loader_reuses_mocap_data_loader_not_a_re_implementation`
- `test_c3d_loader_parses_known_file_or_xfail_with_documented_reason`
- `test_c3d_loader_reuses_c3d_reader_not_a_re_implementation`
- `test_synthesize_target_round_trips_known_theta_through_callable_sim`
- `test_validation_rejects_non_unit_quaternion_with_clear_error`
- `test_validation_rejects_non_monotonic_time`
- `test_validation_rejects_butt_clubhead_distance_outside_plausible_shaft_range`
- `test_sha256_in_provenance_matches_file_contents`
- `test_clubtarget_is_frozen_attempting_to_set_attribute_raises`

## DbC contract

Use `@precondition` and `@postcondition` from
`src.shared.python.core.contracts`:

Preconditions:

- `path.exists()`.
- For Excel: `sheet in {"TW_wiffle", "TW_ProV1", "GW_wiffle", "GW_ProV11"}`.
- `opts.sample_rate_hz > 0`.

Postconditions (each loader must satisfy `CLUB_IK_SPEC.md` §"Validation rules"):

- `time` strictly increasing, starts at 0.
- All trajectory arrays share `time`'s row count.
- No NaN/Inf in position arrays; magnitudes plausible.
- Quaternions unit-norm to within `1e-6`.
- `1 <= impact_idx <= N`.
- `source.sha256` matches a freshly computed hash of the source file.

## Acceptance Criteria

- [ ] `ClubTarget` frozen dataclass implemented.
- [ ] Three loader functions implemented; each delegates to the existing parser
      (`mocap_data_loader.py` or `c3d_reader.py`) rather than re-parsing.
- [ ] All listed tests pass via `python3 -m pytest tests/motion_matching/`.
- [ ] DbC decorators applied; postconditions verified by tests.
- [ ] `ruff check` and `ruff format --check` clean.
- [ ] No file exceeds 1200 lines.
- [ ] No `print()`; use `get_logger`.
- [ ] No TODO/FIXME without a tracked issue link.

## Labels

`motion-matching`, `shared`, `python`, `tdd`, `dbc`, `infra`

## Effort estimate

M (1-3 days). Reuses existing parsers; the bulk of the work is the validation
layer and time-alignment helpers.

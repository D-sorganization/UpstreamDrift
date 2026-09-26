# Implementation Handoff — Versioned Pre-Impact Bundle (IA-U2)

## Identity

- Repository: D-sorganization/UpstreamDrift
- Working directory: `C:/Users/diete/Repositories/UpstreamDrift-worktrees/strong-9703`
- Branch: `feat/9703-pre-impact-bundle`
- Baseline commit: `ff9fbee62`
- Implementation commit: `SELF`
- Pull request: draft, opened from this branch (number recorded in DL-#9703 on the next commit)
- Governing issue: #9703 (IA-U2, parent #9700)
- Development log entry: DL-#9703

## Objective and Status

- Objective: slices 1–3 of #9703 — decide placement, implement a versioned
  immutable `PreImpactBundle` with fail-closed contracts, and test it first.
- Status: implementation complete, draft PR under review. Engine adapters and
  installed-wheel fixtures are out of scope and remain open under #9703.

## Files and Decisions

- `src/shared/python/physics/pre_impact_bundle.py`: bundle records, wire
  mapping (`to_dict`/`from_dict`/JSON, version 1 only), `project_onto_basis`,
  `grip_pose_from_delivery_sample`.
- `src/shared/python/physics/_pre_impact_contracts.py`: classified
  `PreImpactBundleError`/`AbsentFieldError`, `FieldOrigin`, `Quantity`,
  explicit-raise validators (survive `python -O`).
- `src/shared/python/physics/_pre_impact_frames.py`: `Pose` and
  power-consistent twist/wrench transforms reusing `spatial_algebra.transforms`.
- `tests/shared_contracts/test_pre_impact_bundle.py`: 48 tests.
- `docs/development/impact_acoustics_program.md`: "Pre-Impact Bundle Version 1
  — Placement" decision.
- Decision: consumer-side record in UpstreamDrift composing Tools conventions;
  no Tools import in the module. Bundle, modal-state record, origin enum and
  hand-wrench record should be upstreamed to Tools, as should public rotation
  and inertia validators.

## Validation

- `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg python -m pytest tests/shared_contracts/test_pre_impact_bundle.py -q -p no:cacheprovider` — 48 passed.
- `python -m pytest tests/shared_contracts -q -p no:cacheprovider` — 85 collected: 84 passed, 1 skipped, 0 failed.
- `ruff check` / `ruff format --check` on the four new files — pass.
- `mypy` on the three new modules — no issues.
- `shared_scripts/development_log.py` — no DL-#9703 findings; pre-existing
  WIP/size-ceiling findings on main are unchanged.

## Blockers and Risks

- Tools has no public modal energy/projection API; the bundle declares its own
  quadratic form. Revisit when Tools exposes one.
- Quaternion unit tolerance (1e-10) is stricter than the Tools delivery wire
  (1e-6); such samples are refused, not normalized.

## Next Steps

1. Review the draft PR; record its number in DL-#9703.
2. Implement the first engine adapter producing a `PreImpactBundle`.
3. Add installed-wheel fixtures at the reviewed Tools pin.
4. File the Tools upstreaming issue for the bundle wire and public validators.

## Change Log

- `SELF` — Declare `FloatArray` as an explicit `TypeAlias` so the pre-push mypy hook (no numpy in its env) accepts it (#9703).
- `6c80d5665` — Versioned PreImpactBundle v1 with fail-closed contracts (#9703).

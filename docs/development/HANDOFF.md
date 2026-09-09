# Attributed Club Catalog Handoff

## Identity and Scope

- Repository: D-sorganization/UpstreamDrift.
- Working directory: `C:/Users/diete/Repositories/Worktrees/UpstreamDrift-player-clubs`.
- Branch: `feat/9903-player-club-catalog`; implementation commit: SELF.
- Base: `18c8f922e87c92f6f518da05c0819f71ce3193ba` (merged capture-profile PR #9910).
- Issues: #9903 and #9904; epic #9902; entries DL-#9903 and DL-#9904.
- Pull request: not created; contract da340df45 published through normal hooks.
- Session: `capture-product-01a08427-club-catalog`; central lease and presence active.

Extend the existing club-data authority so unknown measurements, source attribution,
custom builds and conflicting claims survive exchange and cannot silently become
model inputs. The larger capture-product goal remains active: everyday calibration
#9897, club sources/bag #9902, guided wizard #9906 and fleet rollout all remain open.
Gasification mapping is planned for future cheaper agents, per user direction.

## Implementation and Compatibility

- `club_data/catalog.py`: immutable optional attributed claims, component identity,
  source/license metadata, deterministic build IDs and content revisions; explicit
  SI consumption rejects unverified/suggested values, ambiguous MOI and conflicts.
- `catalog_io.py`: bounded versioned JSON/CSV, validates all records before returning,
  no automatic overwrite. CSV has flat identity/property columns and JSON source cells.
- `catalog_legacy.py`: old default-filled ClubSpecification values stay unverified;
  do not infer which values were measured. The old loader behavior is unchanged.
- Public facade retains legacy names lazily; catalog imports avoid PyQt/pandas/openpyxl.
- Unit factors reuse the pinned Tools `sidekick.utils.unit_constants` authority.
- Guide and generated capability map describe the data contract only. The bag/capture
  UI remains #9905 and the qualified public source catalog/update process is implemented under #9904 and awaits publication qualification.
- No scientific solver or calculation inventory approval is claimed. Existing
  `blocked-inventory-required` manual release state and UP-D0/UP-D1 remain authoritative.

## Validation

- TDD: first test collection failed because `club_data.catalog` did not exist.
- `python3 -m pytest tests/unit/test_club_catalog.py tests/unit/test_club_data_loader.py -q --no-cov -o addopts=''`: 63 passed, 8 existing import deprecation warnings.
- `python3 -m mypy src/shared/python/club_data/catalog.py src/shared/python/club_data/catalog_io.py src/shared/python/club_data/catalog_legacy.py --follow-imports=silent --ignore-missing-imports`: passes after exchange typing corrections.
- Scoped Ruff lint/format, architecture, map and document budgets pass. Whole-source
  LoD reports three unchanged main capture chains fixed by pending PR #9917; no club
  source violations. Normal commit/pre-push hooks remain to run. No PR has been published for this contract.

## Concurrent Work and Risks

Capture UX PR #9917 is owned separately in `UpstreamDrift-capture-setup`. Preserve
the currently launched app in `UpstreamDrift-ubuntu-ci`, PID50860. Reference agent
#9914 owns headless C3D fitting; impact agent #9912 owns provider pinning. Shared
SPEC/development-log conflicts must preserve each issue's entry.

This isolated worktree contains only task-owned changes. Do not alter other agents'
branches or the vendored Tools checkout. Capture UX #9917 merged to remote main at8fce9f238ce89876dd363fb41b4ba1169a87d1b6 and is integrated by normal merge; source files were conflict-free. Its445 capture/parity tests passed; generated maps are regenerated after combining source registries. Latest visible test launch is Python child PID61500 (venv launcher PID30900) in UpstreamDrift-capture-setup (old PID50860 exited).

The shared reference solver is Tools PR
#5140; numerical repair is #5136. Their CI is still pending and private downstream
Gasification checkout remains an external credential issue. Fleet adoption is39/41,
with replacement owners now confirmed: context/Obsidian task will carry the canonical policy in Tools#5138 and Gasification#4944; audit adoption after merge.

## Next Steps

1. Complete staged validation and normal hooks; publish #9903 contract PR.
2. Integrate qualified public source entries/update procedure (#9904).
3. Add bag editing and capture-bound source snapshots (#9905), then verify visible
   workflow evidence before closing the contract issue and epic.
4. Continue #9917 protected merge, shared calibration consumer UI, wizard and fleet rollout.

## Change Log

- SELF: implement and qualify optional attributed club contracts and lossless exchange;
  update DL-#9903, SPEC, public facade and generated architecture references.

- SELF: add three offline manufacturer builds with per-field source/units/status, read-only deterministic review command and separate player overrides. 69 catalog/source/legacy tests and two-module mypy pass; source review uses Titleist T150 2023 and PING G440 MAX HL official tables. Read-only CLI regression also passes. Update DL-#9904 and preserve earlier entries.

- SELF: integrate capture UX501092b27; preserve both issue-keyed logs and all source. Regenerate conflicting map outputs from combined registries.

- SELF: integrate remote main8fce9f238 (tree-identical to501092b27); mark DL-#9913 shipped. Full LoD now passes3013 files with no baseline growth. Preserve club continuation and regenerated maps.

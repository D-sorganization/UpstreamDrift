# Agent Handoff — Proximal-Distal Program

Last updated: 2026-08-24

## Canonical Authority

- Epic #8557 is the single completion authority. UpstreamDrift owns scientific
  sources, models, evidence registers, claims, and release artifacts.
  AffineDrift publishes an immutable revision-pinned projection. Tools owns
  reusable consumers; do not copy its implementations into UpstreamDrift or
  edit `vendor/ud-tools`.
- Preserve the distinction between mechanics, model-conditional results,
  empirical context, provisional interpretation, and unavailable human data.
  Synthetic traces never substitute for participant validation.
- UpstreamDrift requires human review. Never force-push, admin-merge, bypass
  checks, add quarantine debt, or change shared runners to force progress.

## Active #9026 Evidence Bridge

- Worktree:
  `C:\Users\diete\Repositories\UpstreamDrift-worktrees\9026-biomechanics-evidence-bridge`.
  Branch: `research/9026-biomechanics-evidence-bridge`.
- Machine authorities are
  `data/biomechanics_source_register.json` and
  `data/biomechanics_evidence_bridge.json`. The source register covers 16
  independently authored works and 16 anatomical, apparatus, population,
  equipment, and task domains. The bridge covers seven modalities, nine
  mechanisms, 41 linked claims, and nine transport dimensions.
- Bilateral allocation remains structurally unidentified without independent
  bilateral six-axis measurements. Human validation remains externally
  blocked. Five of seven modalities retain explicit source/data gaps.
- The paper and reviewer projections are generated and freshness checked.
  Dense paper tables were replaced with wrapped records; reviewer Markdown
  retains the full tables. PDF pages 166--187 were rasterized and visually
  inspected with no overlap or clipping.
- Claim inventory reviews 1,137/1,137 candidates through 304 claims: 284
  supported, 5 inconclusive, and 15 untested. `PD-CLAIM-305` binds the bridge
  chapter to local digest authorities. It deliberately does not treat all 16
  papers as direct support for the umbrella bridge claim.
- The evidence manifest validates 2,139 references, 328 local artifacts, and
  78 externally adjudicated URLs. The release validates 600 artifacts.
- The optimized paper is 245 pages, 1,897,607 bytes, SHA-256
  `16e5e16b1c5d539135e48c8211ad7080491f2721b59559319729690fc4ffe4ac`,
  with 197 URI links, 248 outline entries, fast-web access, and extractable
  text on every page. Computational qualification passes. Archival
  qualification remains false because the PDF is untagged and retains Type 3
  and two unembedded font resources.
- A bibliography defect was corrected: `deRugy2018` now identifies Borzelli et
  al., DOI `10.1371/journal.pone.0205911`, rather than unrelated DOI
  `10.1371/journal.pone.0205538`. Higdon et al. fatigue metadata was added.
- Local focused gates pass: 20 bridge tests, 55 final claim/release/PDF tests,
  Ruff, title capitalization, file-size budget, all-page rendering, and exact
  release qualification. Finish rebase, rerun changed gates, push, open the
  full PR, and shepherd protected CI plus human review.

## ControlTower #8800 Recovery Boundary

- Tailscale and noninteractive SSH from Deskcomputer are verified. A separate
  Codex session on ControlTower is unnecessary.
- Campaign `UpstreamDrift-8800-1bd4d57da` is stopped. No campaign Python
  process is running. The status file saying `running` is stale.
- Windows checkpoints are preserved: shaft 48/48 and ground 45/48. The frozen
  launch authority binds source
  `1bd4d57da7bd257b76b42b3cc19524b283b5f748`, image
  `sha256:b40d91fe2326c5fae288e4a853377fb164aa0a6ba1de62cb28aba15d65500a1e`,
  two workers, two CPUs, 56 GB RAM, and atomic pathway checkpoints.
- `ControlTower-SSD` is stopped. Windows previously reported its approximately
  342 GiB `F:\WSL\ControlTower-SSD\ext4.vhdx` as corrupted and unreadable
  (`0x80070570`). Preserve the VHDX and C: checkpoints.
- Do not retry WSL, run CHKDSK, repair/mount the VHDX, restart services, or
  resume the campaign without explicit approval of the VHDX safety copy,
  recovery, and one-worker resume. After recovery, verify all 93 checkpoints
  and run only the three missing ground evaluations.

## Dependencies and Open Scientific Gates

- Upstream PR #9022 merged as remote-main commit `76bf6ab1d`; its provider-mode
  isolation and recovery record must remain.
- PR #9017 merged at `ce6fce1c2b8a6e50e410d16d31e219fabcb154e1`
  and supplies fail-closed participant split, processing, frame-transform, and
  event authorities for #9004.
- #9004 remains open without a qualifying governed participant trajectory.
  #8556 remains open without synchronized participant-level bilateral six-axis
  grip wrenches. Simscape exports, fixtures, tutorials, and paper curves are
  not substitutes.
- #8800 blocks the final #8752/#8668 audit until its complete checkpoint set is
  recovered and qualified. Preserve adverse/null results.
- Tools protected work remains dependency ordered. Do not create redundant
  reruns for unchanged capacity-bound work, and do not pin a Tools revision
  until its protected merge is verified. Staged consumer worktree
  `8358-tools-variation-adapter-prep` remains unpublishable until dependencies
  merge. AffineDrift #3930 remains downstream of a qualified UpstreamDrift
  release; never project a moving or partial campaign.
- The model ladder is synthetic and conditional. It does not establish human
  mechanics, anatomy, physiology, injury, equipment calibration, coaching
  strategy, or a universal speed benefit. Keep energy transfer, momentum
  redistribution, joint work, constraint forces, and clubhead speed distinct.

## Validation

```powershell
python3 -m pytest -n 0 -q tests/unit/research/test_biomechanics_source_register.py tests/unit/research/test_biomechanics_evidence_bridge.py tests/unit/research/test_biomechanics_evidence_surfaces.py tests/unit/research/test_biomechanics_claim_registration.py
python3 -m scripts.research.proximal_distal_energy.biomechanics_source_register validate
python3 -m scripts.research.proximal_distal_energy.biomechanics_evidence_bridge validate
python3 -m scripts.research.proximal_distal_energy.biomechanics_evidence_surfaces validate
python3 -m scripts.research.proximal_distal_energy.claim_adjudication_summary validate
python3 -m scripts.research.proximal_distal_energy.claim_audit validate
python3 -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python3 -m scripts.research.proximal_distal_energy.external_source_review validate
python3 -m scripts.research.proximal_distal_energy.qualify_open_release validate
python3 scripts/check_document_title_case.py --changed-from origin/main
python3 scripts/ci/check_file_size_budget.py
```

Use TDD, DbC, DRY, and LoD. Edit canonical sources, regenerate governed
outputs, and verify exact PR head, reviews, checks, merge SHA, remote-main
ancestry, and a clean checkout before reporting completion.

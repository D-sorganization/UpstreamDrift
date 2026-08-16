# Agent Handoff — UpstreamDrift

Last updated: 2026-08-16

This is current operational state. Historical detail belongs in git/GitHub.

## Program Authority

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal program; #8595 retains the photographed agenda.
- #8668 governs subject-scaled articulated contact. Children #8676, #8678,
  #8680, and #8682 completed inertia, contact projection, bounded bilateral
  forwarding, and typed slack/contact.
- [#8684](https://github.com/D-sorganization/UpstreamDrift/issues/8684)
  governs distributed grip, shaft, and ground pathways. #8685 completed grip
  discretization. [#8697](https://github.com/D-sorganization/UpstreamDrift/issues/8697)
  is the active shaft child on branch `research/8684-articulated-shaft-8697`.
  Protected PR [#8715](https://github.com/D-sorganization/UpstreamDrift/pull/8715)
  was opened from rebased head `7cb5f8b78` onto main `7298dd811`.
- #8556 remains open: no governed participant dataset contains synchronized
  bilateral six-axis grip wrenches. Synthetic traces cannot replace it.
- NotebookLM review remains blocked on manual Google reauthentication. Never
  automate credentials, authentication dialogs, CAPTCHA, or two-factor steps.

## Qualified Articulated Baseline

- Six profiles, three grip spans, and 13 phases define 234 closed states.
  Prescribed hands miss grips by 0.171–0.616 m; the reduced-tree solve closes
  to `1.16e-10 m`. Broad bounds/spheres are not clinical ranges or anatomy.
- Native MuJoCo and robotics Pinocchio independently qualify the 20-coordinate
  rigid tree. Bilateral point, typed-slack, and distributed-fiber tiers retain
  power, passivity, energy, refinement, geometry, and engine-parity controls.
- The distributed atlas covers 12 states, one/three/five fibers, two velocity
  signs, two steps, two engines, and nested 4/10/25/50 ms observations. It
  establishes discretization sensitivity, not measured pressure or benefit.

## Active Shaft Slice — Complete Locally

- Added passive bending/torsion constitutive, forward, atlas, frozen-basis,
  limiting-step diagnostic, figure, registration, release, and test layers.
- Coordinates are two tip-normalized first bending modes plus tip twist.
  Bending inherits the 24-element FE authority at `5.2399 Hz`; declared tapered
  hollow-section torsion is `70.1260 Hz`. Damping ratio is 0.018. These are
  synthetic structural references, not equipment calibration.
- Native WSL cannot solve the FE eigenproblem with its lean SciPy/NumPy stack.
  Regenerate `articulated_shaft_structural_basis.{json,npz}` under Windows;
  native runs hash-check that frozen basis without importing SciPy.
- Coarse-step falsification is retained: 1.0 ms leaves the linear domain for
  state `(0,0)`/negative velocity; 0.50 ms does so for `(8,0)`/positive
  velocity. The limiting 0.25/0.125/0.0625 ms residuals refine
  `0.00776 → 0.00390 → 0.00195 J` while remaining bounded through 50 ms.
- The registered 0.25/0.125 ms atlas covers 384 trajectories and 1,536 nested
  summaries. Every domain, power, energy, activation, and parity gate passes.
  Maximum bend is `1.696 mm`, twist `0.001855 rad`, energy residual
  `0.007798 → 0.003900`, trajectory parity `3.84e-13`, and force parity
  `2.15e-10`.
- Of 384 coupled-versus-rigid cells, 126 match within 5% for peak load and
  dissipated work. Speed differences span `-0.0285` to `+0.0212 m/s`
  (82 negative, 44 positive), rejecting a universal passive-shaft speed benefit.
- The paper is 229 pages and 1,733,358 bytes with 189 URI links and 246 outline
  entries. New body pages 138–140 and the mechanism figure were visually
  inspected. Inventory/claim/release totals are 1,047/291/39; all are reviewed.

## Immediate Next Steps

1. Shepherd #8715 without bypassing protection, preserve squash auto-merge,
   and verify its merge commit on remote main.
2. Close #8697 only after merge. Update #8684 with mixed matched outcomes and
   retained calibration/higher-mode boundaries.
3. Create the finite ground/free-moment child under #8684. Qualify it separately
   with the same state, energy, load/work, reversal, killswitch, refinement,
   and two-engine controls before combining ground, grip, and shaft pathways.
4. Continue to full-delivery matching/uncertainty and governed human holdout;
   do not close #8556 without qualifying participant data.

## Reproduction and Release Gates

Use Windows Python for the FE basis and WSL `python3` for native atlas runs.
The full shaft atlas takes about 34 minutes with four ordered workers.

```powershell
python -m scripts.research.proximal_distal_energy.generate_articulated_shaft_structural_basis
wsl.exe bash -lc "cd /mnt/c/Users/diete/Repositories/UpstreamDrift-worktrees/articulated-shaft-8697 && python3 -m scripts.research.proximal_distal_energy.run_articulated_shaft_time_step_diagnostic"
wsl.exe bash -lc "cd /mnt/c/Users/diete/Repositories/UpstreamDrift-worktrees/articulated-shaft-8697 && python3 -m scripts.research.proximal_distal_energy.run_articulated_shaft_atlas"
python -m scripts.research.proximal_distal_energy.make_articulated_shaft_figure
python -m scripts.research.proximal_distal_energy.claim_audit inventory
python scripts/research/proximal_distal_energy/register_articulated_shaft_claims.py
python -m scripts.research.proximal_distal_energy.release_claim_review write
python -m scripts.research.proximal_distal_energy.qualify_open_release write
python -m pytest tests/research/test_articulated_shaft.py tests/research/test_articulated_shaft_forward.py tests/research/test_articulated_shaft_atlas.py tests/research/test_proximal_distal_release_bundle.py -q
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/check_doc_size_budget.py
pre-commit run --hook-stage pre-push --all-files
```

Do not infer human technique, physiology, injury, timing demand, or coaching
advice; close #8556/#8557; bypass protection; force-push; admin-merge; or rerun
unchanged runner-capacity failures.

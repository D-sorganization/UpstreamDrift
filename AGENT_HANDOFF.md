# Agent Handoff — UpstreamDrift

Last updated: 2026-08-17

This is current operational state. Historical detail belongs in git/GitHub.

## Program Authority

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal program; #8595 retains the photographed agenda.
- #8668 governs subject-scaled articulated contact. Children #8676, #8678,
  #8680, and #8682 completed inertia, contact projection, bounded bilateral
  forwarding, and typed slack/contact.
- [#8684](https://github.com/D-sorganization/UpstreamDrift/issues/8684)
  governs distributed grip, shaft, and ground pathways. **All three filed
  children are merged**: grip discretization (#8685/#8696), passive shaft
  (#8697 via PR #8715, `0c988f05a`, with SPEC follow-up #8717 `051f8dccc`), and
  finite ground/free moment (#8719 via PR #8723, `a1a61399`). All are verified
  ancestors of remote main.
- **#8668 and #8684 remain open on purpose.** Every filed child is closed, but
  both epics declare design elements that have not shipped, so neither is
  complete against its own acceptance criteria. Do not close them on
  child-count alone — see Declared Gaps below.
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

## Qualified Shaft Slice

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
- The paper is 231 pages and 1,764,016 bytes with 192 URI links and 246 outline
  entries. Finite-ground body pages 140–143, front matter, availability, and
  end matter were visually inspected. Inventory/claim/release totals are
  1,063/295/40; all are reviewed.

## Qualified Ground Slice

- `articulated_ground.py` adds `fixed`, `translation`, `free_moment`, and
  `coupled` base pathways while the separately rooted club remains grip-coupled.
  The passive law exposes reaction force, intrinsic/transported moment, energy,
  damping, and closure; CoP reversal changes transport only.
- Full non-club inertia/cross terms, Christoffel bias, gravity, common-coordinate
  grip Jacobians, and forward integration are implemented. Fixed base reduces
  exactly to the shaft tier; positive-definite inertia, energy, power, contact,
  domain, and force/moment records are covered by 30 regression tests.
- A dependency-free Newton/line-search solver balances ground, grip, and
  gravity at fixed posture. Retain natural-zero, gravity-only, and full
  conditional-equilibrium initializations as separate sensitivity branches.
- The 42-trace two-engine diagnostic has 14 monotonically refining energy
  series, trajectory error below `2.5e-12`, and force error below `1.1e-10`.
- Initialization is not innocuous. At 0.125 ms, natural-zero versus gravity-
  only versus conditional-base starts produced peak ground forces of 32.8,
  565.5, and 510.3 N and 4 ms club speeds of 0.264, 1.908, and 0.946 m/s.
  The conditional solve balances only base generalized forces, not the full
  mechanism; retain this limitation and use natural-zero for exact-state atlas
  killswitch comparisons.
- The completed atlas contains 384 primary traces (12 states, four pathways,
  velocity reversal, two steps/engines) plus 192 rigid-shaft and horizontal-
  restraint-removed controls at 4/10/25/50 ms. All 576 trajectories pass the
  registered domain, refinement, energy, and native-engine gates. Worst energy
  residual refines `0.01986 → 0.00995`; maximum trajectory, grip-force, and
  ground-force discrepancies are `1.77e-10`, `6.41e-10`, and `2.28e-10`.
- The preregistered 5% peak-grip-load plus total-dissipated-work screen admits
  **0/384** coupled--fixed cells; total-work discrepancy is 1.72--2.00 because
  only the coupled path contains ground damping. Do not interpret unmatched
  positive speed differences as a ground-pathway benefit.
- A labeled post-hoc non-ground-dissipation screen admits 60 cells: 20 speed
  differences are positive and 40 negative (`-0.00075` to `+0.01394 m/s`);
  it is sensitivity evidence, not a replacement for the registered estimand.
- JSON/NPZ evidence and the six-panel PDF/SVG figure are generated. Atlas and
  post-hoc evidence tests pass. The 30-test scoped suite, claim/release/
  readiness audits, title case, source-size gate, and PDF QA pass. The full
  pre-push hook still exposes unrelated repository-wide Ruff/mypy baseline debt;
  its unrelated formatter edits were restored.

## Declared Gaps — Why #8668 and #8684 Are Still Open

These are read from the epics' own Required Design sections against committed
evidence, not from opinion. Each is a reason the parent epic cannot close.

1. **Grip friction is not modelled.** #8684 requires "friction bounds" and
   #8668 requires distributed contact with "friction"; the grip atlas records
   `friction_tissue_shaft_or_ground: not_modeled`, and the ground law is
   explicitly "not unilateral normal contact, Coulomb friction, foot
   segmentation, or measured pressure".
2. **Loss of contact is unexercised at the distributed tier.** Both epics
   require unilateral opening and reattachment there. The distributed grip
   atlas reports `maximum_transition_count: 0` — no station ever opened. Open
   and reattach events exist only at the single-point typed-slack tier
   (#8683: 108 opening and 216 reattachment cells), so the distributed claim is
   untested rather than supported.
3. **No manufactured-solution cases for the articulated tier.** #8668 requires
   "manufactured free-body and constrained-motion cases". No
   `articulated_*` manufactured artifact exists; the only manufactured/
   uncertainty evidence on main predates the articulated tier.
4. **No articulated parameter-uncertainty or failure map.** #8668 requires
   sensitivity to joint-limit, anthropometric, grip, shaft, and ground
   parameters plus uncertainty and failure maps. Only the planar-tier
   `uncertainty_control_*` studies exist.

## Immediate Next Steps

1. Close the declared gaps above before proposing closure of #8684 or #8668.
   They are filed as
   [#8751](https://github.com/D-sorganization/UpstreamDrift/issues/8751)
   (grip friction and distributed loss of contact, gaps 1-2) and
   [#8752](https://github.com/D-sorganization/UpstreamDrift/issues/8752)
   (manufactured solutions and articulated parameter uncertainty, gaps 3-4).
   #8751 changes scientific conclusions; #8752 is verification debt.
2. Continue to calibrated unilateral 3D contact, full-delivery matching/
   uncertainty, and governed human holdout; do not close #8556 without
   qualifying participant data.
3. Do not close #8557. It governs the whole program and outlives these tiers.

## Reproduction and Release Gates

Use Windows Python for the FE basis and WSL `python3` for native atlas runs.

```powershell
python -m scripts.research.proximal_distal_energy.generate_articulated_shaft_structural_basis
wsl.exe bash -lc "cd /mnt/c/Users/diete/Repositories/UpstreamDrift-worktrees/articulated-shaft-8697 && python3 -m scripts.research.proximal_distal_energy.run_articulated_shaft_time_step_diagnostic"
wsl.exe bash -lc "cd /mnt/c/Users/diete/Repositories/UpstreamDrift-worktrees/articulated-shaft-8697 && python3 -m scripts.research.proximal_distal_energy.run_articulated_shaft_atlas"
wsl.exe bash -lc "cd /mnt/c/Users/diete/Repositories/UpstreamDrift-worktrees/ground-free-moment-8719 && python3 -m scripts.research.proximal_distal_energy.run_articulated_ground_diagnostic"
wsl.exe bash -lc "cd /mnt/c/Users/diete/Repositories/UpstreamDrift-worktrees/ground-free-moment-8719 && python3 -m scripts.research.proximal_distal_energy.run_articulated_ground_atlas"
python -m scripts.research.proximal_distal_energy.run_articulated_ground_posthoc_sensitivity
python -m scripts.research.proximal_distal_energy.make_articulated_ground_figure
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

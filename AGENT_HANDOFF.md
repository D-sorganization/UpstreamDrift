# Agent Handoff — UpstreamDrift

Last updated: 2026-08-15

This is a current-state handoff. Historical detail belongs in git and GitHub.

## Program Authority

- Epic [#8557](https://github.com/D-sorganization/UpstreamDrift/issues/8557)
  governs the proximal-to-distal program; [#8595](https://github.com/D-sorganization/UpstreamDrift/issues/8595)
  retains the photographed nine-point agenda.
- [#8668](https://github.com/D-sorganization/UpstreamDrift/issues/8668) is the
  subject-scaled articulated-contact tier. Children #8676, #8678, and #8680
  completed inertia, contact projection, and bounded bilateral forwarding.
- [#8682](https://github.com/D-sorganization/UpstreamDrift/issues/8682) adds
  architecture-budget remediation and typed unilateral slack/contact.
- [#8556](https://github.com/D-sorganization/UpstreamDrift/issues/8556)
  remains open: no governed participant dataset contains synchronized bilateral
  six-axis grip wrenches. Synthetic traces cannot replace it.
- NotebookLM review is blocked on manual Google reauthentication. Never
  automate credentials, authentication dialogs, CAPTCHA, or two-factor steps.

## Governed Evidence State

- The inventory contains 1,029 adjudicated candidates, 284 atomic claims, and
  37 release claims; no candidate or release review is open.
- Evidence integrity covers 1,908 references, 250 hash-pinned local artifacts,
  and 78 URLs representing 56 works. Link availability is not validation.
- Eight of nine photographed-agenda points have bounded/partial model answers.
  MTQ-06—whether passive transfer reduces timing precision beyond the adverse
  planar comparison and in people—remains unresolved.
- Higher proximal rate is not universally beneficial. Matching choice and
  torso, arm, and wrist killswitches retain favorable and adverse outcomes.

## Articulated Geometry and Dynamics

- Six engineering profiles, three grip spans, and 13 phases define 234 states.
  Prescribed anatomical hand points miss grip contacts by 0.171–0.616 m even
  though both contact Jacobians can have rank six.
- The reduced-tree closed solve reaches `1.15817e-10 m` worst closure. Broad
  joint bounds and spheres are not clinical ranges or mesh anatomy.
- Native MuJoCo and robotics Pinocchio independently assemble the same
  20-coordinate tree at all 234 closed states. Mass, bias, inverse-dynamics,
  symmetry, and positive-definiteness gates pass below `1.8e-12` relative.
- Bilateral Kelvin–Voigt forces project through hand/grip Jacobians at all 234
  states. Action–reaction, virtual power, passivity, geometry reversal, and
  native initial-acceleration gates pass.
- The reduced hand-carriage reference passes 2,160 paired cases through 4, 10,
  25, and 50 ms; absence of a first failure is right-censored at 50 ms.

## Bounded Bilateral and Typed-Slack Results

- The bilateral articulated gate covers 18 states, seven branches, three time
  steps, and two engines: 756 five-millisecond trajectories. All retention,
  power, energy, refinement, and parity gates pass; the worst normalized energy
  residual falls from `0.00738` to `0.000854`.
- The typed atlas covers bilateral, tension-only, 0.5 mm and 1.5 mm dead-zone
  laws; common-displacement and matched-extension preload; velocity reversal;
  two event probes; three steps; and two engines: 1,944 trajectories.
- All typed-law numerical and parity gates pass. Worst force is `23.643 N`,
  normalized energy residual refines `0.01923 → 0.01097 → 0.00480`, trajectory
  parity is `1.32e-15`, and active-set parity failures are zero.
- Natural branches show no transition before 5 ms. The 1.5 mm common 1 mm
  displacement stays open; matched extension stays taut. Isolated probes yield
  108 opening and 216 reattachment cells, qualifying event logic only.
- No slack benefit, necessity, intent, timing economy, self-correction,
  delivery advantage, human transfer, or coaching rule is supported.

## Next Scientific Gates

1. Extend typed laws beyond 5 ms with distributed grip pressure, calibrated
   friction, shaft bending/torsion, tissue, ground coupling, and uncertainty.
2. Repeat geometry, reversal, killswitch, matched-work/load, refinement,
   virtual-power, work–energy, and two-engine event controls.
3. Add full-delivery state-matched rate, acceleration, and timing interventions
   plus continuous attraction regions with identified observers.
4. Qualify MT-E07 using a traceable bilateral six-axis device and execute
   MT-H01 only after governed acquisition and a frozen participant holdout.

## Reproduction and Release Gates

```powershell
python -m scripts.research.proximal_distal_energy.run_articulated_contact_projection
python -m scripts.research.proximal_distal_energy.run_articulated_forward_contact
python -m scripts.research.proximal_distal_energy.run_articulated_slack_atlas
python -m scripts.research.proximal_distal_energy.make_articulated_slack_figure
python scripts/research/proximal_distal_energy/register_articulated_slack_claims.py
python -m pytest tests/research/test_articulated_contact_projection.py `
  tests/research/test_articulated_forward_contact.py `
  tests/research/test_articulated_slack_contact.py `
  tests/research/test_articulated_slack_forward.py `
  tests/research/test_proximal_distal_release_bundle.py -q
python -m scripts.research.proximal_distal_energy.claim_audit validate
python -m scripts.research.proximal_distal_energy.claim_evidence_integrity validate
python -m scripts.research.proximal_distal_energy.momentum_question_readiness validate
python -m scripts.research.proximal_distal_energy.qualify_open_release validate
python scripts/ci/check_architecture_budget.py
python scripts/check_document_title_case.py --changed-from origin/main
python scripts/check_doc_size_budget.py
pre-commit run --hook-stage pre-push --all-files
```

Render and inspect the Quarto PDF before protected squash merge. Do not infer
human technique, physiology, injury, timing demand, or coaching advice; treat a
net wrench as bilateral allocation; close #8556/#8557; bypass protection;
force-push; admin-merge; or rerun unchanged capacity failures.

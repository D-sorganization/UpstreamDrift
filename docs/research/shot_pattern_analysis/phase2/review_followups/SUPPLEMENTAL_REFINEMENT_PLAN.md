# Supplemental Numerical Refinement Plan

This is an adapted, pre-execution deterministic gate for issue #12000. It does not replace the immutable original study acceptance or establish empirical accuracy. No supplemental flights were executed when preparing this script.

## Inputs and Population

Run `validation/supplemental_refinement.py` only after all three 32-shot protocol pilots pass independent complete audits against the accepted new execution snapshot. The input manifest has schema `supplement-refinement-inputs/1`, the exact `execution`, a hash-bound `source_review` (`path`, `sha256`), and `pilots` keyed by the three protocol IDs. Each pilot entry names `root` and hash-bound `audit` (`path`, `sha256`), resolved relative to the input manifest. The source review must have status `independently_accepted` and matching `new_execution`. Pilot plans, cell manifests, and all inventory bytes are checked against the audits. The script refuses existing output receipts and checks source/input drift after execution.

Select the actual saved Straight, Draw, and Fade nominals from rotation-SD 2°, combined2 in each protocol. All profiles within that protocol must share identical nominal records. This gives nine nominal deliveries. No matching is repeated or silently changed.

## Finite Grid and Gates

At each nominal, evaluate zero error; each of five primitive axes at ±2 and ±3 SD; and four signed joint directions at ±2 and ±3 SD. The 37 unique cases use SDs (shaft rotation, path, speed, horizontal strike, vertical strike) = (2°, 0.5°, 0.89408 m/s, 0.00635 m, 0.00635 m). Thus 333 deliveries are compared at six settings, or 1,998 flight evaluations. This is a selected finite sensitivity grid, not a joint probability region or certification of unbounded Gaussian tails.

Reuse the original independent preflight settings and comparison routine, adding the exact primary setting (128 contact steps, 0.02 s flight step). Compare contact 128/256/512 at 0.01 s; flight 0.02/0.01/0.005 s at 512 steps; and the actual primary to 512/0.005. The unchanged gate is endpoint change ≤0.05 m. Also require every fine zero-error nominal endpoint to remain within 0.05 m of its saved target. A repeated identical unsupported reason is reported as outside the supported domain, not numerical success. Status disagreements and numerical exceptions fail the gate. All cases remain in the receipt; no resampling or tolerance adjustment is permitted.

## Reproduction

From the study worktree, after qualifying the real input manifest:

```sh
PYTHONPATH=. QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy /tmp/shot-pattern-validation-venv-20261009/bin/python docs/research/shot_pattern_analysis/phase2/validation/supplemental_refinement.py --inputs QUALIFIED_INPUTS.json --output NEW_IMMUTABLE_RECEIPT.json
```

The script records its own hash, the reused comparison helper hash, execution snapshot, bound input hash, every endpoint/status, and failing case indices. Preparation tests use fake endpoint engines only. Supplemental final acceptance still requires the resulting real receipt and independent review; preparation alone authorizes no Monte Carlo run.

## Preparation Verification

Six fake-engine/qualification rejection tests pass, as do Ruff and a fresh-cache scoped Mypy run (`--python-version 3.12 --follow-imports=silent`). The first default-cache Mypy process crashed with exit 139 and no diagnostics; the fresh-cache retry completed successfully. Neither test execution nor static checking ran a flight simulation.

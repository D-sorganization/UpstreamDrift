# Shot Pattern Analysis Turnover

## Workspace and Authorization

Resume in `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`, branch `codex/shot-pattern-analysis`. Do not edit the primary checkout or `vendor/ud-tools`. The user authorized the personal GitHub identity for this task. Use Gemini 3.8 Flash High through `agy` for issue/publication assistance. Do not bypass hooks, alter repository settings, force-push, or close issues before merged evidence.

## Saved Work and Acceptance State

The core model, launcher tile, configurable GUI/CLI, geometric sensitivity, independent impact/cache/numerical tests, and matrix overview are committed. The study covers 24 cells and 720,000 synthetic shots: driver/7-iron/PW, SD 1°/2°, nominal/doubled curves, fixed loft/assumed shaft rotation. No measured player data or empirically qualified off-center impact model exists. Driver uses tee scoring; irons/PW use approach scoring. Preserve historical failed experiments separately.

SPA-001–014 map to #11840–11853; SPA-015/016 map to #11862/11863. Check `issues/publication_receipt.json` before quoting later issue URLs. No PR has been opened at this checkpoint.

All 24 corrected cells finished with exit code zero. The frozen flight source archive includes 32 source files and the native binary, each hash matching every cell's `run_start.json`. The pre-correction scoring archive includes the prior summary, receipt, baseline, SG report, manifest, and run start for every cell. The V2 rescore completed from saved CSVs without re-simulating a shot. All 24 current manifests were verified file by file, 720,000 scored rows are present, every green expected-strokes knot is at least one, and the V2 baseline hash is `7b0b858eb86581ef26538b1c836271822c98df9d94bc86b4cbb2371543cc1ca1`. Astra independently checked all 24 bundles and metric recomputation.

## Required Remaining Work

1. Finish the final overview and README/research LaTeX text with V2 tables, carry/equal-range controls, paired SG confidence intervals, and qualified conclusions. Refresh the GUI preview.
2. Finish the canonical pre-PR runner and any relevant lint/type gates, then push safely, open a ready PR, attach it to this task, and arm the merge queue through the guard. Keep unresolved gates explicit. The scientific baseline, saved flights, and Astra's independent review are complete.
3. Preserve the two ZIP snapshots and per-cell `run_start.json` when packaging. Do not rerun the full matrix merely to update scores or plots.

## Resume Commands

Use `/home/dieterolson/Repositories/UpstreamDrift/.venv/bin/python` and headless environment variables `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy`.

```bash
git status --short
git log -8 --oneline
python -m src.tools.shot_pattern_analysis.matrix docs/research/shot_pattern_analysis/corrected_results
python -m src.tools.shot_pattern_analysis.overview docs/research/shot_pattern_analysis/corrected_results --output-dir docs/research/shot_pattern_analysis/overview
python -m pytest tests/tools/shot_pattern_analysis tests/ui/tools/shot_pattern_analysis
python /home/dieterolson/Repositories/Repository_Management/scripts/pre_pr.py --base-ref origin/main
```

The matrix command now skips only verified complete bundles. Before running it, inspect the latest commits and this document so existing valid runs are preserved. Never run the expensive full matrix merely to regenerate scores or plots.

To reproduce the V2 postflight scoring from saved CSVs after preserving a copy of current score artifacts, run from the repository root:

```bash
python - <<'PY'
import hashlib, json
from pathlib import Path
from src.tools.shot_pattern_analysis.scoring import score_saved_bundle
root = Path('docs/research/shot_pattern_analysis/corrected_results')
manifests = sorted(root.glob('*/manifest.json'))
assert len(manifests) == 24
for manifest_path in manifests:
    cell = manifest_path.parent
    manifest = json.loads(manifest_path.read_text())
    flight_sha = manifest['files_sha256']['shots.csv']
    assert hashlib.sha256((cell / 'shots.csv').read_bytes()).hexdigest() == flight_sha
    score_saved_bundle(cell)
    receipt_path = cell / 'receipt.json'
    receipt = json.loads(receipt_path.read_text())
    receipt.setdefault('rescoring', {})['shots_csv_sha256'] = flight_sha
    receipt_path.write_text(json.dumps(receipt, indent=2) + '\n')
    manifest['scoring_revision'] = 'table9-full-plus-anchor-reconciled-putting/2'
    for name in manifest['files_sha256']:
        manifest['files_sha256'][name] = hashlib.sha256((cell / name).read_bytes()).hexdigest()
    assert manifest['files_sha256']['shots.csv'] == flight_sha
    manifest_path.write_text(json.dumps(manifest, indent=2) + '\n')
PY
```

## Known External Gates

Standalone LaTeX compilation is unverified: the built-in compiler could not download its uncached Tectonic bundle. The canonical engineering manual remains `blocked-inventory-required`; this separate research reference does not qualify that release. Full-repository title-case debt is pre-existing and must not be described as fixed. Earlier mypy processes crashed under concurrent simulation load; a successful final type check is still required.

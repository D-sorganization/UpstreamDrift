# Shot Pattern Analysis Turnover

## Workspace and Authorization

Resume in `/home/dieterolson/.codex/worktrees/shot-pattern-analysis/UpstreamDrift`, branch `codex/shot-pattern-analysis`. Do not edit the primary checkout or `vendor/ud-tools`. The user authorized the personal GitHub identity for this task. Use Gemini 3.8 Flash High through `agy` for issue/publication assistance. Do not bypass hooks, alter repository settings, force-push, or close issues before merged evidence.

## Saved Work and Acceptance State

The core model, launcher tile, configurable GUI/CLI, geometric sensitivity, independent impact/cache/numerical tests, and matrix overview are committed. The study covers 24 cells and 720,000 synthetic shots: driver/7-iron/PW, SD 1°/2°, nominal/doubled curves, fixed loft/assumed shaft rotation. No measured player data or empirically qualified off-center impact model exists. Driver uses tee scoring; irons/PW use approach scoring. Preserve historical failed experiments separately.

SPA-001–014 map to #11840–11853; SPA-015/016 map to #11862/11863. Check `issues/publication_receipt.json` before quoting later issue URLs. Publication is the final step: open the ready PR with a blocked merge label; its URL is recorded in the task attachment and GitHub branch metadata.

All 24 corrected cells finished with exit code zero. The frozen flight source archive includes 32 source files and the native binary, each hash matching every cell's `run_start.json`. The pre-correction scoring archive includes the prior summary, receipt, baseline, SG report, manifest, and run start for every cell. The V2 rescore completed from saved CSVs without re-simulating a shot. All 24 current manifests were verified file by file, 720,000 scored rows are present, every green expected-strokes knot is at least one, and the V2 baseline hash is `7b0b858eb86581ef26538b1c836271822c98df9d94bc86b4cbb2371543cc1ca1`. Astra independently checked all 24 bundles and metric recomputation.

## Release State

The final overview, 72-row statistics, corrected README/LaTeX tables, GUI preview,
and independent Astra review are complete. All 21 analysis issues are published and remain
open pending a merged implementing PR. No flights need rerunning.

The branch was rebased only to resolve actual conflicts with current main.
Shared generated maps are regenerated and their integration boundary is explicitly
revalidated. Final evidence is in `final_validation_receipt.json`, `pre_pr_validation.txt`, and `feature_validation.txt`. The final feature/UI lane passes 139 tests including slow tests. Six canonical gate groups pass (Semgrep advisory skipped; consistency collection has no marked tests), while affected tests and fleet policy fail on unchanged baseline artifacts. The stale research SHA is separately tracked as #11940; the affected lane reached 543 passed and 12 skipped before that failure. Do not claim full-repository acceptance.

## Resume Commands

Use `/home/dieterolson/Repositories/UpstreamDrift/.venv/bin/python` and headless environment variables `QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy`.

```bash
git status --short
git log -8 --oneline
python -m src.tools.shot_pattern_analysis.matrix docs/research/shot_pattern_analysis/corrected_results
python -m src.tools.shot_pattern_analysis.overview docs/research/shot_pattern_analysis/corrected_results --output-dir docs/research/shot_pattern_analysis/overview_v2
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

Standalone LaTeX compilation is unverified: the built-in compiler could not download its uncached Tectonic bundle. The canonical engineering manual remains `blocked-inventory-required`; this separate research reference does not qualify that release. Full-repository title-case debt is pre-existing and must not be described as fixed. Earlier mypy processes crashed under concurrent simulation load; targeted changed-source type checks have passed; canonical release checks are recorded separately.

## Resumed Validation

The user resumed the goal after the connection-loss checkpoint. Final headless
tool/UI suite including slow: 134 passed. Launcher/registry suites: 108 passed.
Impact/landing Python suite: 105 passed. Native Rust: 99 unit and 3 integration tests
passed. Astra independently verified all committed CSV/archive/artifact hashes.

The prior DiffMypy failure had conflicting package roots. Use `MYPYPATH=$PWD`
with the canonical runner. The shared host environment contains NumPy 2.5 stubs
requiring Python 3.12 while the project intentionally supports Python 3.11;
validation uses an isolated environment with compatible NumPy rather than changing
project support or suppressing type errors. Exact final environment/gates follow.

## Validation Environment Reproduction

Use a separate Python 3.12.13 environment; retain the original environment unchanged.
Install `numpy==2.2.6`, `scipy-stubs==1.17.1.0`, `optype==0.17.0`, and
`numpy-typing-compat==20251206.2.2` with `--no-deps`, inheriting the existing
project dependencies read-only through a `.pth` file. The resulting environment
is `/tmp/shot-pattern-validation-venv-20261009`, with mypy 1.20.1. Full changed-source
type checking passes 24 files without exclusions or suppressions.

```bash
QT_QPA_PLATFORM=offscreen MPLBACKEND=Agg MUJOCO_GL=egl SDL_VIDEODRIVER=dummy \
MYPYPATH="$PWD" PYTHONPATH=/home/dieterolson/Repositories/Repository_Management \
/tmp/shot-pattern-validation-venv-20261009/bin/python \
/home/dieterolson/Repositories/Repository_Management/scripts/pre_pr.py \
  --from-ref "$(git merge-base origin/main HEAD)" --to-ref HEAD
```

SPA-019 (#11930) records the broad-test import collision. A package marker fixes
the new test namespace; simultaneous collection of both `test_core.py` modules
passes. This preserves existing test behavior rather than ignoring collection errors.

## Clean-Checkout and Native Provenance Corrections

SPA-020 (#11936) removes the assumed ignored Cargo.lock requirement from both
run and export paths through one canonical snapshot helper. Presence or absence
is explicit; tracked scientific sources and the native binary remain mandatory.
The original lock remains in the frozen source archive. A generated root copy
was preserved outside the repository to satisfy the existing clutter contract.

SPA-021 (#11937) supports direct and packaged .so/.pyd native module layouts
while rejecting missing or ambiguous binaries. Ten filename/discovery unit tests
pass; this does not claim actual Windows flight execution.

Baseline Drake abort (#11934) and unchanged development-log fleet-policy debt
(#11935), and stale unrelated research provenance (#11940) remain separately tracked. The feature PR must be ready for review with
a blocked merge label while these broader release gates remain unresolved.

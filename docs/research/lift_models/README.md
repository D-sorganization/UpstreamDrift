# Lift Model Packs Exploratory Modelling Reference

Start with the [maintained LaTeX reference](lift_models.tex). It is an
editable standalone research source (see `AGENTS.md`, Modeling Reference
Documentation) for epic #11740, child LIFT-11 (#11751); the engineering
design-manual source remains `manuals/upstreamdrift`, and the governed
calculation for this material is Chapter 15, "Lift Model Pack Audit"
(`manuals/upstreamdrift/chapters/15-lift-pack-audit.qmd`), itself evidence
class "derived, provisional" and not yet approved. No lift motion-capture or
force-plate data exist, so nothing here is scientific validation.

| Item                                                                                   | Status                                                                                     |
| -------------------------------------------------------------------------------------- | ------------------------------------------------------------------------------------------ |
| Model inventory per pack (body, barbell, grip, foot/bench contact)                     | Documented                                                                                 |
| Coordinate frames, units, reference lifter                                             | Documented (cites Chapter 15; not re-derived)                                              |
| Phase definitions and per-pack phase counts                                            | Documented, measured (PACK_PARITY_BASELINE.md)                                             |
| Shared phase/exercise standard (LIFT-2, #11742)                                        | Open — board proposal pending owner approval                                               |
| Inverse dynamics / IK / feedback tracking / open-loop replay for the four engine packs | Not run as of this revision                                                                |
| Movement Optimizer inverse dynamics and spine load                                     | Documented, cites `spine_loads.py`                                                         |
| Muscle redundancy                                                                      | Open — depends on LIFT-6 (#11746)                                                          |
| Measured cross-engine parity (segments, hands, feet, bar centre, lifter CoM)           | Measured, LIFT-1 baseline receipt (#11771)                                                 |
| Discrepancies and tracking issues                                                      | Documented, derived programmatically (`gaps.py`)                                           |
| Failed experiments / negative results                                                  | Documented (floor-pull start pose, right-hand attachment, grip width, bench contact force) |

## Build and Reproduce

```bash
cd docs/research/lift_models && pdflatex -interaction=nonstopmode -halt-on-error lift_models.tex
export PYTHONPATH=.:src MUJOCO_GL=egl MPLBACKEND=Agg QT_QPA_PLATFORM=offscreen
python3 scripts/lifting/run_pack_parity_baseline.py
```

The generated PDF is not committed.

## Build Locally

CI typesets this reference with a pinned LaTeX toolchain
(`.github/workflows/latex-references.yml`) whenever a `docs/research/**/*.tex`
file changes. To reproduce it:

```bash
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build docs/research/lift_models/lift_models.tex
pdflatex -interaction=nonstopmode -halt-on-error -output-directory build docs/research/lift_models/lift_models.tex
# or, with Tectonic:
tectonic -X compile --outdir build docs/research/lift_models/lift_models.tex
```

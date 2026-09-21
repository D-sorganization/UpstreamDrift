# Neural Motion Matching Artifact Audit (NM-00)

Governing issue: [#10615](https://github.com/D-sorganization/UpstreamDrift/issues/10615).
Parent epic: [#10603](https://github.com/D-sorganization/UpstreamDrift/issues/10603).
Schema: `neural-artifact-audit/1.0.0`.

## Purpose

Inventory existing datasets, checkpoints and training claims before any new neural
training campaign. Fail closed on absent files, synthetic fixtures and note-only
plateau prose. Do not invent TenThousandFiles training results.

## How to Reproduce

```powershell
python -c "from pathlib import Path; from src.shared.python.neural_motion import audit_neural_artifacts; r=audit_neural_artifacts(Path('.')); r.write_json('docs/plans/neural_motion_matching/evidence/nm00_artifact_audit_receipt.json'); print(r.source_revision, len(r.artifacts))"
```

Coverage matrix (one cell per TB-00 `GolfModelIdentity`):

```powershell
python -c "from pathlib import Path; from src.shared.python.neural_motion import generate_neural_coverage_matrix; import json; m=generate_neural_coverage_matrix(Path('.')); print(len(m), m[0].disposition.value)"
```

Unit tests:

```powershell
python -m pytest tests/unit/neural_motion/test_artifact_audit.py -q -n 0 --no-cov --timeout=60
```

## Disposition Summary

| Artifact ID                                                       | Disposition | Claim Status           | Notes                                       |
| ----------------------------------------------------------------- | ----------- | ---------------------- | ------------------------------------------- |
| `corpus.ten_thousand_files`                                       | quarantine  | unsupported            | Documented path absent on this host         |
| `fixture.sweep_synthetic`                                         | retain      | software_contract_only | Tracked toy fixture; not native supervision |
| `checkpoint.surrogate_best_default`                               | quarantine  | unsupported            | Default path absent                         |
| `checkpoint.surrogate_production_default`                         | quarantine  | unsupported            | Default path absent                         |
| `claim.inverse_cvae_mean_baseline_plateau`                        | quarantine  | note_only              | Source note; no curves                      |
| `claim.inverse_regressor_mean_baseline_plateau`                   | quarantine  | note_only              | Source note; no curves                      |
| `claim.historical_design_*` (#4075/#4076/#3999/#4000/#6014/#5419) | quarantine  | unsupported            | Prior design only                           |

## Reproducible Capability (Current)

- Sweep parquet loader and synthetic fixture generators work for software contracts.
- Safe checkpoint loader (`weights_only=True`) refuses unsafe pickle payloads.
- Forward surrogate, inverse CVAE/regressor and timestep inverse modules exist as code.
- No qualified native corpus, no production checkpoint hash and no reproduced
  training curves are claimed by this audit.

## Unsupported Claims

- Any 10k-scale training accuracy or speed result.
- Mean-baseline plateau text as evidence of a completed training campaign.
- Synthetic `data/sweep_synthetic` as physical/native supervision.
- Historical issues #4075/#4076/#3999/#4000/#6014/#5419 as current qualification.

## Next Action

Dispatch [#10617](https://github.com/D-sorganization/UpstreamDrift/issues/10617) (NM-02):
make native dataset labels complete and semantically correct under the NM-01
frozen learning-task and benefit-experiment contracts.

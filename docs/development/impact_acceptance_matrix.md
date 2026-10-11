# Impact Acceptance Matrix (Issue #9550, Epic #9546)

Machine-readable record: [`src/config/impact_acceptance.json`](../../src/config/impact_acceptance.json),
gated by `tests/config/impact_acceptance/test_impact_acceptance_matrix.py`.
This page restates it for review; the JSON is authoritative.

| Field                                               | Value                                                                                                                      |
| --------------------------------------------------- | -------------------------------------------------------------------------------------------------------------------------- |
| Audit snapshot (UpstreamDrift / Tools)              | `1f69a51fce997932f04a6ad1dd95bf4d065ba971` / `3d93bb2c89813e17551814d3be7e895f791e29af` (2026-09-04)                       |
| Reconciled against (UpstreamDrift base / Tools pin) | `df9f3b95386433fe4c3a45559498b3e3f501cfdf` / `86d0f28b1cc5acf61185e07e320c816c2d005512` (2026-10-10 UTC; F09o integration) |
| Release claim                                       | `code_verified_only`; no predictive-accuracy claim                                                                         |
| Supported Python matrix                             | 3.11 and 3.12 (CI Standard); 3.12 for the lock and Docker image                                                            |

## Model-Capability Matrix

### Current F09o Pin Reconciliation

F09o #12044 integrates the F01b/private-consumer branch `d232efcc69` into the
F09n base shown above. Before advancing the JSON, all 26 applicable capability
probes passed and the new Tools bundle was built and actually served/verified:
76 assets, five JavaScript assets, 2,801,426 bytes, exact revision/mount/digests,
and the missing-file 404. The complete matrix suite then passed all 27 tests.
Retain the original pin-drift RED receipt. The new served receipt is
`docs/development/feedback_controls/F09O-IMPACT-SERVED-RECEIPT.json`.

The diagnostic build used DeskComputer's Node 25.9.0/npm 11.12.1. Its existing
native SDK environment lacked FastAPI, so the transferred built bundle was
verified locally with Python 3.13/FastAPI 0.128.0/httpx 0.28.1. No native SDK
environment was modified. These actual versions are separate from the CI
authority's Node 22/Python 3.11/FastAPI 0.136.3; no toolchain-equivalence claim
is made. The F09o turnover binds the source archive, integration inputs and
executed evidence. This remains software correctness evidence only.

Reproduction from a complete checkout of the admitted Tools pin:

```powershell
$env:ROC_RELEASE_REVISION = '86d0f28b1cc5acf61185e07e320c816c2d005512'
Push-Location vendor/ud-tools/src/rate_of_closure/web
npm.cmd ci --no-audit --no-fund
npm.cmd run build -- --base=/impact-explorer-app/
node.exe release/generateReleaseArtifacts.mjs
Pop-Location
python3 scripts/ci/verify_impact_explorer_bundle.py `
  --dist vendor/ud-tools/src/rate_of_closure/web/dist `
  --expected-revision 86d0f28b1cc5acf61185e07e320c816c2d005512
python3 -m pytest tests/config/impact_acceptance/test_impact_acceptance_matrix.py
```

### Previous T01 Pin Reconciliation

On 2026-10-09, the exact merged Tools pin
`2e7665111b06f92ffbfe178b92d74d6a81c95388` was rechecked from UpstreamDrift
`62951330bb` before advancing JSON reconciliation metadata. All 26 applicable
matrix/capability checks passed, followed by a clean web build and actual
served-bundle verification: 76 assets, five JavaScript assets, 2,801,426 bytes,
the declared mount/revision and a missing-artifact 404. The receipt is
`docs/development/feedback_controls/T01_IMPACT_SERVED_RECEIPT.json`.
This updates exact-pin software evidence; predictive accuracy, physical
reference data and cross-runtime numerical scenario parity remain open.

Library: `src/shared/python/physics/impact_model` (SI units; caller's frame,
face normal `n`; friction spin axis `t x n`). Each row is probed against the
shipping code by the gate, so the table cannot describe a model that no
longer exists.

| Model           | Contact                                   | Tangential spin               | Gear effect                           | Inertia                         | COR                         | Shaft / grip | Numerical limits                                                                                                                                       |
| --------------- | ----------------------------------------- | ----------------------------- | ------------------------------------- | ------------------------------- | --------------------------- | ------------ | ------------------------------------------------------------------------------------------------------------------------------------------------------ |
| `rigid_body`    | instantaneous (0 s)                       | supported (Coulomb, capped)   | empirical overlay via solver API only | scalar MOI effective mass       | constant (`params.cor`)     | not modelled | closed form; friction impulse capped at 2/7·m·v_t; needs non-zero face normal                                                                          |
| `spring_damper` | finite, integrated (~1.26 ms at defaults) | **unavailable / normal-only** | empirical overlay via solver API only | ignored (offset and MOI unused) | constant (parameter unused) | not modelled | semi-implicit Euler, dt 1e-7 s, 5 ms / 1e5 N caps; converges to rtol 1e-4 at dt 5e-8; default k = 1e6 N/m gives smash 0.82, i.e. uncalibrated defaults |
| `finite_time`   | finite, **reported only**                 | supported (as rigid_body)     | empirical overlay via solver API only | scalar MOI effective mass       | constant                    | not modelled | delegates to rigid_body; duration is the input parameter (0.5 ms default)                                                                              |

A model lacking spin support returns the pre-impact spin unchanged; the
product must present that as _unavailable_, never as a predicted zero-spin
result. The gear-effect overlay is `offset × speed × scale` with a
user-supplied factor — an empirical relation, not a rigid-body prediction —
and is applied only through `ImpactSolverAPI.solve_pre_impact_state`.

## Golden Coverage (Item 2)

Covered in the gate, all SI: centered vs 20 mm toe strike, no-hit (zero
approach → zero launch and spin), 20 vs 60 m/s (constant smash factor),
backspin sign for a lofted strike, normal-direction momentum conservation,
passivity (kinetic energy never increases) and spring-damper time-step
convergence. Not covered here: left/right and loft/path/attack
decompositions, imperial conversion, external impulses and instantaneous
limits across the Tools explorer models — provider-owned goldens (Tools #4251).

## Installed Product Evidence (Item 4)

CI Standard `impact-explorer-web-build` builds the pinned Tools bundle from a
clean checkout with `--base=/impact-explorer-app/`, stamps Tools' release
artifacts with the gitlink (`ROC_RELEASE_REVISION`), then
`scripts/ci/verify_impact_explorer_bundle.py` serves `dist` through the
`/impact-explorer-app` mount contract and checks what a client receives:
`index.html` stamped with the pinned revision, every manifest asset served
with its SHA-256 (76 assets, 5 JavaScript at this pin), all references under
the base path, and a 404 — not the index fallback — for a missing artifact.
The receipt is uploaded as `impact-explorer-bundle-receipt-<sha>`.

Historical September 18 reproduction (Windows, Node 25; CI's Node 22 is the
authority):

```text
npm ci && MSYS_NO_PATHCONV=1 npm run build -- --base=/impact-explorer-app/
ROC_RELEASE_REVISION=62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1 node release/generateReleaseArtifacts.mjs
python3 scripts/ci/verify_impact_explorer_bundle.py --dist vendor/ud-tools/src/rate_of_closure/web/dist \
    --expected-revision 62e8cdbf9c9f5f8a43a0342059f825e8fa78f8e1
PASS: {"mount": "/impact-explorer-app", "assets_verified": 76, "javascript_assets": 5, "total_bytes": 2743316, "missing_artifact_status": 404, ...}
```

Without `MSYS_NO_PATHCONV=1` the Git-Bash shell rewrites the base path to
`/Program Files/Git/impact-explorer-app/` and the verifier fails — the
mis-based bundle that an HTTP 200 on `index.html` would have hidden.

## Outstanding Items

| Item | Status  | Blockers           | Owner      |
| ---- | ------- | ------------------ | ---------- |
| 2    | partial | #9546, Tools #4251 | unassigned |
| 3    | open    | #9546, Tools #4251 | unassigned |
| 4    | partial | #9417              | unassigned |
| 5    | open    | #9546              | unassigned |
| 6    | partial | #9546              | unassigned |

Item 6 changed `tools.rate_of_closure` in the feature-parity registry from
`parity` to `gap` (#9546): both runtimes execute the same pinned Tools
revision and the web mount serves verified artifacts, but no saved scenario
has been compared numerically across the PyQt and web runtimes yet.

## Claim Boundary

Synthetic fixtures and agreement with another implementation are code
verification. Nothing here — and none of the audit's 16 library tests —
supports a polished or predictive delivery claim; that requires the physical
reference datasets and predeclared error criteria of item 3, recorded
through the design-manual governance pathway.

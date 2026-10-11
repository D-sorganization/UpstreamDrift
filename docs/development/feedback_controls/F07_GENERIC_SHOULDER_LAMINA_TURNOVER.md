# F07 Generic Shoulder Triangle-Lamina Turnover

## Provider Pin Impact Reconciliation Follow-Up

PR #12246's full CI suite found one stale exact-pin impact acceptance record:
24,025 tests passed and one failed because the record still referenced Tools
`86d0f28b1cc5acf61185e07e320c816c2d005512`. The failure was reproduced locally
before updating metadata. All 26 remaining capability probes passed against
the new pin. DeskComputer built an archive of exact Tools commit
`6b6e4be55e7086a33db43b36d2632765be619bde`; the transferred bundle was served
and verified locally, covering 76 assets, five JavaScript assets, 2,801,426
bytes and the missing-artifact 404. See
[`F07_PROVIDER_12246_IMPACT_SERVED_RECEIPT.json`](F07_PROVIDER_12246_IMPACT_SERVED_RECEIPT.json)
and the [impact acceptance record](../impact_acceptance_matrix.md) for exact
toolchains, archive hashes and reproduction. Historical receipts and all
scientific qualification limits remain unchanged.

Issue #12211 adds a **new** candidate plant, not an edit or accepted repair
of the original derived 557-muscle source. The exact original source and
author-right CEINMS mesh SHA-256s are fixed in
`tour_matching/generic_shoulder_lamina.py`; the reviewed entry point rejects
other bytes. The missing BUET author mesh bytes are not inferred. Right mesh
placement is a declared identity-frame hypothesis and left local vertices are
z-reflected with reversed winding and reflected normals. Native frames,
muscle paths, joints, wraps, force laws and scalar body masses remain in the
source XML. Any other semantic XML edit rejects.

The provider is Tools #5503 `shared.python.humanoid_character_builder.mesh`
`triangle_lamina`, with source hashes for its kernel and inertia result
representation. The pinned Tools tree is the provider authority. Because the
UpstreamDrift checkout still owns a local `humanoid_character_builder` package,
the established seam deliberately refuses to fill missing modules from Tools;
doing so would create a mixed package. The producer therefore invokes a small
isolated worker with only `vendor/ud-tools/src` first on that child process's
import path. It exchanges numeric input/output through NPZ/JSON files and
returns the exact kernel, inertia-representation and worker source hashes.
Generation and subsequent candidate admission independently recheck those
identities. Neither the parent process's `sys.path` nor `sys.modules` is
modified, and no lamina tensor formula is copied into UpstreamDrift. The exact
mesh is open/nonmanifold, so the solid-volume route stays failed. The
zero-thickness triangle-multiset recipe is a separate fixed-mass mechanical
hypothesis.

TDD began with a missing-module RED. Synthetic four-body tests now cover
hash-mismatched source/mesh, altered source frame/scale, undeclared mass XML
edits, and mirrored COM/full tensor at the same mass. Actual OpenSim 4.6
execution against the final three-reduction XML SHA
`453e09c4e42dcc3f4b74a3b2efeff063e96c3820b3eee1efe95eef249579010d`
produced candidate SHA
`5f2cdd22013c2700f95d625a4c42eb488ed6c5267721cd0d2acde38544754326`.
The fresh 40-body physical audit passes; the original fails exactly four
bilateral clavicle/scapula bodies. Three sampled native poses (initialized
pelvis tilt, SC_x offsets ±0.01) show 0 maximum body-transform and QErr/UErr
difference. The immutable local receipt is
`docs/development/feedback_controls_planning/generic_shoulder_lamina_12211_v4/candidate-receipt.json`
in the fleet workspace, SHA-256
`562186fe895c0f107756e78001e486534d1f379b8dac2da6d289e01d81ec82cf`.
It binds runtime, model, provider, generated mesh and sampled pose identities.
The final Tools kernel source SHA-256 is
`d058b059644c88b382d49348ca8c96b50d4176cf13d8f10789e013598b1462de`;
the inertia representation dependency SHA-256 is
`60a45fe471c2e1cbfb2ae80d9a6b3760b62b07f0fd5327a4d5e652a49f467e10`.
These are conditional diagnostic results; visual meshes for other bodies were
not present during the OpenSim reload and do not enter this mass derivation.

Reproduction uses
`scripts/opensim/derive_generic_shoulder_lamina.py --source <exact-final-557.osim> --right-clavicle <exact-clavicle.vtp> --right-scapula <exact-scapula.vtp> --output <new-directory>`
with the OpenSim 4.6 Python environment and its native `CASADIPATH`. It needs
no development Tools root or test-only import injection. The same producer
path serves the focused tests and CLI, and fails closed unless both pinned
provider files and the worker match their recorded SHA-256 values. Keep the PR
draft/unarmed while remaining integration prerequisites are unresolved.

Remaining science gates include actual source-region/mesh ownership, full
native force/work/acceleration chart evidence, passive muscle/tendon laws,
contact and grip, independent head/marker geometry, private full-horizon
capture matching and fresh excitation replay. The original 557 source stays
physically blocked. All 17 variants and six ecosystems remain in the
denominator; no physiological or publication acceptance is asserted.

## Handoff Status (2026-10-10)

The actual OpenSim 4.6 focused suite passed 9 tests with no skips against
the final Tools provider source hash above. The source-bound v4 receipt and
generated XML hash are preserved in fleet planning. The generic candidate is
diagnostic and must remain a draft, unarmed PR until Tools #5503 is merged,
pinned, and revalidated; #12207 is also an ancestry prerequisite. No private
capture or production fit was run for this candidate.

The earlier reduction PR #12181 is still open with a ledger-freshness unit
failure and a real branch conflict. Those are not corrected by #12211 and
must be reconciled separately before treating its ancestry as accepted.
This branch must not be presented as a replacement for the original 557
source's failed physical-body audit or as completed full-swing matching.

## Pinned Provider and CLI Revalidation (2026-10-10)

The consumer now pins Tools gitlink `vendor/ud-tools` at
`6b6e4be55e7086a33db43b36d2632765be619bde`. The isolated worker resolves
`triangle_lamina.py` and `inertia_calculator.py` only from that gitlink, checks
their exact paths and bytes, and binds its own source digest into each
candidate. Synthetic tests exercise the provider without modifying parent
import state; a dedicated test snapshots and verifies the parent `sys.path`
and Tools mesh-module entries remain unchanged. The OpenSim 4.6 focused suite
passes all ten tests against the pinned provider.

The default CLI, without a `--development-tools-root`, reproduced candidate
v6 model SHA-256
`5f2cdd22013c2700f95d625a4c42eb488ed6c5267721cd0d2acde38544754326`, unchanged
from v5. Kernel SHA-256 is
`3fd45937bed011c90b4d4e3fdf8bcf932a8bfd9a0816441ee708cc93cd4080ee`, the
inertia representation SHA-256 remains
`60a45fe471c2e1cbfb2ae80d9a6b3760b62b07f0fd5327a4d5e652a49f467e10`, and the
worker SHA-256 is
`4309279807a6ac439b3b215dc3cf2414f4a93edaa445030c5a7dd085fc8e1bbf`. Its
40-body physical audit and three-pose comparison passed with zero maximum
transform and constraint-error difference. Receipt bytes are preserved in
fleet planning as `generic_shoulder_lamina_12211_v6/candidate-receipt.json`,
SHA-256 `248065dd5a5338b1907b56a5777d69447efb044b4242d95c6def0b141e15994f`.
The earlier v4/v5 receipts remain unchanged as historical records.

After the worker added an explicit fail-closed check for imports without source
file paths, the no-override CLI was rerun into a distinct v7 directory. Its
candidate model SHA remains
`5f2cdd22013c2700f95d625a4c42eb488ed6c5267721cd0d2acde38544754326`; the final
worker SHA is
`66e65bcfacbfb668d5b3bfadd3a76147a490f409f07e6d98d052fd4815279200`, with the
same Tools kernel SHA `3fd45937bed011c90b4d4e3fdf8bcf932a8bfd9a0816441ee708cc93cd4080ee`
and inertia representation SHA
`60a45fe471c2e1cbfb2ae80d9a6b3760b62b07f0fd5327a4d5e652a49f467e10`. The
40-body native audit again reported zero maximum transform and constraint-error
difference across three poses. The separate v7 receipt is
`generic_shoulder_lamina_12211_v7/candidate-receipt.json`, SHA-256
`76d89fdf93632217cbf9345d806c32cc728cae9bfb82b58c9d9bdd42b9c9087f`; v6
remains the historical receipt for the prior worker bytes.

The native test command used the dedicated OpenSim environment:

```powershell
$env:GENERIC_SHOULDER_FIXTURE_ROOT='C:\Users\diete\Repositories\docs\development\feedback_controls_planning'
$env:CASADIPATH='C:\Users\diete\Repositories\.venv-feedback-opensim\Lib\site-packages\opensim'
& 'C:\Users\diete\Repositories\.venv-feedback-opensim\Scripts\python.exe' -m pytest -q -o addopts= tests/opensim/test_generic_shoulder_lamina.py
```

The CLI v6 command used the same source and retained author-right meshes, with
no development provider override:

```powershell
$env:PYTHONPATH='C:\Users\diete\Repositories\Worktrees\feedback-controls-main-integration-12202'
$env:CASADIPATH='C:\Users\diete\Repositories\.venv-feedback-opensim\Lib\site-packages\opensim'
& 'C:\Users\diete\Repositories\.venv-feedback-opensim\Scripts\python.exe' scripts/opensim/derive_generic_shoulder_lamina.py --source 'C:\Users\diete\Repositories\docs\development\feedback_controls_planning\composed_reduction_12161_v2\03-custom-mtp.osim' --right-clavicle 'C:\Users\diete\Repositories\docs\development\feedback_controls_planning\generic_shoulder_mesh_triage_v1\clavicle.vtp' --right-scapula 'C:\Users\diete\Repositories\docs\development\feedback_controls_planning\generic_shoulder_mesh_triage_v1\scapula.vtp' --output 'C:\Users\diete\Repositories\docs\development\feedback_controls_planning\generic_shoulder_lamina_12211_v6'
```

## Provider Source Revalidation (2026-10-10)

The Tools provider source was narrowed for an explicit tuple type annotation in
`triangle_lamina.py`; the prior source commit remains preserved in Tools
history. The updated provider commit is
`920e6fd633e796cf311cfd16f294fd45f5ad8cb1`, with kernel SHA-256
`3fd45937bed011c90b4d4e3fdf8bcf932a8bfd9a0816441ee708cc93cd4080ee`.
The inertia representation dependency is unchanged at SHA-256
`60a45fe471c2e1cbfb2ae80d9a6b3760b62b07f0fd5327a4d5e652a49f467e10`.

Against that exact Tools checkout, the OpenSim 4.6 focused suite passed all
nine tests. A new native derivation used the reviewed source and mesh inputs,
producing the same candidate model SHA-256
`5f2cdd22013c2700f95d625a4c42eb488ed6c5267721cd0d2acde38544754326`.
Its 40-body physical audit and three-pose native comparison passed with zero
maximum transform and constraint-error difference. The source model remains
physically rejected for the same four bilateral clavicle/scapula bodies. The
new immutable receipt is
`docs/development/feedback_controls_planning/generic_shoulder_lamina_12211_v5/candidate-receipt.json`
in the fleet workspace, SHA-256
`74b205b34f81df5ee2f476a9aa44f7e3d1e15493e1ed67a9c55f6eefbf41554a`.
The earlier v4 receipt remains unchanged as the historical source-hash record.

This revalidation used the external development Tools root because Tools #5503
has not yet merged; the consumer gitlink therefore remains unchanged. Repeat
the native suite and receipt derivation against the merged Tools pin before
arming or merging the consumer PR. All scientific limitations and open gates
above remain in force.

## Current Main Integration

The provider follow-up normally merges main `9e5f87c4dd31fc3137fc204d98337633fa5c4855`. Generated divergence inventory is rebuilt against the unchanged exact Tools pin `6b6e4be55e7086a33db43b36d2632765be619bde`; context views are regenerated, with all recorded boundary inputs verified. The source-bound remote web build and served-asset receipt are retained. No physical qualification or private trajectory result is introduced by resolving generated-document conflicts.

Validation after this main merge: all five central gates passed, including full diff mypy on six production files and affected tests (28 passed, eight optional/external-fixture cases skipped under system Python). The pinned SDK ran nine provider tests successfully; the exact 557-muscle test was then run separately with GENERIC_SHOULDER_FIXTURE_ROOT and passed, giving ten distinct provider tests with actual native boundary evidence. The portable acceptance/pin/inertia selection passed 39 tests. No prior private capture acceptance is inferred.

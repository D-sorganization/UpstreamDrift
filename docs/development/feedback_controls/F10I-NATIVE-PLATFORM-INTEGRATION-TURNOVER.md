# Native Platform Integration Turnover — F10i #12060

## Implementation and Preserved Authorities

The owned `feat/f10i-native-platform-integration-12060` branch integrates published
guarded controls head `ad2cefcbe3b4f2e23dde96e04b39607ddfe379f8` and native platform
head `f5554f91b46a27f5625f85268ed668b27f3293ef`. Existing Moco preparation, numerical
seed, solve/export and independent muscle replay now coexist with six-engine
replay entry points and guarded native MyoSuite command optimization. No second
optimizer or replay-contract implementation is introduced. Keep the admitted
Tools pin `86d0f28b1cc5acf61185e07e320c816c2d005512`, explicit private contract
loader, concrete muscle-law restrictions and pre-execution integrity guards.

Registry conflicts were merged by blocker ID, preserving independent additions
and scientific exclusions. Generated views are regenerated from current inputs.
Reference conventions moves from filename prefix 38 to 40; chapter 38 remains
the project MyoSuite task reference. Historical evidence identities remain intact.

## TDD and Actual Runtime Evidence

The original eight-case availability contract failed four cases before integration:
missing Moco preparation/replay, owned Simscape replay, and CLI route. Its guessed
CLI flags were corrected to the existing provider's public `--output-dir` and
`--source-sha256`; no provider API was altered to satisfy the guessed fixture.
Availability proves imports only. The combined actual OpenSim 4.6/MuJoCo 3.8
campaign passes 328 tests with 25 explicit skips and zero
failures/errors. All 4509 declared Python source/test files have matching
before/after hashes. SDK-dependent Drake/Pinocchio skips are not native evidence;
Python Simscape protocol checks are not MATLAB R2025b execution.

Broad initial selection: 495 passed, 10 failed, 27 skipped. One actual integration
regression required a test correction: 17-digit TRC export can have zero rounding
error. The corrected test checks nonnegative bounded error, original time, observed
positions and a genuinely missing sample after independent readback. The other
failures were missing optional C3D runtime and an intentionally opt-in absent
Rajagopal fixture. Installing ezc3d revealed a same-process Windows DLL collision
with OpenSim's bundled ezc3d.dll. The subsequent broad attempt retained eight
capture-load failures; its environment changed during execution and it is not
accepted as stable runtime attestation. Final qualified checks use a frozen
environment and the explicit affected test selection. The SDK environment retains
ezc3d 1.6.3; no source loader bypass, fabricated capture, SDK reinstall or relaxed
physical criterion was used. Capture preparation must remain isolated from native
execution until compatible DLL loading is established.

The prior F09p actual MyoSuite 3.0/MuJoCo 3.6 176-test receipt remains historical:
all 21 declared executed source/test files are byte-identical after integration.
This comparison does not attest additional transitive modules or a fresh 3.6 run.

## Reproduction and Evidence Location

Use the existing Python 3.12 OpenSim environment with `CASADIPATH` pointing to its
`Lib/site-packages/opensim`; retain the admitted Tools path and repository root in
`PYTHONPATH`. Run `pytest --noconftest -o addopts=''` on the integration contract,
native Moco runner/guess/initial-binding/muscle bundle/passive/reference/replay/
constraint/marker suites, TRC and pelvis contracts, four native engine replay
protocol modules and the pinned/private loader tests. The exact selected test
modules are represented in retained JUnit. Runtime files live in the fleet
workspace `docs/development/feedback_controls_planning`: `f10i-platform-red.xml`,
`f10i-native-opensim-integration.xml`, broad failed receipts,
`f10i-native-qualified-campaign.xml`, before/after source manifests and
`f10i-native-integration-receipt.json`. They contain no new private motion arrays.

## Scientific Scope and Next Work

Keep Astra's qualification firewall and full seventeen-model/six-ecosystem
denominator. Full private-reference muscle-driven OpenSim matching, complete-state
independent excitation replay, anatomy, passive-load readiness, native contact/grip
and assistance accounting remain open. Source preparation must retain registration,
marker coverage, passive, assistance, constraint and complete bounded state blockers
before Moco construction. Numerical seeds are not observed motion. Integration,
native fixture success, optimizer convergence and animations do not close F10.
Reuse existing SDKs/checkouts; retain Desktop previews with their current explicit
nominal-viability labels. No new fit video or qualified private match is claimed.

## Publication Gates and Baseline Diagnostics

Normal pre-push Ruff, policy, title/manual, Prettier, mypy, Bandit and configured
unit-test hooks pass. The first mypy attempt failed with a stale-cache NumPy
cross-reference assertion; the full hook passed with a fresh version-specific
`MYPY_CACHE_DIR`. The existing cache was retained. No validation hook was skipped.
The runtime manifest corresponds exactly to 4,507 Git blobs. Two unchanged loader
test files have CRLF runtime bytes versus LF Git blobs; their code is identical
after newline normalization, with both hashes retained explicitly. All 4,509
runtime file bytes remained unchanged before/after the accepted campaign.

The central helper's generic `cli.py` test mapping also selects unrelated CLI
tests. Its initial broad test run returned 332 pass / 3 fail / 118 skip; two
motion-pipeline help checks pass with `PYTHONUTF8=1`, while the unchanged shot
optimizer test still uses an obsolete invocation. A separate UTF-8 diagnostic
reproduces one failure and four passes. Source and tests for that optimizer are
unchanged from the guarded-controls parent. Track that baseline in #12064;
the OpenSim/C3D mixed-process DLL issue is #12065. The initial central helper also
hit a Windows output-decoding failure after its affected-test run; it is not an
all-green central-gate receipt. Preserve these failures alongside the passing
actual native campaign and normal hooks.

The final availability contract uses explicit public imports; the dynamic-import
form triggered Semgrep's non-literal-import rule despite its fixed parametrization.
No exemption or suppression was added. The revised test passes all eight cases in
the SDK-free Python 3.13 process and the frozen OpenSim native campaign again
passes 328 tests with 25 explicit skips. Retain `f10i-native-static-final.xml` and
`f10i-static-source-before.json`/`f10i-static-source-after.json`: all 4,509 hashes
match within this final execution; the only changed Python file relative to the
earlier campaign is the availability test.

The final central affected-test selection returns 334 pass / 1 fail / 118 skip.
Its remaining failure is the unchanged shot-optimizer baseline #12064. The two
motion-pipeline subprocess failures were missing `src.shared` in their inherited
import path; explicitly retaining the repository root beside the fleet helper in
`PYTHONPATH` resolves them. UTF-8 also avoids the helper's prior output-decoding
failure. Keep the initial failures and the corrected-environment evidence; the
central helper remains non-green because #12064 is unresolved.

## Current Main Integration

Merge current `main` `26316150998ede86d08b59f1686d8b80ff9f8bdf` after GitHub
reported a conflicting base and therefore withheld ordinary PR workflows.
Preserve both independent handoff additions and regenerate divergence views.
Production changes inherited from main include Pinocchio's public velocity/body
placement accessors, shared bushing-law/prescribed-motion exports, and per-engine
grip adapters. They do not replace the guarded controls or native Moco path.

Repeat the affected native campaign on this integrated source and retain
`f10i-main-native-final.xml` with its new source manifests. The separate actual
OpenSim 4.6/MuJoCo 3.8 grip suite passes 77 tests with 12 explicit SDK skips;
record `f10i-main-grip-native.xml`. Canonical chapter 11 now records its bushing
equations, frames, rotational-rate mapping and prescribed-motion scope. These
checks do not promote full-body muscular own-contact or private-fit acceptance.
UI/documentation changes inherited from main are not new F10i feature claims.

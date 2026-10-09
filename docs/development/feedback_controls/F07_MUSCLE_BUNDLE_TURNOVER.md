# Native Muscle Bundle Turnover

## Current Checkpoint

Issue #11908 is leased by root/codex. Branch
`feat/feedback-opensim-bundle-11908` is based on native Pinocchio PR #11906 and
therefore inherits native Drake #11896 and MuJoCo #11836 dependencies. No PR
has been published for this child. RED commit `a8e63b4cda` preceded implementation.
Eight actual OpenSim4.6 tests pass, including physical units and execution
from owned frozen source bytes. All five central pre-PR gates, scoped mypy,
DRY, LoD, architecture, canonical governance and generated context pass.
Central mypy uses MYPYPATH equal to the repository root: its default dual
src/root path assigns two module names to the same OpenSim facade; the single
canonical package root passes without suppressions. Eight default-Python native
test skips are excluded from physics evidence.

## Native Reproduction

Use the root-owned Python3.12 OpenSim environment with worktree root,
`src/shared/python`, and `vendor/ud-tools/src` on `PYTHONPATH`. Run
`python -m pytest --noconftest -o addopts= -q tests/opensim/test_native_muscle_bundle.py`.
The existing fixture is imported as a pytest plugin to reuse its actual native
compliant-muscle model. Original RED JUnit lives in the fleet planning staging
folder. Do not run native evidence under the global mock-provider conftest.

## Ownership and Next Work

The adapter consumes Tools T01 and the existing native muscle kernel. It
records registered discrete/modeling-option path/value sets and muscle flags;
continuous `Y` coverage is insufficient to claim arbitrary complete State.
The component subset is deliberately narrow. Source review does not prove
binary equivalence. Add negative capability, input/policy, native clamp and
runtime cases, initialization derivative/force checks, law coverage only after
the exact-law dependency is integrated, and canonical governance gates before
publication. Owned immutable source snapshots and the returned model digest bind actual replay.
The source-replacement regression verifies that changes to the caller file after
admission do not change executed bytes.

Keep native geometry #11903 and FK #11907 separate from replay evidence. Do
not promote the 520-muscle source while locks, couplers, assistance, anatomy,
passive force, world registration, contact or private evaluation gates remain.
No capture matching or production parity is established here.

## Storage

This checkout borrows primary Git objects. Its sparse Tools checkout borrows
the retained primary `UpstreamDrift-10614-co10` submodule object store at exact
T01 commit `2e7665111b06f92ffbfe178b92d74d6a81c95388`. Audit alternates before
retiring either checkout. No private capture files or videos were copied.

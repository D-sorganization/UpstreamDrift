# F07 Frozen Marker Holdout Turnover

Issue #11899 delivers a pure scoring boundary using the existing TourCapture,
marker placements and body-pose contracts. It does not close F07/F08 or freeze
the private D02 protocol. The historical exploratory holdout calibration API
continues to refit withheld offsets and is explicitly documented as such.

## Evidence

TDD first failed because the independent scorer was absent. The regression
shows a systematic 0.2 m withheld displacement erased by nuisance refitting
and retained by frozen scoring. Invalid transforms, missing support/placements,
masking, clock preservation and pooled Euclidean RMS are covered. No private
capture or native model is needed for these original synthetic tests.

Run `python3 -m pytest -o addopts='' tests/opensim/test_frozen_marker_scoring.py tests/opensim/test_marker_calibration.py`.

Validation: the scoring/calibration plus design-manual contract run passed 29
tests. Central `pre_pr.py --base-ref origin/main` passed all five gates; its
affected-test selection passed 24 tests and skipped two existing Pinocchio/Drake
provider checks unavailable in that interpreter. No native OpenSim acceptance
is inferred from this engine-independent scoring change.

## Remaining Gates

Inputs alone cannot prove training-only provenance. Freeze source/model hashes,
anthropometry, marker attachments and offsets, preprocessing, noise, event,
alignment and split policies before independent evaluation. Preserve actual
subject/trial/source grouping and native-grid versus observation-grid identity.
No inferred anatomical body aliases, capture qualification or model acceptance
is supplied. The full muscle/contact/grip/coupler and replay program remains open.

Canonical authority: chapter 13, Feedback-Control Comparison Admission;
registry inventory and generated publication remain blocked.

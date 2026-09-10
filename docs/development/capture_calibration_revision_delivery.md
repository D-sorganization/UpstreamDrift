# Calibration Revision Status Delivery

Tracked by calibration epic #9899 and the guided capture workflow. This branch
builds on camera setup PR #9954; both changes must reach protected main before
the combined product is considered delivered.

Delivery PR: #9959. The successor starts with
[Capture Product Turnover](capture_product_turnover.md). All 26 focused controls
pass; the extracted-helper/model subset passes six. CI architecture and document
size failures were reproduced and corrected without widening either budget.

## Player Behavior

A reconstruction records the SHA-256 of the selected camera calibration source.
The capture wizard compares that evidence with the calibration the player has
currently reviewed. A matching result remains complete. A changed calibration or
legacy result without that evidence requires **Reconstruct Again**. Model steps
inherit that prerequisite through the existing workflow planner and its links.
The status check does not delete or rewrite previous results.

After a new reconstruction is available, triangulated model fits are checked
against the reconstruction-summary fingerprint already recorded by the model
writer. An old fit remains stale until refitted. The check reads only the known
capture-relative summary; it never follows arbitrary paths from result metadata.
Image-space fits retain their separate observation/camera input contract.

The command captures source identity before loading cameras and lens correction,
then verifies it before reconstruction and before publishing the summary. A
missing or changed source produces an actionable error. Direct API callers may
still supply camera objects without a file; those results have no file-based
calibration association and are not silently treated as reviewed capture output.

## Evidence and Remaining Work

Missing-module and missing-function tests failed before implementation. The
reconstruction command test verifies the persisted fingerprint against the actual
camera source. Tests also cover same-path changes, missing source files, legacy
results, matching revisions and real wizard refresh after a new review of changed
source bytes. Existing wizard evidence and reconstruction pipeline tests pass.

The calibrated source fingerprint establishes lineage, not physical accuracy.
Optical zoom still requires the matching lens profile and fresh player review.
The numerical ruler-scale provider remains Tools PR #5169; its worker/editor
integration and physical-camera qualification remain separate acceptance items.
The installed candidate `7c7950574` is frozen and must not be modified by this
branch's source tests or dependency installation.

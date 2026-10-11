# F01c Initial-State Replay Binding Turnover (#11854)

The F01 replay comparison contract now declares `feedback-comparison/1.1.0` and
requires `initial_state_sha256` for independent replay admission. The caller
copies this digest from the Tools T01
`ExperimentReplayBundle.integrity.initial_state_sha256`; F01 validates its
shape and rejects a mismatched digest across a same-input comparison. This is
an additive evidence extension: an unversioned receipt can still document
identity or transcription, but cannot claim independently replayed dynamics.

The negative tests first failed because `ComparisonEvidence` lacked the field.
They now reject missing/malformed payload digests, old versions, and different
initial physical states under a common state schema, while preserving the
non-replay admission levels. Run
`python -m pytest tests/unit/engines/test_feedback_comparison.py -q --no-cov`.
The registry and design manual remain blocked for scientific release; no native
trajectory, contact, physiological or real-capture accuracy is asserted here.

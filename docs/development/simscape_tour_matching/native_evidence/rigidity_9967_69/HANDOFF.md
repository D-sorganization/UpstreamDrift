# Full-Capture Rigidity Floor

The actual25-marker assignment places HeadTop/HeadFront/HeadSide on Hub together
with back markers. The native specification has no independent head joint.
Using existing rigid_attachment_residuals with fixed candidate62 offsets and
all654 capture frames gives a17.393 mm whole-swing RMS lower bound and24.279 mm
terminal lower bound. At0.85 s the terminal bound is23.677 mm, versus run62's
actual66.713 mm. Thus rigidity does not explain all remaining forward error.

A diagnostic allowing an independent six-DOF head pose lowers this relaxation
to4.471 mm whole/9.177 mm terminal. This is a changed physical assumption, not
an implemented neck model or a permitted way to pass the original acceptance.
No markers, weights, offsets, constraints or engine models were changed.
The original25 mm whole/35 mm terminal gates therefore have limited headroom
under current attachment assumptions; rotations/quaternions cannot remove it.

Reproduce from repository root:
`python -m docs.development.simscape_tour_matching.native_evidence.rigidity_9967_69.audit_rigidity`.
Report records source/provider/input hashes, marker count, full clock, per-body
and prefix metrics. It reads the actual raw capture from archived run63 and the
convenient candidate62. Missing observations retain their masks. Every body pose
is independently relaxed per frame; connectivity, dynamics and continuity are
ignored. Offsets or assignment recalibration would change this conditional bound.

Keep original-equivalent engine work intact. Any future articulated head variant
requires explicit physical-model identity, same variant in all engines and
separate acceptance; do not silently drop head markers or relabel them.

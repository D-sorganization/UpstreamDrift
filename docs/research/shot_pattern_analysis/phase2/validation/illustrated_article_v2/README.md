# Illustrated Article Presentation Revision

## Scope

The October 11, 2026 user-requested revision puts right-positive lateral
position on the horizontal plotting axis and downrange distance on the
vertical plotting axis. The simulation coordinate frame and all saved numerical
results are unchanged. The original published edition and its verification
receipts remain historical evidence; they do not certify this revised edition.

The editable source remains `../../../shot_pattern_analysis.tex` relative
to this directory. New plots are derived from saved shots and flight samples
under `../../presentation_v2/`; no flight or impact simulation is rerun.
The report also expands the delivery, two-body impact, flight, and scoring
explanation and adds a graphical findings summary.

## Project Acceptance Boundary

The research analysis and artifact delivery are substantially complete within
the stated geometric simulation scope. Repository integration is not complete:
PR #12278 was verified open and blocked at head
`7287f6b25523c7900a164114973174f25bee36cc` during this revision.
The independent validation follow-up saved local repairs through `e426bac0dd`,
with 168 focused tests passing, but its canonical delta gate still reported
1,501 passes and 14 failures plus development-log policy failure.
This presentation revision does not clear those gates.

Empirical model qualification (#11984) remains deferred and is distinct from
the requested geometric study. The prior built-in editor compiler failure
(#12120) is separate from successful local CLI PDF generation. Current revision
compilation and visual-review results belong in `review.json` alongside this
file, when available.

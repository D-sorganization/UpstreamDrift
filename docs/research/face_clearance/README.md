# Static Finite-Face Clearance

`motion_matching.face_clearance` computes the instantaneous distance from a
sphere centre to the nearest point of a finite triangle patch, minus sphere
radius. A separate guard rejects zero, negative, or insufficient initial gaps.
The caller must supply face triangles and the centre in the same frame, in
metres, and choose any required positive safety margin from justified evidence.

This is an unsigned point-to-surface distance, not a signed solid distance.
Positive gap does not establish that the sphere is outside a closed clubhead,
on the front side, approaching, or that a first contact root has been found.
No interval coverage or physiological feasibility is claimed. Initial overlap
and unsearched trajectory intervals stay unqualified. The geometry is the
rendered-face assumption, not a measured face.

Run `python -m pytest -q tests/unit/motion_matching/test_face_clearance_15493.py`.
See the editable LaTeX reference for equations and numerical restrictions.

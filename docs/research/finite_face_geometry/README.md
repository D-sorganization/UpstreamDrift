# Finite Face Geometry

This experimental geometry interface exposes the existing rendered clubface
patch in a requested model frame. It is not a contact detector: it cannot
certify initial separation, a first root, impact, or physiological feasibility.
An initial overlap or unsearched interval remains unqualified and must not be
turned into an accepted contact. The rendered mesh and face mask are modeling
choices, not measured clubface certification. See the editable LaTeX reference
for the selection and coordinate transform.

Run the headless regression with `python -m pytest -q
tests/unit/motion_matching/test_v2_finite_face_geometry_15493.py`.

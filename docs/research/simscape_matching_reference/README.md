# Simscape Matching Reference

Start with the [standalone LaTeX reference](simscape_matching_reference.tex). It
documents the model ladder, coordinate frames, joint conventions, capture
processing, geometry and inertia assumptions, inverse kinematics, controllers,
contact diagnostics, qualification gates and reproducible exports.

The [refinement record](MATCHING_REFINEMENT.md) and
[current turnover](../../development/HANDOFF.md) identify tested experiments,
rejected candidates and outstanding work. The engine's
[reference pointer](../../../src/engines/Simscape_Multibody_Models/3D_Golf_Model/matlab/exploratory_gs3dx/docs/REFERENCE.md)
links here so the modeling account has one maintained LaTeX source.

This is a separate research and experiment record. The canonical engineering
design manual remains governed by the QMD sources under
[manuals/upstreamdrift](../../../manuals/upstreamdrift); its generated artifacts
must not be edited directly. The repository's
[modeling documentation policy](../../../AGENTS.md) requires source references,
equations, assumptions, units, reproduction steps and evidence whenever the
modeling process changes.

Capture A is the public tour reference; capture O is the owner evaluation.
Avatar distortions cannot establish playing ability. Current exports use IK
poses and are labeled **IK / DYNAMICS UNQUALIFIED**. Historical feedback-assisted
simulations and new IK exports must be identified separately. Issues #11156,
#11160 and #11173 remain open scientific gates; successful software execution
does not establish physical qualification.

Future shareable matches use the Human ellipsoid model with capture-specific
longitudinal geometry and calibrated foot orientation. Artistic transverse
proportions, incomplete head orientation constraints and subject inertia
personalization remain limitations. Position RMS excludes the foot orientation
term, which is reported separately. Flat soles at address are a calibration
assumption, not measured ground contact.

The reviewed baseline is `144e81188dd7bb106f81d89b7a7330dd20cce511`.
Execution provenance must identify the actual capture and model hashes, all
solve/render dependencies, fitted geometry, sampling and native receipt. A
scratch mirror may report an unknown Git commit; exact file hashes must then
identify its executed inputs. Public records use capture aliases and hashes;
raw private captures, absolute private paths and pose caches remain outside Git.

The `.tex` source is standalone and can be opened in Codex's LaTeX editor or
compiled with an existing LaTeX installation:

```sh
pdflatex -interaction=nonstopmode -halt-on-error simscape_matching_reference.tex
pdflatex -interaction=nonstopmode -halt-on-error simscape_matching_reference.tex
```

Structural documentation checks:

```sh
python3 -m pytest -q docs/research/simscape_matching_reference/test_simscape_matching_reference.py
```

These checks verify source presence, standalone structure, the governance
notice and obvious private-path leakage. They do not validate equations or
physical claims. Native Simscape probes, mathematical tests, compiled PDF
inspection and complete video decoding provide distinct evidence.

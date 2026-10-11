# Simscape Matching Reference

The editable LaTeX research reference is
[simscape_matching_reference.tex](../../../../../../../docs/research/simscape_matching_reference/simscape_matching_reference.tex).
This is a separate research and experiment record, not the engineering design
manual. The canonical manual remains `manuals/upstreamdrift` (QMD).

The reference covers topology, frames and units, marker processing, geometry
calibration, IK, torque mapping, ILC, foot-pose feedback, contact diagnostics,
acceptance gates, limitations, and reproducible execution.
Maintain those equations in the LaTeX source rather than duplicating them here.

Reviewed historical source: `144e81188`. Current `GS3DX_Human` remains
unqualified: 365 BW and 22.6 BW contact loads reject physical qualification.
Controller conflict is a hypothesis, not a demonstrated causal explanation.
Issues #11156, #11160, and #11173 remain open; a qualified open-loop full swing
has not been established.

The club metric requires explicit `phases.contact` and face error at most
2 degrees. Peak club speed is a distinct event. A video export does not pass
that gate or establish anatomical or dynamic admissibility.

Use `tools/run_matlab_locked.ps1` for native jobs. Record capture/model/source
hashes, fitted workspace geometry, sampled coverage, physical frame timing,
and process receipts. A completion marker does not prove process success.
IK visualizations, feedback-assisted dynamics, and open-loop dynamics must
retain distinct labels in the video and provenance.

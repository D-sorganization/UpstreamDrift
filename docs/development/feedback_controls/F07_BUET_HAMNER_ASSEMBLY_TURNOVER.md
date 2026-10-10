# BUET–Hamner Native Assembly Turnover

Issue #12157 is a source-preserving OpenSim mechanical candidate. Branch
`feat/f07-buet-hamner-assembly-12157` is stacked on zero-MTP PR #12153 to
reuse its finite native mass, body-velocity, constraint and actual
$J_{\mathrm{system}}^\mathsf{T}F_{\mathrm{body}}+f_{\mathrm{mobility}}$
observers. The exact BUET/Hamner source artifacts remain outside this public
change. `native_buet_hamner_assembly.py` accepts only their reviewed XML
SHA-256 values, verifies the common pelvis/femur properties and hip
SpatialTransforms, clones 8 distal bodies/joints and 84 lower muscles onto the
BUET model, then serializes and independently reloads the result. The output
model is a derived diagnostic, not a replacement for either donor.

The [native receipt](F07_BUET_HAMNER_ASSEMBLY_RECEIPT.json) binds BUET
`b66a31e0…e79bf6`, Hamner `349ce7a4…687cf`, derived XML
`8d4d3497…1d34a3`, adapter `7a2050d6…281209f`, OpenSim 4.6 build and three
loaded extension binaries. The 2,273,988-byte derived XML is retained in
owned planning staging, not committed. Fresh load has 40 bodies and 557
muscles, including all 473 BUET muscles and exact 84 retained Hamner muscles.
Native component-property dumps match source for retained bodies, joints,
muscles and BUET constraints. At three declared hip poses and nonzero speeds,
all sampled body transform/velocity and muscle length/speed differences were
zero. Native QErr/UErr were zero. Minimum sampled mass eigenvalue is
`1.3362969448668327e-06`; observed total applied generalized-force norm is
`54563.69711509691`, both in mixed native units. These numerical values
carry no physiological admission threshold.

Qualified replay of the calculation, with read-only local donor paths:

```powershell
$env:CASADIPATH='C:/Users/diete/Repositories/.venv-feedback-opensim/Lib/site-packages/opensim'
$env:PYTHONPATH=(Get-Location).Path
$env:UD_BUET_SOURCE='C:/Users/diete/Repositories/docs/development/feedback_controls_planning/buet_raw_ranges_v1/source/Bilateral_Upper_Extremity_Trunk_Model_Markered.osim'
$env:UD_HAMNER_SOURCE='C:/Users/diete/Repositories/UpstreamDrift/shared/models/opensim/opensim-models/Models/Hamner/FullBodyModel_Hamner2010_v2_0.osim'
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m pytest tests/opensim/test_native_buet_hamner_assembly.py -q
& 'C:/Users/diete/Repositories/.venv-feedback-opensim/Scripts/python.exe' -m scripts.opensim.emit_buet_hamner_assembly_receipt --buet-source $env:UD_BUET_SOURCE --buet-sha256 b66a31e08087327f8756d587f471b0494a3d9d6f7f6d510de66beba4a9e79bf6 --hamner-source $env:UD_HAMNER_SOURCE --hamner-sha256 349ce7a44f1541794f2283cab73bb954270228f2f50808de549ec5097a1687cf --derived-output <fresh-owned-staging>/derived.osim --receipt-output <fresh-owned-staging>/receipt.json
```

The emitter requires both output paths to be fresh. Existing output and
changed/unreviewed source bytes fail closed. OpenSim emitted missing visual
mesh warnings; the entrypoint XML hashes do not prove resource closure. A
source ownership match does not establish anatomy, active-wrap behavior,
passive-force suitability, physiological muscle work, contact/grip, full
native-state replay, native Moco convergence or capture tracking. BUET's
partially locked Abdjnt coordinates require a separate explicit derived
reduction before that model class is ready for native Moco. The Rajagopal
zero-MTP reduction is another separate policy, not applied by this assembler.

TDD sequence: the native test first failed on missing module. The
source-version negative then failed because updating a changed XML digest
would have admitted it; version-one exact pins corrected that. New velocity,
path-speed, mass/force/constraint and native-binary receipt assertions each
failed before their observations were added. The final suite passes three
actual OpenSim cases plus four portable contract cases. Chapter 44 is the
calculation authority; the registry stays blocked with no approved
calculation entries. No public source assets, golfer markers or captures were
added.

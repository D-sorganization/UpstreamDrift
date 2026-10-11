# Drake Candidate-Lift Zero-Effort Observation

This separate attempt folder preserves one source-bound Drake zero-effort
acceleration observation for the manufactured candidate shared with the
MuJoCo and Pinocchio diagnostics. `artifact-manifest.json` binds each archived
receipt, log, metadata file, and inert observer snapshot to its original path
and SHA-256. The 52-entry archive at
`../manufactured_native_diagnostics_20261011/` and its manifest are unchanged.

The run used the source-only UpstreamDrift archive at commit
`9e5f87c4dd31fc3137fc204d98337633fa5c4855`, manifest SHA-256
`c5ec40129d3b28368142eadfb33e8e1464f7c6e4d4505dd5a6d2f86870d4f569`, and the
single reviewed lazy-facade overlay at commit
`92738d3cab3865bf727ba96a9cca9388d52ef955`. The exact model spec SHA-256 is
`09cff1393a48b7f44e2ceaac9132473c64a33f9f853823c5e7d7fef80802e6b3`; the
external candidate SHA-256 is
`2caa3ce94e596dd61d50172229ad04a98c5235be6388566b4376be4ba41e2fcf`, bound
to the MuJoCo geometry receipt SHA-256
`9986c0791134234f3d3b4e19b2784b48d3758965f1f5fbac23261211f5922ac3`.

The run used Python 3.12.12 and Drake 1.57.0 from the archive's `.venv312`.
Its exact 41-coordinate q was bit-identical to the retained Pinocchio and
MuJoCo inputs; v and generalized effort were exactly zero for all coordinates
(six root and 35 actuated channels). Before dynamics, the kinematic observer
recorded a finite 6×41 weld Jacobian of rank 6, condition number
`8.88008731610288`, and maximum pose residual `5.54899034626804e-13`. Four
shared sphere samples were finite. The near-floor right forefoot sample had a
normal force of `1.1275702593849246e-11 N`; the other three samples had zero
normal force. This is a near-touch, not supported equilibrium.

Exactly one native adapter acceleration call returned a finite 41-coordinate
qdd vector. Total model mass reported by the adapter was
`108.32071866402505 kg`. Drake's root translation-coordinate accelerations
were `-4.903325000000071` and `-8.49280802602257` in the source generalized
coordinate basis per second squared. The output is not reinterpreted as a
different engine's spatial/world acceleration basis. Descriptive maximum
absolute qdd differences were `8.819362580158242e-11` versus the retained
Pinocchio vector and `2.369282030043434e-10` versus MuJoCo, both at
`mtp_angle_r`. These numbers have no acceptance threshold and do not claim
parity or qualification. The contact state is consistent with a manufactured
free-fall observation, not a supported pose or physical equilibrium.

The bounded run completed in 3 seconds with sampled peak process-group RSS of
`306648 KiB` and exit code 0. The source and input hashes were unchanged. The
supervisor recorded no cleanup action; an independent post-run process-group
query found no remaining process. Raw receipt, metadata, log, runner snapshot,
and pure-test snapshot are preserved byte-for-byte.

The observation is diagnostic only. It does not establish calibrated contact,
anatomical fidelity, physical qualification, cross-engine parity, or any of the
17 production model/variant rows. All 17 remain unqualified. No private capture
or runtime binary is included.

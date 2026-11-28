# WP-0 F0 Reference Record

This record preserves the reference values stated in [Epic #15805](https://github.com/D-sorganization/UpstreamDrift/issues/15805).
It does not mark WP-0 complete: the issue's raw summary JSON and rendered MP4
were not present in this checkout or in the local `Videos`, `OneDrive`,
UpstreamDrift worktree, or Runner Dashboard session directories.

The issue identifies source revision `ffa19cf8ccc36b3ae1b492ef32b123ecf14e557a`,
`upstream-physics` 2.1.3 revision 3, preset `acushnet_example_1`,
`default_sand`, and a 25.0 m/s swing. Its reported summary values are copied
into `wp0_f0_reference.json` with `summary_evidence` set to
`issue_report_only`; the source did not provide the original JSON bytes or a
digest for them.

The local headless replay used Python 3.14.3 and installed
`upstream-physics` 2.1.3 revision 0. Strict Rust parity refused a carry result.
Its force and impulse differ from the issue's record, so those local values
are retained only as diagnostic evidence and are not substituted for the
reference summary.

The source renderer currently defaults to the Down-the-Line camera preset
(12 degree elevation, 180 degree azimuth). Those code defaults do not establish
the camera used for the missing MP4. WP-0 remains blocked until the exact
original MP4 and raw summary are supplied or recovered, with their camera
extrinsics, view bounds, file digest, and container metadata.

The JSON record and test deliberately fail closed: missing render evidence
cannot be represented as a completed baseline freeze.

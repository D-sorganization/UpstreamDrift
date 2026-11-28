# Additional Golfer Reference Footage

Epic #15894 uses the linked [acquisition receipt](golfer-expansion-acquisition-2026-11-27.json) for O’Grady, McIlroy, Rahm, Johnson, García, Watson, 1986 Greg Norman, Moe Norman, Knudson, Schlee, Miller and Trevino. Video files are in private non-Git storage. Sixteen receipt rows cover fifteen unique files, including some short excerpts; two rows share the USGA 1973 highlight.

Each retained entry records its source URL, claimed uploader, relative filename, SHA-256, stream metadata and provenance limits. Media hashes and sizes were independently rechecked. Source rights and permission to redistribute have not been established. Uploader attribution is not equivalent to independently verified identity or capture date: Schlee is not visually confirmed in the sampled 1973 USGA highlights, and his separate instructional clip relies on uploader attribution. The claimed 1986 O’Grady upload is low resolution and its date is unverified. Modern O’Grady footage is a separate reference.

The `ffprobe` frame rates describe the encoded rendition; archival edits, slow motion, repeats and variable playback mean they do not establish a physical capture clock. Clips and visual observations are references for appearance and future analysis, not fitted swing motion, synchronized marker captures or qualified training data. Check the underlying rights and timing before later use.

For a privately acquired file, independently recheck its receipt with `sha256sum FILE` and inspect stream/timing information with `ffprobe -show_streams -show_format FILE`. Reacquisition depends on changing source availability and does not guarantee identical formats or hashes.

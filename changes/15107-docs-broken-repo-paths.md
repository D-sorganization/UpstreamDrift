# Documentation Broken Repo Paths (#15107)

- Fixed relocated/missing repository file and directory citations across testing guides (`docs/testing/`), `docs/user_guide/mesh_backends_setup.md`, `docs/README.md`, `docs/architecture/C4.md`, `docs/user_guide/body_part_viz/quickstart.md`, and the user manual chapters/appendices.
- Extended `scripts/check_spec_paths.py` (`check_doc_paths`) and `tests/scripts/test_check_spec_paths.py` to fail CI if curated testing, architecture, or user-guide documentation cites missing bare directories or non-`.py` repo paths.

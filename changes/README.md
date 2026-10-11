# Change Fragments

Each pull request adds its own `changes/<issue>-<slug>.md` instead of editing
the `SPEC.md` change log, `docs/development/DEVELOPMENT_LOG.md` or the handoff
directly, so concurrently queued pull requests never conflict over those files
([Repository_Management#1894](https://github.com/D-sorganization/Repository_Management/issues/1894),
rolled out here by
[Repository_Management#1976](https://github.com/D-sorganization/Repository_Management/issues/1976)).

```bash
python3 shared_scripts/changes_fragment.py new --issue N --summary "One line"
# live work also records the development-log state:
python3 shared_scripts/changes_fragment.py new --issue N --summary "One line" \
  --dl-state in_review --next-step "Merge the PR." --branch feat/x
python3 shared_scripts/changes_fragment.py validate
```

A valid fragment satisfies the SPEC.md freshness check
(`scripts/ci/check_spec_freshness.py`). After merge, `collate-changes.yml`
writes the SPEC.md row keyed by the pull request, updates the `DL-#<issue>`
entry in place and deletes the fragment.

The fragment tooling in `shared_scripts/` is vendored byte-identical from
Repository_Management; `docs/development/change-fragment-bundle.json` pins
the digests. Change it upstream, never here.

This README keeps the directory tracked between collations and is never
treated as a fragment.

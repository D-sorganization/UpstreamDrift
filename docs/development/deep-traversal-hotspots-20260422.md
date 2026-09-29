# Deep Traversal Hotspot Review - 2026-04-22

Issue: https://github.com/D-sorganization/UpstreamDrift/issues/2958

## Summary

The generated assessment flagged repeated three-hop member chains in the Sphinx
static documentation bundle and in `build_hooks.py`. The review found no
first-party runtime Law-of-Demeter violation that should be refactored in this
slice.

## Findings

| Path                                                          | Disposition    | Reason                                                                                                                                                                                                          |
| ------------------------------------------------------------- | -------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `docs/sphinx/_static/_sphinx_javascript_frameworks_compat.js` | Justified      | Sphinx-generated compatibility JavaScript; already documented in `docs/sphinx/_static/VENDORED.md`.                                                                                                             |
| `docs/sphinx/_static/jquery.js`                               | Justified      | Vendored jQuery distribution; editing would fork upstream library behavior.                                                                                                                                     |
| `docs/sphinx/_static/js/badge_only.js`                        | Justified      | Vendored Read the Docs theme asset; upgrade through theme regeneration.                                                                                                                                         |
| `docs/sphinx/_static/js/theme.js`                             | Justified      | Vendored Read the Docs theme asset; upgrade through theme regeneration.                                                                                                                                         |
| `docs/sphinx/_static/js/versions.js`                          | Justified      | Vendored Read the Docs version flyout asset; upgrade through theme regeneration.                                                                                                                                |
| `docs/sphinx/_static/searchtools.js`                          | Justified      | Sphinx-generated search implementation; upgrade through Sphinx regeneration.                                                                                                                                    |
| `docs/sphinx/_static/sphinx_highlight.js`                     | Justified      | Sphinx-generated highlighting implementation; upgrade through Sphinx regeneration.                                                                                                                              |
| `build_hooks.py`                                              | False positive | Matches occur in documentation text and the `hatchling.builders.hooks.plugin.interface` import path, not in runtime object-graph traversal. Runtime access stays within the build hook object and direct paths. |

## Boundary Decision

No code boundary extraction is needed for this issue. The first-party
`UIBuildHook` methods expose focused wrapper properties for `ui/` and
`ui/dist/`, keep subprocess behavior inside `_run_npm_build`, and convert
subprocess exceptions at the build-hook boundary.

Future work should treat findings under `docs/sphinx/_static/` as vendored
asset maintenance unless a local patch is explicitly required. In that case,
regenerate the Sphinx bundle from pinned dependencies instead of editing the
generated files directly.

## Validation

- Reviewed the flagged files from issue #2958.
- Confirmed vendored asset policy exists in `docs/sphinx/_static/VENDORED.md`.
- No behavioral tests were added because this is a documentation and audit
  justification change only.

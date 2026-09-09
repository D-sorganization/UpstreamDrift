# Reviewed Renderer Compatibility — #9783

Tools PR #5090 passes production Linux visual workflow `34217794994`.
The downstream job `102039836077` in run `34217794975` instead fails the
UpstreamDrift analysis-policy contract: it hard-codes the old PyQt variation
reference hash. Thirteen other contracts pass. This is independent of the
Rust debounce failure (Tools #5095) and the private Gasification checkout 404.

The focused repair accepts exactly these source-artifact/variation-image pairs:

| Source Artifact Commit                     | Variation PNG SHA-256                                              |
| ------------------------------------------ | ------------------------------------------------------------------ |
| `be71b03676eda7bbfa40c880ded3a3bb7112b868` | `650267b346dab8651b6163d83046ed46cbe604c83c71908cca2b1168e84a78cd` |
| `df4101f2825b3b2d255dad1d6f8746818fc82812` | `22f6640e896e9ea5c740e9db7e3d3201cdf7264f2cf1cff33966b539354f1d40` |

The old pair is present in the existing Tools gitlink `eab74a901a7c8467e1997049a73e2cfd2df74428`.
The candidate pair is published at Tools `b8c6e6013ad9dab6eab0bbfe0096b449feb16abb`.
Its repeat captures and individual image review are recorded in that commit's
`docs/development/rate-pyqt-renderer-4844-reference-review.md`. The ten PyQt
PNGs are byte-identical in two Linux capture attempts. Protected merge is the
provider's approval event; this compatibility test does not supply that approval.

Unknown source commits, unknown hashes and swapped known pairs are refused.
Analysis-policy behavior, configured-provider ownership and all three variation
tolerances remain unchanged: channel threshold 1, mean delta 200 microunits,
changed-pixel fraction 250 microunits. No production module or vendor pin changes.

## Verification and Next Step

The existing test first reproduced the real candidate hash failure locally.
Four new refusal cases then failed on the missing assertion helper before its
implementation. The updated provider file passes all fifteen cases against
both real trees: the initialized immutable vendor pin and the exact candidate
checkout selected through `TOOLS_REPO_PATH`. Existing import deprecation warnings
remain separate from this result. Scope is software compatibility, not a golf
physics or acoustics finding.

The full `tests/shared_contracts` folder also passes all eighteen cases in each
provider mode. Pinned Ruff 0.15.17 reports no lint errors and all 6,681 files
formatted. SPEC duplicate checks and design-manual governance pass; the existing
manual release block remains. An initial run used the environment's Ruff 0.16.4
and reported six unrelated findings; the repository and CI explicitly pin
0.15.17. Normal commit/push hooks and protected CI remain delivery gates. After merge, retry the relevant Tools consumer lane against merged main.
Keep #9700 and all unqualified physical/perceptual gates open.

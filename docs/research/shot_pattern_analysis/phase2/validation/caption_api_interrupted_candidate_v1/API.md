# Selected-Pair Renderer API Specification

## Module
`src.tools.shot_pattern_analysis.matched_supplement_figures`

## Function Signature
```python
def render_selected_pair_figures(
    pair: dict[str, Any],
    output_dir: Path,
    *,
    figure_kinds: frozenset[str],
) -> list[dict[str, Any]]:
```

## Closed Figure Kinds
The set of allowable figure kinds is strictly bounded to the following 7 kinds:
- `density_png`
- `density_pdf`
- `density_svg`
- `overhead_png`
- `overhead_pdf`
- `overhead_svg`
- `overhead_social_png`

## Behavior & Invariants
1. **Rejection of Invalid Kind Sets**:
   - `figure_kinds` must be an instance of `frozenset`.
   - `figure_kinds` must not be empty.
   - Any unknown kind (not in the closed set of 7) raises `ValueError`.
2. **Selective Export & No Untracked Outputs**:
   - Only requested figure formats are rendered and written to disk.
   - No untracked sibling formats (e.g. generating svg when only png is requested) are created.
3. **No Numerical / Resampling Work**:
   - The selected renderer never calls pair statistics, bootstrap calculations, or simulations.
   - It never writes social density figures (the 9 pattern cards) or cell assets.
4. **Metadata & Path Parity**:
   - Ledger entries for generated figures have exact schema, keys, and values matching ordinary `_render_pair_figures`.
   - Path naming, directory layout (`comparisons/`), and SHA-256 calculation remain identical.
5. **Caption Wrapping & Bounding Box Containment**:
   - Full covariance captions wrap within figure width.
   - Positive vertical clearance is maintained between all header lines, legends, and plot subplots.

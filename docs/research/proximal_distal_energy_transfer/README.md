# Proximal-to-Distal Energy Transfer in the Golf Swing

Extended-length scientific article investigating proximal-to-distal (P→D)
energy transfer in the golf swing: where it occurs, how it differs between
skilled and less-skilled players, and whether a player should transfer
energy distally **early** in the downswing or **retain it proximally early
and accelerate the transfer late** — the question motivated by this
repository's ZTCF/ZVCF counterfactual work.

Tracked by epic
[#8373](https://github.com/D-sorganization/UpstreamDrift/issues/8373)
(sub-issues #8374–#8379). Destined for eventual publication in the
`affinedrift` repository (issue
[#8379](https://github.com/D-sorganization/UpstreamDrift/issues/8379)).

## Files

| File | What it is |
| --- | --- |
| [`proximal_distal_energy_transfer.qmd`](proximal_distal_energy_transfer.qmd) | Quarto source (single source of truth) |
| [`references.bib`](references.bib) | Verified BibTeX bibliography (every entry checked against a real publication record; DOIs/stable URLs included) |
| [`proximal_distal_energy_transfer.tex`](proximal_distal_energy_transfer.tex) | LaTeX generated from the Quarto source (`keep-tex: true`) |
| [`proximal_distal_energy_transfer.pdf`](proximal_distal_energy_transfer.pdf) | Rendered PDF (15 pp.) |

## Building

Requires [Quarto](https://quarto.org) and a LaTeX distribution (TinyTeX or
TeX Live with `lmodern`):

```bash
cd docs/research/proximal_distal_energy_transfer
quarto render proximal_distal_energy_transfer.qmd --to pdf
```

The `.tex` file is regenerated on every render; edit the `.qmd`, never the
`.tex`.

## Content honesty

The article is a literature synthesis and modeling perspective. It reports
**no new experimental data**; quantitative claims are sourced to the cited
literature or to inspectable code in this repository. The quantitative
counterfactual experiments it proposes are tracked in issue
[#8377](https://github.com/D-sorganization/UpstreamDrift/issues/8377) and
must be executed before any results are added.

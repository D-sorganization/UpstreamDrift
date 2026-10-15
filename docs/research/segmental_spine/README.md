# Segmental Spine Research Notes

Editable research source: [segmental_spine.tex](segmental_spine.tex)
(compile with `tectonic`; the PDF is not committed). Epic #12735, scaffold
issue #12738. Binding conventions: ADR-0054
(`docs/adr/0054-segmental-spine-kinematics-and-loads.md`). Code lives in
`src/shared/python/spine/`.

## Layout

- `segmental_spine.tex` assembles one subsection per child issue from
  `sections/<key>.tex` (for example `sections/spn-1a.tex`) with
  `\IfFileExists`; a missing file renders as "Pending".
- Each child writes only its own `sections/<key>.tex`, starting with a
  Title Case `\subsection{...}`. The main file is edited only to add sections.
- `sections/.gitkeep` keeps the directory tracked.
- The calculation-level engineering design manual is separate: provisional
  chapter `manuals/upstreamdrift/chapters/66-segmental-spine.qmd`.

## Compile

```bash
cd docs/research/segmental_spine && tectonic segmental_spine.tex
```

Result status: not qualified; no scientific result is claimed here.
